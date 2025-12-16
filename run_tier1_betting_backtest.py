"""
TIER 1 Betting Backtest
=======================
Compare betting performance: Baseline vs TIER 1 features
- Uses 2024 as test, 2025 as validation
- Evaluates spread, totals, and moneyline bets
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import xgboost as xgb
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR
from src.tier1_features import add_all_tier1_features

print("=" * 80)
print("TIER 1 BETTING BACKTEST")
print("=" * 80)

# Load data
games_epa = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")
games_2025 = pd.read_parquet(PROCESSED_DATA_DIR.parent / "2025" / "completed_2025_with_epa.parquet")

# Add TIER 1 features
games_epa = add_all_tier1_features(games_epa)
games_2025 = add_all_tier1_features(games_2025)

# Define feature sets
BASELINE = ['spread_line', 'total_line', 'elo_diff', 'elo_prob',
            'home_rest', 'away_rest', 'rest_advantage',
            'is_dome', 'is_cold', 'div_game', 'home_implied_prob']

TIER1 = BASELINE + ['is_primetime', 'is_grass', 'bad_weather',
                    'home_off_epa_3wk', 'away_off_epa_3wk', 'epa_diff_3wk',
                    'home_def_epa_3wk', 'away_def_epa_3wk']

# Split data
train = games_epa[(games_epa['season'] >= 2006) & (games_epa['season'] <= 2023)].copy()
test_2024 = games_epa[games_epa['season'] == 2024].copy()
val_2025 = games_2025.copy()

# Fill missing
for col in TIER1:
    if col in train.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test_2024[col] = test_2024[col].fillna(med)
        val_2025[col] = val_2025[col].fillna(med) if col in val_2025.columns else med

print(f"Train: {len(train)} | Test 2024: {len(test_2024)} | Val 2025: {len(val_2025)}")

# Betting simulation function
def simulate_betting(model_margin, model_total, df, min_edge=0.02):
    """Simulate spread and totals betting with flat $100 bets."""
    SPREAD_STD = 12.48
    TOTAL_STD = 12.55
    
    results = {'spread_bets': 0, 'spread_wins': 0, 'spread_pnl': 0,
               'total_bets': 0, 'total_wins': 0, 'total_pnl': 0}
    
    for idx, row in df.iterrows():
        # Spread betting
        pred_margin = model_margin.predict(row[features].values.reshape(1, -1))[0]
        spread = row.get('spread_line', 0)
        actual_margin = row['home_score'] - row['away_score']
        
        # Prob home covers
        home_cover_prob = norm.cdf((pred_margin + spread) / SPREAD_STD)
        
        if home_cover_prob > 0.5 + min_edge:
            results['spread_bets'] += 1
            won = actual_margin + spread > 0
            results['spread_wins'] += int(won)
            results['spread_pnl'] += 91 if won else -100
        elif home_cover_prob < 0.5 - min_edge:
            results['spread_bets'] += 1
            won = actual_margin + spread < 0
            results['spread_wins'] += int(won)
            results['spread_pnl'] += 91 if won else -100
        
        # Totals betting
        pred_total = model_total.predict(row[features].values.reshape(1, -1))[0]
        line = row.get('total_line', 44)
        actual_total = row['home_score'] + row['away_score']
        
        over_prob = norm.cdf((pred_total - line) / TOTAL_STD)
        
        if over_prob > 0.5 + min_edge:
            results['total_bets'] += 1
            won = actual_total > line
            results['total_wins'] += int(won)
            results['total_pnl'] += 91 if won else -100
        elif over_prob < 0.5 - min_edge:
            results['total_bets'] += 1
            won = actual_total < line
            results['total_wins'] += int(won)
            results['total_pnl'] += 91 if won else -100
    
    return results

# Train and evaluate for each feature set
print("\n" + "=" * 80)
print("BETTING RESULTS")
print("=" * 80)

for name, features in [('Baseline', BASELINE), ('TIER1', TIER1)]:
    avail = [f for f in features if f in train.columns and f in val_2025.columns]
    
    X_train = train[avail].fillna(0)
    y_margin = train['home_score'] - train['away_score']
    y_total = train['home_score'] + train['away_score']
    
    # Train margin model
    margin_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
    margin_model.fit(X_train, y_margin)
    
    # Train totals model
    total_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)
    total_model.fit(X_train, y_total)
    
    # Evaluate on 2024 and 2025
    for dataset_name, dataset in [('2024', test_2024), ('2025', val_2025)]:
        dataset_eval = dataset.copy()
        for col in avail:
            dataset_eval[col] = dataset_eval[col].fillna(0) if col in dataset_eval.columns else 0
        
        res = simulate_betting(margin_model, total_model, dataset_eval, min_edge=0.02)
        
        spread_wr = res['spread_wins']/res['spread_bets']*100 if res['spread_bets'] > 0 else 0
        spread_roi = res['spread_pnl']/(res['spread_bets']*100)*100 if res['spread_bets'] > 0 else 0
        total_wr = res['total_wins']/res['total_bets']*100 if res['total_bets'] > 0 else 0
        total_roi = res['total_pnl']/(res['total_bets']*100)*100 if res['total_bets'] > 0 else 0
        
        print(f"\n{name} - {dataset_name}:")
        print(f"  Spread: {res['spread_bets']} bets, {spread_wr:.1f}% WR, ${res['spread_pnl']:+.0f} ({spread_roi:+.1f}% ROI)")
        print(f"  Totals: {res['total_bets']} bets, {total_wr:.1f}% WR, ${res['total_pnl']:+.0f} ({total_roi:+.1f}% ROI)")

print("\n" + "=" * 80)

