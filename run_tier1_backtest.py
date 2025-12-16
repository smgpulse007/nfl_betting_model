"""
TIER 1 Feature Backtest
=======================
Test enhanced model with TIER 1 features:
- Train: 1999-2023
- Test: 2024
- Validation: 2025

Compare baseline vs enhanced model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import xgboost as xgb
from scipy.stats import norm

import sys
sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR
from src.tier1_features import add_all_tier1_features
from src.feature_engineering import compute_elo_features

print("=" * 80)
print("TIER 1 FEATURE BACKTEST")
print("=" * 80)

# SECTION 1: Load Data
print("\n[1/5] Loading data...")
games_epa = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")
games_2025 = pd.read_parquet(PROCESSED_DATA_DIR.parent / "2025" / "completed_2025.parquet")

print(f"  Historical: {len(games_epa)} games")
print(f"  2025:       {len(games_2025)} games")

# SECTION 2: Add TIER 1 Features
print("\n[2/5] Adding TIER 1 features...")
games_epa = add_all_tier1_features(games_epa)
games_2025 = add_all_tier1_features(games_2025)

# SECTION 3: Prepare Feature Sets
BASELINE_FEATURES = [
    'spread_line', 'total_line', 'elo_diff', 'elo_prob',
    'home_rest', 'away_rest', 'rest_advantage',
    'is_dome', 'is_cold', 'div_game', 'home_implied_prob'
]

TIER1_FEATURES = BASELINE_FEATURES + [
    'is_primetime', 'is_grass', 'is_very_windy', 'bad_weather',
    'home_off_epa_3wk', 'away_off_epa_3wk', 'epa_diff_3wk',
    'home_def_epa_3wk', 'away_def_epa_3wk'
]

# Filter to games with EPA data (2006+)
games_train = games_epa[(games_epa['season'] >= 2006) & (games_epa['season'] <= 2023)].copy()
games_test = games_epa[games_epa['season'] == 2024].copy()

# Handle missing values
for col in TIER1_FEATURES:
    if col in games_train.columns:
        median = games_train[col].median()
        games_train[col] = games_train[col].fillna(median)
        games_test[col] = games_test[col].fillna(median)
        if col in games_2025.columns:
            games_2025[col] = games_2025[col].fillna(median)
        else:
            games_2025[col] = median

print(f"  Train: {len(games_train)} games (2006-2023)")
print(f"  Test:  {len(games_test)} games (2024)")
print(f"  Val:   {len(games_2025)} games (2025)")

# SECTION 4: Train and Evaluate
print("\n[3/5] Training models...")

# Target
y_train = games_train['home_win']
y_test = games_test['home_win']
y_val = games_2025['home_win'] if 'home_win' in games_2025.columns else (games_2025['home_score'] > games_2025['away_score']).astype(int)

results = {}

for name, features in [('Baseline', BASELINE_FEATURES), ('TIER1', TIER1_FEATURES)]:
    # Filter to available features
    avail = [f for f in features if f in games_train.columns]
    
    X_train = games_train[avail]
    X_test = games_test[avail]
    X_val = games_2025[[f for f in avail if f in games_2025.columns]]
    
    # Fill any remaining NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_val = X_val.fillna(0)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Predict probabilities
    p_test = model.predict_proba(X_test)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    results[name] = {
        'test_acc': accuracy_score(y_test, (p_test > 0.5).astype(int)),
        'test_brier': brier_score_loss(y_test, p_test),
        'val_acc': accuracy_score(y_val, (p_val > 0.5).astype(int)),
        'val_brier': brier_score_loss(y_val, p_val),
    }
    
    # Feature importance (top 10)
    if name == 'TIER1':
        imp = pd.DataFrame({'feature': avail, 'importance': model.feature_importances_})
        imp = imp.sort_values('importance', ascending=False).head(10)
        print(f"\n  {name} Top 10 Features:")
        for _, row in imp.iterrows():
            print(f"    {row['feature']:25} {row['importance']:.4f}")

# SECTION 5: Results Comparison
print("\n" + "=" * 80)
print("[4/5] RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Model':<12} | {'2024 Acc':>10} | {'2024 Brier':>12} | {'2025 Acc':>10} | {'2025 Brier':>12}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<12} | {r['test_acc']*100:>9.1f}% | {r['test_brier']:>12.4f} | {r['val_acc']*100:>9.1f}% | {r['val_brier']:>12.4f}")

# Improvement
delta_acc = (results['TIER1']['val_acc'] - results['Baseline']['val_acc']) * 100
delta_brier = results['Baseline']['val_brier'] - results['TIER1']['val_brier']
print(f"\n  TIER1 Improvement (2025):")
print(f"    Accuracy: {delta_acc:+.1f}%")
print(f"    Brier:    {delta_brier:+.4f} (lower is better)")

print("\n" + "=" * 80)
print("[5/5] TIER 1 BACKTEST COMPLETE")
print("=" * 80)

