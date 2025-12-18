"""
TIER S+A Backtest Runner
========================
Runs backtests with all TIER S+A features on:
- 2024 season (test set)
- 2025 season Weeks 1-15 (validation set)

Outputs:
- Spread, Totals, Moneyline performance
- Week-by-week breakdown
- Week 15 deep dive
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from src.tier_sa_features import (
    compute_all_tier_sa_features, 
    merge_features_to_games,
    add_rest_day_features
)
from src.tier1_features import add_all_tier1_features
from src.feature_engineering import compute_elo_features
from src.data_loader import load_all_schedules, create_game_features
from config import PROCESSED_DATA_DIR

import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression


def load_and_prepare_data():
    """Load all data with TIER S+A features."""
    print("="*60)
    print("LOADING DATA WITH TIER S+A FEATURES")
    print("="*60)
    
    # Load schedules for all years including 2025
    years = list(range(1999, 2026))
    print(f"\nLoading schedules for {min(years)}-{max(years)}...")
    schedules = nfl.import_schedules(years)
    
    # Create basic features
    games = create_game_features(schedules)
    
    # Compute Elo features
    games = compute_elo_features(games)
    
    # Add TIER 1 features
    games = add_all_tier1_features(games)
    
    # Add rest day features
    games = add_rest_day_features(games)
    
    # Compute and merge TIER S+A features (2016+ for NGS, 2018+ for PFR)
    tier_sa_years = list(range(2016, 2026))
    tier_sa_features = compute_all_tier_sa_features(tier_sa_years)
    games = merge_features_to_games(games, tier_sa_features)
    
    # Filter to completed games for training/test
    completed = games[games['result'].notna()].copy()
    
    print(f"\nTotal games: {len(games)}")
    print(f"Completed games: {len(completed)}")
    print(f"2024 games: {len(games[games['season'] == 2024])}")
    print(f"2025 games: {len(games[games['season'] == 2025])}")
    
    return games, completed


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for modeling."""
    # Base features
    features = [
        'elo_diff', 'elo_prob', 'spread_line', 'total_line',
        'rest_advantage', 'div_game', 'is_dome', 'is_cold', 'is_windy',
        'home_implied_prob', 'away_implied_prob',
    ]
    
    # TIER 1 features
    tier1 = ['is_primetime', 'is_grass', 'bad_weather', 'home_short_week', 'away_short_week']
    
    # TIER S+A features
    tier_sa = [
        'home_cpoe_3wk', 'away_cpoe_3wk', 'cpoe_diff',
        'home_pressure_rate_3wk', 'away_pressure_rate_3wk', 'pressure_diff',
        'home_injury_impact', 'away_injury_impact', 'injury_diff',
        'home_qb_out', 'away_qb_out',
        'home_ryoe_3wk', 'away_ryoe_3wk', 'ryoe_diff',
        'home_separation_3wk', 'away_separation_3wk', 'separation_diff',
        'home_time_to_throw_3wk', 'away_time_to_throw_3wk',
    ]
    
    all_features = features + tier1 + tier_sa
    
    # Filter to columns that exist
    available = [f for f in all_features if f in df.columns]
    print(f"\nUsing {len(available)} features:")
    print(f"  Available: {available}")
    
    return available


def train_models(train_df: pd.DataFrame, features: list):
    """Train spread, totals, and moneyline models."""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Prepare training data
    X = train_df[features].copy()
    
    # Fill missing values
    X = X.fillna(X.median())
    
    models = {}
    
    # 1. Spread model (predict home margin)
    y_spread = train_df['result']
    models['spread'] = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    models['spread'].fit(X, y_spread)
    print(f"  Spread model trained on {len(X)} samples")
    
    # 2. Totals model (predict combined score)
    y_totals = train_df['game_total']
    models['totals'] = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    models['totals'].fit(X, y_totals)
    print(f"  Totals model trained on {len(X)} samples")
    
    # 3. Moneyline model (predict home win)
    y_ml = train_df['home_win']
    models['moneyline'] = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
    )
    models['moneyline'].fit(X, y_ml)
    print(f"  Moneyline model trained on {len(X)} samples")
    
    return models


def predict_and_evaluate(models: dict, test_df: pd.DataFrame, features: list, dataset_name: str):
    """Make predictions and evaluate betting performance."""
    print(f"\n" + "="*60)
    print(f"EVALUATING ON {dataset_name.upper()}")
    print("="*60)
    
    # Prepare test data
    df = test_df.copy()
    X = df[features].fillna(df[features].median())
    
    # Make predictions
    df['pred_margin'] = models['spread'].predict(X)
    df['pred_total'] = models['totals'].predict(X)
    df['pred_win_prob'] = models['moneyline'].predict_proba(X)[:, 1]
    
    results = {'dataset': dataset_name, 'games': len(df)}
    
    # Spread betting
    df['bet_home_cover'] = df['pred_margin'] > df['spread_line']
    df['home_covered'] = df['result'] > df['spread_line']
    spread_correct = (df['bet_home_cover'] == df['home_covered']).sum()
    results['spread_wr'] = spread_correct / len(df)
    results['spread_roi'] = (results['spread_wr'] * 1.909 - 1) * 100
    
    # Totals betting
    df['bet_over'] = df['pred_total'] > df['total_line']
    df['went_over'] = df['game_total'] > df['total_line']
    totals_correct = (df['bet_over'] == df['went_over']).sum()
    results['totals_wr'] = totals_correct / len(df)
    results['totals_roi'] = (results['totals_wr'] * 1.909 - 1) * 100
    
    # Moneyline betting (bet when edge > 5%)
    MIN_EDGE = 0.05

    # Compute implied probabilities from moneyline odds if available
    if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
        # Convert American odds to implied probability
        def ml_to_prob(ml):
            if pd.isna(ml):
                return 0.5
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)

        df['home_implied_prob'] = df['home_moneyline'].apply(ml_to_prob)
        df['away_implied_prob'] = df['away_moneyline'].apply(ml_to_prob)
    else:
        # Estimate from spread if moneylines not available
        df['home_implied_prob'] = 0.5 + (df['spread_line'] / 100)
        df['away_implied_prob'] = 1 - df['home_implied_prob']

    df['ml_edge'] = df['pred_win_prob'] - df['home_implied_prob']
    df['bet_home_ml'] = df['ml_edge'] > MIN_EDGE
    df['bet_away_ml'] = df['ml_edge'] < -MIN_EDGE

    # Track which games we bet on and actual outcomes
    df['ml_bet'] = df['bet_home_ml'] | df['bet_away_ml']
    df['ml_correct'] = (
        (df['bet_home_ml'] & (df['result'] > 0)) |
        (df['bet_away_ml'] & (df['result'] < 0))
    )

    ml_bets = df[df['ml_bet']].copy()
    if len(ml_bets) > 0:
        results['ml_wr'] = ml_bets['ml_correct'].mean()
        results['ml_bets'] = len(ml_bets)

        # Calculate actual ROI using real odds
        ml_profits = []
        for _, row in ml_bets.iterrows():
            if row['bet_home_ml']:
                if row['result'] > 0:  # Home win
                    if 'home_moneyline' in row and not pd.isna(row.get('home_moneyline')):
                        ml = row['home_moneyline']
                        profit = 100/ml if ml > 0 else abs(ml)/100
                    else:
                        profit = 0.909  # -110 odds
                else:
                    profit = -1
            else:  # Bet away
                if row['result'] < 0:  # Away win
                    if 'away_moneyline' in row and not pd.isna(row.get('away_moneyline')):
                        ml = row['away_moneyline']
                        profit = 100/ml if ml > 0 else abs(ml)/100
                    else:
                        profit = 0.909
                else:
                    profit = -1
            ml_profits.append(profit)

        results['ml_roi'] = (sum(ml_profits) / len(ml_profits)) * 100
        results['ml_profit_units'] = sum(ml_profits)
    else:
        results['ml_wr'] = 0
        results['ml_bets'] = 0
        results['ml_roi'] = 0
        results['ml_profit_units'] = 0

    # Also track pure win prediction accuracy (no edge filter)
    df['pred_home_win'] = df['pred_win_prob'] > 0.5
    df['actual_home_win'] = df['result'] > 0
    results['win_pred_accuracy'] = (df['pred_home_win'] == df['actual_home_win']).mean()

    print(f"\n📊 {dataset_name} Results ({len(df)} games):")
    print(f"  Spread: {results['spread_wr']*100:.1f}% WR, {results['spread_roi']:+.1f}% ROI")
    print(f"  Totals: {results['totals_wr']*100:.1f}% WR, {results['totals_roi']:+.1f}% ROI")
    print(f"  Moneyline: {results['ml_wr']*100:.1f}% WR on {results['ml_bets']} bets, {results['ml_roi']:+.1f}% ROI")
    print(f"  Win Prediction Accuracy: {results['win_pred_accuracy']*100:.1f}%")

    return results, df


def week_by_week_analysis(df: pd.DataFrame, dataset_name: str):
    """Analyze performance week by week including moneyline."""
    print(f"\n📅 Week-by-Week Analysis ({dataset_name}):")
    print("-" * 80)

    weekly = []
    for week in sorted(df['week'].unique()):
        week_df = df[df['week'] == week]

        # Spread
        spread_wr = (week_df['bet_home_cover'] == week_df['home_covered']).mean()

        # Totals
        totals_wr = (week_df['bet_over'] == week_df['went_over']).mean()

        # Moneyline
        ml_week = week_df[week_df['ml_bet']]
        ml_bets = len(ml_week)
        ml_wr = ml_week['ml_correct'].mean() if ml_bets > 0 else 0
        ml_correct = ml_week['ml_correct'].sum() if ml_bets > 0 else 0

        # Win prediction accuracy (all games)
        win_acc = (week_df['pred_home_win'] == week_df['actual_home_win']).mean()

        weekly.append({
            'week': week,
            'games': len(week_df),
            'spread_wr': spread_wr,
            'totals_wr': totals_wr,
            'ml_bets': ml_bets,
            'ml_wr': ml_wr,
            'ml_correct': ml_correct,
            'win_accuracy': win_acc
        })

        print(f"  Week {week:2d}: {len(week_df):2d} games | Spread: {spread_wr*100:5.1f}% | Totals: {totals_wr*100:5.1f}% | ML: {ml_correct}/{ml_bets} ({ml_wr*100:5.1f}%) | Win Acc: {win_acc*100:5.1f}%")

    return pd.DataFrame(weekly)


def week15_deep_dive(df: pd.DataFrame):
    """Detailed analysis of Week 15 predictions."""
    print("\n" + "="*60)
    print("WEEK 15 DEEP DIVE")
    print("="*60)

    week15 = df[df['week'] == 15].copy()

    if len(week15) == 0:
        print("No Week 15 data available")
        return None

    print(f"\n📊 Week 15: {len(week15)} games\n")

    # Game-by-game breakdown
    cols = ['away_team', 'home_team', 'result', 'spread_line', 'pred_margin',
            'game_total', 'total_line', 'pred_total']

    for idx, row in week15.iterrows():
        spread_correct = (row['bet_home_cover'] == row['home_covered'])
        totals_correct = (row['bet_over'] == row['went_over'])

        print(f"{row['away_team']} @ {row['home_team']}")
        print(f"  Score: {int(row['away_score'])} - {int(row['home_score'])} (Margin: {int(row['result'])})")
        print(f"  Spread: Line={row['spread_line']:+.1f}, Pred={row['pred_margin']:+.1f} → {'✅' if spread_correct else '❌'}")
        print(f"  Total: Line={row['total_line']:.1f}, Pred={row['pred_total']:.1f}, Actual={int(row['game_total'])} → {'✅' if totals_correct else '❌'}")
        print()

    # Summary
    spread_wr = (week15['bet_home_cover'] == week15['home_covered']).mean()
    totals_wr = (week15['bet_over'] == week15['went_over']).mean()

    print(f"Week 15 Summary:")
    print(f"  Spread: {spread_wr*100:.1f}% ({int(spread_wr*len(week15))}/{len(week15)})")
    print(f"  Totals: {totals_wr*100:.1f}% ({int(totals_wr*len(week15))}/{len(week15)})")

    return week15


def main():
    """Main backtest runner."""
    print("="*60)
    print("TIER S+A BACKTEST - 2024 TEST & 2025 VALIDATION")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    all_games, completed = load_and_prepare_data()

    # Get feature columns
    features = get_feature_columns(completed)

    # Split data
    # Train: 1999-2023 (use 2018+ for full feature availability)
    # Test: 2024
    # Validation: 2025

    train_df = completed[(completed['season'] >= 2018) & (completed['season'] <= 2023)].copy()
    test_2024 = completed[completed['season'] == 2024].copy()
    val_2025 = completed[completed['season'] == 2025].copy()

    print(f"\n📊 Data Split:")
    print(f"  Train (2018-2023): {len(train_df)} games")
    print(f"  Test (2024): {len(test_2024)} games")
    print(f"  Validation (2025): {len(val_2025)} games")

    # Train models
    models = train_models(train_df, features)

    # Evaluate on 2024
    results_2024, pred_2024 = predict_and_evaluate(models, test_2024, features, "2024 Test")
    weekly_2024 = week_by_week_analysis(pred_2024, "2024")

    # Evaluate on 2025
    if len(val_2025) > 0:
        results_2025, pred_2025 = predict_and_evaluate(models, val_2025, features, "2025 Validation")
        weekly_2025 = week_by_week_analysis(pred_2025, "2025")
        week15_df = week15_deep_dive(pred_2025)
    else:
        results_2025 = None
        weekly_2025 = None
        print("\n⚠️ No 2025 validation data available")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Extract feature importances from each model
    feature_importance = {}
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(features, model.feature_importances_.tolist()))
            feature_importance[model_name] = importance

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'TIER_S+A',
        'features': features,
        'feature_importance': feature_importance,
        'train_years': '2018-2023',
        'train_size': len(train_df),
        'results_2024': results_2024,
        'results_2025': results_2025,
    }

    with open(results_dir / "tier_sa_backtest_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save predictions
    pred_2024.to_parquet(results_dir / "tier_sa_predictions_2024.parquet", index=False)
    if len(val_2025) > 0:
        pred_2025.to_parquet(results_dir / "tier_sa_predictions_2025.parquet", index=False)

    # Save weekly stats
    weekly_2024.to_csv(results_dir / "tier_sa_weekly_2024.csv", index=False)
    if weekly_2025 is not None:
        weekly_2025.to_csv(results_dir / "tier_sa_weekly_2025.csv", index=False)

    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print(f"  {results_dir / 'tier_sa_backtest_results.json'}")
    print(f"  {results_dir / 'tier_sa_predictions_2024.parquet'}")
    print(f"  {results_dir / 'tier_sa_predictions_2025.parquet'}")

    # Final comparison
    print("\n" + "="*60)
    print("TIER S+A vs BASELINE COMPARISON")
    print("="*60)
    print("""
    | Model    | Dataset | Spread WR | Spread ROI | Totals WR | Totals ROI |
    |----------|---------|-----------|------------|-----------|------------|
    | Baseline | 2024    | 49.4%     | -5.6%      | ~50%      | -5%        |
    | TIER S+A | 2024    | {:.1f}%    | {:+.1f}%    | {:.1f}%    | {:+.1f}%    |
    | TIER S+A | 2025    | {:.1f}%    | {:+.1f}%    | {:.1f}%    | {:+.1f}%    |
    """.format(
        results_2024['spread_wr']*100, results_2024['spread_roi'],
        results_2024['totals_wr']*100, results_2024['totals_roi'],
        results_2025['spread_wr']*100 if results_2025 else 0,
        results_2025['spread_roi'] if results_2025 else 0,
        results_2025['totals_wr']*100 if results_2025 else 0,
        results_2025['totals_roi'] if results_2025 else 0,
    ))

    return summary


if __name__ == "__main__":
    main()

