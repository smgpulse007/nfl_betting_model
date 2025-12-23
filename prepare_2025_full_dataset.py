"""
Prepare complete 2025 Week 1-16 dataset using nfl-data-py + ESPN validation

Strategy:
1. Use nfl-data-py for all historical game data (scores, odds, schedule)
2. Use ESPN API to validate current week odds
3. Compute all features including injuries, EPA, etc.
4. Generate predictions and evaluate accuracy
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR
from src.tier_sa_features import (
    compute_all_tier_sa_features,
    merge_features_to_games,
    add_rest_day_features
)
from src.tier1_features import add_all_tier1_features
from src.feature_engineering import compute_elo_features
from src.data_loader import create_game_features
from src.models import NFLBettingModels
from espn_game_api import ESPNGameAPI

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")

def prepare_features_for_all_years():
    """
    Prepare features for all years (1999-2025) using TIER S+A feature engineering

    This follows the same pattern as run_tier_sa_backtest.py
    """
    print("\n" + "="*100)
    print("PREPARING TIER S+A FEATURES FOR ALL YEARS")
    print("="*100)

    # Load schedules for all years including 2025
    years = list(range(1999, 2026))
    print(f"\n[1/5] Loading schedules for {min(years)}-{max(years)}...")
    schedules = nfl.import_schedules(years)

    # Create basic features
    print("\n[2/5] Creating basic game features...")
    games = create_game_features(schedules)

    # Compute Elo features
    print("\n[3/5] Computing Elo ratings...")
    games = compute_elo_features(games)

    # Add TIER 1 features
    print("\n[4/5] Adding TIER 1 features...")
    games = add_all_tier1_features(games)

    # Add rest day features
    games = add_rest_day_features(games)

    # Compute and merge TIER S+A features (2016+ for NGS, 2018+ for PFR)
    print("\n[5/5] Computing and merging TIER S+A features...")
    tier_sa_years = list(range(2016, 2026))
    tier_sa_features = compute_all_tier_sa_features(tier_sa_years)
    games = merge_features_to_games(games, tier_sa_features)

    # Filter to completed games
    completed = games[games['result'].notna()].copy()

    print(f"\n✅ Total games: {len(games)}")
    print(f"✅ Completed games: {len(completed)}")
    print(f"✅ 2024 games: {len(games[games['season'] == 2024])}")
    print(f"✅ 2025 games: {len(games[games['season'] == 2025])}")
    print(f"✅ 2025 completed: {len(completed[completed['season'] == 2025])}")

    return games, completed

def evaluate_completed_games(historical_df, completed_2025):
    """
    Evaluate model accuracy on completed 2025 games (Weeks 1-16)
    
    This is the key function for checking accuracy!
    """
    print("\n" + "="*100)
    print("EVALUATING MODEL ACCURACY ON COMPLETED 2025 GAMES")
    print("="*100)
    
    # Train on historical data only (no 2025)
    print(f"\n[1/2] Training on historical data ({len(historical_df)} games)...")
    models = NFLBettingModels()
    models.fit(historical_df)
    
    # Predict on completed 2025 games
    print(f"\n[2/2] Predicting {len(completed_2025)} completed 2025 games...")
    predictions = models.predict(completed_2025)
    
    # Merge predictions with actuals
    results = completed_2025.merge(
        predictions,
        on=['game_id', 'season', 'week', 'home_team', 'away_team'],
        how='left'
    )
    
    # Calculate accuracy metrics
    print("\n" + "="*100)
    print("ACCURACY RESULTS")
    print("="*100)
    
    # Moneyline accuracy
    results['xgb_ml_pick'] = (results['xgb_win_prob'] > 0.5).astype(int)
    results['lr_ml_pick'] = (results['lr_win_prob'] > 0.5).astype(int)

    # Vegas baseline (from moneyline odds)
    if 'home_moneyline' in results.columns:
        # Convert American odds to implied probability
        def ml_to_prob(ml):
            if pd.isna(ml):
                return np.nan
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return -ml / (-ml + 100)

        results['vegas_prob'] = results['home_moneyline'].apply(ml_to_prob)
        results['vegas_ml_pick'] = (results['vegas_prob'] > 0.5).astype(int)
        vegas_ml_acc = (results['vegas_ml_pick'] == results['home_win']).mean()
    else:
        vegas_ml_acc = None

    xgb_ml_acc = (results['xgb_ml_pick'] == results['home_win']).mean()
    lr_ml_acc = (results['lr_ml_pick'] == results['home_win']).mean()

    print(f"\n📊 MONEYLINE ACCURACY:")
    print(f"   XGBoost:  {xgb_ml_acc:.1%} ({(results['xgb_ml_pick'] == results['home_win']).sum()}/{len(results)})")
    print(f"   Logistic: {lr_ml_acc:.1%} ({(results['lr_ml_pick'] == results['home_win']).sum()}/{len(results)})")
    if vegas_ml_acc is not None:
        print(f"   Vegas:    {vegas_ml_acc:.1%} ({(results['vegas_ml_pick'] == results['home_win']).sum()}/{len(results)})")
    
    # Spread accuracy
    results['xgb_spread_pick'] = (results['home_cover_prob'] > 0.5).astype(int)
    
    xgb_spread_acc = (results['xgb_spread_pick'] == results['home_cover']).mean()
    
    print(f"\n📊 SPREAD ACCURACY:")
    print(f"   XGBoost: {xgb_spread_acc:.1%} ({(results['xgb_spread_pick'] == results['home_cover']).sum()}/{len(results)})")
    
    # Totals accuracy
    results['xgb_total_pick'] = (results['over_prob'] > 0.5).astype(int)
    
    xgb_total_acc = (results['xgb_total_pick'] == results['over_hit']).mean()
    
    print(f"\n📊 TOTALS (O/U) ACCURACY:")
    print(f"   XGBoost: {xgb_total_acc:.1%} ({(results['xgb_total_pick'] == results['over_hit']).sum()}/{len(results)})")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    results.to_csv(output_dir / '2025_week1_16_evaluation.csv', index=False)
    print(f"\n✅ Saved detailed results to: {output_dir / '2025_week1_16_evaluation.csv'}")
    
    return results

if __name__ == "__main__":
    # Prepare all features
    all_games, completed_games = prepare_features_for_all_years()

    # Split by year
    historical = completed_games[completed_games['season'] < 2025].copy()
    completed_2025 = completed_games[completed_games['season'] == 2025].copy()

    print(f"\n📊 Data Split:")
    print(f"  Historical (1999-2024): {len(historical)} games")
    print(f"  2025 Completed: {len(completed_2025)} games")

    # Evaluate on completed 2025 games
    if len(completed_2025) > 0:
        evaluation_results = evaluate_completed_games(historical, completed_2025)
    else:
        print("\n⚠️ No completed 2025 games to evaluate")

