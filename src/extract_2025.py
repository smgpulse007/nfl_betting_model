"""
Extract and Prepare 2025 Season Data for Predictions

This script:
1. Loads 2025 schedule from nfl-data-py
2. Separates completed vs upcoming games
3. Applies Elo ratings (continuing from 2024)
4. Prepares features for model predictions
5. Saves to data/2025/ folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR
from src.feature_engineering import EloRatingSystem, compute_elo_features
from src.data_loader import create_game_features

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")


def main():
    print("=" * 70)
    print("EXTRACTING 2025 SEASON DATA")
    print("=" * 70)
    
    # Create 2025 data folder
    DATA_2025 = PROCESSED_DATA_DIR.parent / "2025"
    DATA_2025.mkdir(parents=True, exist_ok=True)
    
    # 1. Load 2025 schedule
    print("\n[1/5] Loading 2025 schedule...")
    schedule_2025 = nfl.import_schedules([2025])
    print(f"  Total games: {len(schedule_2025)}")
    
    # 2. Separate completed vs upcoming
    print("\n[2/5] Separating completed vs upcoming games...")
    completed = schedule_2025[schedule_2025['home_score'].notna()].copy()
    upcoming = schedule_2025[schedule_2025['home_score'].isna()].copy()
    
    print(f"  Completed games: {len(completed)}")
    print(f"  Upcoming games: {len(upcoming)}")
    
    if len(upcoming) > 0:
        print(f"\n  Next upcoming games:")
        next_games = upcoming.sort_values(['week', 'game_id']).head(10)
        for _, g in next_games.iterrows():
            print(f"    Week {g['week']}: {g['away_team']} @ {g['home_team']} ({g['gameday']})")
    
    # 3. Load historical data to continue Elo
    print("\n[3/5] Loading historical data for Elo continuation...")
    historical = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    print(f"  Historical games: {len(historical)}")
    
    # 4. Create features for 2025 games
    print("\n[4/5] Creating features for 2025...")
    
    # Apply same feature engineering as training data
    schedule_2025_features = create_game_features(schedule_2025)
    
    # Combine historical + 2025 for continuous Elo calculation
    all_games = pd.concat([historical, schedule_2025_features], ignore_index=True)
    
    # Recompute Elo including 2025 games
    all_games_elo = compute_elo_features(all_games)
    
    # Extract just 2025 with updated Elo
    games_2025 = all_games_elo[all_games_elo['season'] == 2025].copy()
    
    # Re-separate completed vs upcoming (now with Elo)
    completed_2025 = games_2025[games_2025['home_score'].notna()].copy()
    upcoming_2025 = games_2025[games_2025['home_score'].isna()].copy()
    
    # 5. Save to files
    print("\n[5/5] Saving data...")
    
    games_2025.to_parquet(DATA_2025 / "all_games_2025.parquet", index=False)
    completed_2025.to_parquet(DATA_2025 / "completed_2025.parquet", index=False)
    upcoming_2025.to_parquet(DATA_2025 / "upcoming_2025.parquet", index=False)
    
    # Also save as CSV for easy viewing
    upcoming_2025[['game_id', 'week', 'gameday', 'away_team', 'home_team', 
                   'spread_line', 'total_line', 'home_moneyline', 'away_moneyline',
                   'home_elo', 'away_elo', 'elo_diff', 'elo_prob']].to_csv(
        DATA_2025 / "upcoming_games.csv", index=False
    )
    
    print(f"\n  Saved to {DATA_2025}:")
    print(f"    - all_games_2025.parquet ({len(games_2025)} games)")
    print(f"    - completed_2025.parquet ({len(completed_2025)} games)")
    print(f"    - upcoming_2025.parquet ({len(upcoming_2025)} games)")
    print(f"    - upcoming_games.csv (for viewing)")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("2025 SEASON SUMMARY")
    print("=" * 70)
    
    print(f"\nCompleted Games by Week:")
    print(completed_2025.groupby('week').size())
    
    if len(upcoming_2025) > 0:
        print(f"\nUpcoming Games with Elo Predictions:")
        display_cols = ['game_id', 'away_team', 'home_team', 'spread_line', 
                       'elo_diff', 'elo_prob']
        print(upcoming_2025[display_cols].head(16).to_string(index=False))
    
    # Betting lines availability
    has_spread = upcoming_2025['spread_line'].notna().sum()
    has_ml = upcoming_2025['home_moneyline'].notna().sum()
    print(f"\nBetting Lines Available for Upcoming Games:")
    print(f"  Spread: {has_spread}/{len(upcoming_2025)}")
    print(f"  Moneyline: {has_ml}/{len(upcoming_2025)}")
    
    return games_2025, completed_2025, upcoming_2025


if __name__ == "__main__":
    all_2025, completed, upcoming = main()

