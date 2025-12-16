"""
Compute Rolling EPA for 2025 Season
====================================
Uses cached historical EPA + computes 2025 EPA from PBP.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")

print("=" * 70)
print("COMPUTING 2025 ROLLING EPA")
print("=" * 70)

# Load cached historical EPA
CACHE_DIR = RAW_DATA_DIR / "cache"
historical_epa = pd.read_parquet(CACHE_DIR / "team_rolling_epa.parquet")
print(f"Loaded historical EPA: {len(historical_epa)} team-games")

# Compute 2025 EPA from PBP
print("\nLoading 2025 play-by-play...")
try:
    pbp_2025 = nfl.import_pbp_data([2025], downcast=True)
    plays = pbp_2025[pbp_2025['play_type'].isin(['pass', 'run'])].copy()
    print(f"  Loaded {len(plays)} plays")
except Exception as e:
    print(f"Error loading 2025 PBP: {e}")
    plays = pd.DataFrame()

if len(plays) > 0:
    # Offensive EPA
    off = plays.groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'success': 'mean',
    }).reset_index()
    off.columns = ['game_id', 'team', 'off_epa', 'off_success']
    
    # Defensive EPA
    def_ = plays.groupby(['game_id', 'defteam']).agg({
        'epa': 'mean',
    }).reset_index()
    def_.columns = ['game_id', 'team', 'def_epa_allowed']
    
    # Merge
    team_2025 = off.merge(def_, on=['game_id', 'team'], how='outer')
    team_2025['season'] = 2025
    team_2025['week'] = team_2025['game_id'].str.extract(r'_(\d+)_').astype(int)
    
    # Compute rolling
    team_2025 = team_2025.sort_values(['team', 'week'])
    
    for team in team_2025['team'].unique():
        mask = team_2025['team'] == team
        team_2025.loc[mask, 'off_epa_3wk'] = (
            team_2025.loc[mask, 'off_epa'].shift(1).rolling(3, min_periods=1).mean()
        )
        team_2025.loc[mask, 'off_epa_5wk'] = (
            team_2025.loc[mask, 'off_epa'].shift(1).rolling(5, min_periods=1).mean()
        )
        team_2025.loc[mask, 'def_epa_3wk'] = (
            team_2025.loc[mask, 'def_epa_allowed'].shift(1).rolling(3, min_periods=1).mean()
        )
        team_2025.loc[mask, 'def_epa_5wk'] = (
            team_2025.loc[mask, 'def_epa_allowed'].shift(1).rolling(5, min_periods=1).mean()
        )
        team_2025.loc[mask, 'success_3wk'] = (
            team_2025.loc[mask, 'off_success'].shift(1).rolling(3, min_periods=1).mean()
        )
    
    print(f"\n2025 EPA computed: {len(team_2025)} team-games")
    
    # Merge with 2025 games
    DATA_2025 = PROCESSED_DATA_DIR.parent / "2025"
    games_2025 = pd.read_parquet(DATA_2025 / "completed_2025.parquet")
    
    # Home EPA
    home_epa = team_2025[['game_id', 'team', 'off_epa_3wk', 'off_epa_5wk', 
                          'def_epa_3wk', 'def_epa_5wk', 'success_3wk']].copy()
    home_epa.columns = ['game_id', 'home_team', 'home_off_epa_3wk', 'home_off_epa_5wk',
                        'home_def_epa_3wk', 'home_def_epa_5wk', 'home_success_3wk']
    
    # Away EPA
    away_epa = team_2025[['game_id', 'team', 'off_epa_3wk', 'off_epa_5wk',
                          'def_epa_3wk', 'def_epa_5wk', 'success_3wk']].copy()
    away_epa.columns = ['game_id', 'away_team', 'away_off_epa_3wk', 'away_off_epa_5wk',
                        'away_def_epa_3wk', 'away_def_epa_5wk', 'away_success_3wk']
    
    games_2025 = games_2025.merge(home_epa, on=['game_id', 'home_team'], how='left')
    games_2025 = games_2025.merge(away_epa, on=['game_id', 'away_team'], how='left')
    
    # Differential
    games_2025['epa_diff_3wk'] = games_2025['home_off_epa_3wk'] - games_2025['away_off_epa_3wk']
    games_2025['epa_diff_5wk'] = games_2025['home_off_epa_5wk'] - games_2025['away_off_epa_5wk']
    
    # Save
    games_2025.to_parquet(DATA_2025 / "completed_2025_with_epa.parquet", index=False)
    print(f"\nSaved to {DATA_2025 / 'completed_2025_with_epa.parquet'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("2025 EPA FEATURE SUMMARY")
    print("=" * 70)
    for col in ['home_off_epa_3wk', 'away_off_epa_3wk', 'epa_diff_3wk']:
        miss = games_2025[col].isna().sum()
        mean = games_2025[col].mean()
        print(f"  {col}: mean={mean:.4f}, missing={miss}/{len(games_2025)}")

