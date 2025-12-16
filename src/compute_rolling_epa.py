"""
Compute Rolling EPA Features
============================
Creates rolling EPA metrics (3-week, 5-week) for all teams
and merges with game-level data.

This is the most computationally expensive TIER 1 feature.
Run once and cache results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")


def compute_team_epa_by_game(years: list) -> pd.DataFrame:
    """Compute EPA per game for each team (offense and defense)."""
    print(f"Computing game-level EPA for {min(years)}-{max(years)}...")
    
    all_epa = []
    
    for year in tqdm(years, desc="Loading PBP"):
        try:
            pbp = nfl.import_pbp_data([year], downcast=True)
            plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
            
            if len(plays) == 0:
                continue
            
            # Offensive EPA
            off = plays.groupby(['game_id', 'posteam']).agg({
                'epa': 'mean',
                'success': 'mean',
            }).reset_index()
            off.columns = ['game_id', 'team', 'off_epa', 'off_success']
            
            # Defensive EPA (lower = better defense)
            def_ = plays.groupby(['game_id', 'defteam']).agg({
                'epa': 'mean',
            }).reset_index()
            def_.columns = ['game_id', 'team', 'def_epa_allowed']
            
            # Merge
            team_game = off.merge(def_, on=['game_id', 'team'], how='outer')
            team_game['season'] = year
            
            # Extract week from game_id
            team_game['week'] = team_game['game_id'].str.extract(r'_(\d+)_').astype(int)
            
            all_epa.append(team_game)
            
        except Exception as e:
            print(f"Error {year}: {e}")
            continue
    
    return pd.concat(all_epa, ignore_index=True)


def compute_rolling_features(team_epa: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages for each team."""
    print("Computing rolling averages (3wk, 5wk)...")
    
    team_epa = team_epa.sort_values(['season', 'team', 'week'])
    
    result = []
    for (season, team), group in tqdm(team_epa.groupby(['season', 'team']), desc="Rolling"):
        g = group.copy()
        
        # Shift to not include current game
        g['off_epa_3wk'] = g['off_epa'].shift(1).rolling(3, min_periods=1).mean()
        g['off_epa_5wk'] = g['off_epa'].shift(1).rolling(5, min_periods=1).mean()
        g['def_epa_3wk'] = g['def_epa_allowed'].shift(1).rolling(3, min_periods=1).mean()
        g['def_epa_5wk'] = g['def_epa_allowed'].shift(1).rolling(5, min_periods=1).mean()
        g['success_3wk'] = g['off_success'].shift(1).rolling(3, min_periods=1).mean()
        
        result.append(g)
    
    return pd.concat(result, ignore_index=True)


def merge_epa_with_games(games: pd.DataFrame, epa: pd.DataFrame) -> pd.DataFrame:
    """Merge rolling EPA with games dataframe."""
    print("Merging EPA features with games...")
    
    # Prepare EPA for home team
    home_epa = epa[['game_id', 'team', 'off_epa_3wk', 'off_epa_5wk', 
                    'def_epa_3wk', 'def_epa_5wk', 'success_3wk']].copy()
    home_epa.columns = ['game_id', 'home_team', 'home_off_epa_3wk', 'home_off_epa_5wk',
                        'home_def_epa_3wk', 'home_def_epa_5wk', 'home_success_3wk']
    
    # Prepare EPA for away team
    away_epa = epa[['game_id', 'team', 'off_epa_3wk', 'off_epa_5wk',
                    'def_epa_3wk', 'def_epa_5wk', 'success_3wk']].copy()
    away_epa.columns = ['game_id', 'away_team', 'away_off_epa_3wk', 'away_off_epa_5wk',
                        'away_def_epa_3wk', 'away_def_epa_5wk', 'away_success_3wk']
    
    # Merge with games
    games = games.merge(home_epa, on=['game_id', 'home_team'], how='left')
    games = games.merge(away_epa, on=['game_id', 'away_team'], how='left')
    
    # Create differential features
    games['epa_diff_3wk'] = games['home_off_epa_3wk'] - games['away_off_epa_3wk']
    games['epa_diff_5wk'] = games['home_off_epa_5wk'] - games['away_off_epa_5wk']
    
    return games


if __name__ == "__main__":
    # Compute for years 2006-2024 (PBP available from 2006)
    YEARS = list(range(2006, 2025))
    
    # Step 1: Compute game-level EPA
    team_epa = compute_team_epa_by_game(YEARS)
    print(f"Computed EPA for {len(team_epa)} team-games")
    
    # Step 2: Compute rolling features
    epa_rolling = compute_rolling_features(team_epa)
    
    # Save intermediate result
    CACHE_DIR = RAW_DATA_DIR / "cache"
    CACHE_DIR.mkdir(exist_ok=True)
    epa_rolling.to_parquet(CACHE_DIR / "team_rolling_epa.parquet", index=False)
    print(f"Saved rolling EPA to {CACHE_DIR / 'team_rolling_epa.parquet'}")
    
    # Step 3: Merge with games
    games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    games_with_epa = merge_epa_with_games(games, epa_rolling)
    
    # Save enhanced games
    games_with_epa.to_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet", index=False)
    print(f"Saved games with EPA to {PROCESSED_DATA_DIR / 'games_with_epa.parquet'}")
    
    # Summary
    print("\n" + "="*60)
    print("EPA Feature Summary")
    print("="*60)
    epa_cols = [c for c in games_with_epa.columns if 'epa' in c.lower() and '3wk' in c]
    for col in epa_cols:
        miss = games_with_epa[col].isna().sum()
        mean = games_with_epa[col].mean()
        print(f"  {col}: mean={mean:.4f}, missing={miss}")

