"""
TIER 1 Feature Engineering - Quick Wins
========================================
Features:
1. Rolling EPA (3-week, 5-week) for offense and defense
2. Primetime flags (TNF, SNF, MNF)
3. Surface type (grass vs turf)
4. Extreme weather flags (high wind, precipitation)
5. QB consistency flag (same QB as last week)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")


def add_primetime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add primetime game indicators."""
    df = df.copy()
    
    # TNF = Thursday games
    df['is_tnf'] = (df['weekday'] == 'Thursday').astype(int)
    
    # MNF = Monday games
    df['is_mnf'] = (df['weekday'] == 'Monday').astype(int)
    
    # SNF = Sunday night (usually 8:20pm ET games)
    df['is_snf'] = 0
    if 'gametime' in df.columns:
        df['is_snf'] = (
            (df['weekday'] == 'Sunday') & 
            (df['gametime'].str.contains('20:', na=False) | df['gametime'].str.contains('8:', na=False))
        ).astype(int)
    
    # Any primetime
    df['is_primetime'] = (df['is_tnf'] | df['is_mnf'] | df['is_snf']).astype(int)
    
    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add surface type features."""
    df = df.copy()
    
    # Is grass (vs any type of turf)
    df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
    
    # Is artificial turf
    df['is_turf'] = (~df['surface'].str.lower().str.contains('grass', na=False)).astype(int)
    
    return df


def add_extreme_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add extreme weather condition flags."""
    df = df.copy()
    
    # Already have is_cold (temp < 40)
    # Add more granular weather flags
    df['is_freezing'] = (df['temp'] < 32).astype(int)
    df['is_very_cold'] = (df['temp'] < 20).astype(int)
    
    # Wind flags
    df['is_windy'] = (df['wind'] > 15).astype(int)
    df['is_very_windy'] = (df['wind'] > 20).astype(int)
    
    # Combined bad weather
    df['bad_weather'] = ((df['is_cold'] == 1) | (df['is_windy'] == 1)).astype(int)
    
    return df


def compute_rolling_epa(years: list, window: int = 3) -> pd.DataFrame:
    """
    Compute rolling EPA metrics from play-by-play data.
    Returns team-week level EPA stats.
    """
    print(f"Computing {window}-week rolling EPA for {min(years)}-{max(years)}...")
    
    all_epa = []
    
    for year in tqdm(years, desc=f"Rolling EPA ({window}wk)"):
        try:
            pbp = nfl.import_pbp_data([year], downcast=True)
            plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
            
            if len(plays) == 0:
                continue
            
            # Offensive EPA by team-game
            off_epa = plays.groupby(['game_id', 'posteam', 'week']).agg({
                'epa': 'mean'
            }).reset_index()
            off_epa.columns = ['game_id', 'team', 'week', 'off_epa_game']
            
            # Defensive EPA (EPA allowed)
            def_epa = plays.groupby(['game_id', 'defteam', 'week']).agg({
                'epa': 'mean'
            }).reset_index()
            def_epa.columns = ['game_id', 'team', 'week', 'def_epa_game']
            
            # Merge off and def
            team_epa = off_epa.merge(def_epa, on=['game_id', 'team', 'week'], how='outer')
            team_epa['season'] = year
            
            # Sort and compute rolling average
            team_epa = team_epa.sort_values(['team', 'week'])
            
            # Rolling mean (shift by 1 to not include current game)
            for team in team_epa['team'].unique():
                mask = team_epa['team'] == team
                team_epa.loc[mask, f'off_epa_{window}wk'] = (
                    team_epa.loc[mask, 'off_epa_game']
                    .shift(1).rolling(window, min_periods=1).mean()
                )
                team_epa.loc[mask, f'def_epa_{window}wk'] = (
                    team_epa.loc[mask, 'def_epa_game']
                    .shift(1).rolling(window, min_periods=1).mean()
                )
            
            all_epa.append(team_epa)
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    if not all_epa:
        return pd.DataFrame()
    
    result = pd.concat(all_epa, ignore_index=True)
    return result


def add_all_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all TIER 1 features to a games dataframe."""
    df = add_primetime_flags(df)
    df = add_surface_features(df)
    df = add_extreme_weather_flags(df)
    return df


if __name__ == "__main__":
    from config import PROCESSED_DATA_DIR
    
    # Test on existing data
    games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    
    print(f"Original columns: {len(games.columns)}")
    games = add_all_tier1_features(games)
    print(f"After TIER 1: {len(games.columns)}")
    
    # Show new features
    new_cols = ['is_tnf', 'is_mnf', 'is_snf', 'is_primetime', 
                'is_grass', 'is_turf', 'is_freezing', 'is_very_cold',
                'is_windy', 'is_very_windy', 'bad_weather']
    print("\nNew feature distributions:")
    for col in new_cols:
        if col in games.columns:
            print(f"  {col}: {games[col].mean()*100:.1f}% positive")

