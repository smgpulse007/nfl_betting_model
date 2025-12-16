"""
Data Loader - Loads and prepares all NFL data for modeling

This script:
1. Loads historical schedules (includes scores, odds, weather)
2. Loads play-by-play for EPA calculations
3. Creates train/test splits (train: 1999-2023, test: 2024)
4. Saves processed datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_YEARS, TEST_YEARS


def load_all_schedules() -> pd.DataFrame:
    """Load all NFL schedules from 1999-2024."""
    years = list(range(1999, 2025))
    print(f"Loading schedules for {min(years)}-{max(years)}...")
    
    schedules = nfl.import_schedules(years)
    
    # Filter to completed games only (have scores)
    completed = schedules[schedules['home_score'].notna()].copy()
    print(f"Loaded {len(completed)} completed games")
    
    return completed


def load_team_epa_by_season(years: list) -> pd.DataFrame:
    """
    Load play-by-play and compute season-to-date EPA metrics.
    Returns rolling EPA stats for each team-week.
    """
    print(f"Loading play-by-play for EPA calculations ({min(years)}-{max(years)})...")
    
    all_stats = []
    
    for year in tqdm(years, desc="Processing seasons"):
        try:
            pbp = nfl.import_pbp_data([year], downcast=True)
            
            # Filter to real plays
            plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
            
            if len(plays) == 0:
                continue
            
            # Aggregate by team-game
            team_game = plays.groupby(['game_id', 'posteam', 'week']).agg({
                'epa': ['sum', 'mean', 'count'],
                'success': 'mean',
                'yards_gained': 'mean',
            }).reset_index()
            
            team_game.columns = ['game_id', 'team', 'week', 
                                'total_epa', 'epa_play', 'plays',
                                'success_rate', 'ypp']
            team_game['season'] = year
            
            all_stats.append(team_game)
            
        except Exception as e:
            print(f"Error loading {year}: {e}")
            continue
    
    if not all_stats:
        return pd.DataFrame()
    
    return pd.concat(all_stats, ignore_index=True)


def create_game_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Create modeling features from schedule data.
    Includes: spread, total, rest, weather, etc.
    """
    df = schedules.copy()
    
    # Target variables
    df['home_win'] = (df['result'] > 0).astype(int)
    # FIXED: spread_line is "points home gives" (positive = home favorite)
    # Home covers if actual margin > spread_line
    df['home_cover'] = (df['result'] > df['spread_line']).astype(int)
    df['game_total'] = df['home_score'] + df['away_score']
    df['over_hit'] = (df['game_total'] > df['total_line']).astype(int)
    
    # Rest advantage
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    
    # Weather features (handle missing)
    df['temp'] = df['temp'].fillna(70)  # Dome games ~70
    df['wind'] = df['wind'].fillna(0)
    df['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
    df['is_cold'] = (df['temp'] < 40).astype(int)
    df['is_windy'] = (df['wind'] > 15).astype(int)
    
    # Division game
    df['div_game'] = df['div_game'].astype(int)
    
    # Implied probabilities from moneylines
    def american_to_prob(odds):
        if pd.isna(odds):
            return np.nan
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    df['home_implied_prob'] = df['home_moneyline'].apply(american_to_prob)
    df['away_implied_prob'] = df['away_moneyline'].apply(american_to_prob)
    
    return df


def prepare_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (1999-2023) and test (2024)."""
    train = df[df['season'].isin(TRAIN_YEARS)].copy()
    test = df[df['season'].isin(TEST_YEARS)].copy()
    
    print(f"Train set: {len(train)} games ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
    print(f"Test set: {len(test)} games ({TEST_YEARS})")
    
    return train, test


def main():
    """Main data loading pipeline."""
    print("=" * 60)
    print("NFL BETTING MODEL - DATA LOADER")
    print("=" * 60)
    
    # 1. Load schedules
    schedules = load_all_schedules()
    
    # 2. Create features
    games = create_game_features(schedules)
    
    # 3. Split train/test
    train, test = prepare_train_test_split(games)
    
    # 4. Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(PROCESSED_DATA_DIR / "train_games.parquet", index=False)
    test.to_parquet(PROCESSED_DATA_DIR / "test_games.parquet", index=False)
    games.to_parquet(PROCESSED_DATA_DIR / "all_games.parquet", index=False)
    
    print(f"\nSaved to {PROCESSED_DATA_DIR}")
    
    # 5. Summary stats
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    # Betting data availability
    has_spread = games['spread_line'].notna().sum()
    has_ml = games['home_moneyline'].notna().sum()
    has_total = games['total_line'].notna().sum()
    
    print(f"Games with spread line: {has_spread} ({has_spread/len(games)*100:.1f}%)")
    print(f"Games with moneyline: {has_ml} ({has_ml/len(games)*100:.1f}%)")
    print(f"Games with total line: {has_total} ({has_total/len(games)*100:.1f}%)")
    
    return games, train, test


if __name__ == "__main__":
    games, train, test = main()

