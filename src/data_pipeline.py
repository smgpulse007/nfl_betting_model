"""
Data Pipeline for NFL Betting Model

Connects to all data sources and prepares datasets for modeling.
Sources:
- nfl-data-py: Play-by-play, schedules, rosters
- Kaggle NFL Dataset: Historical betting odds (must be downloaded separately)
- The Odds API: Live odds (requires API key)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import requests
from tqdm import tqdm
import warnings

# Import nfl_data_py
try:
    import nfl_data_py as nfl
except ImportError:
    raise ImportError("Please install nfl-data-py: pip install nfl-data-py")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    TRAIN_YEARS, TEST_YEARS, NFL_DATA_YEARS,
    ODDS_API_KEY, ODDS_API_BASE_URL, ODDS_API_SPORT
)


class NFLDataPipeline:
    """Main data pipeline for NFL betting model."""
    
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # NFL-DATA-PY METHODS
    # =========================================================================
    
    def load_schedules(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Load NFL schedules with game results."""
        years = years or NFL_DATA_YEARS
        print(f"Loading schedules for {min(years)}-{max(years)}...")
        
        schedules = nfl.import_schedules(years)
        print(f"Loaded {len(schedules)} games")
        return schedules
    
    def load_pbp(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Load play-by-play data with EPA metrics."""
        years = years or NFL_DATA_YEARS
        print(f"Loading play-by-play for {min(years)}-{max(years)}...")
        
        # This can be large, load in chunks
        pbp = nfl.import_pbp_data(years, downcast=True)
        print(f"Loaded {len(pbp)} plays")
        return pbp
    
    def load_weekly_data(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Load weekly player stats."""
        years = years or NFL_DATA_YEARS
        print(f"Loading weekly data for {min(years)}-{max(years)}...")
        
        weekly = nfl.import_weekly_data(years)
        print(f"Loaded {len(weekly)} player-weeks")
        return weekly
    
    def load_team_descriptions(self) -> pd.DataFrame:
        """Load team info (names, abbreviations, divisions)."""
        return nfl.import_team_desc()
    
    def load_rosters(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Load roster data."""
        years = years or NFL_DATA_YEARS
        return nfl.import_rosters(years)
    
    # =========================================================================
    # GAME-LEVEL AGGREGATIONS
    # =========================================================================
    
    def aggregate_team_stats(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate play-by-play to team-game level stats.
        Computes EPA/play, success rate, etc.
        """
        print("Aggregating team stats from play-by-play...")
        
        # Filter to real plays (no timeouts, penalties, etc.)
        plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
        
        # Offensive stats by team-game
        off_stats = plays.groupby(['game_id', 'posteam', 'season']).agg({
            'epa': ['sum', 'mean', 'count'],
            'success': 'mean',
            'yards_gained': ['sum', 'mean'],
            'pass': 'sum',
            'rush': 'sum',
            'first_down': 'sum',
        }).reset_index()
        
        # Flatten column names
        off_stats.columns = ['game_id', 'team', 'season', 
                            'total_epa', 'epa_per_play', 'total_plays',
                            'success_rate', 'total_yards', 'yards_per_play',
                            'pass_plays', 'rush_plays', 'first_downs']
        
        print(f"Aggregated stats for {len(off_stats)} team-games")
        return off_stats
    
    # =========================================================================
    # THE ODDS API METHODS
    # =========================================================================
    
    def fetch_live_odds(self) -> Optional[Dict]:
        """Fetch current NFL odds from The Odds API."""
        if not ODDS_API_KEY:
            print("Warning: No ODDS_API_KEY set. Skipping live odds.")
            return None
        
        url = f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching odds: {response.status_code}")
            return None
    
    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    
    def save_processed_data(self, df: pd.DataFrame, name: str):
        """Save processed dataframe to parquet."""
        path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    # Test the pipeline
    pipeline = NFLDataPipeline()
    
    # Load schedules (lightweight test)
    schedules = pipeline.load_schedules([2023, 2024])
    print(schedules.head())
    print(f"\nColumns: {schedules.columns.tolist()}")

