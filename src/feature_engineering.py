"""
Feature Engineering for NFL Betting Model

Creates predictive features:
1. Elo ratings (with QB adjustment)
2. Rolling EPA metrics
3. Opponent-adjusted stats
4. Situational features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ELO_CONFIG, ROLLING_WINDOWS


class EloRatingSystem:
    """
    Elo rating system for NFL teams.
    Based on FiveThirtyEight methodology.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or ELO_CONFIG
        self.ratings: Dict[str, float] = {}
        self.history: list = []
    
    def _init_rating(self, team: str) -> float:
        """Initialize team rating."""
        if team not in self.ratings:
            self.ratings[team] = self.config['initial_rating']
        return self.ratings[team]
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for team A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def _margin_multiplier(self, margin: int) -> float:
        """
        Margin of victory multiplier.
        Uses log transform to reduce impact of blowouts.
        """
        return np.log(abs(margin) + 1) * (2.2 / (1 + 0.001 * abs(margin)))
    
    def update(self, home_team: str, away_team: str, 
               home_score: int, away_score: int,
               is_playoff: bool = False) -> Tuple[float, float]:
        """
        Update Elo ratings after a game.
        Returns (home_new_rating, away_new_rating)
        """
        # Get current ratings
        home_elo = self._init_rating(home_team)
        away_elo = self._init_rating(away_team)
        
        # Add home field advantage
        home_elo_adj = home_elo + self.config['home_advantage']
        
        # Expected scores
        home_expected = self._expected_score(home_elo_adj, away_elo)
        away_expected = 1 - home_expected
        
        # Actual scores (1 for win, 0.5 for tie, 0 for loss)
        margin = home_score - away_score
        if margin > 0:
            home_actual, away_actual = 1, 0
        elif margin < 0:
            home_actual, away_actual = 0, 1
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # K-factor with margin multiplier
        k = self.config['k_factor']
        if is_playoff:
            k *= self.config['playoff_multiplier']
        
        mov_mult = self._margin_multiplier(margin)
        
        # Update ratings
        home_delta = k * mov_mult * (home_actual - home_expected)
        away_delta = k * mov_mult * (away_actual - away_expected)
        
        self.ratings[home_team] = home_elo + home_delta
        self.ratings[away_team] = away_elo + away_delta
        
        return self.ratings[home_team], self.ratings[away_team]
    
    def regress_to_mean(self, factor: float = None):
        """Regress all ratings toward mean at season start."""
        factor = factor or self.config['mean_reversion']
        mean = self.config['initial_rating']
        
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - factor) + mean * factor
            )
    
    def get_rating(self, team: str) -> float:
        """Get current rating for a team."""
        return self._init_rating(team)
    
    def get_win_prob(self, home_team: str, away_team: str) -> float:
        """Get win probability for home team."""
        home_elo = self.get_rating(home_team) + self.config['home_advantage']
        away_elo = self.get_rating(away_team)
        return self._expected_score(home_elo, away_elo)


def compute_elo_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Elo ratings for all games chronologically.
    Returns games with Elo features added.
    """
    print("Computing Elo ratings...")
    
    # Sort by date
    df = games.sort_values(['season', 'week', 'game_id']).copy()
    
    elo = EloRatingSystem()
    
    # Store pre-game Elo for each game
    home_elos = []
    away_elos = []
    elo_diffs = []
    win_probs = []
    
    current_season = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Elo"):
        # Regress at season start
        if current_season != row['season']:
            if current_season is not None:
                elo.regress_to_mean()
            current_season = row['season']
        
        # Get pre-game ratings
        home_elo = elo.get_rating(row['home_team'])
        away_elo = elo.get_rating(row['away_team'])
        win_prob = elo.get_win_prob(row['home_team'], row['away_team'])

        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)
        win_probs.append(win_prob)

        # Update ratings after game (only if game is completed)
        if pd.notna(row['home_score']) and pd.notna(row['away_score']):
            is_playoff = row['game_type'] != 'REG'
            elo.update(
                row['home_team'], row['away_team'],
                int(row['home_score']), int(row['away_score']),
                is_playoff=is_playoff
            )
    
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = elo_diffs
    df['elo_prob'] = win_probs
    
    return df


if __name__ == "__main__":
    from config import PROCESSED_DATA_DIR
    
    # Load games
    games = pd.read_parquet(PROCESSED_DATA_DIR / "all_games.parquet")
    
    # Compute Elo
    games_with_elo = compute_elo_features(games)
    
    # Save
    games_with_elo.to_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    print(f"\nSaved games with Elo to {PROCESSED_DATA_DIR / 'games_with_features.parquet'}")
    
    # Quick validation
    print("\nElo Feature Summary:")
    print(games_with_elo[['home_elo', 'away_elo', 'elo_diff', 'elo_prob']].describe())

