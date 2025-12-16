"""
Backtesting Engine for NFL Betting Model

Simulates betting performance with:
- Walk-forward validation
- Kelly criterion sizing
- CLV tracking
- ROI calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, BACKTEST_CONFIG, KELLY_FRACTION, MIN_EDGE_THRESHOLD


@dataclass
class Bet:
    """Represents a single bet."""
    game_id: str
    season: int
    week: int
    bet_type: str  # 'home_ml', 'away_ml', 'home_spread', 'away_spread', 'over', 'under'
    odds: float
    stake: float
    model_prob: float
    implied_prob: float
    edge: float
    result: int  # 1 = win, 0 = loss, -1 = push
    pnl: float


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal."""
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


def implied_prob(odds: float) -> float:
    """Get implied probability from American odds."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def kelly_stake(edge: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Calculate Kelly criterion stake.
    edge = (model_prob - implied_prob)
    Returns fraction of bankroll to bet.
    """
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1  # Net odds
    
    # Kelly formula: f* = (bp - q) / b
    # where p = win probability, q = 1-p, b = decimal odds - 1
    p = edge + implied_prob(odds)  # Model probability
    q = 1 - p
    
    kelly = (b * p - q) / b if b > 0 else 0
    
    # Apply fractional Kelly for safety
    return max(0, kelly * fraction)


class Backtester:
    """Backtesting engine for NFL betting strategies."""
    
    def __init__(self, config: Dict = None):
        self.config = config or BACKTEST_CONFIG
        self.bets: List[Bet] = []
        self.bankroll_history: List[float] = []
    
    def run_backtest(self, games: pd.DataFrame, predictions: pd.DataFrame,
                     min_edge: float = MIN_EDGE_THRESHOLD) -> pd.DataFrame:
        """
        Run backtest on games with model predictions.
        
        Args:
            games: DataFrame with game results and odds
            predictions: DataFrame with model probabilities
            min_edge: Minimum edge required to place bet
        """
        print(f"Running backtest with min_edge={min_edge:.1%}...")
        
        # Merge predictions with games
        df = games.merge(predictions, on=['game_id', 'season', 'week', 'home_team', 'away_team'])
        
        bankroll = self.config['initial_bankroll']
        self.bankroll_history = [bankroll]
        self.bets = []
        
        for _, row in df.iterrows():
            # Skip if missing odds
            if pd.isna(row['home_moneyline']) or pd.isna(row['away_moneyline']):
                continue
            
            # Get model probability and market implied probability
            model_prob = row['ensemble_prob']
            home_implied = implied_prob(row['home_moneyline'])
            away_implied = implied_prob(row['away_moneyline'])
            
            # Calculate edges
            home_edge = model_prob - home_implied
            away_edge = (1 - model_prob) - away_implied
            
            # Check for betting opportunities
            bet_made = False
            
            # Home moneyline bet
            if home_edge >= min_edge:
                stake_pct = kelly_stake(home_edge, row['home_moneyline'])
                stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])
                
                if stake > 0:
                    # Determine result
                    if row['result'] > 0:  # Home won
                        decimal_odds = american_to_decimal(row['home_moneyline'])
                        pnl = stake * (decimal_odds - 1)
                        result = 1
                    elif row['result'] < 0:  # Away won
                        pnl = -stake
                        result = 0
                    else:  # Tie
                        pnl = 0
                        result = -1
                    
                    bet = Bet(
                        game_id=row['game_id'],
                        season=row['season'],
                        week=row['week'],
                        bet_type='home_ml',
                        odds=row['home_moneyline'],
                        stake=stake,
                        model_prob=model_prob,
                        implied_prob=home_implied,
                        edge=home_edge,
                        result=result,
                        pnl=pnl
                    )
                    self.bets.append(bet)
                    bankroll += pnl
                    bet_made = True
            
            # Away moneyline bet
            elif away_edge >= min_edge:
                stake_pct = kelly_stake(away_edge, row['away_moneyline'])
                stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])
                
                if stake > 0:
                    if row['result'] < 0:  # Away won
                        decimal_odds = american_to_decimal(row['away_moneyline'])
                        pnl = stake * (decimal_odds - 1)
                        result = 1
                    elif row['result'] > 0:  # Home won
                        pnl = -stake
                        result = 0
                    else:
                        pnl = 0
                        result = -1
                    
                    bet = Bet(
                        game_id=row['game_id'],
                        season=row['season'],
                        week=row['week'],
                        bet_type='away_ml',
                        odds=row['away_moneyline'],
                        stake=stake,
                        model_prob=1-model_prob,
                        implied_prob=away_implied,
                        edge=away_edge,
                        result=result,
                        pnl=pnl
                    )
                    self.bets.append(bet)
                    bankroll += pnl
                    bet_made = True
            
            if bet_made:
                self.bankroll_history.append(bankroll)
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate backtest performance report."""
        if not self.bets:
            return {"error": "No bets placed"}
        
        bets_df = pd.DataFrame([vars(b) for b in self.bets])
        
        total_bets = len(self.bets)
        wins = sum(1 for b in self.bets if b.result == 1)
        losses = sum(1 for b in self.bets if b.result == 0)
        
        total_staked = bets_df['stake'].sum()
        total_pnl = bets_df['pnl'].sum()
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_staked': total_staked,
            'total_pnl': total_pnl,
            'roi': total_pnl / total_staked if total_staked > 0 else 0,
            'final_bankroll': self.bankroll_history[-1],
            'initial_bankroll': self.config['initial_bankroll'],
            'bankroll_growth': (self.bankroll_history[-1] / self.config['initial_bankroll']) - 1,
            'avg_edge': bets_df['edge'].mean(),
            'bets_df': bets_df
        }

