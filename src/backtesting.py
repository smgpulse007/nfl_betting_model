"""
Backtesting Engine for NFL Betting Model

Simulates betting performance with:
- Walk-forward validation
- Kelly criterion sizing
- CLV tracking
- ROI calculation

Supports:
- Moneyline betting (home/away ML)
- Spread betting (ATS - against the spread)
- Totals betting (over/under)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, BACKTEST_CONFIG, KELLY_FRACTION, MIN_EDGE_THRESHOLD


# Standard spread/totals odds (typically -110)
STANDARD_SPREAD_ODDS = -110
STANDARD_TOTAL_ODDS = -110


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
    line: Optional[float] = None  # Spread or total line


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
                     min_edge: float = MIN_EDGE_THRESHOLD,
                     bet_types: List[str] = None) -> Dict:
        """
        Run backtest on games with model predictions.

        Args:
            games: DataFrame with game results and odds
            predictions: DataFrame with model probabilities
            min_edge: Minimum edge required to place bet
            bet_types: List of bet types to include ['moneyline', 'spread', 'totals']
                       Default is ['moneyline'] for backward compatibility
        """
        if bet_types is None:
            bet_types = ['moneyline']

        print(f"Running backtest with min_edge={min_edge:.1%}, bet_types={bet_types}...")

        # Merge predictions with games
        df = games.merge(predictions, on=['game_id', 'season', 'week', 'home_team', 'away_team'])

        bankroll = self.config['initial_bankroll']
        self.bankroll_history = [bankroll]
        self.bets = []

        for _, row in df.iterrows():
            bets_this_game = []

            # =====================
            # MONEYLINE BETS
            # =====================
            if 'moneyline' in bet_types:
                if not pd.isna(row.get('home_moneyline')) and not pd.isna(row.get('away_moneyline')):
                    model_prob = row['ensemble_prob']
                    home_implied = implied_prob(row['home_moneyline'])
                    away_implied = implied_prob(row['away_moneyline'])

                    home_edge = model_prob - home_implied
                    away_edge = (1 - model_prob) - away_implied

                    # Home ML
                    if home_edge >= min_edge:
                        stake_pct = kelly_stake(home_edge, row['home_moneyline'])
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0 and row['result'] != 0:  # Skip ties
                            if row['result'] > 0:
                                pnl = stake * (american_to_decimal(row['home_moneyline']) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='home_ml', odds=row['home_moneyline'], stake=stake,
                                model_prob=model_prob, implied_prob=home_implied,
                                edge=home_edge, result=result, pnl=pnl
                            ))

                    # Away ML
                    elif away_edge >= min_edge:
                        stake_pct = kelly_stake(away_edge, row['away_moneyline'])
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0 and row['result'] != 0:
                            if row['result'] < 0:
                                pnl = stake * (american_to_decimal(row['away_moneyline']) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='away_ml', odds=row['away_moneyline'], stake=stake,
                                model_prob=1-model_prob, implied_prob=away_implied,
                                edge=away_edge, result=result, pnl=pnl
                            ))

            # =====================
            # SPREAD BETS (ATS)
            # =====================
            if 'spread' in bet_types:
                # Handle column naming after merge (spread_line_x from games, spread_line_y from preds)
                spread_col = 'spread_line_x' if 'spread_line_x' in row.index else 'spread_line'
                spread = row.get(spread_col)

                if not pd.isna(spread) and 'home_cover_prob' in row.index:
                    spread_implied = 0.5  # Standard -110 odds imply 50%
                    home_cover_prob = row['home_cover_prob']
                    away_cover_prob = row['away_cover_prob']

                    home_spread_edge = home_cover_prob - spread_implied
                    away_spread_edge = away_cover_prob - spread_implied

                    # Actual result
                    actual_margin = row['result']
                    home_covered = (actual_margin + spread) > 0
                    away_covered = (actual_margin + spread) < 0
                    is_push = (actual_margin + spread) == 0

                    # Home spread bet
                    if home_spread_edge >= min_edge and not is_push:
                        stake_pct = kelly_stake(home_spread_edge, STANDARD_SPREAD_ODDS)
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0:
                            if home_covered:
                                pnl = stake * (american_to_decimal(STANDARD_SPREAD_ODDS) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='home_spread', odds=STANDARD_SPREAD_ODDS, stake=stake,
                                model_prob=home_cover_prob, implied_prob=spread_implied,
                                edge=home_spread_edge, result=result, pnl=pnl, line=spread
                            ))

                    # Away spread bet
                    elif away_spread_edge >= min_edge and not is_push:
                        stake_pct = kelly_stake(away_spread_edge, STANDARD_SPREAD_ODDS)
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0:
                            if away_covered:
                                pnl = stake * (american_to_decimal(STANDARD_SPREAD_ODDS) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='away_spread', odds=STANDARD_SPREAD_ODDS, stake=stake,
                                model_prob=away_cover_prob, implied_prob=spread_implied,
                                edge=away_spread_edge, result=result, pnl=pnl, line=-spread
                            ))

            # =====================
            # TOTALS BETS (O/U)
            # =====================
            if 'totals' in bet_types:
                # Handle column naming after merge
                total_col = 'total_line_x' if 'total_line_x' in row.index else 'total_line'
                total_line = row.get(total_col)

                if not pd.isna(total_line) and 'over_prob' in row.index:
                    total_implied = 0.5  # Standard -110 odds
                    over_prob = row['over_prob']
                    under_prob = row['under_prob']

                    over_edge = over_prob - total_implied
                    under_edge = under_prob - total_implied

                    # Actual result (total_line already set above)
                    actual_total = row['home_score'] + row['away_score']
                    went_over = actual_total > total_line
                    went_under = actual_total < total_line
                    is_push = actual_total == total_line

                    # Over bet
                    if over_edge >= min_edge and not is_push:
                        stake_pct = kelly_stake(over_edge, STANDARD_TOTAL_ODDS)
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0:
                            if went_over:
                                pnl = stake * (american_to_decimal(STANDARD_TOTAL_ODDS) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='over', odds=STANDARD_TOTAL_ODDS, stake=stake,
                                model_prob=over_prob, implied_prob=total_implied,
                                edge=over_edge, result=result, pnl=pnl, line=total_line
                            ))

                    # Under bet
                    elif under_edge >= min_edge and not is_push:
                        stake_pct = kelly_stake(under_edge, STANDARD_TOTAL_ODDS)
                        stake = min(bankroll * stake_pct, bankroll * self.config['max_bet_pct'])

                        if stake > 0:
                            if went_under:
                                pnl = stake * (american_to_decimal(STANDARD_TOTAL_ODDS) - 1)
                                result = 1
                            else:
                                pnl = -stake
                                result = 0

                            bets_this_game.append(Bet(
                                game_id=row['game_id'], season=row['season'], week=row['week'],
                                bet_type='under', odds=STANDARD_TOTAL_ODDS, stake=stake,
                                model_prob=under_prob, implied_prob=total_implied,
                                edge=under_edge, result=result, pnl=pnl, line=total_line
                            ))

            # Process all bets for this game
            for bet in bets_this_game:
                self.bets.append(bet)
                bankroll += bet.pnl
                self.bankroll_history.append(bankroll)

        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate backtest performance report with breakdown by bet type."""
        if not self.bets:
            return {"error": "No bets placed"}

        bets_df = pd.DataFrame([vars(b) for b in self.bets])

        total_bets = len(self.bets)
        wins = sum(1 for b in self.bets if b.result == 1)
        losses = sum(1 for b in self.bets if b.result == 0)

        total_staked = bets_df['stake'].sum()
        total_pnl = bets_df['pnl'].sum()

        # Breakdown by bet type
        by_type = {}
        for bet_type in bets_df['bet_type'].unique():
            type_df = bets_df[bets_df['bet_type'] == bet_type]
            type_wins = (type_df['result'] == 1).sum()
            type_total = len(type_df)
            type_staked = type_df['stake'].sum()
            type_pnl = type_df['pnl'].sum()

            by_type[bet_type] = {
                'bets': type_total,
                'wins': type_wins,
                'losses': type_total - type_wins,
                'win_rate': type_wins / type_total if type_total > 0 else 0,
                'staked': type_staked,
                'pnl': type_pnl,
                'roi': type_pnl / type_staked if type_staked > 0 else 0
            }

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
            'by_type': by_type,
            'bets_df': bets_df
        }

