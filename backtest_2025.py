"""
Backtest 2025 Season - Betting Performance

Simulates actual betting on 2025 completed games
using our model predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels
from src.backtesting import Backtester, implied_prob, american_to_decimal

DATA_2025 = PROCESSED_DATA_DIR.parent / "2025"


def main():
    print("=" * 70)
    print("2025 SEASON BETTING BACKTEST")
    print("=" * 70)

    # 1. Load and train models on 1999-2024
    print("\n[1/4] Training models on historical data (1999-2024)...")
    historical = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    
    models = NFLBettingModels()
    models.fit(historical)

    # 2. Load 2025 completed games
    print("\n[2/4] Loading 2025 completed games...")
    completed_2025 = pd.read_parquet(DATA_2025 / "completed_2025.parquet")
    print(f"  Games with moneyline odds: {completed_2025['home_moneyline'].notna().sum()}")

    # 3. Generate predictions
    print("\n[3/4] Generating predictions...")
    preds_2025 = models.predict(completed_2025)

    # 4. Run backtest with different edge thresholds
    print("\n[4/4] Running backtest simulations...")
    
    backtester = Backtester()
    
    all_results = []
    for min_edge in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        report = backtester.run_backtest(completed_2025, preds_2025, min_edge=min_edge)
        
        if 'error' not in report:
            all_results.append({
                'min_edge': f"{min_edge:.0%}",
                'total_bets': report['total_bets'],
                'wins': report['wins'],
                'losses': report['losses'],
                'win_rate': f"{report['win_rate']:.1%}",
                'total_staked': f"${report['total_staked']:,.0f}",
                'total_pnl': f"${report['total_pnl']:+,.0f}",
                'roi': f"{report['roi']:+.1%}",
                'final_bankroll': f"${report['final_bankroll']:,.0f}",
            })
    
    print("\n" + "=" * 70)
    print("BETTING RESULTS BY EDGE THRESHOLD")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Detailed breakdown for 2% edge
    print("\n" + "=" * 70)
    print("DETAILED BET LOG (2% Minimum Edge)")
    print("=" * 70)
    
    report = backtester.run_backtest(completed_2025, preds_2025, min_edge=0.02)
    
    if 'bets_df' in report:
        bets = report['bets_df'].copy()
        bets['result_str'] = bets['result'].map({1: '✅ WIN', 0: '❌ LOSS', -1: '⬜ PUSH'})
        
        # Weekly breakdown
        bets['week'] = bets['game_id'].str.split('_').str[1].astype(int)
        
        weekly_pnl = bets.groupby('week').agg({
            'stake': 'sum',
            'pnl': 'sum',
            'result': lambda x: (x == 1).sum()
        }).rename(columns={'result': 'wins'})
        weekly_pnl['bets'] = bets.groupby('week').size()
        weekly_pnl['roi'] = (weekly_pnl['pnl'] / weekly_pnl['stake'] * 100).round(1)
        
        print("\nWeekly P&L:")
        print(weekly_pnl.to_string())
        
        print(f"\nTotal Bets: {len(bets)}")
        print(f"Win Rate: {report['win_rate']:.1%}")
        print(f"Total Staked: ${report['total_staked']:,.2f}")
        print(f"Total P&L: ${report['total_pnl']:+,.2f}")
        print(f"ROI: {report['roi']:+.1%}")
        print(f"Starting Bankroll: $10,000")
        print(f"Final Bankroll: ${report['final_bankroll']:,.2f}")
        
        # Show individual bets
        print("\n" + "-" * 70)
        print("ALL BETS PLACED")
        print("-" * 70)
        
        display_bets = bets[['game_id', 'bet_type', 'odds', 'edge', 'stake', 'result_str', 'pnl']].copy()
        display_bets['odds'] = display_bets['odds'].apply(lambda x: f"{x:+.0f}")
        display_bets['edge'] = display_bets['edge'].apply(lambda x: f"{x:.1%}")
        display_bets['stake'] = display_bets['stake'].apply(lambda x: f"${x:.0f}")
        display_bets['pnl'] = display_bets['pnl'].apply(lambda x: f"${x:+,.0f}")
        
        print(display_bets.to_string(index=False))


def show_current_capabilities():
    """Show what the model currently predicts."""
    print("\n" + "=" * 70)
    print("CURRENT MODEL CAPABILITIES")
    print("=" * 70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    WHAT WE PREDICT                              │
    ├─────────────────────────────────────────────────────────────────┤
    │  ✅ WIN PROBABILITY    → Moneyline betting                      │
    │     - Predicts P(home_win) from 0-1                             │
    │     - Compares to implied odds for edge                         │
    │     - Uses Kelly criterion for bet sizing                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  ❌ SPREAD PREDICTION  → Not yet implemented                    │
    │     - Would predict margin of victory                           │
    │     - Compare to Vegas spread for ATS betting                   │
    ├─────────────────────────────────────────────────────────────────┤
    │  ❌ TOTALS PREDICTION  → Not yet implemented                    │
    │     - Would predict combined score                              │
    │     - Compare to Vegas total for O/U betting                    │
    └─────────────────────────────────────────────────────────────────┘
    
    TO ADD SPREAD/TOTALS, WE NEED:
    1. XGBoost regressor trained on (result + spread) for ATS
    2. XGBoost regressor trained on (home_score + away_score) for O/U
    3. Convert point predictions to probabilities using historical variance
    """)


if __name__ == "__main__":
    main()
    show_current_capabilities()

