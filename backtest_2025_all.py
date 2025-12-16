"""
2025 Season Backtest - All Bet Types

Tests moneyline, spread, and totals betting on completed 2025 games.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels
from src.backtesting import Backtester


def run_2025_backtest():
    """Run backtest on 2025 completed games."""
    
    # Load training data (1999-2024)
    print("Loading data...")
    games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    train = games  # Use all historical data for training
    
    # Load 2025 completed games
    completed_2025 = pd.read_parquet(Path("data/2025/completed_2025.parquet"))
    print(f"Train: {len(train)} games (1999-2024)")
    print(f"2025 Completed: {len(completed_2025)} games")
    
    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    models = NFLBettingModels()
    models.fit(train)
    
    # Generate predictions for 2025
    print("\nGenerating predictions for 2025...")
    predictions = models.predict(completed_2025)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("MODEL EVALUATION - 2025 Season (Weeks 1-15)")
    print("=" * 70)
    results = models.evaluate(completed_2025, predictions)
    
    print("\n--- MONEYLINE ---")
    for model, metrics in results['moneyline'].items():
        name = model.replace('_prob', '').replace('_win', '')
        print(f"  {name}: {metrics['accuracy']:.1%} accuracy, {metrics['brier_score']:.4f} Brier")
    
    print("\n--- SPREAD (ATS) ---")
    if results['spread']:
        print(f"  Margin MAE: {results['spread']['margin_mae']:.2f} points")
        print(f"  Cover Accuracy: {results['spread']['cover_accuracy']:.1%}")
        print(f"  Cover Brier: {results['spread']['cover_brier']:.4f}")
    
    print("\n--- TOTALS (O/U) ---")
    if results['totals']:
        print(f"  Total MAE: {results['totals']['total_mae']:.2f} points")
        print(f"  O/U Accuracy: {results['totals']['over_accuracy']:.1%}")
        print(f"  O/U Brier: {results['totals']['over_brier']:.4f}")
    
    # Run backtests
    print("\n" + "=" * 70)
    print("BETTING BACKTEST - 2025 Season")
    print("=" * 70)
    
    backtester = Backtester()
    
    configs = [
        {'name': 'Moneyline Only', 'types': ['moneyline']},
        {'name': 'Spread Only', 'types': ['spread']},
        {'name': 'Totals Only', 'types': ['totals']},
        {'name': 'All Bet Types', 'types': ['moneyline', 'spread', 'totals']},
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        report = backtester.run_backtest(
            completed_2025, predictions, 
            min_edge=0.02, 
            bet_types=config['types']
        )
        
        if 'error' in report:
            print(f"  {report['error']}")
            continue
        
        print(f"  Total Bets: {report['total_bets']}")
        print(f"  Win Rate: {report['win_rate']:.1%}")
        print(f"  Total Staked: ${report['total_staked']:,.0f}")
        print(f"  P&L: ${report['total_pnl']:+,.0f}")
        print(f"  ROI: {report['roi']:+.1%}")
        
        if 'by_type' in report:
            print("\n  By Bet Type:")
            for bt, stats in report['by_type'].items():
                print(f"    {bt}: {stats['bets']} bets, {stats['win_rate']:.1%} win, {stats['roi']:+.1%} ROI")
        
        all_results[config['name']] = {
            'bets': report['total_bets'],
            'win_rate': report['win_rate'],
            'roi': report['roi'],
            'pnl': report['total_pnl']
        }
    
    # Save results
    results_path = Path("results/backtest_2025_all_types.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return all_results


if __name__ == "__main__":
    run_2025_backtest()

