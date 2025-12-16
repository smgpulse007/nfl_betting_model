"""
NFL Betting Model - 2025 Season Predictions

This script:
1. Loads trained models (from 1999-2024 data)
2. Generates predictions for upcoming 2025 games
3. Identifies betting opportunities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels
from src.backtesting import implied_prob, kelly_stake, american_to_decimal


def main():
    print("=" * 70)
    print("NFL BETTING MODEL - 2025 PREDICTIONS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    DATA_2025 = PROCESSED_DATA_DIR.parent / "2025"
    
    # 1. Load and train models on all historical data (1999-2024)
    print("\n[1/4] Training models on historical data...")
    historical = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    completed_2025 = pd.read_parquet(DATA_2025 / "completed_2025.parquet")
    
    # Use ALL completed games for training (including 2024 and completed 2025)
    train_data = pd.concat([historical, completed_2025], ignore_index=True)
    print(f"  Training on {len(train_data)} completed games")
    
    models = NFLBettingModels()
    models.fit(train_data)
    
    # 2. Load upcoming games
    print("\n[2/4] Loading upcoming 2025 games...")
    upcoming = pd.read_parquet(DATA_2025 / "upcoming_2025.parquet")
    print(f"  Found {len(upcoming)} upcoming games")
    
    # 3. Generate predictions
    print("\n[3/4] Generating predictions...")
    predictions = models.predict(upcoming)
    
    # Merge with game details
    results = upcoming.merge(predictions, on=['game_id', 'season', 'week', 'home_team', 'away_team'])
    
    # 4. Identify betting opportunities
    print("\n[4/4] Analyzing betting opportunities...")
    
    # Calculate edges for games with moneyline odds
    results['home_implied'] = results['home_moneyline'].apply(
        lambda x: implied_prob(x) if pd.notna(x) else np.nan
    )
    results['away_implied'] = results['away_moneyline'].apply(
        lambda x: implied_prob(x) if pd.notna(x) else np.nan
    )
    
    results['home_edge'] = results['ensemble_prob'] - results['home_implied']
    results['away_edge'] = (1 - results['ensemble_prob']) - results['away_implied']
    
    # Display results
    print("\n" + "=" * 70)
    print("UPCOMING GAMES WITH PREDICTIONS")
    print("=" * 70)
    
    display_cols = [
        'game_id', 'gameday', 'away_team', 'home_team',
        'spread_line', 'ensemble_prob', 'elo_prob_x'
    ]
    
    for week in sorted(results['week'].unique()):
        week_games = results[results['week'] == week].copy()
        print(f"\n--- WEEK {week} ---")
        
        for _, game in week_games.iterrows():
            home_prob = game['ensemble_prob']
            away_prob = 1 - home_prob
            
            print(f"\n{game['away_team']} @ {game['home_team']} ({game['gameday']})")
            print(f"  Vegas Spread: {game['spread_line']}")
            print(f"  Model: {game['home_team']} {home_prob:.1%} | {game['away_team']} {away_prob:.1%}")
            
            # Show betting opportunity if odds available
            if pd.notna(game['home_moneyline']):
                print(f"  Moneylines: {game['home_team']} {game['home_moneyline']:+.0f} | {game['away_team']} {game['away_moneyline']:+.0f}")
                
                if game['home_edge'] > 0.02:
                    print(f"  🔥 BET {game['home_team']} ML - Edge: {game['home_edge']:.1%}")
                elif game['away_edge'] > 0.02:
                    print(f"  🔥 BET {game['away_team']} ML - Edge: {game['away_edge']:.1%}")
                else:
                    print(f"  ❌ No edge (home: {game['home_edge']:.1%}, away: {game['away_edge']:.1%})")
    
    # Summary of betting opportunities
    print("\n" + "=" * 70)
    print("BETTING OPPORTUNITIES (Edge >= 2%)")
    print("=" * 70)
    
    bets = []
    for _, game in results.iterrows():
        if pd.notna(game['home_moneyline']):
            if game['home_edge'] >= 0.02:
                bets.append({
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'date': game['gameday'],
                    'bet': f"{game['home_team']} ML",
                    'odds': game['home_moneyline'],
                    'model_prob': game['ensemble_prob'],
                    'implied_prob': game['home_implied'],
                    'edge': game['home_edge']
                })
            elif game['away_edge'] >= 0.02:
                bets.append({
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'date': game['gameday'],
                    'bet': f"{game['away_team']} ML",
                    'odds': game['away_moneyline'],
                    'model_prob': 1 - game['ensemble_prob'],
                    'implied_prob': game['away_implied'],
                    'edge': game['away_edge']
                })
    
    if bets:
        bets_df = pd.DataFrame(bets).sort_values('edge', ascending=False)
        print(bets_df.to_string(index=False))
    else:
        print("No betting opportunities with >= 2% edge found.")
    
    # Save predictions
    results.to_parquet(DATA_2025 / "predictions_2025.parquet", index=False)
    print(f"\nPredictions saved to {DATA_2025 / 'predictions_2025.parquet'}")
    
    return results


if __name__ == "__main__":
    results = main()

