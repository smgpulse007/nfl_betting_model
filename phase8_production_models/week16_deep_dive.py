"""
================================================================================
WEEK 16 DEEP DIVE ANALYSIS
================================================================================

Comprehensive analysis of 2024 Week 16 performance and 2025 Week 16 predictions.

Author: NFL Betting Model v0.4.0
Date: 2025-12-27
================================================================================
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("WEEK 16 DEEP DIVE ANALYSIS")
print("=" * 120)

# =============================================================================
# PART 1: 2024 WEEK 16 RETROSPECTIVE
# =============================================================================
print(f"\n{'='*120}")
print("PART 1: 2024 WEEK 16 RETROSPECTIVE (ACTUAL RESULTS)")
print("=" * 120)

# Load 2024 analysis
df_2024 = pd.read_csv('../results/phase8_results/2024_week16_17_analysis.csv')
w16_2024 = df_2024[df_2024['week'] == 16].copy()

print(f"\nTotal Games: {len(w16_2024)}")
print(f"Correct Predictions: {w16_2024['correct'].sum()}")
print(f"Accuracy: {w16_2024['correct'].mean():.1%}")

print(f"\n{'─'*120}")
print("GAME-BY-GAME RESULTS")
print("─" * 120)
print(f"{'Away Team':<12} {'@':<3} {'Home Team':<12} {'Score':<12} {'Predicted':<12} {'Confidence':<12} {'Result':<10}")
print("─" * 120)

for idx, row in w16_2024.iterrows():
    score = f"{row['away_score']:.0f}-{row['home_score']:.0f}"
    result = "✅ CORRECT" if row['correct'] == 1 else "❌ WRONG"
    print(f"{row['away_team']:<12} @ {row['home_team']:<12} {score:<12} "
          f"{row['predicted_winner']:<12} {row['confidence']:.1%}{'':>6} {result:<10}")

# Analyze misses
print(f"\n{'─'*120}")
print("MISSED PREDICTIONS ANALYSIS")
print("─" * 120)

misses = w16_2024[w16_2024['correct'] == 0].copy()
if len(misses) > 0:
    print(f"\nTotal Misses: {len(misses)}")
    print(f"\nDetailed Miss Analysis:")
    for idx, row in misses.iterrows():
        print(f"\n  Game: {row['away_team']} @ {row['home_team']}")
        print(f"  Predicted: {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
        print(f"  Actual: {row['actual_winner']} won {row['away_score']:.0f}-{row['home_score']:.0f}")
        print(f"  Model Probabilities:")
        print(f"    - XGBoost: {row['XGBoost_prob']:.1%}")
        print(f"    - LightGBM: {row['LightGBM_prob']:.1%}")
        print(f"    - CatBoost: {row['CatBoost_prob']:.1%}")
        print(f"    - RandomForest: {row['RandomForest_prob']:.1%}")
        print(f"    - Ensemble: {row['Ensemble_prob']:.1%}")
else:
    print("\n🎯 PERFECT WEEK! All predictions correct!")

# Confidence analysis
print(f"\n{'─'*120}")
print("CONFIDENCE ANALYSIS")
print("─" * 120)

print(f"\nAverage Confidence: {w16_2024['confidence'].mean():.1%}")
print(f"Highest Confidence: {w16_2024['confidence'].max():.1%}")
print(f"Lowest Confidence: {w16_2024['confidence'].min():.1%}")

# Accuracy by confidence level
high_conf = w16_2024[w16_2024['confidence'] >= 0.65]
med_conf = w16_2024[(w16_2024['confidence'] >= 0.55) & (w16_2024['confidence'] < 0.65)]
low_conf = w16_2024[w16_2024['confidence'] < 0.55]

print(f"\nAccuracy by Confidence Level:")
if len(high_conf) > 0:
    print(f"  High (≥65%): {high_conf['correct'].mean():.1%} ({high_conf['correct'].sum()}/{len(high_conf)} games)")
if len(med_conf) > 0:
    print(f"  Medium (55-65%): {med_conf['correct'].mean():.1%} ({med_conf['correct'].sum()}/{len(med_conf)} games)")
if len(low_conf) > 0:
    print(f"  Low (<55%): {low_conf['correct'].mean():.1%} ({low_conf['correct'].sum()}/{len(low_conf)} games)")

# =============================================================================
# PART 2: 2025 WEEK 16 PREVIEW
# =============================================================================
print(f"\n{'='*120}")
print("PART 2: 2025 WEEK 16 PREVIEW (UPCOMING PREDICTIONS)")
print("=" * 120)

# Load 2025 predictions
df_2025 = pd.read_csv('../results/phase8_results/2025_predictions.csv')
w16_2025 = df_2025[df_2025['week'] == 16].copy()

print(f"\nTotal Games: {len(w16_2025)}")
print(f"Average Confidence: {w16_2025['confidence'].mean():.1%}")

print(f"\n{'─'*120}")
print("2025 WEEK 16 PREDICTIONS")
print("─" * 120)
print(f"{'Date':<12} {'Away Team':<12} {'@':<3} {'Home Team':<12} {'Predicted Winner':<18} {'Confidence':<12}")
print("─" * 120)

for idx, row in w16_2025.iterrows():
    date = row['gameday'][:10] if pd.notna(row['gameday']) else 'TBD'
    print(f"{date:<12} {row['away_team']:<12} @ {row['home_team']:<12} "
          f"{row['predicted_winner']:<18} {row['confidence']:.1%}")

# Top picks
print(f"\n{'─'*120}")
print("TOP 5 MOST CONFIDENT PICKS")
print("─" * 120)

top_picks = w16_2025.nlargest(5, 'confidence')
for i, (idx, row) in enumerate(top_picks.iterrows(), 1):
    print(f"{i}. {row['predicted_winner']} over {row['away_team'] if row['predicted_winner'] == row['home_team'] else row['home_team']} "
          f"({row['confidence']:.1%} confidence)")

# Load betting recommendations
df_bets = pd.read_csv('../results/phase8_results/2025_betting_recommendations.csv')
w16_bets = df_bets[df_bets['week'] == 16].copy()

print(f"\n{'─'*120}")
print("TOP 5 BETTING OPPORTUNITIES (by Expected Value)")
print("─" * 120)

top_bets = w16_bets.nlargest(5, 'expected_value')
for i, (idx, row) in enumerate(top_bets.iterrows(), 1):
    print(f"{i}. {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
    print(f"   Edge: {row['edge']:.2%} | EV: {row['expected_value']:.2%}")
    print(f"   Kelly Bet: ${row['kelly_bet']:.2f} | Fixed Stake: ${row['fixed_stake_bet']:.2f}")

print(f"\n{'='*120}")
print("DEEP DIVE COMPLETE")
print("=" * 120)

