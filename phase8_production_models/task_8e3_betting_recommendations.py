"""
================================================================================
TASK 8E.3: EXPORT BETTING RECOMMENDATIONS
================================================================================

Generate betting recommendations for 2025 season based on model predictions.

Applies 4 betting strategies:
1. Kelly Criterion
2. Fixed Stake
3. Confidence Threshold
4. Proportional Betting

Author: NFL Betting Model v0.4.0
Date: 2025-12-27
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("TASK 8E.3: EXPORT BETTING RECOMMENDATIONS")
print("=" * 120)

# =============================================================================
# STEP 1: LOAD 2025 PREDICTIONS
# =============================================================================
print(f"\n[1/4] Loading 2025 predictions...")

df = pd.read_csv('../results/phase8_results/2025_predictions.csv')

print(f"  ✅ Loaded {len(df)} predictions")
print(f"  ✅ Columns: {df.shape[1]}")

# Filter to upcoming games only (no scores yet)
upcoming = df[df['home_score'].isna()].copy()
completed = df[df['home_score'].notna()].copy()

print(f"\n  📊 Game Status:")
print(f"     - Completed: {len(completed)}")
print(f"     - Upcoming: {len(upcoming)}")

# =============================================================================
# STEP 2: CALCULATE BETTING METRICS
# =============================================================================
print(f"\n[2/4] Calculating betting metrics...")

# Assume standard -110 odds (1.91 decimal odds)
# This means you need to bet $110 to win $100
standard_odds = 1.91
implied_prob = 1 / standard_odds  # ~0.5236

# Calculate edge (model probability - implied probability)
upcoming['edge'] = upcoming['confidence'] - implied_prob

# Calculate expected value (EV)
# EV = (prob_win * profit) - (prob_loss * stake)
# For -110 odds: profit = 100/110 = 0.909, stake = 1
upcoming['expected_value'] = (upcoming['confidence'] * 0.909) - ((1 - upcoming['confidence']) * 1)

# Kelly Criterion: fraction of bankroll to bet
# Kelly = (p * b - q) / b, where p=win_prob, q=loss_prob, b=odds-1
# For -110 odds: b = 0.909
upcoming['kelly_fraction'] = ((upcoming['confidence'] * 0.909) - (1 - upcoming['confidence'])) / 0.909

# Cap Kelly at 5% to avoid over-betting
upcoming['kelly_fraction'] = upcoming['kelly_fraction'].clip(0, 0.05)

print(f"  ✅ Calculated edge, EV, and Kelly fractions")
print(f"  ✅ Positive EV bets: {len(upcoming[upcoming['expected_value'] > 0])}")

# =============================================================================
# STEP 3: APPLY BETTING STRATEGIES
# =============================================================================
print(f"\n[3/4] Applying betting strategies...")

# Assume $10,000 bankroll
bankroll = 10000

# Strategy 1: Kelly Criterion
upcoming['kelly_bet'] = upcoming['kelly_fraction'] * bankroll
upcoming['kelly_bet'] = upcoming['kelly_bet'].round(2)

# Strategy 2: Fixed Stake (2% of bankroll)
fixed_stake = bankroll * 0.02
upcoming['fixed_stake_bet'] = np.where(upcoming['expected_value'] > 0, fixed_stake, 0)

# Strategy 3: Confidence Threshold (only bet if confidence > 65%)
threshold = 0.65
upcoming['threshold_bet'] = np.where(upcoming['confidence'] > threshold, fixed_stake, 0)

# Strategy 4: Proportional (bet size proportional to confidence)
# Scale from 0% to 5% of bankroll based on confidence (60% to 80%)
min_conf = 0.60
max_conf = 0.80
upcoming['proportional_bet'] = np.where(
    upcoming['confidence'] > min_conf,
    ((upcoming['confidence'] - min_conf) / (max_conf - min_conf)) * bankroll * 0.05,
    0
)
upcoming['proportional_bet'] = upcoming['proportional_bet'].clip(0, bankroll * 0.05).round(2)

print(f"  ✅ Applied 4 betting strategies")
print(f"  ✅ Kelly bets: {len(upcoming[upcoming['kelly_bet'] > 0])}")
print(f"  ✅ Fixed stake bets: {len(upcoming[upcoming['fixed_stake_bet'] > 0])}")
print(f"  ✅ Threshold bets: {len(upcoming[upcoming['threshold_bet'] > 0])}")
print(f"  ✅ Proportional bets: {len(upcoming[upcoming['proportional_bet'] > 0])}")

# =============================================================================
# STEP 4: CREATE RECOMMENDATIONS AND EXPORT
# =============================================================================
print(f"\n[4/4] Creating betting recommendations...")

# Select columns for export
recommendations = upcoming[[
    'game_id', 'season', 'week', 'gameday', 'weekday',
    'home_team', 'away_team', 'predicted_winner',
    'home_win_probability', 'away_win_probability', 'confidence',
    'edge', 'expected_value',
    'kelly_bet', 'fixed_stake_bet', 'threshold_bet', 'proportional_bet',
    'confidence_category'
]].copy()

# Sort by expected value (descending)
recommendations = recommendations.sort_values('expected_value', ascending=False)

# Save to CSV
output_path = '../results/phase8_results/2025_betting_recommendations.csv'
recommendations.to_csv(output_path, index=False)

print(f"  ✅ Created recommendations: {recommendations.shape}")
print(f"  ✅ Saved to: {output_path}")

# Print summary statistics
print(f"\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print(f"\n📊 Betting Recommendations:")
print(f"   - Total upcoming games: {len(upcoming)}")
print(f"   - Positive EV bets: {len(recommendations[recommendations['expected_value'] > 0])}")
print(f"   - Average confidence: {recommendations['confidence'].mean():.1%}")
print(f"   - Average edge: {recommendations['edge'].mean():.2%}")

print(f"\n💰 Recommended Bet Amounts (Total):")
print(f"   - Kelly Criterion: ${recommendations['kelly_bet'].sum():,.2f}")
print(f"   - Fixed Stake: ${recommendations['fixed_stake_bet'].sum():,.2f}")
print(f"   - Confidence Threshold: ${recommendations['threshold_bet'].sum():,.2f}")
print(f"   - Proportional: ${recommendations['proportional_bet'].sum():,.2f}")

print(f"\n✅ Betting recommendations saved to: {output_path}")
print("=" * 120)

