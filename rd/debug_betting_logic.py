"""
DEBUG: BETTING LOGIC
====================
Something is wrong. 74% WR across ALL seasons is impossible.
Let's trace through specific examples to find the bug.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

print("=" * 80)
print("DEBUGGING BETTING LOGIC")
print("=" * 80)

# Load data
games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")
test = games[games['season'] == 2024].head(20).copy()

ELO_TO_MARGIN = 25.0
test['elo_margin'] = test['elo_diff'] / ELO_TO_MARGIN
test['vegas_margin'] = -test['spread_line']
test['actual_margin'] = test['home_score'] - test['away_score']

print("\n  SAMPLE GAMES (2024 - First 20)")
print("=" * 120)
print(f"  {'Home':<5} {'Away':<5} | {'Spread':>7} | {'Vegas M':>8} | {'Elo Diff':>10} | {'Elo M':>7} | {'Actual':>7} | {'Bet':>6} | {'Result':>8}")
print("=" * 120)

wins, losses = 0, 0
for idx, row in test.iterrows():
    elo_m = row['elo_margin']
    vegas_m = row['vegas_margin']
    spread = row['spread_line']
    actual = row['actual_margin']
    
    # Betting logic
    if elo_m > vegas_m:
        bet = "HOME"
        won = actual + spread > 0  # Home covers if actual + spread > 0
    else:
        bet = "AWAY"
        won = actual + spread < 0  # Away covers if actual + spread < 0
    
    result = "✅ WIN" if won else "❌ LOSS"
    if won:
        wins += 1
    else:
        losses += 1
    
    print(f"  {row['home_team']:<5} {row['away_team']:<5} | {spread:>7.1f} | {vegas_m:>8.1f} | {row['elo_diff']:>10.0f} | {elo_m:>7.1f} | {actual:>7.0f} | {bet:>6} | {result}")

print(f"\n  Sample WR: {wins}/{wins+losses} = {wins/(wins+losses)*100:.1f}%")

# ============================================================================
# SANITY CHECK: What should the baseline be?
# ============================================================================
print("\n" + "=" * 80)
print("SANITY CHECK: RANDOM BASELINE")
print("=" * 80)

all_games = games[games['spread_line'].notna()].copy()
all_games['home_covered'] = (all_games['home_score'] - all_games['away_score'] + all_games['spread_line']) > 0

print(f"\n  Overall home cover rate: {all_games['home_covered'].mean()*100:.1f}%")
print(f"  This should be close to 50%!")

# Check by season
print(f"\n  Home cover rate by season:")
for season in range(2015, 2026):
    season_df = all_games[all_games['season'] == season]
    if len(season_df) > 0:
        hcr = season_df['home_covered'].mean()
        print(f"    {season}: {hcr*100:.1f}%")

# ============================================================================
# THE REAL QUESTION: Is our Elo model ACTUALLY predictive?
# ============================================================================
print("\n" + "=" * 80)
print("ELO MODEL ACCURACY (Not betting, just prediction)")
print("=" * 80)

all_games['elo_margin'] = all_games['elo_diff'] / 25.0
all_games['vegas_margin'] = -all_games['spread_line']
all_games['actual_margin'] = all_games['home_score'] - all_games['away_score']

# Elo prediction: Does the home team cover?
# Elo says home covers if elo_margin > vegas_margin
all_games['elo_says_home_cover'] = all_games['elo_margin'] > all_games['vegas_margin']

# Accuracy: How often is Elo correct about home covering?
all_games['elo_correct'] = all_games['elo_says_home_cover'] == all_games['home_covered']

print(f"\n  Elo prediction accuracy: {all_games['elo_correct'].mean()*100:.1f}%")

# Wait - this is the same as the betting WR!
# The question is: WHY is Elo so good at predicting covers?

# Let's check if there's a correlation issue
print(f"\n  Correlation Analysis:")
print(f"    elo_margin vs vegas_margin: {all_games['elo_margin'].corr(all_games['vegas_margin']):.3f}")
print(f"    elo_margin vs actual_margin: {all_games['elo_margin'].corr(all_games['actual_margin']):.3f}")
print(f"    vegas_margin vs actual_margin: {all_games['vegas_margin'].corr(all_games['actual_margin']):.3f}")

# ============================================================================
# HYPOTHESIS: Elo is computed AFTER the game in some seasons
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING FOR TEMPORAL LEAKAGE IN ELO")
print("=" * 80)

# For Elo to be valid, it should be computed BEFORE the game
# Let's check if elo_diff predicts the game TOO well

# In a fair Elo system, the correlation should be around 0.3-0.4
# If it's >0.6, something is wrong

sample = all_games[all_games['season'] == 2024].copy()
corr = sample['elo_diff'].corr(sample['actual_margin'])
print(f"\n  Elo_diff correlation with actual_margin (2024): {corr:.3f}")

if corr > 0.5:
    print(f"  ⚠️ WARNING: Correlation too high! Possible temporal leakage.")
else:
    print(f"  ✅ Correlation is in expected range.")

# Check if Elo updates are happening correctly
print(f"\n  Checking Elo progression for a team...")
team = 'KC'  # Kansas City
team_games = sample[sample['home_team'] == team].sort_values('week')[['week', 'home_team', 'away_team', 'home_elo', 'away_elo', 'home_score', 'away_score']]
print(f"\n  {team} Home Games in 2024:")
print(team_games.to_string(index=False))

# ============================================================================
# KEY INSIGHT: Check the ELO calculation
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING ELO CALCULATION METHOD")
print("=" * 80)

# Load raw games before Elo
raw_games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
sample_raw = raw_games[raw_games['season'] == 2024][['game_id', 'week', 'home_team', 'away_team', 'home_elo', 'away_elo']].head(10)
print(f"\n  Raw games (with Elo) - First 10 of 2024:")
print(sample_raw.to_string(index=False))
