"""
DEBUG: SPREAD DATA QUALITY
==========================
Home cover rate should be ~50% if spreads are set correctly.
59.4% home cover rate suggests data quality issues.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

print("=" * 80)
print("DEBUGGING SPREAD DATA")
print("=" * 80)

# Load data
games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")

# Filter to games with spread data
games = games[games['spread_line'].notna()].copy()

# Calculate actual margin
games['actual_margin'] = games['home_score'] - games['away_score']
games['home_covered'] = (games['actual_margin'] + games['spread_line']) > 0

# ============================================================================
# KEY INSIGHT: What does spread_line actually represent?
# ============================================================================
print("\n" + "=" * 80)
print("UNDERSTANDING spread_line")
print("=" * 80)

# Sample some games
sample = games[games['season'] == 2024].head(10)[['home_team', 'away_team', 'spread_line', 'home_score', 'away_score', 'actual_margin']]
sample['home_covered'] = (sample['actual_margin'] + sample['spread_line']) > 0

print("\n  Sample games:")
print(sample.to_string(index=False))

# What is the spread line?
# If spread_line is POSITIVE, home team is UNDERDOG
# If spread_line is NEGATIVE, home team is FAVORITE

# Example: spread_line = -7 means home is favored by 7
#          Home wins by 8 → actual_margin = 8
#          Home covers: 8 + (-7) = 1 > 0 ✓

# Example: spread_line = 3 means home is underdog by 3
#          Home loses by 2 → actual_margin = -2
#          Home covers: -2 + 3 = 1 > 0 ✓

print(f"\n  Spread line distribution:")
print(f"    Mean: {games['spread_line'].mean():.2f}")
print(f"    Positive (home underdog): {(games['spread_line'] > 0).mean()*100:.1f}%")
print(f"    Negative (home favorite): {(games['spread_line'] < 0).mean()*100:.1f}%")

# ============================================================================
# THE REAL ISSUE: What's causing 59% home cover rate?
# ============================================================================
print("\n" + "=" * 80)
print("INVESTIGATING HIGH HOME COVER RATE")
print("=" * 80)

print(f"\n  Overall home cover rate: {games['home_covered'].mean()*100:.1f}%")

# Is it the PUSH handling?
# A push is when actual_margin + spread_line = 0
games['is_push'] = (games['actual_margin'] + games['spread_line']) == 0
print(f"\n  Pushes: {games['is_push'].sum()} ({games['is_push'].mean()*100:.1f}%)")

# Exclude pushes
non_push = games[~games['is_push']]
print(f"  Home cover rate (excluding pushes): {non_push['home_covered'].mean()*100:.1f}%")

# ============================================================================
# CRITICAL: Check if spread_line sign is correct
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING SPREAD LINE SIGN CONVENTION")
print("=" * 80)

# If home team is strong, spread should be NEGATIVE
# Let's check correlation
games['elo_diff'] = games['home_elo'] - games['away_elo']
corr = games['spread_line'].corr(games['elo_diff'])
print(f"\n  Correlation: spread_line vs elo_diff = {corr:.3f}")

# If correlation is NEGATIVE, spread_line is in "Vegas" format
# (-7 for home favorites, +7 for home underdogs)
# If correlation is POSITIVE, it's inverted

if corr < -0.5:
    print(f"  ✅ Spread line appears to be in standard Vegas format")
    print(f"     (negative = home favorite, positive = home underdog)")
elif corr > 0.5:
    print(f"  ⚠️ Spread line appears INVERTED!")
else:
    print(f"  ⚠️ Weak correlation - unclear format")

# ============================================================================
# SIMULATING CORRECT BETTING
# ============================================================================
print("\n" + "=" * 80)
print("SIMULATING WITH CORRECTED UNDERSTANDING")
print("=" * 80)

# The spread_line from nfl-data-py is the AWAY team spread
# So if spread_line = 3, the AWAY team is getting +3 (home is -3)
# Wait, let me verify this...

# Looking at the data: KC vs BAL in Week 1 2024
# KC is clearly the favorite. If spread_line = 3, that means...
kc_game = games[(games['home_team'] == 'KC') & (games['season'] == 2024) & (games['week'] == 1)]
print(f"\n  KC vs BAL Week 1, 2024:")
print(f"    spread_line: {kc_game['spread_line'].values[0]}")
print(f"    KC final: {kc_game['home_score'].values[0]}")
print(f"    BAL final: {kc_game['away_score'].values[0]}")
print(f"    Actual margin: {kc_game['actual_margin'].values[0]}")

# Check the actual spread for this game (KC was -3)
# So spread_line = 3 means HOME is -3 (favored by 3)

# WAIT - I think the issue is interpretation:
# spread_line in this dataset is the POINT SPREAD from HOME team perspective
# So spread_line = 3 means home gives 3 points (home is favorite)

# Let's re-check
print("\n  Reinterpreting spread_line...")
print(f"  If spread_line represents 'points home gives':")
print(f"    KC -3 → spread_line = 3? Let's check...")

# In standard betting:
# KC -3 means KC must win by more than 3
# BAL +3 means BAL can lose by up to 3

# If spread_line = 3 for a game where home is -3:
# Home covers if actual_margin > 3
# But our formula is: actual_margin + spread_line > 0
# So: actual_margin + 3 > 0 → actual_margin > -3

# Wait, that's WRONG! 
# If home is -3 (must win by >3 to cover):
# Home covers if actual_margin > 3
# But our formula gives: home covers if actual_margin > -3

print("\n  ⚠️ FOUND THE BUG!")
print(f"  Current formula: home_covered = actual_margin + spread_line > 0")
print(f"  This is WRONG if spread_line is 'points home gives'")
print(f"  Correct formula: home_covered = actual_margin > spread_line")

# Let's verify with corrected formula
games['home_covered_correct'] = games['actual_margin'] > games['spread_line']
print(f"\n  With CORRECTED formula:")
print(f"    Home cover rate: {games['home_covered_correct'].mean()*100:.1f}%")
print(f"    This should be ~50%!")

# ============================================================================
# Final verification with sample game
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION WITH SAMPLE GAME")
print("=" * 80)

print(f"""
  KC vs BAL, Week 1, 2024:
  - spread_line = 3 (KC is -3, BAL is +3)
  - KC scored: 27
  - BAL scored: 20
  - Actual margin: 7

  OLD formula: 7 + 3 = 10 > 0 → Home covers ✓
    (This says home covers regardless of spread!)
    
  CORRECT formula: 7 > 3 → Home covers ✓
    (KC won by 7, needed to win by >3, so they covered)
  
  Both get it right in this case, but let's check a closer game...
""")
