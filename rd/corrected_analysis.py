"""
CORRECTED SPREAD BETTING ANALYSIS
=================================
Bug found: spread_line was interpreted incorrectly.

In nfl-data-py:
- spread_line = 3 means HOME is -3 (favorite by 3)
- HOME covers if actual_margin > spread_line
- AWAY covers if actual_margin < spread_line

Old (WRONG) formula: home_covers = actual_margin + spread_line > 0
New (CORRECT) formula: home_covers = actual_margin > spread_line
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

print("=" * 80)
print("CORRECTED SPREAD BETTING ANALYSIS")
print("=" * 80)

# Load data
games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")
games = games[games['spread_line'].notna()].copy()

# Compute with CORRECT formula
games['actual_margin'] = games['home_score'] - games['away_score']
games['home_covered'] = games['actual_margin'] > games['spread_line']
games['elo_margin'] = games['elo_diff'] / 25.0

print(f"\n  With CORRECTED formula:")
print(f"    Overall home cover rate: {games['home_covered'].mean()*100:.1f}%")
print(f"    Expected: ~48% (slight away advantage due to juice)")

# ============================================================================
# RE-RUN: Multi-season Elo betting simulation
# ============================================================================
print("\n" + "=" * 80)
print("CORRECTED MULTI-SEASON ELO BETTING")
print("=" * 80)

print(f"\n{'Season':<8} | {'Games':<6} | {'Home Cover%':>12} | {'Elo Bets':>10} | {'Elo WR':>8} | {'ROI':>10}")
print("=" * 80)

all_results = []

for season in sorted(games['season'].unique()):
    season_df = games[games['season'] == season].copy()
    if len(season_df) < 100:
        continue
    
    home_cover_rate = season_df['home_covered'].mean()
    
    # Elo betting: bet HOME if Elo thinks home will win by more than spread
    # bet AWAY if Elo thinks home will win by less than spread
    bets, wins, pnl = 0, 0, 0
    
    for idx, row in season_df.iterrows():
        elo_m = row['elo_margin']
        spread = row['spread_line']
        actual = row['actual_margin']
        
        # Elo says home covers if elo_margin > spread
        if elo_m > spread:  # Bet HOME
            bets += 1
            if actual > spread:  # Home covered
                wins += 1
                pnl += 91
            else:
                pnl -= 100
        else:  # Bet AWAY
            bets += 1
            if actual < spread:  # Away covered
                wins += 1
                pnl += 91
            else:
                pnl -= 100
    
    wr = wins / bets * 100 if bets > 0 else 0
    roi = pnl / (bets * 100) * 100 if bets > 0 else 0
    
    all_results.append({'season': season, 'wr': wr, 'roi': roi, 'bets': bets, 'pnl': pnl})
    
    status = "✅" if roi > 0 else "❌"
    print(f"{season:<8} | {len(season_df):<6} | {home_cover_rate*100:>11.1f}% | {bets:>10} | {wr:>7.1f}% | {roi:>9.1f}% {status}")

results_df = pd.DataFrame(all_results)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("CORRECTED SUMMARY STATISTICS")
print("=" * 80)

winning_seasons = (results_df['roi'] > 0).sum()
total_seasons = len(results_df)

print(f"\n  Winning seasons: {winning_seasons}/{total_seasons} ({winning_seasons/total_seasons*100:.0f}%)")
print(f"  Average WR:      {results_df['wr'].mean():.1f}%")
print(f"  Average ROI:     {results_df['roi'].mean():.1f}%")
print(f"  Total P&L:       ${results_df['pnl'].sum():,.0f}")

# Break-even analysis
print(f"\n  Break-even WR for -110 odds: 52.4%")
print(f"  Seasons above break-even: {(results_df['wr'] > 52.4).sum()}/{total_seasons}")

# ============================================================================
# 2024 and 2025 specific
# ============================================================================
print("\n" + "=" * 80)
print("2024-2025 PERFORMANCE (CORRECTED)")
print("=" * 80)

for year in [2024, 2025]:
    row = results_df[results_df['season'] == year]
    if len(row) > 0:
        print(f"\n  {year}:")
        print(f"    Win Rate: {row['wr'].values[0]:.1f}%")
        print(f"    ROI:      {row['roi'].values[0]:.1f}%")
        print(f"    P&L:      ${row['pnl'].values[0]:,.0f}")

# ============================================================================
# COMPARE OLD vs NEW
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: OLD (BUGGY) vs NEW (CORRECTED)")
print("=" * 80)

# Old (buggy) results - from memory
print(f"""
  OLD RESULTS (BUGGY):
    Average WR:     73.9%
    Average ROI:    41.1%
    Winning seasons: 27/27 (100%)
    
  NEW RESULTS (CORRECTED):
    Average WR:     {results_df['wr'].mean():.1f}%
    Average ROI:    {results_df['roi'].mean():.1f}%
    Winning seasons: {winning_seasons}/{total_seasons} ({winning_seasons/total_seasons*100:.0f}%)
    
  INTERPRETATION:
    The "amazing" 74% win rate was a BUG!
    The corrected results show {'a modest edge' if results_df['roi'].mean() > 0 else 'no significant edge'}.
""")
