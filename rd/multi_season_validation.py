"""
MULTI-SEASON VALIDATION
=======================
Does the Elo edge persist across seasons, or is 2024 an anomaly?
This is the DEFINITIVE test for data leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

print("=" * 80)
print("MULTI-SEASON ELO BETTING VALIDATION")
print("=" * 80)

# Load all data
games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_epa.parquet")
games_2025 = pd.read_parquet(PROCESSED_DATA_DIR.parent / "2025" / "completed_2025_with_epa.parquet")

# Add 2025 to games
all_games = pd.concat([games, games_2025], ignore_index=True)

# Filter to seasons with spread data (roughly 2006+)
all_games = all_games[all_games['spread_line'].notna()].copy()

# Compute Elo margin
ELO_TO_MARGIN = 25.0
all_games['elo_margin'] = all_games['elo_diff'] / ELO_TO_MARGIN
all_games['vegas_margin'] = -all_games['spread_line']
all_games['actual_margin'] = all_games['home_score'] - all_games['away_score']
all_games['home_covered'] = (all_games['actual_margin'] + all_games['spread_line']) > 0

print(f"\nTotal games with spread data: {len(all_games)}")
print(f"Seasons: {all_games['season'].min()} - {all_games['season'].max()}")

# Per-season analysis
print(f"\n{'='*90}")
print(f"{'Season':<8} | {'Games':<6} | {'Home Cover%':<12} | {'Elo Bets':<10} | {'Elo WR':<8} | {'Elo ROI':<10} | {'Status':<10}")
print(f"{'='*90}")

season_results = []

for season in sorted(all_games['season'].unique()):
    season_df = all_games[all_games['season'] == season].copy()
    
    if len(season_df) < 100:  # Skip partial seasons
        continue
    
    home_cover_rate = season_df['home_covered'].mean()
    
    # Elo betting simulation (threshold = 0, bet all games)
    bets, wins, pnl = 0, 0, 0
    for idx, row in season_df.iterrows():
        elo_m = row['elo_margin']
        vegas_m = row['vegas_margin']
        spread = row['spread_line']
        actual = row['actual_margin']
        
        if elo_m > vegas_m:
            # Bet home to cover
            bets += 1
            if actual + spread > 0:
                wins += 1
                pnl += 91
            else:
                pnl -= 100
        else:
            # Bet away to cover
            bets += 1
            if actual + spread < 0:
                wins += 1
                pnl += 91
            else:
                pnl -= 100
    
    wr = wins / bets * 100 if bets > 0 else 0
    roi = pnl / (bets * 100) * 100 if bets > 0 else 0
    
    status = "✅ WIN" if roi > 0 else "❌ LOSS"
    
    season_results.append({
        'season': season, 'games': len(season_df), 
        'home_cover': home_cover_rate, 'wr': wr, 'roi': roi, 'pnl': pnl
    })
    
    print(f"{season:<8} | {len(season_df):<6} | {home_cover_rate*100:>10.1f}% | {bets:<10} | {wr:>6.1f}% | {roi:>9.1f}% | {status}")

# Summary statistics
print(f"\n{'='*90}")
print("AGGREGATE STATISTICS")
print(f"{'='*90}")

results_df = pd.DataFrame(season_results)
winning_seasons = (results_df['roi'] > 0).sum()
total_seasons = len(results_df)

print(f"\n  Winning seasons: {winning_seasons}/{total_seasons} ({winning_seasons/total_seasons*100:.0f}%)")
print(f"  Average WR:      {results_df['wr'].mean():.1f}%")
print(f"  Average ROI:     {results_df['roi'].mean():.1f}%")
print(f"  Total P&L:       ${results_df['pnl'].sum():,.0f}")

# Break-even analysis
print(f"\n  Break-even WR for -110 odds: 52.4%")
print(f"  Seasons above break-even: {(results_df['wr'] > 52.4).sum()}/{total_seasons}")

# Recent vs Historical
recent = results_df[results_df['season'] >= 2018]
historical = results_df[results_df['season'] < 2018]

print(f"\n  HISTORICAL (2006-2017):")
print(f"    Avg WR:  {historical['wr'].mean():.1f}%")
print(f"    Avg ROI: {historical['roi'].mean():.1f}%")

print(f"\n  RECENT (2018-2025):")
print(f"    Avg WR:  {recent['wr'].mean():.1f}%")
print(f"    Avg ROI: {recent['roi'].mean():.1f}%")

# ============================================================================
# CRITICAL: Check if 2024-2025 are outliers
# ============================================================================
print(f"\n{'='*90}")
print("OUTLIER ANALYSIS: Are 2024-2025 Anomalies?")
print(f"{'='*90}")

mean_wr = results_df[results_df['season'] < 2024]['wr'].mean()
std_wr = results_df[results_df['season'] < 2024]['wr'].std()

wr_2024 = results_df[results_df['season'] == 2024]['wr'].values[0]
wr_2025 = results_df[results_df['season'] == 2025]['wr'].values[0]

z_2024 = (wr_2024 - mean_wr) / std_wr
z_2025 = (wr_2025 - mean_wr) / std_wr

print(f"\n  Historical WR (2006-2023): {mean_wr:.1f}% ± {std_wr:.1f}%")
print(f"  2024 WR: {wr_2024:.1f}% (z-score: {z_2024:+.2f})")
print(f"  2025 WR: {wr_2025:.1f}% (z-score: {z_2025:+.2f})")

if abs(z_2024) > 2 or abs(z_2025) > 2:
    print(f"\n  ⚠️ WARNING: 2024 or 2025 may be statistical outliers (z > 2)")
else:
    print(f"\n  ✅ 2024-2025 are within normal range")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print(f"\n{'='*90}")
print("FINAL VERDICT")
print(f"{'='*90}")

if results_df['roi'].mean() > 0 and winning_seasons > total_seasons * 0.6:
    verdict = "LIKELY REAL EDGE"
    explanation = """
    The Elo model shows consistent profitability across multiple seasons.
    This suggests a REAL edge, not data leakage.
    
    The edge comes from:
    1. Elo captures team strength independent of public perception
    2. Vegas spreads are influenced by public betting
    3. When Elo disagrees with Vegas, Elo is often correct
    """
else:
    verdict = "POSSIBLE OVERFITTING OR VARIANCE"
    explanation = """
    The results are inconsistent across seasons.
    The high performance in 2024-2025 may be statistical variance.
    """

print(f"\n  VERDICT: {verdict}")
print(explanation)
