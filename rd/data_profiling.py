"""
Data Profiling for NFL Betting Model R&D
========================================
Analyzes:
1. Missing data patterns
2. Feature distributions
3. Data sources and availability
4. Potential new features from nfl-data-py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR
import nfl_data_py as nfl

print('=' * 80)
print('NFL BETTING MODEL - DATA PROFILING REPORT')
print('=' * 80)

# Load main dataset
games = pd.read_parquet(PROCESSED_DATA_DIR / 'games_with_features.parquet')
print(f"\n📊 DATASET OVERVIEW")
print(f"   Total games: {len(games):,}")
print(f"   Seasons: {games['season'].min()} - {games['season'].max()}")
print(f"   Columns: {len(games.columns)}")

# SECTION 1: Missing Data Analysis
print(f"\n{'='*80}")
print("1️⃣  MISSING DATA ANALYSIS")
print('='*80)
missing = games.isnull().sum()
missing_pct = (missing / len(games) * 100).round(2)
missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct})
missing_df = missing_df[missing_df['missing'] > 0].sort_values('pct', ascending=False)
if len(missing_df) > 0:
    print("\nColumns with missing values:")
    for col, row in missing_df.head(20).iterrows():
        print(f"   {col:30} {row['missing']:>6} ({row['pct']:>5.1f}%)")
else:
    print("   No missing values!")

# SECTION 2: Current Features
print(f"\n{'='*80}")
print("2️⃣  CURRENT FEATURES IN MODEL")
print('='*80)
FEATURE_COLS = [
    'spread_line', 'total_line', 'elo_diff', 'elo_prob',
    'home_rest', 'away_rest', 'rest_advantage',
    'temp', 'wind', 'is_dome', 'is_cold', 'div_game', 'home_implied_prob'
]
print("\n   FEATURE               | TYPE      | MISSING | MEAN      | STD")
print("   " + "-"*70)
for col in FEATURE_COLS:
    if col in games.columns:
        dtype = 'numeric' if games[col].dtype in ['float64','int64'] else 'other'
        miss = games[col].isnull().sum()
        if dtype == 'numeric':
            mean = games[col].mean()
            std = games[col].std()
            print(f"   {col:22} | {dtype:9} | {miss:>7} | {mean:>9.2f} | {std:>6.2f}")
        else:
            print(f"   {col:22} | {dtype:9} | {miss:>7} | {'N/A':>9} | {'N/A':>6}")

# SECTION 3: Unused columns (potential features)
print(f"\n{'='*80}")
print("3️⃣  UNUSED COLUMNS (Potential Features)")
print('='*80)
unused = [c for c in games.columns if c not in FEATURE_COLS 
          and c not in ['game_id','season','week','home_team','away_team','home_score','away_score']]
print(f"\n   Found {len(unused)} unused columns. Top candidates:")
for col in unused[:25]:
    if col in games.columns:
        dtype = str(games[col].dtype)[:8]
        miss = games[col].isnull().sum()
        print(f"   {col:30} | {dtype:10} | missing: {miss}")

# SECTION 4: Available data from nfl-data-py
print(f"\n{'='*80}")
print("4️⃣  AVAILABLE DATA SOURCES (nfl-data-py)")
print('='*80)
print("""
   SOURCE                    | DESCRIPTION                        | POTENTIAL FEATURES
   --------------------------|------------------------------------|-----------------------
   import_schedules()        | Game results, stadium, weather     | ✓ Already using
   import_pbp_data()         | Play-by-play with EPA              | EPA/play, success rate
   import_weekly_data()      | Player weekly stats                | QB metrics, RB touches
   import_seasonal_data()    | Season-level aggregations          | Team strength metrics
   import_rosters()          | Player info, depth charts          | Injury proxies
   import_ngs_data()         | Next Gen Stats (speed, separation) | Advanced metrics
   import_combine_data()     | Combine results                    | Player athletic profiles
   import_qbr()              | ESPN QBR metrics                   | QB quality
   import_snap_counts()      | Player snap participation          | Injury/rest signals
   import_injuries()         | Official injury reports            | ⚠️ HIGH VALUE
   import_depth_charts()     | Weekly depth charts                | Lineup changes
""")

# SECTION 5: Feature engineering opportunities
print(f"\n{'='*80}")
print("5️⃣  FEATURE ENGINEERING OPPORTUNITIES")
print('='*80)
print("""
   CATEGORY          | FEATURES                           | EFFORT | EST. VALUE
   ------------------|------------------------------------|---------|-----------
   Rolling EPA       | off_epa_3wk, def_epa_3wk           | Low     | Medium
   QB Metrics        | qb_elo, qb_epa, qb_cpoe            | Medium  | High
   Injuries          | key_player_status, injury_score    | High    | Very High
   Market Movement   | line_movement, sharp_money         | Medium  | High
   Weather Detail    | precip_prob, humidity, wind_chill  | Low     | Low-Medium
   Primetime         | is_primetime, is_mnf, is_tnf       | Low     | Low
   Stadium           | altitude, turf_type                | Low     | Low
   Travel            | travel_distance, timezone_diff     | Low     | Medium
   Opponent Adj      | sos_off, sos_def                   | Medium  | Medium
   Situational       | playoff_clinch, revenge_game       | Medium  | Low
   Public Betting    | ticket_pct, money_pct              | High*   | High
""")

# SECTION 6: Quick data quality checks
print(f"\n{'='*80}")
print("6️⃣  DATA QUALITY CHECKS")
print('='*80)
# Check for games without betting lines
no_spread = games[games['spread_line'].isnull()]
no_total = games[games['total_line'].isnull()]
print(f"\n   Games without spread line: {len(no_spread)} ({len(no_spread)/len(games)*100:.1f}%)")
print(f"   Games without total line:  {len(no_total)} ({len(no_total)/len(games)*100:.1f}%)")

# Check score availability
completed = games[games['home_score'].notna()]
print(f"   Completed games: {len(completed)} ({len(completed)/len(games)*100:.1f}%)")

# Elo sanity check
print(f"\n   Elo rating range: {games['home_elo'].min():.0f} - {games['home_elo'].max():.0f}")
print(f"   Elo diff range: {games['elo_diff'].min():.0f} - {games['elo_diff'].max():.0f}")

print(f"\n{'='*80}")
print("✅ DATA PROFILING COMPLETE")
print('='*80)

