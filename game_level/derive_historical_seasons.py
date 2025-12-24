"""
Phase 5C: Historical Game-Level Derivation (1999-2023)

Derive all 191 approved features for every game in historical seasons.
Expected output: ~12,848 team-games (25 seasons × ~257 games/season × 2 teams)
"""

import pandas as pd
import nfl_data_py as nfl
import json
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from game_level.derive_game_features_complete import derive_game_features_complete
from team_abbreviation_mapping import nfl_data_py_to_espn

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 5C: HISTORICAL GAME-LEVEL DERIVATION (1999-2023)")
print("="*120)

# Configuration
YEARS = list(range(1999, 2024))  # 1999-2023 (25 seasons)
CACHE_DIR = Path('../data/cache')
OUTPUT_DIR = Path('../results/game_level_historical')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n[CONFIG]")
print(f"  - Years: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} seasons)")
print(f"  - Cache directory: {CACHE_DIR}")
print(f"  - Output directory: {OUTPUT_DIR}")

# Track overall statistics
total_games = 0
total_team_games = 0
total_errors = 0
all_errors = []

start_time = datetime.now()

# Process each year
for year in YEARS:
    print(f"\n{'='*120}")
    print(f"PROCESSING YEAR: {year}")
    print(f"{'='*120}")
    
    year_start = datetime.now()
    
    # Load data for this year
    print(f"\n[1/4] Loading {year} season data...")
    
    # Check cache
    pbp_cache = CACHE_DIR / f'pbp_{year}.parquet'
    schedules_cache = CACHE_DIR / f'schedules_{year}.parquet'
    
    if pbp_cache.exists() and schedules_cache.exists():
        print(f"  ✅ Loading from cache...")
        pbp = pd.read_parquet(pbp_cache)
        schedules = pd.read_parquet(schedules_cache)
    else:
        print(f"  ⚠️  Cache not found, loading from nfl_data_py...")
        pbp = nfl.import_pbp_data([year])
        schedules = nfl.import_schedules([year])
        
        # Cache for future use
        pbp.to_parquet(pbp_cache, index=False)
        schedules.to_parquet(schedules_cache, index=False)
        print(f"  ✅ Cached to {pbp_cache}")
    
    print(f"  ✅ Loaded {len(pbp):,} plays")
    print(f"  ✅ Loaded {len(schedules)} games")
    
    # Filter to regular season only
    schedules_reg = schedules[schedules['week'] <= 18].copy()
    print(f"  ✅ Regular season: {len(schedules_reg)} games")
    
    # Get unique game IDs
    game_ids = schedules_reg['game_id'].unique()
    print(f"  ✅ Unique game IDs: {len(game_ids)}")
    
    # Derive features for all games
    print(f"\n[2/4] Deriving features for {year}...")
    year_features = []
    year_errors = []
    
    for game_id in tqdm(game_ids, desc=f"  {year} games"):
        # Get teams for this game
        game_info = schedules_reg[schedules_reg['game_id'] == game_id].iloc[0]
        home_team = game_info['home_team']
        away_team = game_info['away_team']
        
        # Derive features for away team
        try:
            away_features = derive_game_features_complete(
                team=nfl_data_py_to_espn(away_team),
                game_id=game_id,
                pbp=pbp,
                schedules=schedules
            )
            year_features.append(away_features)
        except Exception as e:
            year_errors.append({'year': year, 'game_id': game_id, 'team': away_team, 'error': str(e)})
        
        # Derive features for home team
        try:
            home_features = derive_game_features_complete(
                team=nfl_data_py_to_espn(home_team),
                game_id=game_id,
                pbp=pbp,
                schedules=schedules
            )
            year_features.append(home_features)
        except Exception as e:
            year_errors.append({'year': year, 'game_id': game_id, 'team': home_team, 'error': str(e)})
    
    print(f"\n  ✅ Derived features for {len(year_features)} team-games")
    if year_errors:
        print(f"  ⚠️  Encountered {len(year_errors)} errors")
        all_errors.extend(year_errors)
    
    # Create DataFrame for this year
    print(f"\n[3/4] Creating DataFrame for {year}...")
    df_year = pd.DataFrame(year_features)
    print(f"  ✅ Shape: {df_year.shape[0]} rows × {df_year.shape[1]} columns")
    
    # Save results for this year
    print(f"\n[4/4] Saving {year} results...")
    output_file = OUTPUT_DIR / f'game_level_features_{year}.parquet'
    df_year.to_parquet(output_file, index=False)
    print(f"  ✅ Saved to: {output_file}")
    print(f"     Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Update totals
    total_games += len(game_ids)
    total_team_games += len(year_features)
    total_errors += len(year_errors)
    
    year_elapsed = (datetime.now() - year_start).total_seconds()
    print(f"\n  ⏱️  Year {year} completed in {year_elapsed:.1f} seconds")

# Save all errors
if all_errors:
    errors_file = OUTPUT_DIR / 'derivation_errors.json'
    with open(errors_file, 'w') as f:
        json.dump(all_errors, f, indent=2)
    print(f"\n⚠️  Saved {len(all_errors)} errors to: {errors_file}")

total_elapsed = (datetime.now() - start_time).total_seconds()

print(f"\n{'='*120}")
print("✅ PHASE 5C: HISTORICAL DERIVATION COMPLETE!")
print(f"{'='*120}")
print(f"\nSummary:")
print(f"  - Years processed: {len(YEARS)} ({YEARS[0]}-{YEARS[-1]})")
print(f"  - Total games: {total_games:,}")
print(f"  - Total team-games: {total_team_games:,}")
print(f"  - Total errors: {total_errors}")
print(f"  - Processing time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
print(f"  - Games per second: {total_games/total_elapsed:.1f}")
print(f"\nNext: Combine with 2024 data and create complete dataset")

