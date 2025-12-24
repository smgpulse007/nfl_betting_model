"""
Phase 5B: Full 2024 Season Game-Level Derivation

Derive all 191 approved features for every game in the 2024 season.
Expected output: ~576 rows (288 games × 2 teams per game)
"""

import pandas as pd
import json
from pathlib import Path
import sys
from tqdm import tqdm
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from game_level.derive_game_features_complete import derive_game_features_complete
from team_abbreviation_mapping import nfl_data_py_to_espn

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 5B: FULL 2024 SEASON GAME-LEVEL DERIVATION")
print("="*120)

# Load data
print(f"\n[1/5] Loading 2024 season data...")
pbp = pd.read_parquet('../data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('../data/cache/schedules_2024.parquet')

print(f"  ✅ Loaded {len(pbp):,} plays")
print(f"  ✅ Loaded {len(schedules)} games")

# Filter to regular season only (weeks 1-18)
schedules_reg = schedules[schedules['week'] <= 18].copy()
print(f"  ✅ Regular season: {len(schedules_reg)} games")

# Get unique game IDs
game_ids = schedules_reg['game_id'].unique()
print(f"  ✅ Unique game IDs: {len(game_ids)}")

# Calculate expected rows
expected_rows = len(game_ids) * 2  # 2 teams per game
print(f"  ✅ Expected output: {expected_rows} rows ({len(game_ids)} games × 2 teams)")

# Derive features for all games
print(f"\n[2/5] Deriving features for all games...")
all_features = []
errors = []

for game_id in tqdm(game_ids, desc="Processing games"):
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
        all_features.append(away_features)
    except Exception as e:
        errors.append({'game_id': game_id, 'team': away_team, 'error': str(e)})
    
    # Derive features for home team
    try:
        home_features = derive_game_features_complete(
            team=nfl_data_py_to_espn(home_team),
            game_id=game_id,
            pbp=pbp,
            schedules=schedules
        )
        all_features.append(home_features)
    except Exception as e:
        errors.append({'game_id': game_id, 'team': home_team, 'error': str(e)})

print(f"\n  ✅ Derived features for {len(all_features)} team-games")
if errors:
    print(f"  ⚠️  Encountered {len(errors)} errors")
    for err in errors[:5]:
        print(f"     - {err['game_id']} ({err['team']}): {err['error']}")

# Create DataFrame
print(f"\n[3/5] Creating DataFrame...")
df = pd.DataFrame(all_features)
print(f"  ✅ Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Check for missing values
print(f"\n[4/5] Checking data quality...")
missing_vals = df.isnull().sum()
total_missing = missing_vals.sum()

if total_missing == 0:
    print(f"  ✅ No missing values!")
else:
    print(f"  ⚠️  Found {total_missing} missing values:")
    print(missing_vals[missing_vals > 0].head(10))

# Check feature coverage
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

metadata_cols = {'team', 'game_id'}
derived_features = set(df.columns) - metadata_cols
coverage = len(derived_features & approved_features)
total = len(approved_features)

print(f"\n  Feature Coverage:")
print(f"    - Approved features: {total}")
print(f"    - Derived features: {len(derived_features)}")
print(f"    - Coverage: {coverage}/{total} ({coverage/total*100:.1f}%)")

# Save results
print(f"\n[5/5] Saving results...")
output_dir = Path('../results')
output_dir.mkdir(exist_ok=True)

# Save as parquet (more efficient for large datasets)
parquet_file = output_dir / 'game_level_features_2024.parquet'
df.to_parquet(parquet_file, index=False)
print(f"  ✅ Saved to: {parquet_file}")
print(f"     Size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")

# Also save as CSV for inspection
csv_file = output_dir / 'game_level_features_2024.csv'
df.to_csv(csv_file, index=False)
print(f"  ✅ Saved to: {csv_file}")
print(f"     Size: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")

# Save errors if any
if errors:
    errors_file = output_dir / 'phase5b_errors.json'
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"  ⚠️  Saved errors to: {errors_file}")

print(f"\n{'='*120}")
print("✅ PHASE 5B: FULL 2024 SEASON DERIVATION COMPLETE!")
print(f"{'='*120}")
print(f"\nSummary:")
print(f"  - Games processed: {len(game_ids)}")
print(f"  - Team-games derived: {len(all_features)}")
print(f"  - Features per game: {df.shape[1]}")
print(f"  - Missing values: {total_missing}")
print(f"  - Errors: {len(errors)}")
print(f"\nNext: Run aggregation validation to verify season-level consistency")

