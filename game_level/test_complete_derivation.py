"""
Phase 5A: Test Complete Game-Level Derivation

Test the derive_game_features_complete() function on BAL @ KC, Week 1, 2024.
Verify all 191 approved features are derived.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from game_level.derive_game_features_complete import derive_game_features_complete
from team_abbreviation_mapping import nfl_data_py_to_espn

print("="*120)
print("PHASE 5A: COMPLETE GAME-LEVEL DERIVATION TEST")
print("="*120)

# Configuration
GAME_ID = '2024_01_BAL_KC'

print(f"\n[1/6] Loading data...")
pbp = pd.read_parquet('../data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('../data/cache/schedules_2024.parquet')
print(f"  ✅ Loaded {len(pbp):,} plays, {len(schedules)} games")

# Get game info
game_info = schedules[schedules['game_id'] == GAME_ID].iloc[0]
away_team = game_info['away_team']
home_team = game_info['home_team']

print(f"\n[2/6] Game: {away_team} @ {home_team}")

# Derive features
print(f"\n[3/6] Deriving features for {away_team}...")
away_features = derive_game_features_complete(
    team=nfl_data_py_to_espn(away_team),
    game_id=GAME_ID,
    pbp=pbp,
    schedules=schedules
)
print(f"  ✅ Derived {len(away_features)} features")

print(f"\n[4/6] Deriving features for {home_team}...")
home_features = derive_game_features_complete(
    team=nfl_data_py_to_espn(home_team),
    game_id=GAME_ID,
    pbp=pbp,
    schedules=schedules
)
print(f"  ✅ Derived {len(home_features)} features")

# Check coverage
print(f"\n[5/6] Checking feature coverage...")
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

metadata_cols = {'team', 'game_id'}
derived_features = set(away_features.keys()) - metadata_cols

coverage = len(derived_features & approved_features)
total = len(approved_features)
pct = (coverage / total) * 100

print(f"  Approved features: {total}")
print(f"  Derived features: {len(derived_features)}")
print(f"  Coverage: {coverage}/{total} ({pct:.1f}%)")

missing = approved_features - derived_features
if missing:
    print(f"\n  ❌ Missing {len(missing)} features:")
    for feat in sorted(list(missing))[:10]:
        print(f"     - {feat}")
    if len(missing) > 10:
        print(f"     ... and {len(missing)-10} more")
else:
    print(f"  ✅ All approved features derived!")

# Check for missing values (only in approved features)
print(f"\n[6/6] Checking for missing values in approved features...")
df = pd.DataFrame([away_features, home_features])

# Only check approved features
approved_cols = [col for col in df.columns if col in approved_features]
missing_vals = df[approved_cols].isnull().sum()

if missing_vals.sum() == 0:
    print(f"  ✅ No missing values in approved features!")
else:
    print(f"  ❌ Found {missing_vals.sum()} missing values in approved features:")
    print(missing_vals[missing_vals > 0])

# Save results
output_dir = Path('../results')
csv_file = output_dir / 'phase5a_complete_test.csv'
df.to_csv(csv_file, index=False)
print(f"\n  ✅ Saved to: {csv_file}")

print(f"\n{'='*120}")
if pct >= 100 and missing_vals.sum() == 0:
    print("✅ PHASE 5A VALIDATION PASSED!")
    print("   Ready to proceed to Phase 5B (full 2024 season)")
else:
    print("❌ PHASE 5A VALIDATION FAILED!")
    print(f"   Coverage: {pct:.1f}% (need 100%)")
    print(f"   Missing values: {missing_vals.sum()} (need 0)")
print(f"{'='*120}")

