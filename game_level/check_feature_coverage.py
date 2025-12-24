"""
Check which approved features are derived vs missing
"""
import json
import pandas as pd
from pathlib import Path

# Load approved features
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

# Load test results
test_df = pd.read_csv('../results/phase5a_single_game_test.csv')

# Get derived features (exclude metadata columns)
metadata_cols = {'team', 'game_id', 'win', 'loss', 'tie', 'team_score', 'opp_score', 
                 'point_differential', 'week', 'season', 'is_home'}
derived_features = set(test_df.columns) - metadata_cols

print("="*80)
print("FEATURE COVERAGE ANALYSIS")
print("="*80)

print(f"\nApproved features (r >= 0.85): {len(approved_features)}")
print(f"Derived features: {len(derived_features)}")

# Find missing and extra
missing = approved_features - derived_features
extra = derived_features - approved_features

print(f"\nMissing features: {len(missing)}")
print(f"Extra features (not in approved list): {len(extra)}")

if missing:
    print(f"\n{'='*80}")
    print(f"MISSING FEATURES ({len(missing)}):")
    print(f"{'='*80}")
    for i, feat in enumerate(sorted(missing), 1):
        print(f"{i:3d}. {feat}")

if extra:
    print(f"\n{'='*80}")
    print(f"EXTRA FEATURES ({len(extra)}):")
    print(f"{'='*80}")
    for i, feat in enumerate(sorted(extra), 1):
        print(f"{i:3d}. {feat}")

# Check for missing values
print(f"\n{'='*80}")
print("MISSING VALUES CHECK:")
print(f"{'='*80}")

missing_vals = test_df.isnull().sum()
if missing_vals.sum() == 0:
    print("✅ No missing values found!")
else:
    print(f"❌ Found {missing_vals.sum()} missing values:")
    print(missing_vals[missing_vals > 0])

# Coverage percentage
coverage = (len(derived_features & approved_features) / len(approved_features)) * 100
print(f"\n{'='*80}")
print(f"COVERAGE: {coverage:.1f}% ({len(derived_features & approved_features)}/{len(approved_features)})")
print(f"{'='*80}")

