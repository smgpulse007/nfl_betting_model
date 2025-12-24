"""
Phase 4: Historical Feature Derivation (1999-2023)

Derive the 191 approved features (r >= 0.85) for all seasons from 1999-2023.
This creates a comprehensive training dataset spanning 25 years without imputation.
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import json
from pathlib import Path
from team_abbreviation_mapping import espn_to_nfl_data_py, nfl_data_py_to_espn
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("PHASE 4: HISTORICAL FEATURE DERIVATION (1999-2023)")
print("="*120)

# Configuration
START_YEAR = 1999
END_YEAR = 2023
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Load approved features list
print(f"\n[1/6] Loading approved features list...")
with open('results/approved_features_r085.json', 'r') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

print(f"  ✅ Loaded {len(approved_features)} approved features (r >= 0.85)")
print(f"     - Mean r: {approved_data['metadata']['mean_r']:.4f}")
print(f"     - Median r: {approved_data['metadata']['median_r']:.4f}")
print(f"     - Perfect correlations: {approved_data['metadata']['perfect_correlations']}")

# Import the full derivation function
print(f"\n[2/6] Loading feature derivation engine...")
from full_feature_derivation import derive_all_features

print(f"  ✅ Feature derivation engine loaded")
print(f"     - Function: derive_all_features(team, pbp_reg, schedules_reg)")
print(f"     - Derives ~368 features per team")
print(f"     - Will filter to 191 approved features")

# Create output directory
output_dir = Path('data/derived_features/historical')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[3/6] Preparing to derive features for {len(YEARS)} years ({START_YEAR}-{END_YEAR})...")
print(f"  Output directory: {output_dir}")

# Track progress
all_years_data = []
failed_years = []

print(f"\n[4/6] Deriving features year by year...")
print(f"  (This will take 2-3 hours - processing ~1.25M plays across 25 years)")
print(f"")

for year_idx, year in enumerate(YEARS, 1):
    try:
        print(f"  [{year_idx:2d}/{len(YEARS)}] Processing {year}...", end=" ", flush=True)

        # Load play-by-play data for this year (download if not cached)
        pbp = nfl.import_pbp_data([year])

        # Load schedules for this year
        schedules = nfl.import_schedules([year])
        
        # Filter to regular season only (week <= 18)
        pbp_reg = pbp[pbp['week'] <= 18].copy()
        schedules_reg = schedules[schedules['week'] <= 18].copy()
        
        # Get unique teams for this year
        teams_home = set(schedules_reg['home_team'].unique())
        teams_away = set(schedules_reg['away_team'].unique())
        teams = sorted(teams_home | teams_away)
        
        # Derive features for all teams
        year_features = []
        for team in teams:
            # Convert to ESPN abbreviation
            espn_team = nfl_data_py_to_espn(team)
            
            # Derive all features
            features = derive_all_features(espn_team, pbp_reg, schedules_reg)
            
            # Filter to approved features only
            approved_only = {'team': espn_team, 'year': year}
            for feat in approved_features:
                if feat in features:
                    approved_only[feat] = features[feat]
                else:
                    approved_only[feat] = None  # Missing feature
            
            year_features.append(approved_only)
        
        # Convert to DataFrame
        year_df = pd.DataFrame(year_features)
        
        # Save this year's data
        year_file = output_dir / f'espn_derived_{year}.parquet'
        year_df.to_parquet(year_file, index=False)
        
        # Add to all years
        all_years_data.append(year_df)
        
        print(f"✅ ({len(teams)} teams, {len(pbp_reg):,} plays)")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        failed_years.append((year, str(e)))

print(f"\n[5/6] Combining all years into single dataset...")

if all_years_data:
    # Combine all years
    combined_df = pd.concat(all_years_data, ignore_index=True)
    
    # Save combined dataset
    combined_file = Path('data/derived_features/espn_derived_1999_2023.parquet')
    combined_df.to_parquet(combined_file, index=False)
    
    print(f"  ✅ Combined dataset saved: {combined_file}")
    print(f"     - Shape: {combined_df.shape}")
    print(f"     - Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
    print(f"     - Teams: {combined_df['team'].nunique()} unique")
    print(f"     - Total rows: {len(combined_df):,}")
    print(f"     - Features: {len(approved_features)} approved features")

print(f"\n[6/6] Summary...")
print(f"  ✅ Successfully processed: {len(all_years_data)}/{len(YEARS)} years")
if failed_years:
    print(f"  ❌ Failed years: {len(failed_years)}")
    for year, error in failed_years:
        print(f"     - {year}: {error}")

print(f"\n{'='*120}")
print(f"✅ PHASE 4: HISTORICAL DERIVATION COMPLETE!")
print(f"{'='*120}")

