"""
Combine all game-level features (1999-2024) into a single dataset
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

print("="*120)
print("COMBINING ALL GAME-LEVEL FEATURES (1999-2024)")
print("="*120)

# Load all historical seasons
print(f"\n[1/4] Loading historical seasons (1999-2023)...")
historical_dir = Path('../results/game_level_historical')
historical_files = sorted(historical_dir.glob('game_level_features_*.parquet'))

print(f"  Found {len(historical_files)} historical season files")

dfs = []
for file in historical_files:
    df = pd.read_parquet(file)
    dfs.append(df)
    print(f"  ✅ Loaded {file.name}: {len(df)} rows")

historical_df = pd.concat(dfs, ignore_index=True)
print(f"\n  ✅ Combined historical data: {len(historical_df):,} rows")

# Load 2024 season
print(f"\n[2/4] Loading 2024 season...")
df_2024 = pd.read_parquet('../results/game_level_features_2024.parquet')
print(f"  ✅ Loaded 2024 data: {len(df_2024)} rows")

# Combine all seasons
print(f"\n[3/4] Combining all seasons...")
complete_df = pd.concat([historical_df, df_2024], ignore_index=True)
print(f"  ✅ Complete dataset: {len(complete_df):,} rows × {complete_df.shape[1]} columns")

# Data quality checks
print(f"\n[4/4] Data quality checks...")

# Check for duplicates
duplicates = complete_df.duplicated(subset=['team', 'game_id']).sum()
print(f"  - Duplicate rows: {duplicates}")

# Check missing values in approved features
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

approved_cols = [col for col in complete_df.columns if col in approved_features]
missing_vals = complete_df[approved_cols].isnull().sum().sum()
print(f"  - Missing values (approved features): {missing_vals}")

# Check year distribution
complete_df['year'] = complete_df['game_id'].str[:4].astype(int)
year_counts = complete_df.groupby('year').size()
print(f"\n  Year distribution:")
for year, count in year_counts.items():
    print(f"    {year}: {count:4d} team-games ({count//2:3d} games)")

# Save complete dataset
print(f"\n[5/5] Saving complete dataset...")
output_dir = Path('../results')

# Save as parquet (efficient)
parquet_file = output_dir / 'game_level_features_1999_2024_complete.parquet'
complete_df.to_parquet(parquet_file, index=False)
print(f"  ✅ Saved to: {parquet_file}")
print(f"     Size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")

# Save summary statistics
summary = {
    'total_rows': int(len(complete_df)),
    'total_columns': int(complete_df.shape[1]),
    'total_games': int(len(complete_df) // 2),
    'years': f"{year_counts.index.min()}-{year_counts.index.max()}",
    'total_seasons': int(len(year_counts)),
    'approved_features': len(approved_features),
    'missing_values': int(missing_vals),
    'duplicates': int(duplicates),
    'completeness': float((1 - missing_vals/(len(approved_cols)*len(complete_df)))*100),
    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_file = output_dir / 'game_level_complete_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ Saved summary to: {summary_file}")

# Create sample for inspection
sample_file = output_dir / 'game_level_features_sample.csv'
complete_df.sample(min(100, len(complete_df))).to_csv(sample_file, index=False)
print(f"  ✅ Saved sample to: {sample_file}")

print(f"\n{'='*120}")
print("✅ COMPLETE DATASET CREATED!")
print(f"{'='*120}")
print(f"\nSummary:")
print(f"  - Total rows: {len(complete_df):,}")
print(f"  - Total games: {len(complete_df) // 2:,}")
print(f"  - Years: {year_counts.index.min()}-{year_counts.index.max()} ({len(year_counts)} seasons)")
print(f"  - Features: {complete_df.shape[1]}")
print(f"  - Approved features: {len(approved_features)}")
print(f"  - Missing values: {missing_vals}")
print(f"  - Completeness: {(1 - missing_vals/(len(approved_cols)*len(complete_df)))*100:.1f}%")
print(f"  - File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"\n✅ Ready for Phase 5D (EDA) and Phase 6 (Feature Engineering)")

