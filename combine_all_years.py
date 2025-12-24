"""
Combine historical data (1999-2023) with 2024 data to create complete training dataset
"""
import pandas as pd
import json

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("COMBINING ALL YEARS (1999-2024) INTO COMPLETE TRAINING DATASET")
print("="*120)

# Load approved features list
print(f"\n[1/5] Loading approved features list...")
with open('results/approved_features_r085.json', 'r') as f:
    approved_data = json.load(f)
    approved_features = set(approved_data['features'])

print(f"  ✅ Loaded {len(approved_features)} approved features (r >= 0.85)")

# Load historical data (1999-2023)
print(f"\n[2/5] Loading historical data (1999-2023)...")
historical_df = pd.read_parquet('data/derived_features/espn_derived_1999_2023.parquet')

print(f"  ✅ Historical data loaded")
print(f"     - Shape: {historical_df.shape}")
print(f"     - Years: {historical_df['year'].min()}-{historical_df['year'].max()}")
print(f"     - Teams: {historical_df['team'].nunique()} unique")
print(f"     - Total rows: {len(historical_df):,}")

# Load 2024 data
print(f"\n[3/5] Loading 2024 data...")
df_2024 = pd.read_parquet('data/derived_features/espn_derived_2024.parquet')

# Reset index to make 'team' a column
df_2024 = df_2024.reset_index()

# Filter to approved features only
approved_cols = ['team'] + [f for f in approved_features if f in df_2024.columns]
df_2024_filtered = df_2024[approved_cols].copy()
df_2024_filtered['year'] = 2024

print(f"  ✅ 2024 data loaded")
print(f"     - Shape: {df_2024_filtered.shape}")
print(f"     - Teams: {df_2024_filtered['team'].nunique()}")
print(f"     - Features: {len(approved_cols)-1} (filtered to approved)")

# Combine all years
print(f"\n[4/5] Combining all years...")
combined_df = pd.concat([historical_df, df_2024_filtered], ignore_index=True)

# Sort by year and team
combined_df = combined_df.sort_values(['year', 'team']).reset_index(drop=True)

print(f"  ✅ Combined dataset created")
print(f"     - Shape: {combined_df.shape}")
print(f"     - Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
print(f"     - Teams: {combined_df['team'].nunique()} unique")
print(f"     - Total rows: {len(combined_df):,}")
print(f"     - Features: {len(combined_df.columns)-2} (excluding team, year)")

# Save combined dataset
output_file = 'data/derived_features/espn_derived_1999_2024_complete.parquet'
combined_df.to_parquet(output_file, index=False)

print(f"\n[5/5] Saved complete training dataset...")
print(f"  ✅ File: {output_file}")
print(f"  ✅ Size: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Summary statistics
print(f"\n{'='*120}")
print(f"SUMMARY STATISTICS")
print(f"{'='*120}")

print(f"\n  Years covered: {combined_df['year'].nunique()} years ({combined_df['year'].min()}-{combined_df['year'].max()})")
print(f"  Teams per year:")
for year in sorted(combined_df['year'].unique()):
    year_df = combined_df[combined_df['year'] == year]
    print(f"    {year}: {len(year_df)} teams")

print(f"\n  Unique teams across all years: {combined_df['team'].nunique()}")
print(f"  Teams: {sorted(combined_df['team'].unique())}")

print(f"\n  Features: {len(combined_df.columns)-2}")
print(f"  Total data points: {len(combined_df) * (len(combined_df.columns)-2):,}")

print(f"\n  Missing values:")
missing = combined_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"    Features with missing values: {len(missing)}")
    for feat, count in missing.head(10).items():
        print(f"      {feat}: {count} ({count/len(combined_df)*100:.1f}%)")
else:
    print(f"    ✅ No missing values!")

print(f"\n{'='*120}")
print(f"✅ COMPLETE TRAINING DATASET READY!")
print(f"{'='*120}")
print(f"\n  📁 File: {output_file}")
print(f"  📊 Shape: {combined_df.shape}")
print(f"  📅 Years: {combined_df['year'].min()}-{combined_df['year'].max()} (26 years)")
print(f"  🏈 Teams: {combined_df['team'].nunique()} unique")
print(f"  📈 Features: {len(combined_df.columns)-2} approved (r >= 0.85)")
print(f"  💾 Size: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n{'='*120}")

