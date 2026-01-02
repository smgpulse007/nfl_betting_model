"""
Merge injury and weather features into phase6_game_level dataset
"""

import pandas as pd
import numpy as np

print("="*120)
print("MERGE INJURY + WEATHER FEATURES INTO PHASE6_GAME_LEVEL")
print("="*120)

# Load phase6_game_level (clean, no leakage)
print("\n[1/5] Loading phase6_game_level dataset...")
df_phase6 = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Loaded: {len(df_phase6):,} games × {len(df_phase6.columns):,} columns")
print(f"  ✅ Seasons: {df_phase6['season'].min()}-{df_phase6['season'].max()}")

# Load pregame_features (has injury + weather but also has leakage)
print("\n[2/5] Loading pregame_features for injury + weather data...")
df_pregame = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
print(f"  ✅ Loaded: {len(df_pregame):,} games × {len(df_pregame.columns):,} columns")

# Extract ONLY injury and weather features
injury_weather_cols = ['game_id']

# Injury features
injury_cols = [c for c in df_pregame.columns if 'injury' in c.lower() or 'qb_out' in c.lower()]
injury_weather_cols.extend(injury_cols)

# Weather features
weather_cols = ['temp', 'wind', 'temp_extreme', 'wind_high', 'is_outdoor']
weather_cols = [c for c in weather_cols if c in df_pregame.columns]
injury_weather_cols.extend(weather_cols)

print(f"\n  Extracting features:")
print(f"    • Injury features: {len(injury_cols)}")
for col in sorted(injury_cols):
    print(f"      - {col}")
print(f"    • Weather features: {len(weather_cols)}")
for col in sorted(weather_cols):
    print(f"      - {col}")

# Extract injury + weather data
df_injury_weather = df_pregame[injury_weather_cols].copy()

# Merge with phase6_game_level
print("\n[3/5] Merging injury + weather features...")
df_merged = df_phase6.merge(df_injury_weather, on='game_id', how='left')

print(f"  ✅ Merged dataset: {len(df_merged):,} games × {len(df_merged.columns):,} columns")
print(f"  ✅ Added {len(injury_cols) + len(weather_cols)} features")

# Check coverage
print("\n[4/5] Checking feature coverage...")

for col in injury_cols + weather_cols:
    coverage = df_merged[col].notna().sum() / len(df_merged) * 100
    print(f"    • {col}: {coverage:.1f}% coverage")

# Save
print("\n[5/5] Saving merged dataset...")
output_file = '../results/phase8_results/phase6_game_level_with_injury_weather_1999_2024.parquet'
df_merged.to_parquet(output_file, index=False)

print(f"  ✅ Saved: {output_file}")
print(f"  ✅ Shape: {df_merged.shape}")

# Summary
print(f"\n{'='*120}")
print("SUMMARY")
print("="*120)

print(f"""
MERGED DATASET: phase6_game_level_with_injury_weather_1999_2024.parquet

FEATURES:
  • Original phase6 features: {len(df_phase6.columns)}
  • Injury features added: {len(injury_cols)}
  • Weather features added: {len(weather_cols)}
  • Total features: {len(df_merged.columns)}

DATA QUALITY:
  • No data leakage (uses phase6_game_level as base)
  • Injury data coverage: ~56-59% (2009-2024)
  • Weather data coverage: 100%

NEXT STEPS:
  1. Retrain XGBoost with this dataset
  2. Compare performance with/without injury features
  3. Generate 2025 predictions (need to add 2025 injury/weather data)
""")

print(f"\n{'='*120}")

