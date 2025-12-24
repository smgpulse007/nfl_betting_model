"""
Phase 6A: Integrate TIER S+A Features
======================================

Integrate existing TIER S+A features into game-level dataset.
These features are already computed in tier_sa_features.py for 2016+.

TIER S+A Features (19 total):
- CPOE (3): cpoe_3wk, cpoe_diff
- Pressure (3): pressure_rate_3wk, pressure_diff
- Injury (5): injury_impact, injury_diff, qb_out
- RYOE (3): ryoe_3wk, ryoe_diff
- Separation (3): separation_3wk, separation_diff
- Time to Throw (2): time_to_throw_3wk

Input: game_level_features_with_opponents.parquet
Output: game_level_features_complete_with_tier_sa.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.tier_sa_features import compute_all_tier_sa_features

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 6A: INTEGRATE TIER S+A FEATURES")
print("="*120)

# Load game-level dataset
print(f"\n[1/5] Loading game-level dataset...")
df = pd.read_parquet('../results/game_level_features_with_opponents.parquet')
print(f"  ✅ Loaded {len(df):,} team-games")
print(f"  ✅ Current features: {df.shape[1]}")

# Compute TIER S+A features for all years
print(f"\n[2/5] Computing TIER S+A features (2016-2024)...")
years = list(range(1999, 2025))
tier_sa_features = compute_all_tier_sa_features(years)

# The tier_sa_features are at team-season-week level
# We need to merge them into our team-game dataset

# Extract year and week from game_id
if 'year' not in df.columns:
    df['year'] = df['game_id'].str[:4].astype(int)
if 'week' not in df.columns:
    df['week'] = df['game_id'].str.split('_').str[1].astype(int)

print(f"\n[3/5] Merging TIER S+A features...")

# Merge each feature type
feature_count = 0

# CPOE
if 'cpoe' in tier_sa_features and len(tier_sa_features['cpoe']) > 0:
    cpoe_df = tier_sa_features['cpoe'][['team', 'season', 'week', 'cpoe_3wk', 'time_to_throw_3wk']].copy()
    cpoe_df = cpoe_df.rename(columns={'season': 'year'})
    df = df.merge(cpoe_df, on=['team', 'year', 'week'], how='left')
    feature_count += 2
    print(f"  ✅ Merged CPOE features (2 features)")

# Pressure Rate
if 'pressure' in tier_sa_features and len(tier_sa_features['pressure']) > 0:
    pressure_df = tier_sa_features['pressure'][['team', 'season', 'week', 'pressure_rate_3wk']].copy()
    pressure_df = pressure_df.rename(columns={'season': 'year'})
    df = df.merge(pressure_df, on=['team', 'year', 'week'], how='left')
    feature_count += 1
    print(f"  ✅ Merged Pressure Rate features (1 feature)")

# Injury Impact
if 'injuries' in tier_sa_features and len(tier_sa_features['injuries']) > 0:
    injury_df = tier_sa_features['injuries'][['team', 'season', 'week', 'injury_impact', 'qb_out']].copy()
    injury_df = injury_df.rename(columns={'season': 'year'})
    df = df.merge(injury_df, on=['team', 'year', 'week'], how='left')
    feature_count += 2
    print(f"  ✅ Merged Injury features (2 features)")

# RYOE
if 'ryoe' in tier_sa_features and len(tier_sa_features['ryoe']) > 0:
    ryoe_df = tier_sa_features['ryoe'][['team', 'season', 'week', 'ryoe_3wk']].copy()
    ryoe_df = ryoe_df.rename(columns={'season': 'year'})
    df = df.merge(ryoe_df, on=['team', 'year', 'week'], how='left')
    feature_count += 1
    print(f"  ✅ Merged RYOE features (1 feature)")

# Separation
if 'separation' in tier_sa_features and len(tier_sa_features['separation']) > 0:
    sep_df = tier_sa_features['separation'][['team', 'season', 'week', 'separation_3wk']].copy()
    sep_df = sep_df.rename(columns={'season': 'year'})
    df = df.merge(sep_df, on=['team', 'year', 'week'], how='left')
    feature_count += 1
    print(f"  ✅ Merged Separation features (1 feature)")

print(f"\n  ✅ Total TIER S+A features added: {feature_count}")

# Compute opponent TIER S+A features and differentials
print(f"\n[4/5] Computing opponent TIER S+A and differentials...")

tier_sa_cols = ['cpoe_3wk', 'time_to_throw_3wk', 'pressure_rate_3wk', 
                'injury_impact', 'qb_out', 'ryoe_3wk', 'separation_3wk']

# Filter to columns that exist
tier_sa_cols = [c for c in tier_sa_cols if c in df.columns]

# Create opponent features
opponent_tier_sa = df[['team', 'game_id'] + tier_sa_cols].copy()
opponent_tier_sa = opponent_tier_sa.rename(columns={'team': 'opponent'})
opponent_tier_sa = opponent_tier_sa.rename(columns={c: f'opp_{c}' for c in tier_sa_cols})

# Merge opponent features
df = df.merge(opponent_tier_sa, on=['game_id', 'opponent'], how='left')

# Create differentials
diff_count = 0
for col in tier_sa_cols:
    if f'opp_{col}' in df.columns and col != 'qb_out':  # Don't create diff for binary qb_out
        diff_name = f'diff_{col}'
        df[diff_name] = df[col] - df[f'opp_{col}']
        diff_count += 1

print(f"  ✅ Added {len(tier_sa_cols)} opponent TIER S+A features")
print(f"  ✅ Added {diff_count} differential features")

# Save final dataset
print(f"\n[5/5] Saving final dataset...")

output_file = Path('../results/game_level_features_complete_with_tier_sa.parquet')
df.to_parquet(output_file, index=False)

print(f"  ✅ Saved: {output_file}")
print(f"  ✅ Shape: {df.shape}")
print(f"  ✅ Total features: {df.shape[1]}")

# Analyze coverage
tier_sa_coverage = {}
for col in tier_sa_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        tier_sa_coverage[col] = {
            'non_null': int(non_null),
            'pct': float(pct)
        }

print(f"\n  TIER S+A Feature Coverage:")
for col, stats in tier_sa_coverage.items():
    print(f"    {col:25s}: {stats['non_null']:6,} / {len(df):6,} ({stats['pct']:5.1f}%)")

# Summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'total_features': df.shape[1],
    'tier_sa_features_added': feature_count,
    'opponent_tier_sa_features': len(tier_sa_cols),
    'differential_features': diff_count,
    'tier_sa_coverage': tier_sa_coverage,
    'output_file': str(output_file)
}

with open('../results/phase6a_tier_sa_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*120}")
print("✅ PHASE 6A: COMPLETE!")
print(f"{'='*120}")
print(f"✅ Added {feature_count} TIER S+A features")
print(f"✅ Added {len(tier_sa_cols)} opponent TIER S+A features")
print(f"✅ Added {diff_count} differential features")
print(f"✅ Total features: {df.shape[1]}")
print(f"✅ Coverage: 2016-2024 (NGS/PFR data availability)")
print(f"✅ Ready for final EDA and model training!")

