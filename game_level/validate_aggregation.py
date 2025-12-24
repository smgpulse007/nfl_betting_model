"""
Phase 5B: Aggregation Validation

Validate that game-level features aggregate correctly to season level.
Compare aggregated game-level features with original season-level features.
Expected: r > 0.95 for all counting stats.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 5B: AGGREGATION VALIDATION")
print("="*120)

# Load game-level features (2024)
print(f"\n[1/5] Loading game-level features...")
game_level = pd.read_parquet('../results/game_level_features_2024.parquet')
print(f"  ✅ Loaded {len(game_level)} team-games")
print(f"  ✅ Features: {game_level.shape[1]}")

# Load season-level features (2024)
print(f"\n[2/5] Loading season-level features...")
season_level = pd.read_parquet('../data/derived_features/espn_derived_2024.parquet')
season_level = season_level.reset_index()  # Move 'team' from index to column
print(f"  ✅ Loaded {len(season_level)} team-seasons")
print(f"  ✅ Features: {season_level.shape[1]}")

# Aggregate game-level to season-level
print(f"\n[3/5] Aggregating game-level features to season level...")

# Get numeric columns (exclude metadata)
metadata_cols = {'team', 'game_id'}
numeric_cols = [col for col in game_level.columns if col not in metadata_cols and game_level[col].dtype in ['int64', 'float64']]

# Aggregate by team
aggregated = game_level.groupby('team')[numeric_cols].sum().reset_index()
print(f"  ✅ Aggregated to {len(aggregated)} teams")

# Compare with season-level features
print(f"\n[4/5] Comparing aggregated vs season-level features...")

# Find common features
common_features = set(aggregated.columns) & set(season_level.columns) - {'team'}
print(f"  ✅ Common features: {len(common_features)}")

# Calculate correlations
correlations = []
for feature in sorted(common_features):
    # Merge on team
    merged = aggregated[['team', feature]].merge(
        season_level[['team', feature]], 
        on='team', 
        suffixes=('_game', '_season')
    )
    
    # Calculate correlation
    game_vals = merged[f'{feature}_game'].values
    season_vals = merged[f'{feature}_season'].values
    
    # Skip if all zeros or constant
    if game_vals.std() == 0 or season_vals.std() == 0:
        continue
    
    r, p = pearsonr(game_vals, season_vals)
    
    correlations.append({
        'feature': feature,
        'correlation': r,
        'p_value': p,
        'game_mean': game_vals.mean(),
        'season_mean': season_vals.mean(),
        'game_std': game_vals.std(),
        'season_std': season_vals.std()
    })

# Create correlation DataFrame
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('correlation', ascending=False)

print(f"\n  Correlation Statistics:")
print(f"    - Features compared: {len(corr_df)}")
print(f"    - Mean correlation: {corr_df['correlation'].mean():.4f}")
print(f"    - Median correlation: {corr_df['correlation'].median():.4f}")
print(f"    - Min correlation: {corr_df['correlation'].min():.4f}")
print(f"    - Max correlation: {corr_df['correlation'].max():.4f}")

# Count by correlation threshold
perfect = len(corr_df[corr_df['correlation'] >= 0.999])
excellent = len(corr_df[corr_df['correlation'] >= 0.95])
good = len(corr_df[corr_df['correlation'] >= 0.85])
poor = len(corr_df[corr_df['correlation'] < 0.85])

print(f"\n  Correlation Distribution:")
print(f"    - Perfect (r >= 0.999): {perfect} ({perfect/len(corr_df)*100:.1f}%)")
print(f"    - Excellent (r >= 0.95): {excellent} ({excellent/len(corr_df)*100:.1f}%)")
print(f"    - Good (r >= 0.85): {good} ({good/len(corr_df)*100:.1f}%)")
print(f"    - Poor (r < 0.85): {poor} ({poor/len(corr_df)*100:.1f}%)")

# Show worst correlations
if poor > 0:
    print(f"\n  ⚠️  Features with r < 0.85:")
    worst = corr_df[corr_df['correlation'] < 0.85].head(10)
    for idx, row in worst.iterrows():
        print(f"    - {row['feature']}: r={row['correlation']:.4f}")

# Save results
print(f"\n[5/5] Saving validation results...")
output_dir = Path('../results')

# Save correlation results
corr_file = output_dir / 'phase5b_aggregation_validation.csv'
corr_df.to_csv(corr_file, index=False)
print(f"  ✅ Saved to: {corr_file}")

# Save summary statistics
summary = {
    'total_features_compared': len(corr_df),
    'mean_correlation': float(corr_df['correlation'].mean()),
    'median_correlation': float(corr_df['correlation'].median()),
    'min_correlation': float(corr_df['correlation'].min()),
    'max_correlation': float(corr_df['correlation'].max()),
    'perfect_correlations': int(perfect),
    'excellent_correlations': int(excellent),
    'good_correlations': int(good),
    'poor_correlations': int(poor),
    'validation_passed': poor == 0
}

summary_file = output_dir / 'phase5b_validation_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ Saved to: {summary_file}")

print(f"\n{'='*120}")
if poor == 0:
    print("✅ PHASE 5B: AGGREGATION VALIDATION PASSED!")
    print("   All features have r >= 0.85 when aggregated to season level")
else:
    print("⚠️  PHASE 5B: AGGREGATION VALIDATION NEEDS REVIEW")
    print(f"   {poor} features have r < 0.85 - investigate discrepancies")
print(f"{'='*120}")

