"""
Phase 6A: TIER S+A Feature EDA
===============================

Analyze the newly integrated TIER S+A features for predictive power.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

print("="*120)
print("PHASE 6A: TIER S+A FEATURE EDA")
print("="*120)

# Load dataset
print(f"\n[1/4] Loading dataset...")
df = pd.read_parquet('../results/game_level_features_complete_with_tier_sa.parquet')
print(f"  ✅ Loaded {len(df):,} team-games with {df.shape[1]} features")

# Identify TIER S+A features
tier_sa_base = ['cpoe_3wk', 'time_to_throw_3wk', 'pressure_rate_3wk', 
                'injury_impact', 'qb_out', 'ryoe_3wk', 'separation_3wk']
tier_sa_opp = [f'opp_{c}' for c in tier_sa_base]
tier_sa_diff = [f'diff_{c}' for c in tier_sa_base if c != 'qb_out']

all_tier_sa = tier_sa_base + tier_sa_opp + tier_sa_diff

# Filter to features that exist
all_tier_sa = [f for f in all_tier_sa if f in df.columns]

print(f"\n[2/4] Analyzing {len(all_tier_sa)} TIER S+A features...")

# Compute correlations with winning
results = []
for feature in all_tier_sa:
    # Filter to non-null values
    valid_data = df[[feature, 'win']].dropna()
    
    if len(valid_data) < 100:  # Skip if too few samples
        continue
    
    # Compute correlation
    corr, p_value = stats.pearsonr(valid_data[feature], valid_data['win'])
    
    # Categorize
    if feature.startswith('diff_'):
        category = 'tier_sa_differential'
    elif feature.startswith('opp_'):
        category = 'tier_sa_opponent'
    else:
        category = 'tier_sa_base'
    
    results.append({
        'feature': feature,
        'category': category,
        'correlation': corr,
        'abs_correlation': abs(corr),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'sample_size': len(valid_data),
        'missing_pct': ((len(df) - len(valid_data)) / len(df)) * 100
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('abs_correlation', ascending=False)

# Save results
results_df.to_csv('../results/phase6a_tier_sa_predictive_power.csv', index=False)
print(f"  ✅ Saved: phase6a_tier_sa_predictive_power.csv")

# Category statistics
print(f"\n[3/4] Computing category statistics...")
category_stats = results_df.groupby('category').agg({
    'abs_correlation': ['mean', 'median', 'max', 'count'],
    'significant': 'sum'
}).round(4)

category_stats.columns = ['mean_abs_r', 'median_abs_r', 'max_abs_r', 'total_count', 'significant_count']
category_stats.to_csv('../results/phase6a_tier_sa_category_stats.csv')
print(f"  ✅ Saved: phase6a_tier_sa_category_stats.csv")

# Print summary
print(f"\n{'='*120}")
print("TIER S+A FEATURE ANALYSIS SUMMARY")
print(f"{'='*120}")

print(f"\n📊 Overall Statistics:")
print(f"  Total features analyzed: {len(results_df)}")
print(f"  Significant features (p<0.05): {results_df['significant'].sum()} ({results_df['significant'].sum()/len(results_df)*100:.1f}%)")
print(f"  Mean |r|: {results_df['abs_correlation'].mean():.4f}")
print(f"  Median |r|: {results_df['abs_correlation'].median():.4f}")

print(f"\n📊 Category Performance:")
print(category_stats.to_string())

print(f"\n🏆 Top 10 TIER S+A Features:")
for i, row in results_df.head(10).iterrows():
    sig = "✅" if row['significant'] else "❌"
    print(f"  {i+1:2d}. {row['feature']:30s} | r={row['correlation']:7.4f} | p={row['p_value']:.4e} | {sig}")

print(f"\n📊 Coverage Analysis:")
for feature in tier_sa_base:
    if feature in df.columns:
        non_null = df[feature].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"  {feature:25s}: {non_null:6,} / {len(df):6,} ({pct:5.1f}%)")

# Create summary JSON
print(f"\n[4/4] Creating summary...")
summary = {
    'total_features': len(results_df),
    'significant_features': int(results_df['significant'].sum()),
    'mean_abs_correlation': float(results_df['abs_correlation'].mean()),
    'median_abs_correlation': float(results_df['abs_correlation'].median()),
    'category_stats': category_stats.to_dict(),
    'top_10_features': results_df.head(10)[['feature', 'correlation', 'p_value']].to_dict('records'),
    'coverage': {
        feature: {
            'non_null': int(df[feature].notna().sum()),
            'pct': float((df[feature].notna().sum() / len(df)) * 100)
        }
        for feature in tier_sa_base if feature in df.columns
    }
}

with open('../results/phase6a_tier_sa_eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  ✅ Saved: phase6a_tier_sa_eda_summary.json")

print(f"\n{'='*120}")
print("✅ PHASE 6A EDA: COMPLETE!")
print(f"{'='*120}")
print(f"✅ Analyzed {len(results_df)} TIER S+A features")
print(f"✅ {results_df['significant'].sum()}/{len(results_df)} features significant (p<0.05)")
print(f"✅ Mean |r|: {results_df['abs_correlation'].mean():.4f}")
print(f"✅ Ready for dashboard integration!")

