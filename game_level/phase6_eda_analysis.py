"""
Phase 6: EDA on Engineered Features
====================================

Analyze the engineered features:
- Summary statistics
- Predictive power (correlation with winning)
- Feature importance ranking
- Missing value analysis
- Feature categories breakdown
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 6: EDA ON ENGINEERED FEATURES")
print("="*120)

# Load engineered dataset
print(f"\n[1/6] Loading engineered dataset...")
df = pd.read_parquet('../results/game_level_features_with_opponents.parquet')
print(f"  ✅ Loaded {len(df):,} team-games")
print(f"  ✅ Total features: {df.shape[1]}")

# Identify feature categories
base_features = [c for c in df.columns if not any(x in c for x in ['_roll', '_std', 'opp_', 'diff_', 'win_streak', 'streak_', 'points_', 'point_diff'])]
rolling_features = [c for c in df.columns if '_roll' in c or '_std' in c]
opponent_features = [c for c in df.columns if c.startswith('opp_')]
differential_features = [c for c in df.columns if c.startswith('diff_')]
streak_features = [c for c in df.columns if 'streak' in c or 'points_' in c or 'point_diff' in c]

print(f"\n  Feature Categories:")
print(f"    Base features: {len(base_features)}")
print(f"    Rolling features: {len(rolling_features)}")
print(f"    Opponent features: {len(opponent_features)}")
print(f"    Differential features: {len(differential_features)}")
print(f"    Streak features: {len(streak_features)}")

# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================
print(f"\n[2/6] Analyzing missing values...")

missing_analysis = []
for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    if missing_count > 0:
        missing_analysis.append({
            'feature': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })

missing_df = pd.DataFrame(missing_analysis).sort_values('missing_pct', ascending=False)
missing_df.to_csv('../results/phase6_missing_values.csv', index=False)

print(f"  ✅ Features with missing values: {len(missing_df)}/{df.shape[1]}")
if len(missing_df) > 0:
    print(f"  ⚠️  Top 5 features with most missing:")
    for _, row in missing_df.head(5).iterrows():
        print(f"     - {row['feature']}: {row['missing_pct']:.1f}%")

# =============================================================================
# PREDICTIVE POWER ANALYSIS
# =============================================================================
print(f"\n[3/6] Analyzing predictive power...")

# Ensure we have win indicator
if 'win' not in df.columns:
    df['win'] = (df['total_pointsFor'] > df['total_pointsAgainst']).astype(int)

# Compute correlations with winning
predictive_power = []

# Exclude metadata columns
exclude_cols = ['team', 'game_id', 'year', 'week', 'game_date', 'opponent', 'win', 
                'total_pointsFor', 'total_pointsAgainst', 'scored_20plus', 'scored_30plus']

for col in df.columns:
    if col not in exclude_cols and df[col].dtype in ['int64', 'float64']:
        # Calculate correlation
        valid_data = df[[col, 'win']].dropna()
        
        if len(valid_data) > 100:
            try:
                r, p = pearsonr(valid_data[col], valid_data['win'])
                
                # Determine category
                if col in rolling_features:
                    category = 'rolling'
                elif col in opponent_features:
                    category = 'opponent'
                elif col in differential_features:
                    category = 'differential'
                elif col in streak_features:
                    category = 'streak'
                else:
                    category = 'base'
                
                predictive_power.append({
                    'feature': col,
                    'category': category,
                    'correlation': r,
                    'abs_correlation': abs(r),
                    'p_value': p,
                    'significant': p < 0.05,
                    'n_valid': len(valid_data)
                })
            except:
                pass

pred_df = pd.DataFrame(predictive_power).sort_values('abs_correlation', ascending=False)
pred_df.to_csv('../results/phase6_predictive_power.csv', index=False)

print(f"  ✅ Analyzed {len(pred_df)} features")
print(f"  ✅ Significant features (p<0.05): {pred_df['significant'].sum()}/{len(pred_df)}")
print(f"\n  Top 10 Most Predictive Features:")
for i, row in pred_df.head(10).iterrows():
    print(f"    {i+1}. {row['feature'][:50]:50s} | r={row['correlation']:7.4f} | {row['category']}")

# =============================================================================
# CATEGORY BREAKDOWN
# =============================================================================
print(f"\n[4/6] Category-wise predictive power...")

category_stats = pred_df.groupby('category').agg({
    'abs_correlation': ['mean', 'median', 'max'],
    'significant': 'sum',
    'feature': 'count'
}).round(4)

category_stats.columns = ['mean_abs_r', 'median_abs_r', 'max_abs_r', 'significant_count', 'total_count']
category_stats = category_stats.sort_values('mean_abs_r', ascending=False)

print(f"\n  Category Statistics:")
print(category_stats.to_string())

category_stats.to_csv('../results/phase6_category_stats.csv')

# =============================================================================
# TOP FEATURES BY CATEGORY
# =============================================================================
print(f"\n[5/6] Top features by category...")

top_by_category = {}
for category in pred_df['category'].unique():
    top_features = pred_df[pred_df['category'] == category].head(10)
    top_by_category[category] = top_features[['feature', 'correlation', 'p_value']].to_dict('records')

with open('../results/phase6_top_by_category.json', 'w') as f:
    json.dump(top_by_category, f, indent=2)

print(f"  ✅ Saved top features by category")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print(f"\n[6/6] Creating summary report...")

summary = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'total_features': df.shape[1],
    'feature_categories': {
        'base': len(base_features),
        'rolling': len(rolling_features),
        'opponent': len(opponent_features),
        'differential': len(differential_features),
        'streak': len(streak_features)
    },
    'missing_values': {
        'features_with_missing': len(missing_df),
        'total_features': df.shape[1],
        'pct_complete': ((df.shape[1] - len(missing_df)) / df.shape[1] * 100)
    },
    'predictive_power': {
        'total_analyzed': len(pred_df),
        'significant_features': int(pred_df['significant'].sum()),
        'mean_abs_correlation': float(pred_df['abs_correlation'].mean()),
        'top_10_features': pred_df.head(10)['feature'].tolist()
    }
}

with open('../results/phase6_eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*120}")
print("✅ PHASE 6 EDA: COMPLETE!")
print(f"{'='*120}")
print(f"✅ Total features: {df.shape[1]}")
print(f"✅ Significant predictive features: {pred_df['significant'].sum()}/{len(pred_df)}")
print(f"✅ Mean absolute correlation: {pred_df['abs_correlation'].mean():.4f}")
print(f"✅ Features with <10% missing: {len(missing_df[missing_df['missing_pct'] < 10])}/{len(missing_df)}")
print(f"\n📊 Generated files:")
print(f"   - phase6_missing_values.csv")
print(f"   - phase6_predictive_power.csv")
print(f"   - phase6_category_stats.csv")
print(f"   - phase6_top_by_category.json")
print(f"   - phase6_eda_summary.json")

