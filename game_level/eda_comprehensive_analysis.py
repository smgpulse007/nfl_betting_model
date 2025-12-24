"""
Phase 5D: Comprehensive EDA on Game-Level Data

Generate all analysis results for dashboard integration:
1. Summary statistics
2. Temporal trends
3. Correlation analysis
4. Predictive power analysis
5. Home/away analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 5D: COMPREHENSIVE GAME-LEVEL EDA")
print("="*120)

# Load game-level data
print(f"\n[1/6] Loading game-level dataset...")
df = pd.read_parquet('../results/game_level_features_1999_2024_complete.parquet')
print(f"  ✅ Loaded {len(df):,} team-games")
print(f"  ✅ Features: {df.shape[1]}")

# Load approved features
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = approved_data['features']

print(f"  ✅ Approved features: {len(approved_features)}")

# Extract year from game_id
df['year'] = df['game_id'].str[:4].astype(int)

# Add win indicator (assuming we have total_wins column)
if 'total_wins' in df.columns:
    df['win'] = df['total_wins']
else:
    # Calculate from point differential
    df['win'] = (df['total_pointsFor'] > df['total_pointsAgainst']).astype(int)

# 1. SUMMARY STATISTICS
print(f"\n[2/6] Computing summary statistics...")
summary_stats = []

for feature in approved_features:
    if feature in df.columns:
        data = df[feature].dropna()
        
        summary_stats.append({
            'feature': feature,
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('../results/game_level_eda_summary_statistics.csv', index=False)
print(f"  ✅ Saved summary statistics for {len(summary_df)} features")

# 2. TEMPORAL TRENDS
print(f"\n[3/6] Analyzing temporal trends...")
temporal_trends = []

for feature in approved_features:
    if feature in df.columns:
        # Calculate mean by year
        yearly_means = df.groupby('year')[feature].mean()
        
        if len(yearly_means) > 1:
            # Linear regression
            years = yearly_means.index.values
            values = yearly_means.values
            
            # Remove NaN values
            mask = ~np.isnan(values)
            if mask.sum() > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years[mask], values[mask])
                
                # Calculate percent change
                first_val = yearly_means.iloc[0]
                last_val = yearly_means.iloc[-1]
                pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                
                temporal_trends.append({
                    'feature': feature,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'pct_change': pct_change,
                    'first_year_mean': first_val,
                    'last_year_mean': last_val,
                    'significant': p_value < 0.05
                })

temporal_df = pd.DataFrame(temporal_trends)
temporal_df.to_csv('../results/game_level_eda_temporal_trends.csv', index=False)
print(f"  ✅ Saved temporal trends for {len(temporal_df)} features")
print(f"     Significant trends: {temporal_df['significant'].sum()}/{len(temporal_df)}")

# 3. CORRELATION WITH WINNING
print(f"\n[4/6] Computing correlations with winning...")
correlations = []

for feature in approved_features:
    if feature in df.columns and feature != 'win':
        # Calculate correlation with winning
        data = df[[feature, 'win']].dropna()
        
        if len(data) > 10:
            r, p = pearsonr(data[feature], data['win'])
            
            correlations.append({
                'feature': feature,
                'correlation': r,
                'abs_correlation': abs(r),
                'p_value': p,
                'significant': p < 0.05
            })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
corr_df.to_csv('../results/game_level_eda_predictive_power.csv', index=False)
print(f"  ✅ Saved predictive power for {len(corr_df)} features")
print(f"     Significant correlations: {corr_df['significant'].sum()}/{len(corr_df)}")

# 4. CORRELATION MATRIX (top 50 features)
print(f"\n[5/6] Computing correlation matrix...")
top_features = corr_df.head(50)['feature'].tolist()
corr_matrix = df[top_features].corr()
corr_matrix.to_csv('../results/game_level_eda_correlation_matrix.csv')
print(f"  ✅ Saved correlation matrix for top 50 features")

# Find high correlations
high_corr = []
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr.append({
                'feature1': top_features[i],
                'feature2': top_features[j],
                'correlation': corr_val,
                'abs_correlation': abs(corr_val)
            })

high_corr_df = pd.DataFrame(high_corr).sort_values('abs_correlation', ascending=False)
high_corr_df.to_csv('../results/game_level_eda_high_correlations.csv', index=False)
print(f"  ✅ Found {len(high_corr_df)} high correlation pairs (|r| > 0.7)")

# 5. HOME/AWAY ANALYSIS
print(f"\n[6/6] Analyzing home/away splits...")

# Determine home/away from game_id
df['is_home'] = df.apply(lambda row: row['team'] in row['game_id'].split('_')[-1], axis=1)

home_away_stats = []
for feature in approved_features[:20]:  # Top 20 for now
    if feature in df.columns:
        home_mean = df[df['is_home']][feature].mean()
        away_mean = df[~df['is_home']][feature].mean()
        
        home_away_stats.append({
            'feature': feature,
            'home_mean': home_mean,
            'away_mean': away_mean,
            'difference': home_mean - away_mean,
            'pct_difference': ((home_mean - away_mean) / away_mean * 100) if away_mean != 0 else 0
        })

home_away_df = pd.DataFrame(home_away_stats)
home_away_df.to_csv('../results/game_level_eda_home_away.csv', index=False)
print(f"  ✅ Saved home/away analysis")

print(f"\n{'='*120}")
print("✅ PHASE 5D: COMPREHENSIVE EDA COMPLETE!")
print(f"{'='*120}")
print(f"\nGenerated files:")
print(f"  - game_level_eda_summary_statistics.csv")
print(f"  - game_level_eda_temporal_trends.csv")
print(f"  - game_level_eda_predictive_power.csv")
print(f"  - game_level_eda_correlation_matrix.csv")
print(f"  - game_level_eda_high_correlations.csv")
print(f"  - game_level_eda_home_away.csv")
print(f"\n✅ Ready for dashboard integration")

