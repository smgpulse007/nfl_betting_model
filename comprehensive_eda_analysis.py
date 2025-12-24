"""
Comprehensive EDA Analysis for Phase 4 Historical Dataset (1999-2024)

This script performs deep statistical analysis to inform:
1. Dashboard visualizations
2. Game-level vs season-level decision
3. Feature engineering prioritization
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("COMPREHENSIVE EDA ANALYSIS: PHASE 4 HISTORICAL DATASET (1999-2024)")
print("="*120)

# Load the complete training dataset
print(f"\n[1/10] Loading complete training dataset...")
df = pd.read_parquet('data/derived_features/espn_derived_1999_2024_complete.parquet')

print(f"  ✅ Dataset loaded")
print(f"     - Shape: {df.shape}")
print(f"     - Years: {df['year'].min()}-{df['year'].max()}")
print(f"     - Teams: {df['team'].nunique()} unique")
print(f"     - Features: {len(df.columns)-2} (excluding team, year)")

# Separate features from metadata
feature_cols = [col for col in df.columns if col not in ['team', 'year']]
print(f"     - Feature columns: {len(feature_cols)}")

# ============================================================================
# 1. UNIVARIATE ANALYSIS
# ============================================================================
print(f"\n[2/10] Univariate Analysis...")

# Summary statistics
summary_stats = df[feature_cols].describe().T
summary_stats['skewness'] = df[feature_cols].skew()
summary_stats['kurtosis'] = df[feature_cols].kurtosis()
summary_stats['missing_pct'] = df[feature_cols].isnull().sum() / len(df) * 100

# Identify highly skewed features (|skewness| > 2)
highly_skewed = summary_stats[abs(summary_stats['skewness']) > 2].sort_values('skewness', ascending=False)

print(f"  ✅ Summary statistics calculated")
print(f"     - Highly skewed features (|skew| > 2): {len(highly_skewed)}")
print(f"     - Missing values: {summary_stats['missing_pct'].sum():.2f}% total")

# Save summary statistics
summary_stats.to_csv('results/eda_summary_statistics.csv')
print(f"  ✅ Saved: results/eda_summary_statistics.csv")

# ============================================================================
# 2. TEMPORAL ANALYSIS
# ============================================================================
print(f"\n[3/10] Temporal Analysis (1999-2024)...")

# Calculate year-over-year trends for each feature
temporal_trends = {}
for feature in feature_cols:
    yearly_mean = df.groupby('year')[feature].mean()
    
    # Linear regression to detect trend
    years = yearly_mean.index.values
    values = yearly_mean.values
    
    # Remove NaN values
    mask = ~np.isnan(values)
    if mask.sum() > 2:  # Need at least 3 points
        slope, intercept, r_value, p_value, std_err = stats.linregress(years[mask], values[mask])
        
        temporal_trends[feature] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'pct_change_1999_2024': ((yearly_mean.iloc[-1] - yearly_mean.iloc[0]) / yearly_mean.iloc[0] * 100) if yearly_mean.iloc[0] != 0 else np.nan
        }

temporal_df = pd.DataFrame(temporal_trends).T
temporal_df = temporal_df.sort_values('pct_change_1999_2024', ascending=False, key=abs)

print(f"  ✅ Temporal trends calculated")
print(f"     - Features with significant trends (p < 0.05): {(temporal_df['p_value'] < 0.05).sum()}")
print(f"     - Top 5 increasing features:")
for feat in temporal_df.head(5).index:
    pct = temporal_df.loc[feat, 'pct_change_1999_2024']
    print(f"       {feat}: +{pct:.1f}%")

temporal_df.to_csv('results/eda_temporal_trends.csv')
print(f"  ✅ Saved: results/eda_temporal_trends.csv")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
print(f"\n[4/10] Correlation Analysis...")

# Calculate correlation matrix
corr_matrix = df[feature_cols].corr()

# Find highly correlated pairs (r > 0.90, excluding diagonal)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.90:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)

print(f"  ✅ Correlation matrix calculated")
print(f"     - Highly correlated pairs (|r| > 0.90): {len(high_corr_pairs)}")
print(f"     - Top 5 correlated pairs:")
for idx, row in high_corr_df.head(5).iterrows():
    print(f"       {row['feature1']} <-> {row['feature2']}: r={row['correlation']:.3f}")

# Save correlation matrix and high correlation pairs
corr_matrix.to_csv('results/eda_correlation_matrix.csv')
high_corr_df.to_csv('results/eda_high_correlations.csv', index=False)
print(f"  ✅ Saved: results/eda_correlation_matrix.csv, results/eda_high_correlations.csv")

print(f"\n[5/10] Preparing for predictive power analysis...")
print(f"  (This requires game outcome data - will be calculated in next step)")

# Save analysis metadata
metadata = {
    'dataset_shape': df.shape,
    'years_covered': f"{df['year'].min()}-{df['year'].max()}",
    'unique_teams': int(df['team'].nunique()),
    'total_features': len(feature_cols),
    'highly_skewed_features': len(highly_skewed),
    'significant_temporal_trends': int((temporal_df['p_value'] < 0.05).sum()),
    'high_correlation_pairs': len(high_corr_pairs)
}

with open('results/eda_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*120}")
print(f"✅ COMPREHENSIVE EDA ANALYSIS COMPLETE!")
print(f"{'='*120}")
print(f"\nOutput files:")
print(f"  - results/eda_summary_statistics.csv")
print(f"  - results/eda_temporal_trends.csv")
print(f"  - results/eda_correlation_matrix.csv")
print(f"  - results/eda_high_correlations.csv")
print(f"  - results/eda_metadata.json")
print(f"\n{'='*120}")

