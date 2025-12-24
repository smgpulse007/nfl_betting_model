"""
Predictive Power Analysis: Identify features most correlated with winning

This analysis calculates:
1. Correlation with win percentage
2. Correlation with point differential
3. Feature importance using Random Forest
4. Statistical significance testing
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("PREDICTIVE POWER ANALYSIS: FEATURE CORRELATION WITH WINNING")
print("="*120)

# Load the complete training dataset
print(f"\n[1/5] Loading complete training dataset...")
df = pd.read_parquet('data/derived_features/espn_derived_1999_2024_complete.parquet')

print(f"  ✅ Dataset loaded: {df.shape}")

# Identify target variables
target_vars = {
    'total_winPercent': 'Win Percentage',
    'total_pointDifferential': 'Point Differential',
    'total_wins': 'Total Wins'
}

# Get feature columns (exclude metadata and target variables)
exclude_cols = ['team', 'year'] + list(target_vars.keys())
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"  ✅ Features to analyze: {len(feature_cols)}")
print(f"  ✅ Target variables: {list(target_vars.values())}")

# ============================================================================
# 1. CORRELATION WITH WIN PERCENTAGE
# ============================================================================
print(f"\n[2/5] Calculating correlations with win percentage...")

correlations = []

for feature in feature_cols:
    # Remove NaN values
    mask = ~(df[feature].isna() | df['total_winPercent'].isna())
    
    if mask.sum() > 10:  # Need at least 10 data points
        r, p = pearsonr(df.loc[mask, feature], df.loc[mask, 'total_winPercent'])
        
        correlations.append({
            'feature': feature,
            'r_winpct': r,
            'p_value': p,
            'abs_r': abs(r),
            'significant': p < 0.05
        })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('abs_r', ascending=False)

print(f"  ✅ Correlations calculated")
print(f"     - Significant correlations (p < 0.05): {corr_df['significant'].sum()}/{len(corr_df)}")
print(f"     - Strong correlations (|r| > 0.50): {(corr_df['abs_r'] > 0.50).sum()}")
print(f"     - Moderate correlations (|r| > 0.30): {(corr_df['abs_r'] > 0.30).sum()}")

print(f"\n  🏆 TOP 20 FEATURES CORRELATED WITH WINNING:")
for idx, row in corr_df.head(20).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"     {row['feature']:50s} r={row['r_winpct']:7.4f} {sig}")

# ============================================================================
# 2. CORRELATION WITH POINT DIFFERENTIAL
# ============================================================================
print(f"\n[3/5] Calculating correlations with point differential...")

for idx, row in corr_df.iterrows():
    feature = row['feature']
    mask = ~(df[feature].isna() | df['total_pointDifferential'].isna())
    
    if mask.sum() > 10:
        r, p = pearsonr(df.loc[mask, feature], df.loc[mask, 'total_pointDifferential'])
        corr_df.loc[idx, 'r_ptdiff'] = r
        corr_df.loc[idx, 'p_ptdiff'] = p

print(f"  ✅ Point differential correlations calculated")

# ============================================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print(f"\n[4/5] Calculating Random Forest feature importance...")

# Prepare data for Random Forest
X = df[feature_cols].fillna(df[feature_cols].median())
y = df['total_winPercent'].fillna(df['total_winPercent'].median())

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X, y)

# Get feature importances
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"  ✅ Random Forest trained")
print(f"     - R² score: {rf.score(X, y):.4f}")

print(f"\n  🌲 TOP 20 FEATURES BY RANDOM FOREST IMPORTANCE:")
for idx, row in importances.head(20).iterrows():
    print(f"     {row['feature']:50s} importance={row['importance']:.6f}")

# Merge importance into correlation dataframe
corr_df = corr_df.merge(importances, on='feature', how='left')

# ============================================================================
# 4. COMBINED RANKING
# ============================================================================
print(f"\n[5/5] Creating combined predictive power ranking...")

# Normalize scores to 0-1 range
corr_df['norm_r'] = (corr_df['abs_r'] - corr_df['abs_r'].min()) / (corr_df['abs_r'].max() - corr_df['abs_r'].min())
corr_df['norm_importance'] = (corr_df['importance'] - corr_df['importance'].min()) / (corr_df['importance'].max() - corr_df['importance'].min())

# Combined score (50% correlation + 50% importance)
corr_df['combined_score'] = 0.5 * corr_df['norm_r'] + 0.5 * corr_df['norm_importance']
corr_df = corr_df.sort_values('combined_score', ascending=False)

# Add rank
corr_df['rank'] = range(1, len(corr_df) + 1)

print(f"  ✅ Combined ranking created")

print(f"\n  🎯 TOP 30 FEATURES BY COMBINED PREDICTIVE POWER:")
print(f"     {'Rank':<6} {'Feature':<50} {'r(win%)':<10} {'RF Imp':<10} {'Combined':<10}")
print(f"     {'-'*6} {'-'*50} {'-'*10} {'-'*10} {'-'*10}")
for idx, row in corr_df.head(30).iterrows():
    print(f"     {row['rank']:<6.0f} {row['feature']:<50} {row['r_winpct']:<10.4f} {row['importance']:<10.6f} {row['combined_score']:<10.4f}")

# Save results
corr_df.to_csv('results/eda_predictive_power.csv', index=False)
print(f"\n  ✅ Saved: results/eda_predictive_power.csv")

print(f"\n{'='*120}")
print(f"✅ PREDICTIVE POWER ANALYSIS COMPLETE!")
print(f"{'='*120}")
print(f"\nKey Findings:")
print(f"  - {corr_df['significant'].sum()}/{len(corr_df)} features significantly correlated with winning (p < 0.05)")
print(f"  - {(corr_df['abs_r'] > 0.50).sum()} features with strong correlation (|r| > 0.50)")
print(f"  - Top feature: {corr_df.iloc[0]['feature']} (r={corr_df.iloc[0]['r_winpct']:.4f})")
print(f"  - Random Forest R²: {rf.score(X, y):.4f}")
print(f"\n{'='*120}")

