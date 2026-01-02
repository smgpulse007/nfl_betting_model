"""
Task 8C.3: Feature Correlation & Redundancy Analysis

Analyze feature correlations to identify redundant features and recommend feature reduction strategies.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8C.3: FEATURE CORRELATION & REDUNDANCY ANALYSIS")
print("="*120)

# Load data
print(f"\n[1/5] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

unknown_pregame = [
    'OTLosses', 'losses', 'pointsAgainst', 'pointsFor', 'ties', 'winPercent', 
    'winPercentage', 'wins', 'losses_roll3', 'losses_roll5', 'losses_std',
    'winPercent_roll3', 'winPercent_roll5', 'winPercent_std',
    'wins_roll3', 'wins_roll5', 'wins_std',
    'scored_20plus', 'scored_30plus', 'streak_20plus', 'streak_30plus',
    'vsconf_OTLosses', 'vsconf_leagueWinPercent', 'vsconf_losses', 'vsconf_ties', 'vsconf_wins',
    'vsdiv_OTLosses', 'vsdiv_divisionLosses', 'vsdiv_divisionTies', 
    'vsdiv_divisionWinPercent', 'vsdiv_divisionWins', 'vsdiv_losses', 'vsdiv_ties', 'vsdiv_wins',
    'div_game', 'rest_advantage', 'opponent'
]
pre_game_features.extend(unknown_pregame)

df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

train = df[df['season'] <= 2019].copy()
X_train = train[numeric_pregame].fillna(train[numeric_pregame].median())

print(f"  ✅ Loaded training set: {len(X_train):,} games")
print(f"  ✅ Features: {len(numeric_pregame)}")

# Calculate correlation matrix
print(f"\n[2/5] Calculating correlation matrix...")

corr_matrix = X_train.corr()

print(f"  ✅ Correlation matrix calculated ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")

# Identify highly correlated feature pairs
print(f"\n[3/5] Identifying highly correlated features...")

# Find pairs with correlation > 0.9 (excluding diagonal)
high_corr_threshold = 0.9
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': float(corr_matrix.iloc[i, j])
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)

print(f"  ✅ Found {len(high_corr_pairs)} feature pairs with |correlation| > {high_corr_threshold}")
print(f"\n  Top 10 highly correlated pairs:")
for idx, row in high_corr_df.head(10).iterrows():
    print(f"    {row['feature_1']:<45} <-> {row['feature_2']:<45} {row['correlation']:>7.3f}")

# Identify redundant features
print(f"\n[4/5] Identifying redundant features...")

# Load feature importance from SHAP and permutation
with open('../results/phase8_results/shap_analysis/global_feature_importance.json', 'r') as f:
    shap_importance = json.load(f)

with open('../results/phase8_results/permutation_importance/permutation_importance.json', 'r') as f:
    perm_importance = json.load(f)

# Calculate average importance across all models
all_features_importance = {}

for model_name in shap_importance.keys():
    for feat_dict in shap_importance[model_name]:
        feat = feat_dict['feature']
        if feat not in all_features_importance:
            all_features_importance[feat] = []
        all_features_importance[feat].append(feat_dict['importance'])

for model_name in perm_importance.keys():
    for feat_dict in perm_importance[model_name]:
        feat = feat_dict['feature']
        if feat not in all_features_importance:
            all_features_importance[feat] = []
        all_features_importance[feat].append(feat_dict['importance_mean'])

# Calculate average importance
avg_importance = {feat: np.mean(importances) for feat, importances in all_features_importance.items()}

# For each highly correlated pair, recommend removing the less important feature
redundant_features = set()
redundancy_recommendations = []

for _, row in high_corr_df.iterrows():
    feat1 = row['feature_1']
    feat2 = row['feature_2']
    
    imp1 = avg_importance.get(feat1, 0)
    imp2 = avg_importance.get(feat2, 0)
    
    if imp1 > imp2:
        remove_feat = feat2
        keep_feat = feat1
    else:
        remove_feat = feat1
        keep_feat = feat2
    
    redundant_features.add(remove_feat)
    redundancy_recommendations.append({
        'remove': remove_feat,
        'keep': keep_feat,
        'correlation': float(row['correlation']),
        'remove_importance': float(avg_importance.get(remove_feat, 0)),
        'keep_importance': float(avg_importance.get(keep_feat, 0))
    })

print(f"  ✅ Identified {len(redundant_features)} redundant features to remove")
print(f"  ✅ Recommended feature set size: {len(numeric_pregame) - len(redundant_features)} features")

# Visualizations
print(f"\n[5/5] Creating visualizations...")

import os
os.makedirs('../results/phase8_results/feature_correlation', exist_ok=True)

# 1. Full correlation heatmap (top 50 features by importance)
top_50_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:50]
top_50_feature_names = [f[0] for f in top_50_features]

# Filter correlation matrix to top 50 features
corr_top50 = corr_matrix.loc[top_50_feature_names, top_50_feature_names]

plt.figure(figsize=(20, 18))
sns.heatmap(corr_top50, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (Top 50 Features by Importance)',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('../results/phase8_results/feature_correlation/correlation_heatmap_top50.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: correlation_heatmap_top50.png")
plt.close()

# 2. Highly correlated pairs visualization
if len(high_corr_df) > 0:
    plt.figure(figsize=(14, 10))

    # Show top 20 pairs
    top_pairs = high_corr_df.head(20)

    y_pos = np.arange(len(top_pairs))
    colors = ['red' if abs(c) > 0.95 else 'orange' for c in top_pairs['correlation']]

    plt.barh(y_pos, top_pairs['correlation'].abs(), color=colors, alpha=0.7)

    labels = [f"{row['feature_1'].replace('home_', '').replace('away_', '')[:25]}\n<->\n{row['feature_2'].replace('home_', '').replace('away_', '')[:25]}"
              for _, row in top_pairs.iterrows()]

    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel('|Correlation|', fontsize=12)
    plt.title(f'Top 20 Highly Correlated Feature Pairs (|r| > {high_corr_threshold})',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='|r| = 0.95')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/phase8_results/feature_correlation/highly_correlated_pairs.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: highly_correlated_pairs.png")
    plt.close()

# 3. Feature importance distribution
plt.figure(figsize=(12, 6))

importance_values = list(avg_importance.values())

plt.hist(importance_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Average Feature Importance', fontsize=12)
plt.ylabel('Number of Features', fontsize=12)
plt.title('Distribution of Feature Importance Across All Models', fontsize=14, fontweight='bold')
plt.axvline(x=np.median(importance_values), color='red', linestyle='--',
            label=f'Median: {np.median(importance_values):.4f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/phase8_results/feature_correlation/feature_importance_distribution.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: feature_importance_distribution.png")
plt.close()

# Save results
print(f"\n  Saving results...")

# Save correlation matrix
corr_matrix.to_csv('../results/phase8_results/feature_correlation/correlation_matrix.csv')
print(f"  ✅ Saved: correlation_matrix.csv")

# Save highly correlated pairs
with open('../results/phase8_results/feature_correlation/highly_correlated_pairs.json', 'w') as f:
    json.dump(high_corr_df.to_dict('records'), f, indent=2)
print(f"  ✅ Saved: highly_correlated_pairs.json")

# Save redundancy recommendations
redundancy_summary = {
    'total_features': len(numeric_pregame),
    'redundant_features_count': len(redundant_features),
    'recommended_feature_count': len(numeric_pregame) - len(redundant_features),
    'redundant_features': sorted(list(redundant_features)),
    'redundancy_recommendations': redundancy_recommendations
}

with open('../results/phase8_results/feature_correlation/redundancy_recommendations.json', 'w') as f:
    json.dump(redundancy_summary, f, indent=2)
print(f"  ✅ Saved: redundancy_recommendations.json")

# Save average feature importance
importance_df = pd.DataFrame([
    {'feature': feat, 'avg_importance': imp}
    for feat, imp in avg_importance.items()
]).sort_values('avg_importance', ascending=False)

importance_df.to_csv('../results/phase8_results/feature_correlation/average_feature_importance.csv', index=False)
print(f"  ✅ Saved: average_feature_importance.csv")

print(f"\n{'='*120}")
print("✅ FEATURE CORRELATION & REDUNDANCY ANALYSIS COMPLETE")
print(f"{'='*120}")
print(f"\n📊 Summary:")
print(f"  • Total features: {len(numeric_pregame)}")
print(f"  • Highly correlated pairs (|r| > {high_corr_threshold}): {len(high_corr_pairs)}")
print(f"  • Redundant features identified: {len(redundant_features)}")
print(f"  • Recommended feature set size: {len(numeric_pregame) - len(redundant_features)}")
print(f"  • Potential dimensionality reduction: {len(redundant_features)/len(numeric_pregame)*100:.1f}%")
print(f"\n✅ Generated 3 visualization plots")
print(f"✅ Saved correlation matrix and recommendations")
print(f"\n{'='*120}")

