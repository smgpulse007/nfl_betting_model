"""
Task 8C.2: Permutation Importance Analysis

Calculate permutation importance for all models and compare with SHAP-based importance.
Permutation importance measures the decrease in model performance when a feature's values are randomly shuffled.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8C.2: PERMUTATION IMPORTANCE ANALYSIS")
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

test = df[df['season'] == 2024].copy()

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
X_test = test[numeric_pregame].fillna(train[numeric_pregame].median())
y_test = test['home_win'].values

print(f"  ✅ Loaded test set: {len(y_test):,} games")
print(f"  ✅ Features: {len(numeric_pregame)}")

# Load models
print(f"\n[2/5] Loading all models...")

models = {
    'XGBoost': joblib.load('../models/xgboost_tuned.pkl'),
    'LightGBM': joblib.load('../models/lightgbm_tuned.pkl'),
    'CatBoost': joblib.load('../models/catboost_tuned.pkl'),
    'RandomForest': joblib.load('../models/randomforest_tuned.pkl')
}

print(f"  ✅ Loaded 4 tree-based models")

# Calculate permutation importance
print(f"\n[3/5] Calculating permutation importance...")
print(f"  ℹ️  This may take 5-10 minutes per model...")

perm_importance_results = {}

for name, model in models.items():
    print(f"\n  Processing {name}...")

    if name == 'CatBoost':
        # Manual permutation importance for CatBoost due to sklearn compatibility
        from sklearn.metrics import accuracy_score

        # Baseline score
        baseline_score = accuracy_score(y_test, model.predict(X_test))

        importances = []
        for col in X_test.columns:
            scores = []
            for _ in range(10):  # n_repeats
                X_permuted = X_test.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                score = accuracy_score(y_test, model.predict(X_permuted))
                scores.append(baseline_score - score)  # Decrease in accuracy
            importances.append({
                'feature': col,
                'importance_mean': np.mean(scores),
                'importance_std': np.std(scores)
            })

        perm_df = pd.DataFrame(importances).sort_values('importance_mean', ascending=False)
    else:
        # Use sklearn's permutation_importance for other models
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,  # Number of times to permute each feature
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            scoring='accuracy'
        )

        # Create dataframe with results
        perm_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)

    perm_importance_results[name] = perm_df

    print(f"  ✅ {name} permutation importance calculated")
    print(f"\n  {name} - Top 10 Features:")
    for idx, row in perm_df.head(10).iterrows():
        print(f"    {row['feature']:<45} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

print(f"\n  ✅ All permutation importance calculated")

# Compare with SHAP importance
print(f"\n[4/5] Comparing with SHAP importance...")

# Load SHAP importance
with open('../results/phase8_results/shap_analysis/global_feature_importance.json', 'r') as f:
    shap_importance = json.load(f)

# Create comparison dataframe for each model
comparison_results = {}

for name in models.keys():
    if name in shap_importance:
        # Get SHAP importance
        shap_df = pd.DataFrame(shap_importance[name])
        shap_df = shap_df.rename(columns={'importance': 'shap_importance'})

        # Get permutation importance
        perm_df = perm_importance_results[name][['feature', 'importance_mean']].copy()
        perm_df = perm_df.rename(columns={'importance_mean': 'perm_importance'})

        # Merge
        comparison = pd.merge(shap_df, perm_df, on='feature', how='outer')
        comparison = comparison.fillna(0)

        # Calculate rank correlation
        shap_ranks = comparison['shap_importance'].rank(ascending=False)
        perm_ranks = comparison['perm_importance'].rank(ascending=False)
        rank_corr = shap_ranks.corr(perm_ranks, method='spearman')

        comparison_results[name] = {
            'comparison_df': comparison,
            'rank_correlation': float(rank_corr)
        }

        print(f"\n  {name}:")
        print(f"    Rank correlation (SHAP vs Permutation): {rank_corr:.3f}")

        # Show top 10 features by both methods
        top_shap = set(comparison.nlargest(10, 'shap_importance')['feature'])
        top_perm = set(comparison.nlargest(10, 'perm_importance')['feature'])
        overlap = top_shap & top_perm
        print(f"    Top 10 overlap: {len(overlap)}/10 features")

print(f"\n  ✅ Comparison complete")

# Visualizations
print(f"\n[5/5] Creating visualizations...")

import os
os.makedirs('../results/phase8_results/permutation_importance', exist_ok=True)

# 1. Permutation importance bar plots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Permutation Importance - All Models', fontsize=16, fontweight='bold')

for idx, (name, perm_df) in enumerate(perm_importance_results.items()):
    ax = axes[idx // 2, idx % 2]

    top_features = perm_df.head(15)

    ax.barh(range(len(top_features)), top_features['importance_mean'].values,
            xerr=top_features['importance_std'].values, color='coral', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Importance (Decrease in Accuracy)', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/phase8_results/permutation_importance/permutation_importance_all_models.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: permutation_importance_all_models.png")
plt.close()

# 2. SHAP vs Permutation comparison scatter plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHAP vs Permutation Importance Comparison', fontsize=16, fontweight='bold')

for idx, (name, comp_data) in enumerate(comparison_results.items()):
    ax = axes[idx // 2, idx % 2]

    comp_df = comp_data['comparison_df']
    rank_corr = comp_data['rank_correlation']

    # Scatter plot
    ax.scatter(comp_df['shap_importance'], comp_df['perm_importance'],
               alpha=0.6, s=50, color='steelblue')

    # Add labels for top features
    top_combined = comp_df.nlargest(5, 'shap_importance')
    for _, row in top_combined.iterrows():
        ax.annotate(row['feature'].replace('home_', '').replace('away_', '')[:20],
                   (row['shap_importance'], row['perm_importance']),
                   fontsize=7, alpha=0.7)

    ax.set_xlabel('SHAP Importance', fontsize=11)
    ax.set_ylabel('Permutation Importance', fontsize=11)
    ax.set_title(f'{name} (Rank Corr: {rank_corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/phase8_results/permutation_importance/shap_vs_permutation_comparison.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: shap_vs_permutation_comparison.png")
plt.close()

# Save results
print(f"\n  Saving results...")

# Save permutation importance
perm_importance_json = {}
for name, perm_df in perm_importance_results.items():
    perm_importance_json[name] = perm_df.to_dict('records')

with open('../results/phase8_results/permutation_importance/permutation_importance.json', 'w') as f:
    json.dump(perm_importance_json, f, indent=2)
print(f"  ✅ Saved: permutation_importance.json")

# Save comparison results
comparison_json = {}
for name, comp_data in comparison_results.items():
    comparison_json[name] = {
        'rank_correlation': comp_data['rank_correlation'],
        'top_10_features': comp_data['comparison_df'].nlargest(10, 'perm_importance')['feature'].tolist()
    }

with open('../results/phase8_results/permutation_importance/shap_vs_permutation_comparison.json', 'w') as f:
    json.dump(comparison_json, f, indent=2)
print(f"  ✅ Saved: shap_vs_permutation_comparison.json")

print(f"\n{'='*120}")
print("✅ PERMUTATION IMPORTANCE ANALYSIS COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Analyzed 4 tree-based models")
print(f"✅ Calculated permutation importance for {len(X_test.columns)} features")
print(f"✅ Compared with SHAP importance")
print(f"✅ Generated 2 visualization plots")
print(f"\n{'='*120}")

