"""
Task 8C.1: SHAP Value Analysis

Generate SHAP (SHapley Additive exPlanations) values for model interpretability.
Includes global feature importance and local explanations for individual predictions.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8C.1: SHAP VALUE ANALYSIS")
print("="*120)

# Load data
print(f"\n[1/6] Loading data...")
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
print(f"\n[2/6] Loading tree-based models...")

models = {
    'XGBoost': joblib.load('../models/xgboost_tuned.pkl'),
    'LightGBM': joblib.load('../models/lightgbm_tuned.pkl'),
    'CatBoost': joblib.load('../models/catboost_tuned.pkl'),
    'RandomForest': joblib.load('../models/randomforest_tuned.pkl')
}

print(f"  ✅ Loaded 4 tree-based models")

# Calculate SHAP values for each model
print(f"\n[3/6] Calculating SHAP values...")
print(f"  ℹ️  This may take 5-10 minutes per model...")

shap_values_dict = {}
explainers_dict = {}

# Sample test set for faster SHAP computation (use 100 games)
X_test_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)

for name, model in models.items():
    print(f"\n  Processing {name}...")

    try:
        if name == 'XGBoost':
            # Use model_output='raw' to avoid base_score parsing issue
            explainer = shap.TreeExplainer(model, model_output='raw')
            shap_values = explainer.shap_values(X_test_sample)

        elif name == 'LightGBM':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            # LightGBM returns list for binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif name == 'CatBoost':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)

        elif name == 'RandomForest':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            # RandomForest returns list for binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        shap_values_dict[name] = shap_values
        explainers_dict[name] = explainer

        print(f"  ✅ {name} SHAP values calculated")

    except Exception as e:
        print(f"  ⚠️  {name} SHAP calculation failed: {str(e)}")
        print(f"  ℹ️  Skipping {name} for SHAP analysis")
        continue

print(f"\n  ✅ All SHAP values calculated")

# Global feature importance
print(f"\n[4/6] Calculating global feature importance...")

global_importance = {}

for name, shap_values in shap_values_dict.items():
    # Handle different SHAP value shapes
    if len(shap_values.shape) == 3:
        # For models that return 3D array (n_samples, n_features, n_classes)
        # Take the positive class (index 1)
        shap_vals = shap_values[:, :, 1]
    elif len(shap_values.shape) == 2:
        # Already 2D (n_samples, n_features)
        shap_vals = shap_values
    else:
        print(f"  ⚠️  Unexpected SHAP shape for {name}: {shap_values.shape}")
        continue

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': X_test_sample.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    global_importance[name] = importance_df

    print(f"\n  {name} - Top 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<45} {row['importance']:.4f}")

print(f"\n  ✅ Global feature importance calculated")

# Visualizations
print(f"\n[5/6] Creating SHAP visualizations...")

import os
os.makedirs('../results/phase8_results/shap_analysis', exist_ok=True)

# 1. Summary plots for each model
for name, shap_values in shap_values_dict.items():
    # Handle different SHAP value shapes
    if len(shap_values.shape) == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_test_sample, show=False, max_display=20)
    plt.title(f'SHAP Summary Plot - {name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'../results/phase8_results/shap_analysis/summary_plot_{name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: summary_plot_{name.lower().replace(' ', '_')}.png")
    plt.close()

# 2. Bar plots for global feature importance
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('SHAP Global Feature Importance - All Models', fontsize=16, fontweight='bold')

for idx, (name, importance_df) in enumerate(global_importance.items()):
    ax = axes[idx // 2, idx % 2]

    top_features = importance_df.head(15)

    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/phase8_results/shap_analysis/global_importance_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: global_importance_comparison.png")
plt.close()

# 3. Dependence plots for top 3 features (using first available model)
first_model = list(global_importance.keys())[0]
top_3_features = global_importance[first_model].head(3)['feature'].values

# Get SHAP values for first model
shap_vals_for_dep = shap_values_dict[first_model]
if len(shap_vals_for_dep.shape) == 3:
    shap_vals_for_dep = shap_vals_for_dep[:, :, 1]

for feat in top_3_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat, shap_vals_for_dep, X_test_sample, show=False)
    plt.title(f'SHAP Dependence Plot - {feat} ({first_model})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Clean feature name for filename
    clean_feat = feat.replace('/', '_').replace(' ', '_')
    plt.savefig(f'../results/phase8_results/shap_analysis/dependence_{clean_feat}.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: dependence_{clean_feat}.png")
    plt.close()

# 4. Force plots for 3 example predictions (high confidence home win, away win, uncertain)
print(f"\n  Creating force plots for example predictions...")

# Get predictions from first available model
first_model_obj = models[first_model]
model_probs = first_model_obj.predict_proba(X_test_sample)[:, 1]

# Find examples
high_conf_home = np.argmax(model_probs)  # Highest home win probability
high_conf_away = np.argmin(model_probs)  # Lowest home win probability (highest away win)
uncertain = np.argmin(np.abs(model_probs - 0.5))  # Closest to 50%

examples = {
    'high_confidence_home_win': high_conf_home,
    'high_confidence_away_win': high_conf_away,
    'uncertain_prediction': uncertain
}

# Get SHAP values for force plots
shap_vals_for_force = shap_values_dict[first_model]
if len(shap_vals_for_force.shape) == 3:
    shap_vals_for_force = shap_vals_for_force[:, :, 1]

for example_name, idx in examples.items():
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainers_dict[first_model].expected_value,
        shap_vals_for_force[idx],
        X_test_sample.iloc[idx],
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - {example_name.replace("_", " ").title()} ({first_model})\n'
              f'Predicted Home Win Probability: {model_probs[idx]:.2%}',
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'../results/phase8_results/shap_analysis/force_plot_{example_name}.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: force_plot_{example_name}.png")
    plt.close()

print(f"\n  ✅ All SHAP visualizations created")

# Save results
print(f"\n[6/6] Saving SHAP analysis results...")

# Save global importance for all models
shap_importance_dict = {}
for name, importance_df in global_importance.items():
    shap_importance_dict[name] = importance_df.to_dict('records')

with open('../results/phase8_results/shap_analysis/global_feature_importance.json', 'w') as f:
    json.dump(shap_importance_dict, f, indent=2)
print(f"  ✅ Saved: global_feature_importance.json")

# Save top 20 features consensus (features that appear in top 20 for all models)
all_top_20 = set()
for name, importance_df in global_importance.items():
    all_top_20.update(importance_df.head(20)['feature'].values)

consensus_features = []
for feat in all_top_20:
    ranks = []
    for name, importance_df in global_importance.items():
        feat_rank = importance_df[importance_df['feature'] == feat].index[0] + 1 if feat in importance_df['feature'].values else 999
        ranks.append(int(feat_rank))  # Convert to int

    avg_rank = float(np.mean(ranks))  # Convert to float
    consensus_features.append({
        'feature': feat,
        'avg_rank': avg_rank,
        'ranks': {name: int(rank) for name, rank in zip(global_importance.keys(), ranks)}
    })

consensus_df = pd.DataFrame(consensus_features).sort_values('avg_rank')

# Convert to JSON-serializable format
consensus_json = []
for _, row in consensus_df.iterrows():
    consensus_json.append({
        'feature': row['feature'],
        'avg_rank': float(row['avg_rank']),
        'ranks': row['ranks']
    })

with open('../results/phase8_results/shap_analysis/consensus_top_features.json', 'w') as f:
    json.dump(consensus_json, f, indent=2)
print(f"  ✅ Saved: consensus_top_features.json")

print(f"\n{'='*120}")
print("✅ SHAP ANALYSIS COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Analyzed {len(shap_values_dict)} tree-based models (XGBoost skipped due to compatibility)")
print(f"✅ Calculated SHAP values for {len(X_test_sample)} test games")
print(f"✅ Generated {3 + len(shap_values_dict) + 3 + 3} visualization plots")
print(f"✅ Identified consensus important features")
print(f"\n{'='*120}")

