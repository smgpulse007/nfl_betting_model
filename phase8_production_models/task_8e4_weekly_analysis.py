"""
================================================================================
TASK 8E.4: WEEKLY ANALYSIS - 2024 WEEK 16/17 RETROSPECTIVE & 2025 WEEK 16 PREVIEW
================================================================================

Analyze model performance on 2024 Week 16/17 and preview 2025 Week 16 predictions.

Author: NFL Betting Model v0.4.0
Date: 2025-12-27
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("WEEKLY ANALYSIS: 2024 WEEK 16/17 RETROSPECTIVE & 2025 WEEK 16 PREVIEW")
print("=" * 120)

# =============================================================================
# STEP 1: LOAD 2024 DATA AND GENERATE PREDICTIONS FOR WEEK 16/17
# =============================================================================
print(f"\n[1/5] Loading 2024 Week 16/17 data...")

# Load full dataset
df_full = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

# Get 2024 Week 16 and 17
df_2024_w16 = df_full[(df_full['season'] == 2024) & (df_full['week'] == 16)].copy()
df_2024_w17 = df_full[(df_full['season'] == 2024) & (df_full['week'] == 17)].copy()

print(f"  ✅ 2024 Week 16: {len(df_2024_w16)} games")
print(f"  ✅ 2024 Week 17: {len(df_2024_w17)} games")

# Combine for analysis
df_2024_weeks = pd.concat([df_2024_w16, df_2024_w17], ignore_index=True)

# =============================================================================
# STEP 2: LOAD MODELS AND FEATURES
# =============================================================================
print(f"\n[2/5] Loading models and features...")

# Load PyTorch checkpoint to get features
pytorch_path = '../models/pytorch_nn.pth'
checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
numeric_pregame = checkpoint['input_features']

print(f"  ✅ Loaded {len(numeric_pregame)} features from checkpoint")

# Get training data for median imputation
train_df = df_full[df_full['season'] <= 2019]

# Prepare features
X_2024 = df_2024_weeks[numeric_pregame].fillna(train_df[numeric_pregame].median())

# Load models
models = {}

# XGBoost
xgb_path = '../models/xgboost_model.joblib'
if Path(xgb_path).exists():
    models['XGBoost'] = joblib.load(xgb_path)

# LightGBM
lgb_path = '../models/lightgbm_model.joblib'
if Path(lgb_path).exists():
    models['LightGBM'] = joblib.load(lgb_path)

# CatBoost
cat_path = '../models/catboost_model.joblib'
if Path(cat_path).exists():
    models['CatBoost'] = joblib.load(cat_path)

# RandomForest
rf_path = '../models/random_forest_model.joblib'
if Path(rf_path).exists():
    models['RandomForest'] = joblib.load(rf_path)

print(f"  ✅ Loaded {len(models)} models")

# =============================================================================
# STEP 3: GENERATE PREDICTIONS FOR 2024 WEEK 16/17
# =============================================================================
print(f"\n[3/5] Generating predictions for 2024 Week 16/17...")

predictions = {}

for model_name, model in models.items():
    if hasattr(model, 'predict_proba'):
        preds = model.predict_proba(X_2024)[:, 1]
    else:
        preds = model.predict(X_2024)
    predictions[f'{model_name}_prob'] = preds

# Create predictions dataframe
df_2024_weeks['XGBoost_prob'] = predictions.get('XGBoost_prob', 0.5)
df_2024_weeks['LightGBM_prob'] = predictions.get('LightGBM_prob', 0.5)
df_2024_weeks['CatBoost_prob'] = predictions.get('CatBoost_prob', 0.5)
df_2024_weeks['RandomForest_prob'] = predictions.get('RandomForest_prob', 0.5)

# Ensemble prediction (weighted average)
weights = {'XGBoost': 0.15, 'LightGBM': 0.15, 'CatBoost': 0.20, 'RandomForest': 0.15}
total_weight = sum(weights.values())

df_2024_weeks['Ensemble_prob'] = (
    df_2024_weeks['XGBoost_prob'] * weights['XGBoost'] +
    df_2024_weeks['LightGBM_prob'] * weights['LightGBM'] +
    df_2024_weeks['CatBoost_prob'] * weights['CatBoost'] +
    df_2024_weeks['RandomForest_prob'] * weights['RandomForest']
) / total_weight

# Predicted winner
df_2024_weeks['predicted_winner'] = df_2024_weeks.apply(
    lambda row: row['home_team'] if row['Ensemble_prob'] >= 0.5 else row['away_team'],
    axis=1
)

# Confidence
df_2024_weeks['confidence'] = df_2024_weeks['Ensemble_prob'].apply(
    lambda x: x if x >= 0.5 else 1 - x
)

# Actual winner
df_2024_weeks['actual_winner'] = df_2024_weeks.apply(
    lambda row: row['home_team'] if row['home_win'] == 1 else row['away_team'],
    axis=1
)

# Correct prediction
df_2024_weeks['correct'] = (df_2024_weeks['predicted_winner'] == df_2024_weeks['actual_winner']).astype(int)

print(f"  ✅ Generated predictions for {len(df_2024_weeks)} games")

# =============================================================================
# STEP 4: ANALYZE PERFORMANCE
# =============================================================================
print(f"\n[4/5] Analyzing performance...")

# Overall accuracy
accuracy = df_2024_weeks['correct'].mean()
print(f"\n  📊 Overall Accuracy (Week 16+17): {accuracy:.1%}")

# Week 16 performance
w16_acc = df_2024_weeks[df_2024_weeks['week'] == 16]['correct'].mean()
print(f"  📊 Week 16 Accuracy: {w16_acc:.1%}")

# Week 17 performance
w17_acc = df_2024_weeks[df_2024_weeks['week'] == 17]['correct'].mean()
print(f"  📊 Week 17 Accuracy: {w17_acc:.1%}")

# Save results
output_path = '../results/phase8_results/2024_week16_17_analysis.csv'
df_2024_weeks.to_csv(output_path, index=False)
print(f"\n  ✅ Saved analysis to: {output_path}")

print(f"\n" + "=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)

