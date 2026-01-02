"""
Generate predictions for 2025 Week 17 games
"""

import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path

print("=" * 120)
print("GENERATING WEEK 17 PREDICTIONS")
print("=" * 120)

# Load actual schedule
print("\n[1/5] Loading Week 17 schedule...")
df_schedule = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
week17 = df_schedule[df_schedule['week'] == 17].copy()
week17_upcoming = week17[week17['home_score'].isna()].copy()

print(f"  ✅ Total Week 17 games: {len(week17)}")
print(f"  ✅ Completed: {len(week17) - len(week17_upcoming)}")
print(f"  ✅ Upcoming: {len(week17_upcoming)}")

# Load feature data
print("\n[2/5] Loading feature data...")
df_features = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
print(f"  ✅ Loaded {len(df_features)} games with features")

# Filter to 2025 Week 17
df_week17_features = df_features[
    (df_features['season'] == 2025) & 
    (df_features['week'] == 17)
].copy()
print(f"  ✅ Week 17 features: {len(df_week17_features)} games")

# Load models
print("\n[3/5] Loading models...")
models = {}

# Load PyTorch checkpoint to get feature list
pytorch_path = '../models/pytorch_nn.pth'
checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
numeric_pregame = checkpoint['input_features']
print(f"  ✅ Loaded {len(numeric_pregame)} features from checkpoint")

# XGBoost
xgb_path = '../models/xgboost_tuned.pkl'
if Path(xgb_path).exists():
    models['XGBoost'] = joblib.load(xgb_path)
    print(f"  ✅ Loaded XGBoost")

# LightGBM
lgb_path = '../models/lightgbm_tuned.pkl'
if Path(lgb_path).exists():
    models['LightGBM'] = joblib.load(lgb_path)
    print(f"  ✅ Loaded LightGBM")

# CatBoost
cat_path = '../models/catboost_tuned.pkl'
if Path(cat_path).exists():
    models['CatBoost'] = joblib.load(cat_path)
    print(f"  ✅ Loaded CatBoost")

# RandomForest
rf_path = '../models/random_forest_tuned.pkl'
if Path(rf_path).exists():
    models['RandomForest'] = joblib.load(rf_path)
    print(f"  ✅ Loaded RandomForest")

print(f"  ✅ Total models loaded: {len(models)}")

# Prepare features
print("\n[4/5] Generating predictions...")

# Get features for Week 17
X_week17 = df_week17_features[numeric_pregame].copy()

# Fill missing values with median from training data (1999-2019)
df_train = df_features[(df_features['season'] >= 1999) & (df_features['season'] <= 2019)]
X_train = df_train[numeric_pregame]
medians = X_train.median()
X_week17 = X_week17.fillna(medians)

# Generate predictions from each model
predictions = {}
for model_name, model in models.items():
    pred_proba = model.predict_proba(X_week17)[:, 1]
    predictions[f'{model_name}_prob'] = pred_proba
    print(f"  ✅ {model_name}: {len(pred_proba)} predictions")

# Create predictions dataframe
df_pred = df_week17_features[['season', 'week', 'gameday', 'weekday', 'home_team', 'away_team']].copy()

# Add model predictions
for model_name in predictions:
    df_pred[model_name] = predictions[model_name]

# Calculate ensemble prediction (weighted average)
weights = {
    'XGBoost': 0.15,
    'LightGBM': 0.15,
    'CatBoost': 0.20,
    'RandomForest': 0.15
}

ensemble_prob = np.zeros(len(df_pred))
for model_name, weight in weights.items():
    if f'{model_name}_prob' in df_pred.columns:
        ensemble_prob += weight * df_pred[f'{model_name}_prob'].values

# Normalize weights if not all models available
total_weight = sum(weights.values())
ensemble_prob = ensemble_prob / total_weight

df_pred['Ensemble_prob'] = ensemble_prob
df_pred['home_win_probability'] = ensemble_prob
df_pred['away_win_probability'] = 1 - ensemble_prob

# Determine predicted winner
df_pred['predicted_winner'] = df_pred.apply(
    lambda row: row['home_team'] if row['home_win_probability'] > 0.5 else row['away_team'],
    axis=1
)

# Calculate confidence
df_pred['confidence'] = df_pred.apply(
    lambda row: max(row['home_win_probability'], row['away_win_probability']),
    axis=1
)

# Create game_id
df_pred['game_id'] = (
    df_pred['season'].astype(str) + '_' +
    df_pred['week'].astype(str) + '_' +
    df_pred['away_team'] + '_' +
    df_pred['home_team']
)

# Save predictions
print("\n[5/5] Saving predictions...")
output_path = '../results/phase8_results/2025_week17_predictions.csv'
df_pred.to_csv(output_path, index=False)
print(f"  ✅ Saved to: {output_path}")

# Display predictions
print(f"\n{'='*120}")
print("WEEK 17 PREDICTIONS")
print("=" * 120)
print(f"{'Date':<12} {'Away Team':<12} {'@':<3} {'Home Team':<12} {'Predicted Winner':<18} {'Confidence':<12}")
print("-" * 120)

for idx, row in df_pred.iterrows():
    date = row['gameday'][:10] if pd.notna(row['gameday']) else 'TBD'
    print(f"{date:<12} {row['away_team']:<12} @ {row['home_team']:<12} "
          f"{row['predicted_winner']:<18} {row['confidence']:.1%}")

print(f"\n{'='*120}")
print("WEEK 17 PREDICTIONS COMPLETE")
print("=" * 120)

