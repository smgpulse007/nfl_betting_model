"""
Generate Final Week 16 & 17 Predictions using XGBoost with Injury Features

Uses the clean 2025_complete_dataset_weeks1_17.parquet dataset
"""

import pandas as pd
import numpy as np
import joblib
import json
import nfl_data_py as nfl

print("="*120)
print("FINAL WEEK 16 & 17 PREDICTIONS WITH INJURY FEATURES")
print("="*120)

# Load model
print("\n[1/5] Loading XGBoost model...")
model = joblib.load('../models/xgboost_with_injuries.pkl')

with open('../models/xgboost_with_injuries_features.json', 'r') as f:
    feature_info = json.load(f)
    model_features = feature_info['features']

print(f"  ✅ Loaded model with {len(model_features)} features")

# Load 2025 complete dataset
print("\n[2/5] Loading 2025 complete dataset...")
df_2025 = pd.read_parquet('../results/phase8_results/2025_complete_dataset_weeks1_17.parquet')

print(f"  ✅ Loaded: {df_2025.shape}")
print(f"  ✅ Weeks: {sorted(df_2025['week'].unique())}")

# Filter to Week 16 & 17
df_week16_17 = df_2025[df_2025['week'].isin([16, 17])].copy()

print(f"  ✅ Week 16 & 17 games: {len(df_week16_17)}")

# Prepare features
print("\n[3/5] Preparing features...")

# Check which model features are available
available_features = [f for f in model_features if f in df_week16_17.columns]
missing_features = [f for f in model_features if f not in df_week16_17.columns]

print(f"  ✅ Available features: {len(available_features)}/{len(model_features)}")

if missing_features:
    print(f"  ⚠️  Missing features: {len(missing_features)}")
    # Fill missing features with 0
    for feat in missing_features:
        df_week16_17[feat] = 0

# Prepare X
X = df_week16_17[model_features].fillna(0)

print(f"  ✅ X shape: {X.shape}")

# Generate predictions
print("\n[4/5] Generating predictions...")

y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of home win

df_week16_17['home_win_prob'] = y_pred_proba
df_week16_17['predicted_winner'] = df_week16_17.apply(
    lambda row: row['home_team'] if row['home_win_prob'] >= 0.5 else row['away_team'],
    axis=1
)
df_week16_17['confidence'] = df_week16_17['home_win_prob'].apply(
    lambda p: max(p, 1-p)
)

print(f"  ✅ Predictions generated")

# Create output
print("\n[5/5] Creating output...")

output_cols = ['game_id', 'season', 'week', 'away_team', 'home_team', 
               'predicted_winner', 'home_win_prob', 'confidence']

# Add actual scores if available
if 'away_score' in df_week16_17.columns and 'home_score' in df_week16_17.columns:
    output_cols.extend(['away_score', 'home_score'])
    df_week16_17['actual_winner'] = df_week16_17.apply(
        lambda row: row['home_team'] if pd.notna(row['home_score']) and row['home_score'] > row['away_score'] 
                   else row['away_team'] if pd.notna(row['home_score']) and row['home_score'] < row['away_score']
                   else 'TIE' if pd.notna(row['home_score'])
                   else 'UPCOMING',
        axis=1
    )
    output_cols.append('actual_winner')

df_output = df_week16_17[output_cols].copy()

# Sort by week and confidence
df_output = df_output.sort_values(['week', 'confidence'], ascending=[True, False])

# Save
output_file = '../results/phase8_results/2025_week16_17_final_predictions.csv'
df_output.to_csv(output_file, index=False)

print(f"  ✅ Saved: {output_file}")

# Display summary
print(f"\n{'='*120}")
print("SUMMARY - WEEK 16 & 17 PREDICTIONS")
print("="*120)

for week in [16, 17]:
    df_week = df_output[df_output['week'] == week]
    
    if len(df_week) == 0:
        continue
    
    print(f"\nWEEK {week} ({len(df_week)} games):")
    print(f"  Average confidence: {df_week['confidence'].mean():.1%}")
    print(f"  High confidence (≥65%): {len(df_week[df_week['confidence'] >= 0.65])} games")
    
    # Check if we have actual results
    if 'actual_winner' in df_week.columns:
        completed = df_week[df_week['actual_winner'] != 'UPCOMING']
        if len(completed) > 0:
            correct = completed[completed['predicted_winner'] == completed['actual_winner']]
            print(f"  Completed games: {len(completed)}")
            print(f"  Correct predictions: {len(correct)}/{len(completed)} ({len(correct)/len(completed):.1%})")
    
    print(f"\n  Top 5 predictions:")
    for idx, row in df_week.head(5).iterrows():
        status = ""
        if 'actual_winner' in row and row['actual_winner'] != 'UPCOMING':
            status = " ✅" if row['predicted_winner'] == row['actual_winner'] else " ❌"
        print(f"    • {row['away_team']} @ {row['home_team']}: {row['predicted_winner']} ({row['confidence']:.1%}){status}")

print(f"\n{'='*120}")

