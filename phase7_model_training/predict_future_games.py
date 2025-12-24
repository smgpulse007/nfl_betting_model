"""
Prediction Pipeline for Future Games
=====================================

Predict outcomes for future NFL games (e.g., Week 16 2025).

Usage:
    python predict_future_games.py --week 16 --year 2025
    python predict_future_games.py --model xgboost --week 16 --year 2025
"""

import argparse
import pandas as pd
import joblib
import torch
from pathlib import Path
import json

from config import *
from train_pytorch_nn import NFLPredictor

print("="*120)
print("NFL GAME PREDICTION PIPELINE")
print("="*120)

# Parse arguments
parser = argparse.ArgumentParser(description='Predict NFL game outcomes')
parser.add_argument('--week', type=int, default=16, help='Week number (1-18)')
parser.add_argument('--year', type=int, default=2025, help='Season year')
parser.add_argument('--model', type=str, default='xgboost', 
                    choices=['xgboost', 'lightgbm', 'catboost', 'randomforest', 'pytorch_nn', 'ensemble'],
                    help='Model to use for predictions')
args = parser.parse_args()

print(f"\n📅 Predicting Week {args.week} of {args.year} season")
print(f"🤖 Using model: {args.model}")

# Load feature scaler
scaler_path = MODELS_DIR / 'feature_scaler.pkl'
scaler = joblib.load(scaler_path)
print(f"\n✅ Loaded feature scaler")

# Load selected features
with open(FEATURE_SELECTION_PATH) as f:
    selected_features = json.load(f)
print(f"✅ Loaded {len(selected_features)} selected features")

# Load model
if args.model == 'pytorch_nn':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NFLPredictor(
        input_dim=len(selected_features),
        hidden_dims=NN_PARAMS['hidden_dims'],
        dropout=NN_PARAMS['dropout']
    ).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / 'pytorch_nn_best.pth', weights_only=False))
    model.eval()
    print(f"✅ Loaded PyTorch model on {device}")
elif args.model == 'ensemble':
    models = {}
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
        models[model_name] = joblib.load(MODELS_DIR / f'{model_name}_model.pkl')
    print(f"✅ Loaded 4 models for ensemble")
else:
    model = joblib.load(MODELS_DIR / f'{args.model}_model.pkl')
    print(f"✅ Loaded {args.model} model")

# Load game-level dataset
df = pd.read_parquet(DATA_PATH)
print(f"\n✅ Loaded dataset: {len(df)} games")

# Filter to specified week and year
week_games = df[(df['week'] == args.week) & (df['year'] == args.year)]

if len(week_games) == 0:
    print(f"\n⚠️  No games found for Week {args.week} of {args.year} season")
    print(f"\nℹ️  This is expected if the season hasn't started yet.")
    print(f"\nℹ️  To predict future games, you need to:")
    print(f"   1. Derive features for the current season up to the prediction week")
    print(f"   2. Use the game-level derivation function from Phase 5")
    print(f"   3. Ensure all rolling averages and features are computed from past games only")
    print(f"\n📝 Example workflow:")
    print(f"   1. Run: python derive_game_features_complete.py --year {args.year} --week {args.week}")
    print(f"   2. Run: python predict_future_games.py --week {args.week} --year {args.year}")
    exit(0)

print(f"\n📊 Found {len(week_games)} games for Week {args.week} of {args.year}")

# Extract features
X = week_games[selected_features]
X_scaled = scaler.transform(X)

# Make predictions
if args.model == 'pytorch_nn':
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        y_pred_proba = model(X_tensor).cpu().numpy()
elif args.model == 'ensemble':
    # Average predictions from all models
    predictions = []
    for model_name, model in models.items():
        predictions.append(model.predict_proba(X)[:, 1])
    y_pred_proba = np.mean(predictions, axis=0)
else:
    y_pred_proba = model.predict_proba(X)[:, 1]

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'game_id': week_games['game_id'].values,
    'away_team': week_games['away_team'].values,
    'home_team': week_games['home_team'].values,
    'home_win_probability': y_pred_proba,
    'away_win_probability': 1 - y_pred_proba,
    'predicted_winner': week_games['home_team'].where(y_pred_proba > 0.5, week_games['away_team']).values,
    'confidence': abs(y_pred_proba - 0.5) * 2  # 0 = 50/50, 1 = 100% confident
})

# Sort by confidence (highest first)
predictions_df = predictions_df.sort_values('confidence', ascending=False)

# Display predictions
print(f"\n{'='*120}")
print(f"PREDICTIONS FOR WEEK {args.week} OF {args.year} SEASON")
print(f"{'='*120}")

for idx, row in predictions_df.iterrows():
    print(f"\n{row['away_team']:3s} @ {row['home_team']:3s}")
    print(f"  Predicted Winner: {row['predicted_winner']:3s}")
    print(f"  Home Win Prob:    {row['home_win_probability']:.1%}")
    print(f"  Away Win Prob:    {row['away_win_probability']:.1%}")
    print(f"  Confidence:       {row['confidence']:.1%}")
    
    # Betting recommendation
    if row['confidence'] > 0.40:  # >70% or <30% probability
        if row['home_win_probability'] > 0.70:
            print(f"  💰 BET: {row['home_team']} (Home)")
        elif row['away_win_probability'] > 0.70:
            print(f"  💰 BET: {row['away_team']} (Away)")
    else:
        print(f"  ⚠️  SKIP: Low confidence")

# Save predictions
output_path = RESULTS_DIR / f'predictions_week{args.week}_{args.year}.csv'
predictions_df.to_csv(output_path, index=False)

print(f"\n{'='*120}")
print(f"✅ PREDICTIONS COMPLETE!")
print(f"{'='*120}")
print(f"\n📁 Saved to: {output_path}")
print(f"\n📊 Summary:")
print(f"   Total Games:      {len(predictions_df)}")
print(f"   High Confidence:  {len(predictions_df[predictions_df['confidence'] > 0.40])} (>70% probability)")
print(f"   Medium Confidence: {len(predictions_df[(predictions_df['confidence'] > 0.20) & (predictions_df['confidence'] <= 0.40)])} (60-70% probability)")
print(f"   Low Confidence:   {len(predictions_df[predictions_df['confidence'] <= 0.20])} (<60% probability)")

