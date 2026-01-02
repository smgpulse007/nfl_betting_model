"""
Task 8E.2: Generate 2025 Season Predictions

Generate predictions for all 2025 NFL games using all 6 trained models:
- XGBoost
- LightGBM
- CatBoost
- RandomForest
- PyTorch NN
- TabNet

For each game, generate:
- Individual model predictions
- Ensemble prediction (weighted average)
- Confidence levels
- Predicted winner
"""

import pandas as pd
import numpy as np
import joblib
import json
import torch
import nfl_data_py as nfl
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8E.2: GENERATE 2025 SEASON PREDICTIONS")
print("="*120)

# =============================================================================
# STEP 1: LOAD DATA (1999-2025 INCLUDING PRE-GAME FEATURES)
# =============================================================================
print(f"\n[1/5] Loading data...")

# Load the extended dataset with 2025 pre-game features
df_full = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')

print(f"  ✅ Loaded dataset: {df_full.shape}")
print(f"  ✅ Seasons: {df_full['season'].min()}-{df_full['season'].max()}")

# Extract 2025 data
df_2025 = df_full[df_full['season'] == 2025].copy()
print(f"  ✅ 2025 games: {len(df_2025)}")

# Load 2025 schedule to get scores and game metadata
schedule_2025 = nfl.import_schedules([2025])
schedule_2025 = schedule_2025[schedule_2025['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])].copy()

# Create game_id for merging
schedule_2025['game_id'] = (
    schedule_2025['season'].astype(str) + '_' +
    schedule_2025['week'].astype(str).str.zfill(2) + '_' +
    schedule_2025['away_team'] + '_' +
    schedule_2025['home_team']
)

# Merge to get scores
df_2025 = df_2025.merge(
    schedule_2025[['game_id', 'home_score', 'away_score', 'gameday', 'weekday']],
    on='game_id',
    how='left',
    suffixes=('', '_sched')
)

print(f"  ✅ Merged with schedule: {df_2025.shape}")

# Check if home_score exists
if 'home_score' not in df_2025.columns:
    print(f"  ⚠️  home_score not in columns, checking for alternatives...")
    print(f"  ℹ️  Available score columns: {[c for c in df_2025.columns if 'score' in c.lower()]}")

    # Use the schedule columns if they exist
    if 'home_score_sched' in df_2025.columns:
        df_2025['home_score'] = df_2025['home_score_sched']
        df_2025['away_score'] = df_2025['away_score_sched']
        print(f"  ✅ Using schedule score columns")

# Filter to games without scores (upcoming games)
upcoming_games = df_2025[df_2025['home_score'].isna()].copy()
completed_games = df_2025[df_2025['home_score'].notna()].copy()

print(f"\n  📊 Game Status:")
print(f"     - Completed: {len(completed_games)}")
print(f"     - Upcoming: {len(upcoming_games)}")

# =============================================================================
# STEP 2: LOAD TRAINING FEATURES FROM MODEL CHECKPOINT
# =============================================================================
print(f"\n[2/5] Loading training features...")

# Load PyTorch checkpoint to get the exact features used in training
# Use pytorch_nn.pth which has input_features saved
pytorch_path_features = '../models/pytorch_nn.pth'
checkpoint = torch.load(pytorch_path_features, map_location='cpu', weights_only=False)

if 'input_features' in checkpoint:
    numeric_pregame = checkpoint['input_features']
    print(f"  ✅ Loaded {len(numeric_pregame)} features from PyTorch checkpoint")
else:
    print(f"  ⚠️  No input_features in checkpoint, using all numeric columns")
    # Fallback: use all numeric columns
    metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team',
                     'home_score', 'away_score', 'gameday', 'weekday']
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
    numeric_pregame = [col for col in numeric_cols if col not in metadata_cols]

# Get training data for median imputation (1999-2019)
train_df = df_full[df_full['season'] <= 2019]

# Check which features are available in 2025 data
available_features = [f for f in numeric_pregame if f in df_2025.columns]
missing_features = [f for f in numeric_pregame if f not in df_2025.columns]

print(f"  ✅ Features available in 2025 data: {len(available_features)}/{len(numeric_pregame)}")

if missing_features:
    print(f"  ⚠️  Missing {len(missing_features)} features in 2025 data")
    print(f"  ℹ️  First 10 missing: {missing_features[:10]}")

    # Use only available features
    numeric_pregame = available_features

# Prepare features for 2025 games
X_all = df_2025[numeric_pregame].fillna(train_df[numeric_pregame].median())

print(f"  ✅ Feature matrix: {X_all.shape}")
print(f"  ✅ Missing values filled with training median (1999-2019)")

# =============================================================================
# STEP 3: LOAD MODELS
# =============================================================================
print(f"\n[3/5] Loading trained models...")

models = {}

# Load tree-based models
model_files = {
    'XGBoost': '../models/xgboost_tuned.pkl',
    'LightGBM': '../models/lightgbm_tuned.pkl',
    'CatBoost': '../models/catboost_tuned.pkl',
    'RandomForest': '../models/randomforest_tuned.pkl'
}

for name, path in model_files.items():
    if Path(path).exists():
        models[name] = joblib.load(path)
        print(f"  ✅ Loaded {name}")
    else:
        print(f"  ⚠️  {name} not found at {path}")

# Load PyTorch NN
pytorch_path = '../models/pytorch_nn_best.pth'
if Path(pytorch_path).exists():
    # Define model architecture (must match training)
    import torch.nn as nn
    
    class NFLPredictor(nn.Module):
        def __init__(self, input_dim):
            super(NFLPredictor, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.dropout1 = nn.Dropout(0.3)

            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.dropout2 = nn.Dropout(0.3)

            self.fc3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.dropout3 = nn.Dropout(0.3)

            self.fc4 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
            
            x = torch.sigmoid(self.fc4(x))
            return x
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model = NFLPredictor(len(numeric_pregame)).to(device)

    # Load checkpoint (it's saved as a dict with 'model_state_dict' key)
    checkpoint = torch.load(pytorch_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)

    pytorch_model.eval()
    models['PyTorch NN'] = pytorch_model
    print(f"  ✅ Loaded PyTorch NN")
else:
    print(f"  ⚠️  PyTorch NN not found at {pytorch_path}")

# Load TabNet
tabnet_path = '../models/tabnet_model.zip'
if Path(tabnet_path).exists():
    from pytorch_tabnet.tab_model import TabNetClassifier

    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(tabnet_path)
    models['TabNet'] = tabnet_model
    print(f"  ✅ Loaded TabNet")
else:
    print(f"  ⚠️  TabNet not found at {tabnet_path}")

print(f"\n  ✅ Total models loaded: {len(models)}")

# =============================================================================
# STEP 4: GENERATE PREDICTIONS
# =============================================================================
print(f"\n[4/5] Generating predictions for all 2025 games...")

predictions_dict = {}

for name, model in models.items():
    print(f"\n  Predicting with {name}...")

    if name == 'PyTorch NN':
        # PyTorch prediction
        X_tensor = torch.FloatTensor(X_all.values).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()

    elif name == 'TabNet':
        # TabNet prediction
        preds = model.predict_proba(X_all.values)[:, 1]

    else:
        # Scikit-learn compatible models
        preds = model.predict_proba(X_all)[:, 1]

    predictions_dict[name] = preds
    print(f"    ✅ Generated {len(preds)} predictions")

# Calculate ensemble prediction (weighted average)
# Use best performing models with higher weights
weights = {
    'XGBoost': 0.15,
    'LightGBM': 0.15,
    'CatBoost': 0.20,
    'RandomForest': 0.15,
    'PyTorch NN': 0.175,
    'TabNet': 0.175
}

ensemble_preds = np.zeros(len(X_all))
total_weight = 0

for name, weight in weights.items():
    if name in predictions_dict:
        ensemble_preds += predictions_dict[name] * weight
        total_weight += weight

ensemble_preds /= total_weight

predictions_dict['Ensemble'] = ensemble_preds

print(f"\n  ✅ Ensemble predictions calculated")
print(f"     - Weights: {weights}")
print(f"     - Total weight: {total_weight:.3f}")

# =============================================================================
# STEP 5: CREATE PREDICTIONS DATAFRAME
# =============================================================================
print(f"\n[5/5] Creating predictions dataframe...")

# Start with metadata
results_df = df_2025[['game_id', 'season', 'week', 'gameday', 'weekday',
                      'home_team', 'away_team', 'home_score', 'away_score']].copy()

# Add individual model predictions
for name, preds in predictions_dict.items():
    results_df[f'{name}_prob'] = preds

# Add ensemble prediction as main prediction
results_df['home_win_probability'] = ensemble_preds
results_df['away_win_probability'] = 1 - ensemble_preds

# Determine predicted winner
results_df['predicted_winner'] = results_df.apply(
    lambda row: row['home_team'] if row['home_win_probability'] > 0.5 else row['away_team'],
    axis=1
)

# Add confidence level
results_df['confidence'] = results_df['home_win_probability'].apply(
    lambda p: max(p, 1-p)
)

# Add confidence category
def get_confidence_category(conf):
    if conf >= 0.80:
        return 'Very High (80%+)'
    elif conf >= 0.70:
        return 'High (70-80%)'
    elif conf >= 0.60:
        return 'Medium (60-70%)'
    else:
        return 'Low (<60%)'

results_df['confidence_category'] = results_df['confidence'].apply(get_confidence_category)

# Add actual result for completed games
results_df['actual_winner'] = results_df.apply(
    lambda row: row['home_team'] if pd.notna(row['home_score']) and row['home_score'] > row['away_score']
                else (row['away_team'] if pd.notna(row['away_score']) and row['away_score'] > row['home_score']
                      else None),
    axis=1
)

# Add prediction correctness for completed games
results_df['correct_prediction'] = (
    results_df['predicted_winner'] == results_df['actual_winner']
).where(results_df['actual_winner'].notna(), None)

print(f"  ✅ Created predictions dataframe: {results_df.shape}")

# Save predictions
output_path = '../results/phase8_results/2025_predictions.csv'
results_df.to_csv(output_path, index=False)

print(f"  ✅ Saved to: {output_path}")

# Summary statistics
print(f"\n{'='*120}")
print("SUMMARY")
print(f"{'='*120}")

print(f"\n📊 Prediction Statistics:")
print(f"   - Total games: {len(results_df)}")
print(f"   - Completed games: {results_df['actual_winner'].notna().sum()}")
print(f"   - Upcoming games: {results_df['actual_winner'].isna().sum()}")

if results_df['actual_winner'].notna().sum() > 0:
    accuracy = results_df['correct_prediction'].sum() / results_df['actual_winner'].notna().sum()
    print(f"\n✅ Model Performance on Completed 2025 Games:")
    print(f"   - Accuracy: {accuracy:.2%}")
    print(f"   - Correct: {results_df['correct_prediction'].sum()}")
    print(f"   - Incorrect: {(~results_df['correct_prediction']).sum()}")

print(f"\n📈 Confidence Distribution (All Games):")
conf_dist = results_df['confidence_category'].value_counts().sort_index()
for cat, count in conf_dist.items():
    pct = count / len(results_df) * 100
    print(f"   - {cat}: {count} ({pct:.1f}%)")

print(f"\n🎯 Upcoming Games by Confidence:")
upcoming_df = results_df[results_df['actual_winner'].isna()]
if len(upcoming_df) > 0:
    upcoming_conf = upcoming_df['confidence_category'].value_counts().sort_index()
    for cat, count in upcoming_conf.items():
        pct = count / len(upcoming_df) * 100
        print(f"   - {cat}: {count} ({pct:.1f}%)")

print(f"\n✅ Predictions saved to: {output_path}")
print(f"\n{'='*120}")

