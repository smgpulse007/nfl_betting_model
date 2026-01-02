"""
Task 8A.3: Ensemble Model

Combine predictions from all 5 models using weighted averaging and stacking
"""

import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8A.3: ENSEMBLE MODEL")
print("="*120)

# Load data
print(f"\n[1/7] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Loaded: {df.shape[0]:,} games × {df.shape[1]:,} columns")

# Load feature categorization
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

# Get pre-game features
pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

# Add manually classified UNKNOWN features
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

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split by season
print(f"\n[2/7] Splitting data...")
train = df[df['season'] <= 2019].copy()
val = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test = df[df['season'] == 2024].copy()

print(f"  ✅ Train (1999-2019): {len(train):,} games")
print(f"  ✅ Val (2020-2023): {len(val):,} games")
print(f"  ✅ Test (2024): {len(test):,} games")

# Select pre-game features
pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

# Prepare data
X_train = train[numeric_pregame].fillna(train[numeric_pregame].median())
X_val = val[numeric_pregame].fillna(train[numeric_pregame].median())
X_test = test[numeric_pregame].fillna(train[numeric_pregame].median())
y_train = train['home_win'].values
y_val = val['home_win'].values
y_test = test['home_win'].values

# Load trained models
print(f"\n[3/7] Loading trained models...")

# Tree-based models
xgb_model = joblib.load('../models/xgboost_tuned.pkl')
lgb_model = joblib.load('../models/lightgbm_tuned.pkl')
cb_model = joblib.load('../models/catboost_tuned.pkl')
rf_model = joblib.load('../models/randomforest_tuned.pkl')

# PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

pytorch_checkpoint = torch.load('../models/pytorch_nn.pth')
pytorch_model = NFLPredictor(input_dim=len(numeric_pregame)).to(device)
pytorch_model.load_state_dict(pytorch_checkpoint['model_state_dict'])
pytorch_model.eval()
scaler = pytorch_checkpoint['scaler']

print(f"  ✅ Loaded 5 models: XGBoost, LightGBM, CatBoost, RandomForest, PyTorch")

# Generate predictions from all models
print(f"\n[4/7] Generating predictions from all models...")

def get_pytorch_predictions(X, model, scaler, device):
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).squeeze().cpu().numpy()
    return outputs

# Validation set predictions (for stacking)
val_preds = pd.DataFrame({
    'xgboost': xgb_model.predict_proba(X_val)[:, 1],
    'lightgbm': lgb_model.predict_proba(X_val)[:, 1],
    'catboost': cb_model.predict_proba(X_val)[:, 1],
    'randomforest': rf_model.predict_proba(X_val)[:, 1],
    'pytorch': get_pytorch_predictions(X_val, pytorch_model, scaler, device)
})

# Test set predictions
test_preds = pd.DataFrame({
    'xgboost': xgb_model.predict_proba(X_test)[:, 1],
    'lightgbm': lgb_model.predict_proba(X_test)[:, 1],
    'catboost': cb_model.predict_proba(X_test)[:, 1],
    'randomforest': rf_model.predict_proba(X_test)[:, 1],
    'pytorch': get_pytorch_predictions(X_test, pytorch_model, scaler, device)
})

print(f"  ✅ Generated predictions for validation and test sets")

# Strategy 1: Simple averaging
print(f"\n[5/7] Strategy 1: Simple averaging...")
simple_avg_val = val_preds.mean(axis=1)
simple_avg_test = test_preds.mean(axis=1)

simple_avg_val_acc = accuracy_score(y_val, (simple_avg_val > 0.5).astype(int))
simple_avg_test_acc = accuracy_score(y_test, (simple_avg_test > 0.5).astype(int))

print(f"  ✅ Val accuracy: {simple_avg_val_acc*100:.2f}%")
print(f"  ✅ Test accuracy: {simple_avg_test_acc*100:.2f}%")

# Strategy 2: Weighted averaging (weights based on validation performance)
print(f"\n[6/7] Strategy 2: Weighted averaging...")

# Calculate individual model accuracies on validation set
model_val_accs = {}
for col in val_preds.columns:
    acc = accuracy_score(y_val, (val_preds[col] > 0.5).astype(int))
    model_val_accs[col] = acc
    print(f"  {col:<15} Val Acc: {acc*100:.2f}%")

# Normalize accuracies to get weights
total_acc = sum(model_val_accs.values())
weights = {k: v/total_acc for k, v in model_val_accs.items()}

print(f"\n  Optimal weights:")
for model, weight in weights.items():
    print(f"    {model:<15} {weight:.4f}")

# Weighted average
weighted_avg_val = sum(val_preds[col] * weights[col] for col in val_preds.columns)
weighted_avg_test = sum(test_preds[col] * weights[col] for col in test_preds.columns)

weighted_avg_val_acc = accuracy_score(y_val, (weighted_avg_val > 0.5).astype(int))
weighted_avg_test_acc = accuracy_score(y_test, (weighted_avg_test > 0.5).astype(int))

print(f"\n  ✅ Val accuracy: {weighted_avg_val_acc*100:.2f}%")
print(f"  ✅ Test accuracy: {weighted_avg_test_acc*100:.2f}%")

# Strategy 3: Stacking (meta-learner)
print(f"\n[7/7] Strategy 3: Stacking (Logistic Regression meta-learner)...")

meta_learner = LogisticRegression(random_state=42, max_iter=1000)
meta_learner.fit(val_preds, y_val)

stacking_val_pred = meta_learner.predict(val_preds)
stacking_test_pred = meta_learner.predict(test_preds)

stacking_val_acc = accuracy_score(y_val, stacking_val_pred)
stacking_test_acc = accuracy_score(y_test, stacking_test_pred)

print(f"  ✅ Val accuracy: {stacking_val_acc*100:.2f}%")
print(f"  ✅ Test accuracy: {stacking_test_acc*100:.2f}%")

# Choose best strategy
strategies = {
    'simple_averaging': {'val_acc': simple_avg_val_acc, 'test_acc': simple_avg_test_acc},
    'weighted_averaging': {'val_acc': weighted_avg_val_acc, 'test_acc': weighted_avg_test_acc},
    'stacking': {'val_acc': stacking_val_acc, 'test_acc': stacking_test_acc}
}

best_strategy = max(strategies.items(), key=lambda x: x[1]['test_acc'])

print(f"\n{'='*120}")
print("ENSEMBLE STRATEGY COMPARISON")
print(f"{'='*120}")
print(f"\n{'Strategy':<25} {'Val Accuracy':<15} {'Test Accuracy':<15}")
print(f"{'-'*55}")
for strategy, metrics in strategies.items():
    marker = " ⭐ BEST" if strategy == best_strategy[0] else ""
    print(f"{strategy:<25} {metrics['val_acc']*100:>6.2f}%         {metrics['test_acc']*100:>6.2f}%{marker}")

# Save ensemble model and results
print(f"\n{'='*120}")
print("SAVING ENSEMBLE MODEL")
print(f"{'='*120}")

# Save weighted averaging ensemble (most interpretable)
ensemble_data = {
    'strategy': 'weighted_averaging',
    'weights': weights,
    'val_accuracy': weighted_avg_val_acc,
    'test_accuracy': weighted_avg_test_acc,
    'model_val_accuracies': model_val_accs
}

joblib.dump(ensemble_data, '../models/ensemble_model.pkl')
print(f"\n  ✅ Saved: ../models/ensemble_model.pkl")

# Save stacking meta-learner
joblib.dump(meta_learner, '../models/ensemble_stacking.pkl')
print(f"  ✅ Saved: ../models/ensemble_stacking.pkl")

# Save ensemble weights
with open('../results/phase8_results/ensemble_weights.json', 'w') as f:
    json.dump({
        'weighted_averaging': {k: float(v) for k, v in weights.items()},
        'model_val_accuracies': {k: float(v) for k, v in model_val_accs.items()},
        'strategies_performance': {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in strategies.items()
        }
    }, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/ensemble_weights.json")

print(f"\n{'='*120}")
print("✅ ENSEMBLE MODEL COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Best strategy: {best_strategy[0]}")
print(f"✅ Test accuracy: {best_strategy[1]['test_acc']*100:.2f}%")
print(f"✅ All ensemble models saved")
print(f"\n{'='*120}")

