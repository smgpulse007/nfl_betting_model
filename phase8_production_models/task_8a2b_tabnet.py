"""
Task 8A.2b: Train TabNet Model

TabNet is a neural network architecture specifically designed for tabular data
with interpretable feature selection using sequential attention.
"""

import pandas as pd
import numpy as np
import json
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8A.2b: TABNET MODEL (GPU-ACCELERATED)")
print("="*120)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[0/7] GPU Status: {device}")
if torch.cuda.is_available():
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ CUDA Version: {torch.version.cuda}")
else:
    print(f"  ⚠️  No GPU available, using CPU")

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
print(f"  ✅ Total pre-game features: {len(numeric_pregame)}")

# Prepare data
X_train = train[numeric_pregame].fillna(train[numeric_pregame].median()).values
X_val = val[numeric_pregame].fillna(train[numeric_pregame].median()).values
X_test = test[numeric_pregame].fillna(train[numeric_pregame].median()).values
y_train = train['home_win'].values
y_val = val['home_win'].values
y_test = test['home_win'].values

print(f"  ✅ Data prepared for TabNet")

# Define TabNet model
print(f"\n[3/7] Defining TabNet architecture...")

tabnet_params = {
    'n_d': 64,  # Width of the decision prediction layer
    'n_a': 64,  # Width of the attention embedding for each mask
    'n_steps': 5,  # Number of steps in the architecture (sequential attention)
    'gamma': 1.5,  # Coefficient for feature reusage in the masks
    'n_independent': 2,  # Number of independent Gated Linear Units layers at each step
    'n_shared': 2,  # Number of shared Gated Linear Units at each step
    'lambda_sparse': 1e-4,  # Sparsity regularization
    'momentum': 0.3,  # Momentum for batch normalization
    'clip_value': 2.0,  # Gradient clipping value
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_params': dict(mode='min', patience=5, factor=0.5, min_lr=1e-5),
    'mask_type': 'entmax',  # Mask type for feature selection
    'verbose': 1,
    'device_name': device
}

model = TabNetClassifier(**tabnet_params)

print(f"  ✅ TabNet configured:")
print(f"     - Decision layer width (n_d): {tabnet_params['n_d']}")
print(f"     - Attention width (n_a): {tabnet_params['n_a']}")
print(f"     - Sequential steps: {tabnet_params['n_steps']}")
print(f"     - Feature reusage (gamma): {tabnet_params['gamma']}")
print(f"     - Sparsity regularization: {tabnet_params['lambda_sparse']}")

# Train TabNet
print(f"\n[4/7] Training TabNet...")
print(f"  ℹ️  This may take 5-10 minutes...")

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_name=['val'],
    eval_metric=['accuracy'],
    max_epochs=200,
    patience=20,  # Early stopping patience
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

print(f"\n  ✅ Training complete!")

# Evaluate on validation and test sets
print(f"\n[5/7] Evaluating TabNet...")

# Validation predictions
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"  ✅ Validation accuracy: {val_acc*100:.2f}%")

# Test predictions
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, test_preds)
print(f"  ✅ Test accuracy: {test_acc*100:.2f}%")

# Feature importance
print(f"\n[6/7] Analyzing feature importance...")

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': numeric_pregame,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 most important features:")
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"    {row['feature']:<40} {row['importance']:.4f}")

# Save results
print(f"\n[7/7] Saving TabNet model and results...")

# Save model
model.save_model('../models/tabnet_model')
print(f"  ✅ Saved: ../models/tabnet_model.zip")

# Save feature importance
feature_importance_df.to_csv('../results/phase8_results/tabnet_feature_importance.csv', index=False)
print(f"  ✅ Saved: ../results/phase8_results/tabnet_feature_importance.csv")

# Save results summary
tabnet_results = {
    'val_accuracy': float(val_acc),
    'test_accuracy': float(test_acc),
    'n_epochs': len(model.history['loss']),
    'best_epoch': int(np.argmin(model.history['val_accuracy'])) + 1,
    'hyperparameters': {k: v for k, v in tabnet_params.items() if k not in ['optimizer_fn', 'scheduler_fn']},
    'top_10_features': feature_importance_df.head(10).to_dict('records')
}

with open('../results/phase8_results/tabnet_results.json', 'w') as f:
    json.dump(tabnet_results, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/tabnet_results.json")

# Compare with other models
print(f"\n{'='*120}")
print("MODEL COMPARISON")
print(f"{'='*120}")

# Load previous results
with open('../results/phase8_results/tuning_results.json', 'r') as f:
    tuning_results = json.load(f)

with open('../results/phase8_results/pytorch_training_history.json', 'r') as f:
    pytorch_history = json.load(f)

# Calculate PyTorch test accuracy from saved model
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
pytorch_test_acc = pytorch_checkpoint['test_accuracy']

print(f"\n{'Model':<20} {'Test Accuracy':<15} {'Notes':<50}")
print(f"{'-'*85}")
print(f"{'XGBoost':<20} {tuning_results['XGBoost']['test_accuracy']*100:>6.2f}%         Tree-based, tuned")
print(f"{'LightGBM':<20} {tuning_results['LightGBM']['test_accuracy']*100:>6.2f}%         Tree-based, tuned")
print(f"{'CatBoost':<20} {tuning_results['CatBoost']['test_accuracy']*100:>6.2f}%         Tree-based, tuned")
print(f"{'RandomForest':<20} {tuning_results['RandomForest']['test_accuracy']*100:>6.2f}%         Tree-based, tuned")
print(f"{'PyTorch NN':<20} {pytorch_test_acc*100:>6.2f}%         Feedforward NN (128→64→32→1)")
print(f"{'TabNet':<20} {test_acc*100:>6.2f}%         Sequential attention, interpretable")

# Determine best model
all_models = {
    'XGBoost': tuning_results['XGBoost']['test_accuracy'],
    'LightGBM': tuning_results['LightGBM']['test_accuracy'],
    'CatBoost': tuning_results['CatBoost']['test_accuracy'],
    'RandomForest': tuning_results['RandomForest']['test_accuracy'],
    'PyTorch NN': pytorch_test_acc,
    'TabNet': test_acc
}

best_model = max(all_models.items(), key=lambda x: x[1])

print(f"\n{'='*120}")
print(f"✅ BEST MODEL: {best_model[0]} with {best_model[1]*100:.2f}% test accuracy")
print(f"{'='*120}")

print(f"\n{'='*120}")
print("✅ TABNET TRAINING COMPLETE")
print(f"{'='*120}")
print(f"\n✅ TabNet test accuracy: {test_acc*100:.2f}%")
print(f"✅ Feature importance analyzed and saved")
print(f"✅ Model saved to ../models/tabnet_model.zip")
print(f"✅ Ready to proceed to Phase 8B")
print(f"\n{'='*120}")

