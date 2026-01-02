"""
Task 8B.1: Comprehensive Model Evaluation Metrics

Calculate and visualize comprehensive metrics for all models:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion Matrix
- Classification Report
"""

import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8B.1: COMPREHENSIVE MODEL EVALUATION")
print("="*120)

# Load data
print(f"\n[1/6] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

# Load feature categorization
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

# Get pre-game features
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

# Split data
val = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test = df[df['season'] == 2024].copy()

# Select features
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
X_val = val[numeric_pregame].fillna(train[numeric_pregame].median())
X_test = test[numeric_pregame].fillna(train[numeric_pregame].median())
y_val = val['home_win'].values
y_test = test['home_win'].values

print(f"  ✅ Loaded validation set: {len(y_val):,} games")
print(f"  ✅ Loaded test set: {len(y_test):,} games")

# Load all models
print(f"\n[2/6] Loading all trained models...")

models = {}

# Tree-based models
models['XGBoost'] = joblib.load('../models/xgboost_tuned.pkl')
models['LightGBM'] = joblib.load('../models/lightgbm_tuned.pkl')
models['CatBoost'] = joblib.load('../models/catboost_tuned.pkl')
models['RandomForest'] = joblib.load('../models/randomforest_tuned.pkl')

# PyTorch NN
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

# TabNet
tabnet_model = TabNetClassifier()
tabnet_model.load_model('../models/tabnet_model.zip')

print(f"  ✅ Loaded 6 models: XGBoost, LightGBM, CatBoost, RandomForest, PyTorch NN, TabNet")

# Generate predictions
print(f"\n[3/6] Generating predictions...")

predictions = {}
probabilities = {}

# Tree-based models
for name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
    predictions[name] = models[name].predict(X_test)
    probabilities[name] = models[name].predict_proba(X_test)[:, 1]

# PyTorch NN
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
with torch.no_grad():
    pytorch_probs = pytorch_model(X_test_tensor).squeeze().cpu().numpy()
predictions['PyTorch NN'] = (pytorch_probs > 0.5).astype(int)
probabilities['PyTorch NN'] = pytorch_probs

# TabNet
predictions['TabNet'] = tabnet_model.predict(X_test.values)
probabilities['TabNet'] = tabnet_model.predict_proba(X_test.values)[:, 1]

print(f"  ✅ Generated predictions for all 6 models")

# Calculate comprehensive metrics
print(f"\n[4/6] Calculating comprehensive metrics...")

metrics_results = {}

for name in predictions.keys():
    y_pred = predictions[name]
    y_prob = probabilities[name]

    metrics_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

print(f"  ✅ Calculated metrics for all models")

# Display results
print(f"\n{'='*120}")
print("COMPREHENSIVE METRICS COMPARISON")
print(f"{'='*120}")

print(f"\n{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")
print(f"{'-'*85}")

for name, metrics in metrics_results.items():
    print(f"{name:<15} {metrics['accuracy']*100:>6.2f}%    {metrics['precision']*100:>6.2f}%    "
          f"{metrics['recall']*100:>6.2f}%    {metrics['f1_score']*100:>6.2f}%    "
          f"{metrics['roc_auc']:>6.4f}     {metrics['pr_auc']:>6.4f}")

# Visualizations
print(f"\n[5/6] Creating visualizations...")

# Create output directory
import os
os.makedirs('../results/phase8_results/visualizations', exist_ok=True)

# 1. Metrics comparison bar chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx // 3, idx % 3]

    model_names = list(metrics_results.keys())
    values = [metrics_results[m][metric] for m in model_names]

    bars = ax.bar(range(len(model_names)), values, color='steelblue', alpha=0.7)

    # Highlight best model
    best_idx = np.argmax(values)
    bars[best_idx].set_color('darkgreen')
    bars[best_idx].set_alpha(1.0)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: metrics_comparison.png")
plt.close()

# 2. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

for idx, name in enumerate(predictions.keys()):
    ax = axes[idx // 3, idx % 3]

    cm = np.array(metrics_results[name]['confusion_matrix'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])

    ax.set_title(f'{name}\nAccuracy: {metrics_results[name]["accuracy"]*100:.2f}%')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: confusion_matrices.png")
plt.close()

# 3. ROC curves
plt.figure(figsize=(12, 8))

for name in predictions.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = metrics_results[name]['roc_auc']
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: roc_curves.png")
plt.close()

# 4. Precision-Recall curves
plt.figure(figsize=(12, 8))

for name in predictions.keys():
    precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
    pr_auc = metrics_results[name]['pr_auc']
    plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f})', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/precision_recall_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: precision_recall_curves.png")
plt.close()

# Save results
print(f"\n[6/6] Saving results...")

# Convert numpy arrays to lists for JSON serialization
metrics_json = {
    name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in metrics.items()}
    for name, metrics in metrics_results.items()
}

with open('../results/phase8_results/comprehensive_metrics.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)
print(f"  ✅ Saved: comprehensive_metrics.json")

print(f"\n{'='*120}")
print("✅ COMPREHENSIVE EVALUATION COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Evaluated 6 models on {len(y_test)} test games")
print(f"✅ Generated 4 visualization plots")
print(f"✅ Saved all results to ../results/phase8_results/")
print(f"\n{'='*120}")

