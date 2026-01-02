"""
Task 8B.2: Calibration Analysis

Analyze how well model probabilities match actual outcomes.
Generate calibration curves for various confidence levels (50%, 60%, 70%, 80%, 90%).
"""

import pandas as pd
import numpy as np
import json
import joblib
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8B.2: CALIBRATION ANALYSIS")
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

# Load models and generate probabilities
print(f"\n[2/5] Loading models and generating probabilities...")

probabilities = {}

# Tree-based models
models = {
    'XGBoost': joblib.load('../models/xgboost_tuned.pkl'),
    'LightGBM': joblib.load('../models/lightgbm_tuned.pkl'),
    'CatBoost': joblib.load('../models/catboost_tuned.pkl'),
    'RandomForest': joblib.load('../models/randomforest_tuned.pkl')
}

for name, model in models.items():
    probabilities[name] = model.predict_proba(X_test)[:, 1]

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

X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
with torch.no_grad():
    probabilities['PyTorch NN'] = pytorch_model(X_test_tensor).squeeze().cpu().numpy()

# TabNet
tabnet_model = TabNetClassifier()
tabnet_model.load_model('../models/tabnet_model.zip')
probabilities['TabNet'] = tabnet_model.predict_proba(X_test.values)[:, 1]

print(f"  ✅ Generated probabilities for all 6 models")

# Calculate calibration metrics
print(f"\n[3/5] Calculating calibration metrics...")

calibration_results = {}

for name, probs in probabilities.items():
    # Brier score (lower is better)
    brier = brier_score_loss(y_test, probs)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    calibration_results[name] = {
        'brier_score': float(brier),
        'ece': float(ece),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }

print(f"\n{'Model':<15} {'Brier Score':<15} {'ECE':<15}")
print(f"{'-'*45}")
for name, results in calibration_results.items():
    print(f"{name:<15} {results['brier_score']:>10.4f}      {results['ece']:>10.4f}")

# Analyze confidence levels
print(f"\n[4/5] Analyzing confidence levels (50%, 60%, 70%, 80%, 90%)...")

confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90]
confidence_analysis = {}

for name, probs in probabilities.items():
    confidence_analysis[name] = {}

    for conf_level in confidence_levels:
        # Games where model is confident (prob >= conf_level or prob <= 1-conf_level)
        confident_mask = (probs >= conf_level) | (probs <= 1 - conf_level)

        if confident_mask.sum() > 0:
            confident_probs = probs[confident_mask]
            confident_actual = y_test[confident_mask]

            # Predict home win if prob >= 0.5
            confident_preds = (confident_probs >= 0.5).astype(int)

            accuracy = (confident_preds == confident_actual).mean()
            n_games = confident_mask.sum()

            confidence_analysis[name][f'{int(conf_level*100)}%'] = {
                'n_games': int(n_games),
                'accuracy': float(accuracy),
                'pct_of_total': float(n_games / len(y_test))
            }
        else:
            confidence_analysis[name][f'{int(conf_level*100)}%'] = {
                'n_games': 0,
                'accuracy': 0.0,
                'pct_of_total': 0.0
            }

print(f"\n{'Model':<15} {'Conf Level':<12} {'N Games':<10} {'Accuracy':<12} {'% of Total':<12}")
print(f"{'-'*65}")
for name in probabilities.keys():
    for conf_level in confidence_levels:
        conf_key = f'{int(conf_level*100)}%'
        data = confidence_analysis[name][conf_key]
        print(f"{name:<15} {conf_key:<12} {data['n_games']:<10} {data['accuracy']*100:>6.2f}%       {data['pct_of_total']*100:>6.2f}%")

# Visualizations
print(f"\n[5/5] Creating calibration visualizations...")

# 1. Calibration curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Calibration Curves - All Models', fontsize=16, fontweight='bold')

for idx, name in enumerate(probabilities.keys()):
    ax = axes[idx // 3, idx % 3]

    prob_true = np.array(calibration_results[name]['prob_true'])
    prob_pred = np.array(calibration_results[name]['prob_pred'])

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.plot(prob_pred, prob_true, 'o-', label=f'{name}', linewidth=2, markersize=8)

    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('True Probability', fontsize=11)
    ax.set_title(f'{name}\nBrier: {calibration_results[name]["brier_score"]:.4f}, ECE: {calibration_results[name]["ece"]:.4f}')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/calibration_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: calibration_curves.png")
plt.close()

# 2. Confidence level analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confidence Level Analysis', fontsize=16, fontweight='bold')

# Plot 1: Accuracy vs Confidence Level
ax1 = axes[0]
for name in probabilities.keys():
    conf_levels_pct = [int(c*100) for c in confidence_levels]
    accuracies = [confidence_analysis[name][f'{c}%']['accuracy']*100 for c in conf_levels_pct]
    ax1.plot(conf_levels_pct, accuracies, 'o-', label=name, linewidth=2, markersize=8)

ax1.set_xlabel('Confidence Level (%)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy at Different Confidence Levels')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# Plot 2: Number of Games vs Confidence Level
ax2 = axes[1]
for name in probabilities.keys():
    conf_levels_pct = [int(c*100) for c in confidence_levels]
    n_games = [confidence_analysis[name][f'{c}%']['n_games'] for c in conf_levels_pct]
    ax2.plot(conf_levels_pct, n_games, 'o-', label=name, linewidth=2, markersize=8)

ax2.set_xlabel('Confidence Level (%)', fontsize=12)
ax2.set_ylabel('Number of Games', fontsize=12)
ax2.set_title('Number of Games at Different Confidence Levels')
ax2.legend(loc='best')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/confidence_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: confidence_analysis.png")
plt.close()

# 3. Brier score comparison
plt.figure(figsize=(12, 6))
model_names = list(calibration_results.keys())
brier_scores = [calibration_results[m]['brier_score'] for m in model_names]

bars = plt.bar(range(len(model_names)), brier_scores, color='steelblue', alpha=0.7)

# Highlight best (lowest) Brier score
best_idx = np.argmin(brier_scores)
bars[best_idx].set_color('darkgreen')
bars[best_idx].set_alpha(1.0)

plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.ylabel('Brier Score (lower is better)', fontsize=12)
plt.title('Brier Score Comparison - All Models', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(brier_scores):
    plt.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/brier_score_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: brier_score_comparison.png")
plt.close()

# Save results
with open('../results/phase8_results/calibration_results.json', 'w') as f:
    json.dump({
        'calibration_metrics': calibration_results,
        'confidence_analysis': confidence_analysis
    }, f, indent=2)
print(f"  ✅ Saved: calibration_results.json")

print(f"\n{'='*120}")
print("✅ CALIBRATION ANALYSIS COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Analyzed calibration for 6 models")
print(f"✅ Generated 3 visualization plots")
print(f"✅ Analyzed confidence levels: 50%, 60%, 70%, 80%, 90%")
print(f"\n{'='*120}")

