"""
Betting Simulation with ROI Calculation
========================================

Simulate moneyline betting with Kelly Criterion for bankroll management.
Calculate ROI, Sharpe Ratio, and other betting metrics.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

from config import *
from data_loader import NFLDataLoader

print("="*120)
print("BETTING SIMULATION - ROI CALCULATION")
print("="*120)

# Load data
loader = NFLDataLoader(use_selected_features=True)
data = loader.load_and_prepare()

# Get test data (2024 season)
X_test = data['test']['X']
y_test = data['test']['y']

print(f"\n📊 Test Set: {len(y_test)} games (2024 season)")

# Load all models
models = {}

# Tree-based models
for model_name in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
    model_path = MODELS_DIR / f'{model_name}_model.pkl'
    if model_path.exists():
        models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded {model_name}")

# PyTorch model
import torch
from train_pytorch_nn import NFLPredictor

pytorch_path = MODELS_DIR / 'pytorch_nn_best.pth'
if pytorch_path.exists():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_test.shape[1]
    pytorch_model = NFLPredictor(
        input_dim=input_dim,
        hidden_dims=NN_PARAMS['hidden_dims'],
        dropout=NN_PARAMS['dropout']
    ).to(device)
    pytorch_model.load_state_dict(torch.load(pytorch_path, weights_only=False))
    pytorch_model.eval()
    models['pytorch_nn'] = pytorch_model
    print(f"✅ Loaded pytorch_nn")

print(f"\n📦 Total models loaded: {len(models)}")

# Simulate betting for each model
def simulate_betting(y_true, y_pred_proba, initial_bankroll=10000, kelly_fraction=0.25):
    """
    Simulate moneyline betting with Kelly Criterion.
    
    Assumes American odds:
    - Home favorite: -150 (bet $150 to win $100)
    - Home underdog: +130 (bet $100 to win $130)
    
    For simplicity, we'll use a fixed odds structure based on model confidence.
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = 0
    bets_won = 0
    total_wagered = 0
    total_profit = 0
    
    for i, (true_outcome, pred_proba) in enumerate(zip(y_true, y_pred_proba)):
        # Model's confidence in home team winning
        home_win_prob = pred_proba
        
        # Only bet if model is confident (>60% or <40%)
        if home_win_prob > 0.60:
            # Bet on home team
            bet_prob = home_win_prob
            bet_outcome = 1  # Betting on home
            actual_outcome = true_outcome
        elif home_win_prob < 0.40:
            # Bet on away team
            bet_prob = 1 - home_win_prob
            bet_outcome = 0  # Betting on away
            actual_outcome = 1 - true_outcome
        else:
            # No bet (not confident enough)
            bankroll_history.append(bankroll)
            continue
        
        # Simplified betting: fixed percentage of bankroll
        # In reality, you'd use actual moneyline odds from sportsbooks
        # For simulation, we'll use a fixed 2% of bankroll per bet
        bet_amount = bankroll * 0.02

        # Payout odds (simplified):
        # If model is very confident (>70%), assume favorite odds (-150 = 1.67x)
        # If model is moderately confident (60-70%), assume even odds (2.0x)
        if bet_prob > 0.70:
            payout_multiplier = 1.67  # Favorite odds
        else:
            payout_multiplier = 2.0  # Even odds
        
        # Place bet
        bets_placed += 1
        total_wagered += bet_amount
        
        # Determine outcome
        if actual_outcome == bet_outcome:
            # Win
            profit = bet_amount * (payout_multiplier - 1)
            bankroll += profit
            total_profit += profit
            bets_won += 1
        else:
            # Loss
            bankroll -= bet_amount
            total_profit -= bet_amount
        
        bankroll_history.append(bankroll)
    
    # Calculate metrics
    win_rate = bets_won / bets_placed if bets_placed > 0 else 0
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    final_bankroll = bankroll
    net_profit = final_bankroll - initial_bankroll
    
    # Sharpe Ratio (annualized)
    if len(bankroll_history) > 1:
        returns = np.diff(bankroll_history) / np.array(bankroll_history[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'bets_placed': bets_placed,
        'bets_won': bets_won,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'roi': roi,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': final_bankroll,
        'net_profit': net_profit,
        'sharpe_ratio': sharpe_ratio,
        'bankroll_history': bankroll_history
    }

# Run simulation for each model
print("\n" + "="*120)
print("BETTING SIMULATION RESULTS")
print("="*120)

results = {}

for model_name, model in models.items():
    print(f"\n{'='*120}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*120}")
    
    # Get predictions
    if model_name == 'pytorch_nn':
        X_test_tensor = torch.FloatTensor(data['test']['X_scaled']).to(device)
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy()
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Betting simulation
    betting_results = simulate_betting(y_test.values, y_pred_proba)
    
    print(f"\n📊 Classification Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC:  {roc_auc:.4f}")
    
    print(f"\n💰 Betting Metrics:")
    print(f"   Bets Placed:     {betting_results['bets_placed']}")
    print(f"   Bets Won:        {betting_results['bets_won']}")
    print(f"   Win Rate:        {betting_results['win_rate']:.2%}")
    print(f"   Total Wagered:   ${betting_results['total_wagered']:,.2f}")
    print(f"   Total Profit:    ${betting_results['total_profit']:,.2f}")
    print(f"   ROI:             {betting_results['roi']:.2f}%")
    print(f"   Initial Bankroll: ${betting_results['initial_bankroll']:,.2f}")
    print(f"   Final Bankroll:   ${betting_results['final_bankroll']:,.2f}")
    print(f"   Net Profit:       ${betting_results['net_profit']:,.2f}")
    print(f"   Sharpe Ratio:     {betting_results['sharpe_ratio']:.4f}")
    
    results[model_name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        **betting_results
    }

# Save results
with open(RESULTS_DIR / 'betting_simulation_results.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for model_name, model_results in results.items():
        results_serializable[model_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.float32, np.float64)) else v)
            for k, v in model_results.items()
        }
    json.dump(results_serializable, f, indent=2)

print(f"\n{'='*120}")
print("✅ BETTING SIMULATION COMPLETE!")
print(f"{'='*120}")
print(f"\n📁 Results saved to: {RESULTS_DIR / 'betting_simulation_results.json'}")

