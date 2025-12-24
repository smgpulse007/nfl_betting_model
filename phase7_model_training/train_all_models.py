"""
Train All Models
================

Comprehensive training script for all model architectures:
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- PyTorch Neural Network

Each model is trained, evaluated, and saved with comprehensive metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)

# Import local modules
from config import *
from data_loader import NFLDataLoader

print("="*120)
print("PHASE 7: COMPREHENSIVE MODEL TRAINING")
print("="*120)

# Load data
print("\n" + "="*120)
print("STEP 1: DATA LOADING")
print("="*120)
loader = NFLDataLoader(use_selected_features=True)
data = loader.load_and_prepare()

X_train, y_train = data['train']['X'], data['train']['y']
X_val, y_val = data['val']['X'], data['val']['y']
X_test, y_test = data['test']['X'], data['test']['y']

# Store results
all_results = {}

def evaluate_model(model, X, y, split_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'log_loss': log_loss(y, y_pred_proba)
    }
    
    print(f"\n  {split_name} Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Log Loss:  {metrics['log_loss']:.4f}")
    
    return metrics, y_pred, y_pred_proba

# Train models
models_to_train = [
    ('XGBoost', XGBClassifier(**XGBOOST_PARAMS)),
    ('LightGBM', LGBMClassifier(**LIGHTGBM_PARAMS)),
    ('CatBoost', CatBoostClassifier(**CATBOOST_PARAMS)),
    ('RandomForest', RandomForestClassifier(**RANDOM_FOREST_PARAMS))
]

for model_name, model in models_to_train:
    print("\n" + "="*120)
    print(f"TRAINING: {model_name}")
    print("="*120)
    
    # Train
    print(f"\n[{model_name}] Training...")
    start_time = datetime.now()
    
    if model_name == 'XGBoost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    elif model_name == 'LightGBM':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    else:
        model.fit(X_train, y_train)
    
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    
    # Evaluate
    print(f"\n[{model_name}] Evaluating...")
    train_metrics, _, _ = evaluate_model(model, X_train, y_train, "Train")
    val_metrics, _, _ = evaluate_model(model, X_val, y_val, "Val")
    test_metrics, y_pred_test, y_pred_proba_test = evaluate_model(model, X_test, y_test, "Test")
    
    # Save model
    model_path = MODELS_DIR / f'{model_name.lower()}_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n  ✅ Saved model to {model_path}")
    
    # Store results
    all_results[model_name] = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_time_seconds': train_time,
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat()
    }

# Save all results
results_path = RESULTS_DIR / 'tree_models_results.json'
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*120)
print("✅ ALL TREE-BASED MODELS TRAINED!")
print("="*120)
print(f"\nResults saved to: {results_path}")

# Print summary
print("\n" + "="*120)
print("MODEL COMPARISON SUMMARY (Test Set)")
print("="*120)
summary_df = pd.DataFrame({
    model: results['test_metrics']
    for model, results in all_results.items()
}).T
print(summary_df.to_string())
summary_df.to_csv(RESULTS_DIR / 'tree_models_summary.csv')

