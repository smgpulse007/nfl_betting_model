"""
Training Configuration
=======================

Centralized configuration for all model training experiments.
"""

from pathlib import Path
import json

# Paths
DATA_PATH = Path('../results/game_level_predictions_dataset.parquet')
FEATURE_SELECTION_PATH = Path('../results/phase7_feature_selection/selected_features_clean.json')
MODELS_DIR = Path('../results/phase7_models')
RESULTS_DIR = Path('../results/phase7_results')

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data splits
TRAIN_YEARS = list(range(1999, 2023))  # 1999-2022
VAL_YEAR = 2023
TEST_YEAR = 2024

# Target variable
TARGET = 'home_win'

# Metadata columns
METADATA_COLS = ['game_id', 'game_date', 'year', 'week', 'home_team', 'away_team', 'home_win']

# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Neural network hyperparameters
NN_PARAMS = {
    'hidden_dims': [256, 128, 64],
    'dropout': 0.3,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 15,
    'device': 'cuda'  # RTX 4090
}

# Optuna hyperparameter tuning
OPTUNA_PARAMS = {
    'n_trials': 50,
    'timeout': 3600,  # 1 hour
    'n_jobs': 1  # Sequential for GPU
}

# Evaluation metrics
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'log_loss'
]

# Betting simulation
BETTING_PARAMS = {
    'initial_bankroll': 10000,
    'kelly_fraction': 0.25,  # Quarter Kelly
    'min_edge': 0.02,  # 2% minimum edge
    'max_bet_fraction': 0.05  # Max 5% of bankroll per bet
}

def load_selected_features():
    """Load the selected features from feature selection."""
    with open(FEATURE_SELECTION_PATH, 'r') as f:
        return json.load(f)

def save_model_config(model_name, params):
    """Save model configuration."""
    config_path = MODELS_DIR / f'{model_name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)
    return config_path

def load_model_config(model_name):
    """Load model configuration."""
    config_path = MODELS_DIR / f'{model_name}_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

