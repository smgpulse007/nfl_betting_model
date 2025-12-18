"""
XGBoost Hyperparameter Tuning
=============================
Tune XGBoost hyperparameters to maximize performance with TIER S+A features.

Uses:
- RandomizedSearchCV for efficient hyperparameter search
- TimeSeriesSplit for proper temporal cross-validation
- Multiple scoring metrics (RMSE for regression, log-loss for classification)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from run_tier_sa_backtest import load_and_prepare_data, get_feature_columns, predict_and_evaluate
from version import VERSION, BASELINE_METRICS

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss, make_scorer
import optuna
from optuna.samplers import TPESampler

# Hyperparameter search space
PARAM_DISTRIBUTIONS = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1],
}


def tune_with_optuna(X_train, y_train, model_type='regressor', n_trials=100):
    """Tune using Optuna for more efficient search."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'n_jobs': -1,
        }

        # TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if model_type == 'regressor':
                model = XGBRegressor(**params)
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
                pred = model.predict(X_v)
                score = np.sqrt(mean_squared_error(y_v, pred))
            else:
                params['use_label_encoder'] = False
                params['eval_metric'] = 'logloss'
                model = XGBClassifier(**params)
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
                pred_proba = model.predict_proba(X_v)[:, 1]
                score = log_loss(y_v, pred_proba)

            scores.append(score)

        return np.mean(scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def tune_all_models(train_df, features, n_trials=50):
    """Tune all three models."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*60)

    X = train_df[features].copy().fillna(train_df[features].median())

    results = {}

    # 1. Spread model
    print("\n📊 Tuning SPREAD model...")
    y_spread = train_df['result']
    best_params, best_score = tune_with_optuna(X, y_spread, 'regressor', n_trials)
    results['spread'] = {'params': best_params, 'cv_rmse': best_score}
    print(f"   Best RMSE: {best_score:.4f}")
    print(f"   Best params: {best_params}")

    # 2. Totals model
    print("\n📊 Tuning TOTALS model...")
    y_totals = train_df['game_total']
    best_params, best_score = tune_with_optuna(X, y_totals, 'regressor', n_trials)
    results['totals'] = {'params': best_params, 'cv_rmse': best_score}
    print(f"   Best RMSE: {best_score:.4f}")
    print(f"   Best params: {best_params}")

    # 3. Moneyline model
    print("\n📊 Tuning MONEYLINE model...")
    y_ml = train_df['home_win']
    best_params, best_score = tune_with_optuna(X, y_ml, 'classifier', n_trials)
    results['moneyline'] = {'params': best_params, 'cv_logloss': best_score}
    print(f"   Best Log-Loss: {best_score:.4f}")
    print(f"   Best params: {best_params}")

    return results


def train_tuned_models(train_df, features, tuned_params):
    """Train models with tuned hyperparameters."""
    print("\n" + "="*60)
    print("TRAINING WITH TUNED HYPERPARAMETERS")
    print("="*60)

    X = train_df[features].copy().fillna(train_df[features].median())
    models = {}

    # Spread model
    params = tuned_params['spread']['params'].copy()
    params['random_state'] = 42
    params['n_jobs'] = -1
    models['spread'] = XGBRegressor(**params)
    models['spread'].fit(X, train_df['result'])
    print(f"  Spread: {params}")

    # Totals model
    params = tuned_params['totals']['params'].copy()
    params['random_state'] = 42
    params['n_jobs'] = -1
    models['totals'] = XGBRegressor(**params)
    models['totals'].fit(X, train_df['game_total'])
    print(f"  Totals: {params}")

    # Moneyline model
    params = tuned_params['moneyline']['params'].copy()
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['use_label_encoder'] = False
    params['eval_metric'] = 'logloss'
    models['moneyline'] = XGBClassifier(**params)
    models['moneyline'].fit(X, train_df['home_win'])
    print(f"  Moneyline: {params}")

    return models


def main():
    """Run hyperparameter tuning and compare to baseline."""
    print("="*60)
    print("XGBoost HYPERPARAMETER TUNING EXPERIMENT")
    print(f"Baseline: v{VERSION}")
    print("="*60)

    # Load data
    games, completed = load_and_prepare_data()

    # Training data: 2018-2023 (same as baseline)
    train_df = completed[(completed['season'] >= 2018) & (completed['season'] <= 2023)].copy()
    test_2024 = completed[completed['season'] == 2024].copy()
    test_2025 = games[games['season'] == 2025].copy()

    print(f"\nTraining: {len(train_df)} games (2018-2023)")
    print(f"Test 2024: {len(test_2024)} games")
    print(f"Validation 2025: {len(test_2025)} games")

    # Get features
    features = get_feature_columns(train_df)

    # Tune hyperparameters (50 trials per model = ~5-10 min total)
    tuned_params = tune_all_models(train_df, features, n_trials=50)

    # Train with tuned params
    tuned_models = train_tuned_models(train_df, features, tuned_params)

    # Evaluate on 2024
    print("\n" + "="*60)
    print("EVALUATING TUNED MODELS")
    print("="*60)

    results_2024, _ = predict_and_evaluate(tuned_models, test_2024, features, "2024 Test")
    results_2025, _ = predict_and_evaluate(tuned_models, test_2025, features, "2025 Validation")

    # Compare to baseline
    print("\n" + "="*60)
    print("COMPARISON TO v0.1.0 BASELINE")
    print("="*60)

    baseline = BASELINE_METRICS['2025_validation']
    tuned = results_2025

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Tuned':>12} {'Δ':>12}")
    print("-"*56)

    metrics = [
        ('Spread WR', 'spread_wr', 100),
        ('Spread ROI', 'spread_roi', 1),
        ('Totals WR', 'totals_wr', 100),
        ('Totals ROI', 'totals_roi', 1),
        ('ML WR', 'ml_wr', 100),
        ('ML ROI', 'ml_roi', 1),
        ('Win Accuracy', 'win_accuracy', 100),
    ]

    improvements = []
    for name, key, mult in metrics:
        b_val = baseline.get(key, 0) * mult if mult == 100 else baseline.get(key, 0)
        t_val = tuned.get(key, 0) * mult if mult == 100 else tuned.get(key, 0)
        delta = t_val - b_val
        improvements.append(delta)

        b_str = f"{b_val:.1f}%" if 'WR' in name or 'Accuracy' in name else f"{b_val:+.1f}%"
        t_str = f"{t_val:.1f}%" if 'WR' in name or 'Accuracy' in name else f"{t_val:+.1f}%"
        d_str = f"{delta:+.1f}%"

        indicator = "✅" if delta > 0 else "❌" if delta < 0 else "➖"
        print(f"{name:<20} {b_str:>12} {t_str:>12} {d_str:>10} {indicator}")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'hyperparameter_tuning',
        'baseline_version': VERSION,
        'tuned_params': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv)
                             for kk, vv in v['params'].items()}
                         for k, v in tuned_params.items()},
        'results_2024': results_2024,
        'results_2025': results_2025,
        'baseline_2025': baseline,
        'improvements': dict(zip([m[0] for m in metrics], improvements)),
    }

    with open(results_dir / 'tuning_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ Results saved to results/tuning_results.json")

    # Verdict
    roi_improvement = improvements[1] + improvements[5]  # Spread ROI + ML ROI
    if roi_improvement > 5:
        print(f"\n🎉 TUNING IMPROVED COMBINED ROI BY {roi_improvement:+.1f}%!")
        print("   Consider saving as v0.2.0")
    elif roi_improvement > 0:
        print(f"\n📈 Modest improvement: {roi_improvement:+.1f}% combined ROI")
    else:
        print(f"\n📉 No improvement: {roi_improvement:+.1f}% combined ROI")
        print("   Baseline hyperparameters may already be near-optimal")

    return summary


if __name__ == "__main__":
    main()

