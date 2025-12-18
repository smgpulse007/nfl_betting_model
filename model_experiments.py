"""
Model Experiments - Educational Progression through ML/DL Algorithms

This module provides a framework to experiment with different model architectures,
from simple linear models to deep learning, tracking performance against v0.2.0 baseline.

Progression:
1. Linear Models (interpretable baselines)
   - LinearRegression (spread, totals)
   - LogisticRegression (moneyline) - with odds ratios
   
2. Regularized Linear Models
   - Ridge/Lasso Regression
   - ElasticNet
   
3. Tree-Based Models
   - Decision Trees
   - Random Forests
   
4. Boosting Models (current best)
   - XGBoost (v0.2.0 baseline)
   - LightGBM
   - CatBoost
   
5. Deep Learning (PyTorch)
   - Simple MLP
   - Residual connections
   - Attention mechanisms
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Sklearn
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

# XGBoost (baseline)
from xgboost import XGBRegressor, XGBClassifier

# Data loading
from run_tier_sa_backtest import load_and_prepare_data, get_feature_columns


class ModelExperiment:
    """Base class for model experiments."""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type  # 'spread', 'totals', 'moneyline'
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.feature_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model with optional scaling."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame):
        """For classifiers, get probability estimates."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]
        return self.predict(X)
    
    def get_coefficients(self, feature_names: List[str]) -> pd.DataFrame:
        """Get model coefficients/feature importance."""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_.flatten() if self.model.coef_.ndim > 1 else self.model.coef_
            df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coef,
                'abs_coef': np.abs(coef)
            }).sort_values('abs_coef', ascending=False)
            
            # For logistic regression, compute odds ratios
            if isinstance(self.model, LogisticRegression):
                df['odds_ratio'] = np.exp(df['coefficient'])
            return df
        elif hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()


class LinearSpreadModel(ModelExperiment):
    """Linear Regression for spread prediction."""
    
    def __init__(self, regularization: str = None, alpha: float = 1.0):
        super().__init__(f"Linear_{regularization or 'OLS'}", 'spread')
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif regularization == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            self.model = LinearRegression()


class LinearTotalsModel(ModelExperiment):
    """Linear Regression for totals prediction."""
    
    def __init__(self, regularization: str = None, alpha: float = 1.0):
        super().__init__(f"Linear_{regularization or 'OLS'}", 'totals')
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif regularization == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            self.model = LinearRegression()


class LogisticMLModel(ModelExperiment):
    """Logistic Regression for moneyline prediction with odds ratios."""
    
    def __init__(self, regularization: str = 'l2', C: float = 1.0):
        super().__init__(f"Logistic_{regularization}", 'moneyline')
        self.model = LogisticRegression(
            penalty=regularization if regularization != 'none' else None,
            C=C,
            max_iter=1000,
            solver='lbfgs' if regularization in ['l2', 'none', None] else 'saga'
        )


def run_linear_experiments(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Run all linear model experiments and compare to baseline."""
    print("\n" + "="*60)
    print("LINEAR MODEL EXPERIMENTS")
    print("="*60)

    # Split data
    train_df = df[(df['season'] >= 2018) & (df['season'] <= 2023)].copy()
    test_df = df[df['season'] == 2024].copy()
    val_df = df[df['season'] == 2025].copy()

    X_train = train_df[features].copy().fillna(train_df[features].median())
    X_test = test_df[features].copy().fillna(train_df[features].median())
    X_val = val_df[features].copy().fillna(train_df[features].median())

    results = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'coefficients': {}
    }

    # --- SPREAD MODELS ---
    print("\n📊 SPREAD Models (LinearRegression)")
    for reg in [None, 'ridge', 'lasso']:
        name = f"spread_{reg or 'ols'}"
        model = LinearSpreadModel(regularization=reg)
        model.fit(X_train, train_df['result'])

        # Predictions
        pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(val_df['result'], pred_val))

        # Betting evaluation
        val_df_copy = val_df.copy()
        val_df_copy['pred_spread'] = pred_val
        val_df_copy['bet_home'] = val_df_copy['pred_spread'] > val_df_copy['spread_line']
        val_df_copy['spread_win'] = (
            ((val_df_copy['result'] > val_df_copy['spread_line']) & val_df_copy['bet_home']) |
            ((val_df_copy['result'] < val_df_copy['spread_line']) & ~val_df_copy['bet_home'])
        )
        wr = val_df_copy['spread_win'].mean()
        roi = (wr * 0.91 - (1 - wr)) * 100

        results['models'][name] = {'rmse': rmse, 'wr': wr, 'roi': roi}
        print(f"  {reg or 'OLS':12} | RMSE: {rmse:.2f} | WR: {wr:.1%} | ROI: {roi:+.1f}%")

        # Store coefficients
        coef_df = model.get_coefficients(features)
        results['coefficients'][name] = coef_df.to_dict('records')

    # --- TOTALS MODELS ---
    print("\n📊 TOTALS Models (LinearRegression)")
    for reg in [None, 'ridge', 'lasso']:
        name = f"totals_{reg or 'ols'}"
        model = LinearTotalsModel(regularization=reg)
        model.fit(X_train, train_df['game_total'])

        pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(val_df['game_total'], pred_val))

        val_df_copy = val_df.copy()
        val_df_copy['pred_total'] = pred_val
        val_df_copy['bet_over'] = val_df_copy['pred_total'] > val_df_copy['total_line']
        val_df_copy['total_win'] = (
            ((val_df_copy['game_total'] > val_df_copy['total_line']) & val_df_copy['bet_over']) |
            ((val_df_copy['game_total'] < val_df_copy['total_line']) & ~val_df_copy['bet_over'])
        )
        wr = val_df_copy['total_win'].mean()
        roi = (wr * 0.91 - (1 - wr)) * 100

        results['models'][name] = {'rmse': rmse, 'wr': wr, 'roi': roi}
        print(f"  {reg or 'OLS':12} | RMSE: {rmse:.2f} | WR: {wr:.1%} | ROI: {roi:+.1f}%")

        coef_df = model.get_coefficients(features)
        results['coefficients'][name] = coef_df.to_dict('records')

    # --- MONEYLINE MODELS ---
    print("\n📊 MONEYLINE Models (LogisticRegression) - with Odds Ratios")
    for reg in ['l2', 'l1']:
        name = f"moneyline_{reg}"
        model = LogisticMLModel(regularization=reg, C=0.1)
        model.fit(X_train, train_df['home_win'])

        pred_proba = model.predict_proba(X_val)
        logloss = log_loss(val_df['home_win'], pred_proba)
        accuracy = accuracy_score(val_df['home_win'], (pred_proba > 0.5).astype(int))

        results['models'][name] = {'logloss': logloss, 'accuracy': accuracy}
        print(f"  {reg:12} | LogLoss: {logloss:.4f} | Accuracy: {accuracy:.1%}")

        # Odds ratios
        coef_df = model.get_coefficients(features)
        results['coefficients'][name] = coef_df.to_dict('records')

        # Print top odds ratios
        print(f"\n  Top 10 Odds Ratios ({reg}):")
        for _, row in coef_df.head(10).iterrows():
            or_val = row.get('odds_ratio', np.exp(row['coefficient']))
            print(f"    {row['feature']:30} | OR: {or_val:.3f} | Coef: {row['coefficient']:+.3f}")

    # Save results
    output_path = Path("results/linear_experiments.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {output_path}")

    return results


def main():
    """Run all linear experiments."""
    print("="*60)
    print("MODEL EXPERIMENTS - Linear Models")
    print("="*60)

    # Load data (returns games_df, completed_df)
    _, df = load_and_prepare_data()
    features = get_feature_columns(df)

    # Run experiments
    results = run_linear_experiments(df, features)

    # Compare to v0.2.0 baseline
    print("\n" + "="*60)
    print("COMPARISON TO v0.2.0 BASELINE")
    print("="*60)
    print("\nBaseline (XGBoost v0.2.0):")
    print("  Spread: 52.2% WR, -0.4% ROI")
    print("  Totals: 50.0% WR, -4.5% ROI")
    print("  ML: 64.5% Win Accuracy")

    return results


if __name__ == "__main__":
    main()

