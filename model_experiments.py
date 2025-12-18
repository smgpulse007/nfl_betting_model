"""
Model Experiments - Educational Progression through ML/DL Algorithms
=====================================================================

This module provides a framework to experiment with different model architectures,
from simple linear models to deep learning, tracking performance against v0.2.0 baseline.

=============================================================================
UNDERSTANDING L1 vs L2 REGULARIZATION (Lasso vs Ridge)
=============================================================================

Both are techniques to prevent OVERFITTING by adding a penalty to large coefficients.

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONCEPT: Why Regularization?                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Without regularization, models can have HUGE coefficients that fit the      │
│ training data perfectly but fail on new data (overfitting).                 │
│                                                                             │
│ Example from our OLS results:                                               │
│   pressure_diff coefficient = 431.56 (HUGE!)                                │
│   away_pressure_rate coefficient = 316.12                                   │
│                                                                             │
│ These massive coefficients mean tiny changes in pressure rate cause         │
│ wild swings in predictions - a sign of overfitting to noise.                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ L2 REGULARIZATION (Ridge Regression)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Loss = MSE + λ * Σ(β²)                                                      │
│                                                                             │
│ • Adds the SUM OF SQUARED coefficients as penalty                           │
│ • Shrinks all coefficients toward zero, but never TO zero                   │
│ • Good when you believe ALL features have some predictive power             │
│ • Handles multicollinearity (correlated features) well                      │
│                                                                             │
│ Example from our results (spread_ridge):                                    │
│   pressure_diff: 431.56 (OLS) → 0.16 (Ridge) - DRAMATICALLY reduced!        │
│   RMSE improved: 12.57 (OLS) → 12.53 (Ridge)                                │
│   ROI improved: -6.2% (OLS) → -2.8% (Ridge)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 REGULARIZATION (Lasso Regression)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Loss = MSE + λ * Σ|β|                                                       │
│                                                                             │
│ • Adds the SUM OF ABSOLUTE coefficients as penalty                          │
│ • Can shrink coefficients EXACTLY to zero (feature selection!)              │
│ • Good when you believe SOME features are noise                             │
│ • Produces SPARSE models (fewer features)                                   │
│                                                                             │
│ Example from our results (spread_lasso):                                    │
│   Only 2 non-zero features: spread_line (5.13), home_implied_prob (0.06)    │
│   All other 33 features → 0 (eliminated as noise!)                          │
│                                                                             │
│ This tells us: For predicting spread, Vegas spread_line alone is            │
│ the most important signal. Most of our features are redundant.              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ VISUAL COMPARISON                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     L2 (Ridge)                          L1 (Lasso)                          │
│     ──────────                          ──────────                          │
│        ●                                   ◆                                │
│       ╱│╲                                 ╱│╲                               │
│      ╱ │ ╲    Circular                   ╱ │ ╲   Diamond                    │
│     ╱  │  ╲   constraint                ╱  │  ╲  constraint                 │
│     ╲  │  ╱                             ◆──●──◆                             │
│      ╲ │ ╱                               ╲ │ ╱                              │
│       ╲│╱                                 ╲│╱                               │
│        ●                                   ◆                                │
│                                                                             │
│   Coefficients shrink                 Coefficients hit corners              │
│   but stay non-zero                   (corners = some β = 0)                │
└─────────────────────────────────────────────────────────────────────────────┘

=============================================================================
WHY DID L1 LOGISTIC OUTPERFORM XGBOOST ON WIN ACCURACY? (68% vs 64.5%)
=============================================================================

Several factors explain this surprising result:

1. FEATURE SELECTION → LESS OVERFITTING
   ─────────────────────────────────────
   L1 Logistic kept only 18 of 35 features (17 set to zero)
   XGBoost uses ALL 35 features, potentially fitting noise

   Key L1 survivors: spread_line (OR=2.29), separation_diff, home_pressure_rate
   Eliminated: elo_diff, elo_prob, home_implied_prob, cpoe_diff, injury features

   → L1 discovered that simple market-based signals beat complex feature engineering

2. TASK MISMATCH
   ──────────────
   Win prediction is CLASSIFICATION (yes/no), not regression
   XGBoost was tuned for RMSE/spread prediction, not classification accuracy
   LogisticRegression is purpose-built for binary classification

   XGBoost optimized for: "How many points will home team win by?"
   LogisticRegression optimized for: "Will home team win? (probability)"

3. CALIBRATED PROBABILITIES
   ─────────────────────────
   Logistic regression outputs well-calibrated probabilities
   XGBoost probabilities can be over/under-confident

   LogisticRegression: P(home_win) is directly interpretable
   XGBoost: predict_proba() is an approximation from tree votes

4. SIMPLER DECISION BOUNDARY
   ──────────────────────────
   Win/Loss in NFL often comes down to a few key factors
   L1 found: spread_line alone explains most of the variance (OR=2.29)
   XGBoost may be "overthinking" with complex tree interactions

=============================================================================
ODDS RATIOS EXPLAINED
=============================================================================

For LogisticRegression, we can interpret coefficients as ODDS RATIOS:

   Odds Ratio = e^(coefficient)

   • OR = 1.0  → No effect on outcome
   • OR > 1.0  → Increases probability of home win
   • OR < 1.0  → Decreases probability of home win

Example from L1 Logistic:
   spread_line: coefficient = 0.83, OR = e^0.83 = 2.29

   Interpretation: Each 1-point increase in spread_line (favoring home)
   increases the ODDS of home team winning by 129% (2.29x)

   If home team is -7 vs -6, their odds of winning are 2.29x higher

=============================================================================
MODEL PROGRESSION ROADMAP
=============================================================================

Progression:
1. Linear Models (interpretable baselines) ✅ DONE
   - LinearRegression (spread, totals)
   - LogisticRegression (moneyline) - with odds ratios

2. Regularized Linear Models ✅ DONE
   - Ridge/Lasso Regression
   - ElasticNet (combines L1 + L2)

3. Tree-Based Models
   - Decision Trees (single tree, very interpretable)
   - Random Forests (bagged trees, reduces variance)

4. Boosting Models (current best)
   - XGBoost (v0.2.0 baseline)
   - LightGBM (histogram-based, faster)
   - CatBoost (handles categoricals natively)

5. Deep Learning (PyTorch)
   - Simple MLP (Multi-Layer Perceptron)
   - Residual connections (skip connections)
   - Attention mechanisms (transformer-style)
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

