"""
Prediction Models for NFL Betting

Models:
1. Elo Baseline - Uses Elo win probability
2. XGBoost Classifier - Win probability
3. XGBoost Regressor - Spread prediction (margin of victory)
4. XGBoost Regressor - Totals prediction (combined score)
5. Logistic Regression - Win probability (calibrated)
6. Ensemble - Weighted combination

Betting Types:
- Moneyline: Win probability vs implied odds
- Spread (ATS): Predicted margin vs Vegas spread
- Totals (O/U): Predicted total vs Vegas total
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR


# Feature columns for modeling
FEATURE_COLS = [
    'spread_line',        # Vegas spread (negative = home favored)
    'total_line',         # Vegas total
    'elo_diff',           # Home Elo - Away Elo
    'elo_prob',           # Elo-based win probability
    'home_rest',          # Days of rest
    'away_rest',
    'rest_advantage',     # Home rest - Away rest
    'temp',               # Temperature
    'wind',               # Wind speed
    'is_dome',            # Indoor game
    'is_cold',            # Temp < 40
    'div_game',           # Division rivalry
    'home_implied_prob',  # Market implied probability
]

# Historical standard deviations for probability conversion
# Based on NFL historical data (1999-2024)
SPREAD_STD = 13.5   # Standard deviation of actual margin vs predicted
TOTAL_STD = 10.5    # Standard deviation of actual total vs predicted


def normal_cdf(x: float, mean: float = 0, std: float = 1) -> float:
    """Cumulative distribution function for normal distribution."""
    from scipy.stats import norm
    return norm.cdf(x, mean, std)


class NFLBettingModels:
    """Collection of NFL betting prediction models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_spread = None      # Predicts margin of victory
        self.xgb_win = None         # Predicts win probability
        self.xgb_margin = None      # Predicts actual margin (for spread betting)
        self.xgb_total = None       # Predicts total points
        self.lr_win = None          # Logistic regression for win prob
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

        # Calibration parameters learned from training
        self.spread_std = SPREAD_STD
        self.total_std = TOTAL_STD

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix, handling missing values."""
        X = df[self.feature_cols].copy()

        # Fill missing values
        X['home_implied_prob'] = X['home_implied_prob'].fillna(0.5)
        X['temp'] = X['temp'].fillna(70)
        X['wind'] = X['wind'].fillna(0)
        X = X.fillna(0)

        return X

    def fit(self, train_df: pd.DataFrame):
        """Fit all models on training data."""
        print("Fitting models...")

        # Prepare features
        X = self._prepare_features(train_df)

        # Filter to games with valid targets
        valid_mask = train_df['home_score'].notna() & train_df['spread_line'].notna()
        X_valid = X[valid_mask]
        train_valid = train_df[valid_mask].copy()

        # Targets
        y_win = train_valid['home_win'].values
        y_margin = train_valid['result'].values  # home_score - away_score
        y_total = (train_valid['home_score'] + train_valid['away_score']).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X_valid)

        # 1. XGBoost for margin prediction (for spread betting)
        print("  Training XGBoost margin model (spread betting)...")
        self.xgb_margin = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_margin.fit(X_scaled, y_margin)

        # Calculate residual std for probability conversion
        margin_pred = self.xgb_margin.predict(X_scaled)
        self.spread_std = np.std(y_margin - margin_pred)
        print(f"    Margin model std: {self.spread_std:.2f} points")

        # 2. XGBoost for total points prediction
        print("  Training XGBoost total model (O/U betting)...")
        self.xgb_total = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_total.fit(X_scaled, y_total)

        # Calculate residual std for probability conversion
        total_pred = self.xgb_total.predict(X_scaled)
        self.total_std = np.std(y_total - total_pred)
        print(f"    Total model std: {self.total_std:.2f} points")

        # 3. XGBoost for win probability (existing)
        print("  Training XGBoost win model...")
        self.xgb_win = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.xgb_win.fit(X_scaled, y_win)

        # 4. Calibrated Logistic Regression
        print("  Training Logistic Regression...")
        base_lr = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_win = CalibratedClassifierCV(base_lr, cv=5, method='isotonic')
        self.lr_win.fit(X_scaled, y_win)

        self.is_fitted = True
        print("Models fitted successfully!")

        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for games including spread and totals."""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        preds = df[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()

        # =====================
        # MONEYLINE PREDICTIONS
        # =====================

        # Elo baseline
        preds['elo_prob'] = df['elo_prob'].values

        # XGBoost win probability
        preds['xgb_win_prob'] = self.xgb_win.predict_proba(X_scaled)[:, 1]

        # Logistic Regression win probability
        preds['lr_win_prob'] = self.lr_win.predict_proba(X_scaled)[:, 1]

        # Ensemble win probability
        preds['ensemble_prob'] = (
            preds['elo_prob'] * 0.2 +
            preds['xgb_win_prob'] * 0.4 +
            preds['lr_win_prob'] * 0.4
        )

        # =====================
        # SPREAD PREDICTIONS (ATS)
        # =====================

        # Predicted margin of victory (positive = home wins by X)
        preds['pred_margin'] = self.xgb_margin.predict(X_scaled)

        # Vegas spread (negative = home favored by X)
        preds['spread_line'] = df['spread_line'].values

        # Cover probability: P(actual_margin > -spread_line)
        # If spread is -7 (home favored by 7), home covers if they win by > 7
        # P(margin > -spread) = P(margin > 7) when spread = -7
        # Using normal CDF with predicted margin as mean
        preds['home_cover_prob'] = preds.apply(
            lambda row: normal_cdf(
                x=0,  # We want P(margin + spread > 0)
                mean=-(row['pred_margin'] + row['spread_line']),
                std=self.spread_std
            ) if pd.notna(row['spread_line']) else 0.5,
            axis=1
        )
        preds['away_cover_prob'] = 1 - preds['home_cover_prob']

        # =====================
        # TOTALS PREDICTIONS (O/U)
        # =====================

        # Predicted total points
        preds['pred_total'] = self.xgb_total.predict(X_scaled)

        # Vegas total
        preds['total_line'] = df['total_line'].values

        # Over probability: P(actual_total > vegas_total)
        preds['over_prob'] = preds.apply(
            lambda row: 1 - normal_cdf(
                x=row['total_line'],
                mean=row['pred_total'],
                std=self.total_std
            ) if pd.notna(row['total_line']) else 0.5,
            axis=1
        )
        preds['under_prob'] = 1 - preds['over_prob']

        return preds
    
    def evaluate(self, df: pd.DataFrame, preds: pd.DataFrame) -> Dict:
        """Evaluate model performance for ML, spread, and totals."""
        results = {
            'moneyline': {},
            'spread': {},
            'totals': {}
        }

        # =================
        # MONEYLINE EVALUATION
        # =================
        y_win = df['home_win'].values

        for prob_col in ['elo_prob', 'xgb_win_prob', 'lr_win_prob', 'ensemble_prob']:
            probs = preds[prob_col].values
            preds_binary = (probs > 0.5).astype(int)

            results['moneyline'][prob_col] = {
                'accuracy': accuracy_score(y_win, preds_binary),
                'brier_score': brier_score_loss(y_win, probs),
            }

        # =================
        # SPREAD EVALUATION
        # =================
        # Filter to games with spread data
        valid_spread = df['spread_line'].notna() & df['home_score'].notna()
        if valid_spread.sum() > 0:
            df_spread = df[valid_spread].copy()
            preds_spread = preds[valid_spread].copy()

            # Actual margin
            actual_margin = df_spread['result'].values
            pred_margin = preds_spread['pred_margin'].values

            # Did home cover?
            actual_cover = (actual_margin + df_spread['spread_line'].values) > 0
            pred_cover = preds_spread['home_cover_prob'].values > 0.5

            results['spread'] = {
                'margin_mae': mean_absolute_error(actual_margin, pred_margin),
                'cover_accuracy': accuracy_score(actual_cover, pred_cover),
                'cover_brier': brier_score_loss(actual_cover.astype(int), preds_spread['home_cover_prob'].values),
                'games_evaluated': len(df_spread)
            }

        # =================
        # TOTALS EVALUATION
        # =================
        valid_total = df['total_line'].notna() & df['home_score'].notna()
        if valid_total.sum() > 0:
            df_total = df[valid_total].copy()
            preds_total = preds[valid_total].copy()

            # Actual total
            actual_total = (df_total['home_score'] + df_total['away_score']).values
            pred_total = preds_total['pred_total'].values

            # Did it go over?
            actual_over = actual_total > df_total['total_line'].values
            pred_over = preds_total['over_prob'].values > 0.5

            results['totals'] = {
                'total_mae': mean_absolute_error(actual_total, pred_total),
                'over_accuracy': accuracy_score(actual_over, pred_over),
                'over_brier': brier_score_loss(actual_over.astype(int), preds_total['over_prob'].values),
                'games_evaluated': len(df_total)
            }

        return results


if __name__ == "__main__":
    # Load data
    games = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")

    # Split
    train = games[games['season'] < 2024]
    test = games[games['season'] == 2024]

    print(f"Train: {len(train)} games, Test: {len(test)} games")

    # Fit and evaluate
    models = NFLBettingModels()
    models.fit(train)

    # Predict on test
    test_preds = models.predict(test)
    results = models.evaluate(test, test_preds)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION - 2024 Season")
    print("=" * 70)

    print("\n--- MONEYLINE ---")
    for model, metrics in results['moneyline'].items():
        name = model.replace('_prob', '').replace('_win', '')
        print(f"  {name}: {metrics['accuracy']:.1%} accuracy, {metrics['brier_score']:.4f} Brier")

    print("\n--- SPREAD (ATS) ---")
    if results['spread']:
        print(f"  Margin MAE: {results['spread']['margin_mae']:.2f} points")
        print(f"  Cover Accuracy: {results['spread']['cover_accuracy']:.1%}")
        print(f"  Cover Brier: {results['spread']['cover_brier']:.4f}")
        print(f"  Games: {results['spread']['games_evaluated']}")

    print("\n--- TOTALS (O/U) ---")
    if results['totals']:
        print(f"  Total MAE: {results['totals']['total_mae']:.2f} points")
        print(f"  O/U Accuracy: {results['totals']['over_accuracy']:.1%}")
        print(f"  O/U Brier: {results['totals']['over_brier']:.4f}")
        print(f"  Games: {results['totals']['games_evaluated']}")

