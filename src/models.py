"""
Prediction Models for NFL Betting

Models:
1. Elo Baseline - Uses Elo win probability
2. XGBoost - Spread prediction
3. Logistic Regression - Win probability (calibrated)
4. Ensemble - Weighted combination
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
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


class NFLBettingModels:
    """Collection of NFL betting prediction models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_spread = None
        self.xgb_win = None
        self.lr_win = None
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False
    
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
        X = X[valid_mask]
        train_df = train_df[valid_mask]
        
        # Targets
        y_win = train_df['home_win'].values
        y_cover = train_df['home_cover'].values
        y_spread = (train_df['result'] + train_df['spread_line']).values  # Actual vs spread
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. XGBoost for spread prediction (regression)
        print("  Training XGBoost spread model...")
        self.xgb_spread = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_spread.fit(X_scaled, y_spread)
        
        # 2. XGBoost for win probability
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
        
        # 3. Calibrated Logistic Regression
        print("  Training Logistic Regression...")
        base_lr = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_win = CalibratedClassifierCV(base_lr, cv=5, method='isotonic')
        self.lr_win.fit(X_scaled, y_win)
        
        self.is_fitted = True
        print("Models fitted successfully!")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for games."""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        preds = df[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
        
        # Elo baseline
        preds['elo_prob'] = df['elo_prob'].values
        
        # XGBoost spread prediction
        preds['xgb_spread_pred'] = self.xgb_spread.predict(X_scaled)
        
        # XGBoost win probability
        preds['xgb_win_prob'] = self.xgb_win.predict_proba(X_scaled)[:, 1]
        
        # Logistic Regression win probability
        preds['lr_win_prob'] = self.lr_win.predict_proba(X_scaled)[:, 1]
        
        # Ensemble (simple average)
        preds['ensemble_prob'] = (
            preds['elo_prob'] * 0.2 +
            preds['xgb_win_prob'] * 0.4 +
            preds['lr_win_prob'] * 0.4
        )
        
        return preds
    
    def evaluate(self, df: pd.DataFrame, preds: pd.DataFrame) -> Dict:
        """Evaluate model performance."""
        y_true = df['home_win'].values
        
        results = {}
        
        for prob_col in ['elo_prob', 'xgb_win_prob', 'lr_win_prob', 'ensemble_prob']:
            probs = preds[prob_col].values
            preds_binary = (probs > 0.5).astype(int)
            
            results[prob_col] = {
                'accuracy': accuracy_score(y_true, preds_binary),
                'brier_score': brier_score_loss(y_true, probs),
                'log_loss': log_loss(y_true, probs),
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
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (2024 Season)")
    print("=" * 60)
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

