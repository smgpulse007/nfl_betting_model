"""
Data Loader Module
==================

Load and prepare data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from config import *

class NFLDataLoader:
    """Load and prepare NFL betting data for model training."""
    
    def __init__(self, use_selected_features=True):
        """
        Initialize data loader.
        
        Args:
            use_selected_features: If True, use only selected features from feature selection
        """
        self.use_selected_features = use_selected_features
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def load_data(self):
        """Load the game-level predictions dataset."""
        print(f"\n[DataLoader] Loading data from {DATA_PATH}...")
        df = pd.read_parquet(DATA_PATH)
        print(f"  ✅ Loaded {len(df):,} games")
        return df
    
    def prepare_features(self, df):
        """Prepare feature columns."""
        # Get all feature columns (exclude metadata)
        all_features = [c for c in df.columns if c not in METADATA_COLS]
        
        # Filter to numeric columns only
        numeric_features = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if self.use_selected_features:
            # Load selected features
            selected_features = load_selected_features()
            # Keep only features that exist in the dataset
            self.feature_cols = [f for f in selected_features if f in numeric_features]
            print(f"  ✅ Using {len(self.feature_cols)} selected features")
        else:
            self.feature_cols = numeric_features
            print(f"  ✅ Using all {len(self.feature_cols)} numeric features")
        
        return self.feature_cols
    
    def split_data(self, df):
        """Split data into train/val/test sets."""
        print(f"\n[DataLoader] Splitting data...")
        
        train_df = df[df['year'].isin(TRAIN_YEARS)].copy()
        val_df = df[df['year'] == VAL_YEAR].copy()
        test_df = df[df['year'] == TEST_YEAR].copy()
        
        print(f"  Train: {len(train_df):,} games ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
        print(f"  Val:   {len(val_df):,} games ({VAL_YEAR})")
        print(f"  Test:  {len(test_df):,} games ({TEST_YEAR})")
        
        return train_df, val_df, test_df
    
    def prepare_xy(self, df, fit_scaler=False):
        """
        Prepare X and y from dataframe.
        
        Args:
            df: DataFrame with features and target
            fit_scaler: If True, fit the scaler on this data
        
        Returns:
            X, y: Features and target
        """
        X = df[self.feature_cols].copy()
        y = df[TARGET].copy()
        
        # Fill missing values with 0 (XGBoost/LightGBM handle this natively, but good for NN)
        X = X.fillna(0)
        
        # Scale features for neural networks
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            print(f"  ✅ Fitted scaler on {len(X):,} samples")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X, X_scaled, y
    
    def load_and_prepare(self):
        """
        Load and prepare all data splits.
        
        Returns:
            Dictionary with train/val/test splits and metadata
        """
        # Load data
        df = self.load_data()
        
        # Prepare features
        self.prepare_features(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Prepare X and y
        print(f"\n[DataLoader] Preparing features...")
        X_train, X_train_scaled, y_train = self.prepare_xy(train_df, fit_scaler=True)
        X_val, X_val_scaled, y_val = self.prepare_xy(val_df, fit_scaler=False)
        X_test, X_test_scaled, y_test = self.prepare_xy(test_df, fit_scaler=False)
        
        print(f"  ✅ Train: {X_train.shape}")
        print(f"  ✅ Val:   {X_val.shape}")
        print(f"  ✅ Test:  {X_test.shape}")
        
        # Save scaler
        scaler_path = MODELS_DIR / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"\n  ✅ Saved scaler to {scaler_path}")
        
        return {
            'train': {
                'X': X_train,
                'X_scaled': X_train_scaled,
                'y': y_train,
                'df': train_df
            },
            'val': {
                'X': X_val,
                'X_scaled': X_val_scaled,
                'y': y_val,
                'df': val_df
            },
            'test': {
                'X': X_test,
                'X_scaled': X_test_scaled,
                'y': y_test,
                'df': test_df
            },
            'feature_cols': self.feature_cols,
            'scaler': self.scaler
        }

if __name__ == '__main__':
    # Test data loader
    loader = NFLDataLoader(use_selected_features=True)
    data = loader.load_and_prepare()
    print(f"\n✅ Data loading successful!")
    print(f"   Features: {len(data['feature_cols'])}")
    print(f"   Train samples: {len(data['train']['y'])}")
    print(f"   Val samples: {len(data['val']['y'])}")
    print(f"   Test samples: {len(data['test']['y'])}")

