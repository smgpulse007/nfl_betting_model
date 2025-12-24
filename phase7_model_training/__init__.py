"""
Phase 7: Model Training Module
================================

Comprehensive model training for NFL moneyline predictions using:
- Tree-based models: XGBoost, LightGBM, CatBoost, Random Forest
- Neural networks: PyTorch feedforward networks

Dataset: game_level_predictions_dataset.parquet (6,636 games × 1,160 features)
Target: home_win (binary classification)
Train/Val/Test: 1999-2022 / 2023 / 2024
"""

__version__ = "0.3.1"
__author__ = "NFL Betting Model v0.3.1 - ESPN API Integration"

