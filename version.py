"""
NFL Betting Model Version Information

This file tracks the model version and baseline metrics for comparison.
"""

# Current version
VERSION = "0.2.0"
VERSION_NAME = "Tuned Hybrid"
VERSION_DATE = "2024-12-18"

# Model description
DESCRIPTION = """
Tuned hybrid model with TIER S+A features:
- 35 features including Elo, venue, weather, passing (CPOE, pressure),
  injuries (impact, QB out), and rushing/receiving (RYOE, separation)
- 3 XGBoost models with Optuna-tuned hyperparameters (50 trials each)
- Hybrid approach: Tuned params for Spread/Totals, Baseline for Moneyline
- Training: 2018-2023 (1,696 games) - limited by PFR pressure data availability
- Testing: 2024 (291 games)
- Validation: 2025 (276 games through Week 16)

Changes from v0.1.0:
- Spread: +5.1% ROI improvement (6.3% → 11.4%)
- Totals: +25.4% ROI improvement (-19.6% → +5.8%)
- Moneyline: Kept baseline params (best performer)
"""

# Tuned hyperparameters per model (from Optuna optimization)
TUNED_PARAMS = {
    "spread": {
        "n_estimators": 243,
        "max_depth": 3,
        "learning_rate": 0.0207,
        "min_child_weight": 4,
        "subsample": 0.952,
        "colsample_bytree": 0.653,
        "gamma": 0.032,
        "reg_alpha": 0.861,
        "reg_lambda": 0.841,
    },
    "totals": {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.0332,
        "min_child_weight": 2,
        "subsample": 0.840,
        "colsample_bytree": 0.779,
        "gamma": 0.076,
        "reg_alpha": 0.563,
        "reg_lambda": 0.591,
    },
    "moneyline": {  # Keep baseline - performed best
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
    },
}

# Baseline metrics for comparison (from 2025 validation)
BASELINE_METRICS = {
    "version": VERSION,
    "train_years": "2018-2023",
    "train_size": 1696,
    "features_count": 35,

    # v0.1.0 Baseline Results (for comparison)
    "v0.1.0_baseline": {
        "spread_roi": 6.3,
        "totals_roi": -19.6,
        "ml_roi": 53.0,
        "win_accuracy": 0.645,
    },

    # 2024 Test Results (v0.2.0)
    "2024_test": {
        "games": 291,
        "spread_wr": 0.478,
        "spread_roi": -8.8,
        "totals_wr": 0.509,
        "totals_roi": -2.9,
        "ml_wr": 0.435,
        "ml_roi": 34.7,
        "win_accuracy": 0.687,
    },

    # 2025 Validation Results (v0.2.0)
    "2025_validation": {
        "games": 276,
        "spread_wr": 0.583,
        "spread_roi": 11.4,
        "totals_wr": 0.554,
        "totals_roi": 5.8,
        "ml_wr": 0.493,
        "ml_roi": 53.0,
        "win_accuracy": 0.645,
    },
}

# Feature list
FEATURES = [
    # Base/Elo (11)
    'elo_diff', 'elo_prob', 'spread_line', 'total_line', 'rest_advantage',
    'div_game', 'is_dome', 'is_cold', 'is_windy', 'home_implied_prob', 'away_implied_prob',
    
    # Venue/Weather (5)
    'is_primetime', 'is_grass', 'bad_weather', 'home_short_week', 'away_short_week',
    
    # Passing - TIER S (8)
    'home_cpoe_3wk', 'away_cpoe_3wk', 'cpoe_diff',
    'home_pressure_rate_3wk', 'away_pressure_rate_3wk', 'pressure_diff',
    'home_time_to_throw_3wk', 'away_time_to_throw_3wk',
    
    # Injuries - TIER S (5)
    'home_injury_impact', 'away_injury_impact', 'injury_diff',
    'home_qb_out', 'away_qb_out',
    
    # Rush/Receiving - TIER A (6)
    'home_ryoe_3wk', 'away_ryoe_3wk', 'ryoe_diff',
    'home_separation_3wk', 'away_separation_3wk', 'separation_diff',
]

# Top feature importances (from XGBoost)
TOP_FEATURES = {
    "spread": [
        ('home_implied_prob', 0.207),
        ('away_implied_prob', 0.086),
        ('spread_line', 0.085),
        ('home_ryoe_3wk', 0.045),
        ('ryoe_diff', 0.037),
    ],
    "totals": [
        ('total_line', 0.254),
        ('home_implied_prob', 0.068),
        ('away_implied_prob', 0.065),
        ('home_cpoe_3wk', 0.042),
        ('elo_diff', 0.039),
    ],
    "moneyline": [
        ('home_implied_prob', 0.198),
        ('away_implied_prob', 0.089),
        ('spread_line', 0.076),
        ('elo_prob', 0.058),
        ('elo_diff', 0.045),
    ],
}

