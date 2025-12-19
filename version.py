"""
NFL Betting Model Version Information

This file tracks the model version and baseline metrics for comparison.
"""

# Current version
VERSION = "0.3.1"
VERSION_NAME = "ESPN API Integration"
VERSION_DATE = "2024-12-18"

# Model description
DESCRIPTION = """
v0.3.1 - ESPN API Integration for Independent Edge Research

CRITICAL FINDING: All models are highly correlated with Vegas (r=0.91-0.98)
- Logistic: 100% agreement with Vegas, r=0.980
- RandomForest: 100% agreement, r=0.980
- CatBoost: 94% agreement, r=0.977
- XGBoost: 81% agreement, r=0.914

This means we have NO independent edge - we're just following the market.

NEW IN v0.3.1:
- ESPN API data fetcher for real-time injuries, depth charts, odds
- Week 16 2025 analysis with injury data for all 32 teams
- Research documentation on alternative data sources
- Identified potential edge sources: injuries, line movement, weather

NEXT STEPS (v0.4.0):
- Build features WITHOUT market-derived data (spread_line, implied_prob)
- Track line movement for sharp money indicators
- Integrate real-time injury timing advantage
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

    # v0.2.0 Tuned Hybrid Results
    "v0.2.0_tuned": {
        "spread_roi": 11.4,
        "totals_roi": 5.8,
        "ml_roi": 53.0,
        "win_accuracy": 0.645,
    },

    # 2025 Validation Results (v0.3.0 - Optimal Mix)
    "2025_validation": {
        "games": 276,
        "spread_wr": 0.570,  # XGBoost baseline
        "spread_roi": 8.9,   # XGBoost baseline
        "totals_wr": 0.544,  # OLS
        "totals_roi": 3.9,   # OLS
        "ml_wr": 0.689,      # CatBoost
        "ml_roi": 0.0,       # N/A (classification)
        "win_accuracy": 0.689,  # CatBoost
    },

    # Week 15 2025 Results (most recent)
    "week15_2025": {
        "games": 16,
        "spread_wr": 0.50,
        "totals_wr": 0.625,
        "ml_accuracy": 0.75,
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

# Top feature importances by model type
TOP_FEATURES = {
    "spread": [  # XGBoost baseline
        ('home_implied_prob', 0.141),
        ('spread_line', 0.076),
        ('away_implied_prob', 0.053),
        ('home_ryoe_3wk', 0.037),
        ('is_dome', 0.036),
    ],
    "totals": [  # OLS Linear - coefficients (scaled)
        ('injury_diff', 5.407),
        ('home_implied_prob', 5.138),
        ('spread_line', 5.059),
        ('total_line', 4.553),
        ('pressure_diff', 4.306),
    ],
    "moneyline": [  # CatBoost
        ('away_ryoe_3wk', 11.495),
        ('home_ryoe_3wk', 8.463),
        ('home_implied_prob', 8.243),
        ('spread_line', 7.896),
        ('home_separation_3wk', 6.316),
    ],
}

# Optimal model configuration for v0.3.0
OPTIMAL_MODELS = {
    "spread": {
        "model": "XGBoost",
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
        "performance": {"wr": 0.570, "roi": 8.9}
    },
    "totals": {
        "model": "OLS LinearRegression",
        "params": {},  # No hyperparams
        "performance": {"wr": 0.544, "roi": 3.9}
    },
    "moneyline": {
        "model": "CatBoost",
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
        "performance": {"accuracy": 0.689}
    },
}

