"""
NFL Betting Model Version Information

This file tracks the model version and baseline metrics for comparison.
"""

# Current version
VERSION = "0.1.0"
VERSION_NAME = "Baseline"
VERSION_DATE = "2024-12-17"

# Model description
DESCRIPTION = """
Baseline model with TIER S+A features:
- 35 features including Elo, venue, weather, passing (CPOE, pressure), 
  injuries (impact, QB out), and rushing/receiving (RYOE, separation)
- 3 XGBoost models: Spread (regressor), Totals (regressor), Moneyline (classifier)
- Training: 2018-2023 (1,696 games) - limited by PFR pressure data availability
- Testing: 2024 (291 games)
- Validation: 2025 (228 games through Week 15)
"""

# Baseline metrics for comparison (from 2025 validation)
BASELINE_METRICS = {
    "version": VERSION,
    "train_years": "2018-2023",
    "train_size": 1696,
    "features_count": 35,
    
    # 2024 Test Results
    "2024_test": {
        "games": 291,
        "spread_wr": 0.467,
        "spread_roi": -10.8,
        "totals_wr": 0.478,
        "totals_roi": -8.8,
        "ml_wr": 0.435,
        "ml_roi": 34.7,
        "win_accuracy": 0.629,
    },
    
    # 2025 Validation Results
    "2025_validation": {
        "games": 228,
        "spread_wr": 0.557,
        "spread_roi": 6.3,
        "totals_wr": 0.421,
        "totals_roi": -19.6,
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

