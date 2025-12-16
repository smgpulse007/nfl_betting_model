"""
Configuration for NFL Betting Model
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ============================================================================
# DATA SOURCES
# ============================================================================

# NFL Data Py - Play-by-play data
NFL_DATA_YEARS = list(range(1999, 2025))  # 1999-2024

# Training/Test Split
TRAIN_YEARS = list(range(1999, 2024))  # 1999-2023 for training
TEST_YEARS = [2024]  # 2024 for testing

# ============================================================================
# API KEYS (loaded from .env file)
# ============================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# The Odds API Configuration
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "americanfootball_nfl"
ODDS_API_REGIONS = "us"  # us, uk, eu, au
ODDS_API_MARKETS = "h2h,spreads,totals"  # head-to-head, spreads, totals

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Elo Configuration (based on FiveThirtyEight/nfelo research)
ELO_CONFIG = {
    "initial_rating": 1505,
    "k_factor": 20,
    "home_advantage": 48,  # Elo points
    "travel_factor": 4,    # Points per 1000 miles
    "bye_advantage": 25,   # Points for rest
    "playoff_multiplier": 1.2,
    "mean_reversion": 0.33,  # Regress 1/3 toward mean each season
}

# Rolling Window Sizes (weeks)
ROLLING_WINDOWS = [4, 8, 16]

# Feature Engineering
FEATURE_CONFIG = {
    "use_epa": True,
    "use_elo": True,
    "use_weather": True,
    "use_injuries": True,
    "opponent_adjust": True,
}

# ============================================================================
# BETTING PARAMETERS
# ============================================================================

# Standard vig at -110 odds
STANDARD_VIG = -110
BREAKEVEN_PCT = 0.5238  # Need to win 52.38% at -110 to break even

# Kelly Criterion
KELLY_FRACTION = 0.25  # Use 1/4 Kelly for safety

# Minimum edge to bet
MIN_EDGE_THRESHOLD = 0.02  # 2% minimum edge

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    "initial_bankroll": 10000,
    "unit_size": 100,  # Flat betting unit
    "use_kelly": True,
    "max_bet_pct": 0.05,  # Max 5% of bankroll per bet
}

