# Source Modules

## Module Overview

### `data_pipeline.py`
Connects to `nfl-data-py` package to fetch NFL schedule and game data.
- Fetches seasons 1999-present
- Standardizes team names (e.g., OAK → LV, SD → LAC)
- Adds calculated fields (home_win, result)

### `data_loader.py`
Loads and prepares data for model training.
- Splits data into train/test sets
- Default: Train on 1999-2023, test on 2024

### `feature_engineering.py`
Implements Elo rating system based on FiveThirtyEight methodology.

**Key Classes:**
- `EloRating`: Core Elo calculator with NFL-specific parameters
- `compute_elo_features()`: Adds Elo columns to game dataframe

**Elo Parameters:**
```python
K_FACTOR = 20           # Rating adjustment speed
HOME_ADVANTAGE = 48     # Elo points for home team
INITIAL_RATING = 1505   # Starting Elo for all teams
MEAN_REVERSION = 0.33   # Preseason regression factor
```

### `models.py`
Machine learning models for game prediction.

**Class: `NFLBettingModels`**
- `fit(df)`: Train all models on historical data
- `predict(df)`: Generate win probabilities

**Models:**
1. **Elo Baseline**: Pure Elo-based win probability
2. **XGBoost Spread**: Predicts margin of victory
3. **XGBoost Win**: Binary classification
4. **Logistic Regression**: Calibrated probabilities
5. **Ensemble**: Weighted combination (20% Elo, 40% XGB, 40% LR)

### `backtesting.py`
Betting simulation engine with Kelly criterion sizing.

**Class: `Backtester`**
- `run_backtest(games, predictions, min_edge)`: Simulate betting

**Key Functions:**
- `implied_prob(odds)`: American odds → probability
- `american_to_decimal(odds)`: American → decimal odds
- `kelly_stake(prob, odds)`: Calculate optimal stake size

## Usage Example

```python
from src.data_loader import load_all_data
from src.feature_engineering import compute_elo_features
from src.models import NFLBettingModels
from src.backtesting import Backtester

# Load and prepare data
games = load_all_data()
games = compute_elo_features(games)

# Train models
models = NFLBettingModels()
models.fit(train_data)

# Generate predictions
predictions = models.predict(test_data)

# Backtest
backtester = Backtester()
report = backtester.run_backtest(test_data, predictions, min_edge=0.02)
print(f"ROI: {report['roi']:.1%}")
```

