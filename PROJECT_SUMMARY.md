# NFL Betting Model - Complete Project Summary

## 📋 Project Overview

**Repository**: https://github.com/smgpulse007/nfl_betting_model.git
**Local Path**: `C:\Users\shail\source\Rough\nfl_betting_model`
**Current Branch**: `main`
**Current Version**: v0.3.1 "ESPN API Integration"
**Date**: December 18, 2024

---

## 🏗️ Version History

### v0.1.0 - Foundation (Initial Release)

**Goal**: Build baseline NFL betting prediction system

**Data Pipeline**:
- Used `nfl-data-py` library to fetch NFL data
- Loaded schedules from 1999-2025 (7,263+ games)
- Implemented FiveThirtyEight-style Elo rating system
- Created XGBoost models for spread, totals, and moneyline predictions

**Features Implemented (35 total)**:
```python
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
```

**TIER S Features** (most predictive):
- CPOE (Completion Percentage Over Expected) - 3-week rolling average
- Pressure Rate - QB pressure percentage
- Time to Throw - average seconds before release
- Injury Impact - weighted injury severity score
- QB Out - binary flag if starting QB is injured

**TIER A Features**:
- RYOE (Rush Yards Over Expected) - 3-week rolling
- Separation - receiver separation from defenders

**v0.1.0 Results**:
| Metric | Value |
|--------|-------|
| Spread ROI | +6.3% |
| Totals ROI | -19.6% |
| ML ROI | +53.0% |
| Win Accuracy | 64.5% |

---

### v0.2.0 - Hyperparameter Tuning

**Goal**: Optimize XGBoost hyperparameters using Optuna

**Implementation**:
- Created `tune_hyperparameters.py` with 50 Optuna trials per model
- Hybrid approach: tuned params for Spread/Totals, baseline for Moneyline
- Created `feature/model-experiments` branch for experimentation

**Tuned Hyperparameters**:
```python
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
    "moneyline": {  # Kept baseline - performed best
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
    },
}
```

**v0.2.0 Results**:
| Metric | v0.1.0 | v0.2.0 | Change |
|--------|--------|--------|--------|
| Spread ROI | +6.3% | +11.4% | +5.1% |
| Totals ROI | -19.6% | +5.8% | +25.4% |
| ML ROI | +53.0% | +53.0% | 0% |

---

### v0.3.0 - Optimal Model Mix

**Goal**: Compare multiple ML architectures to find best model per prediction type

**Models Tested**:
- Linear: OLS, Ridge, Lasso (L1), ElasticNet
- Logistic: L1, L2, ElasticNet regularization
- Tree/Boosting: RandomForest, XGBoost, LightGBM, CatBoost

**Key Experiment File**: `model_experiments.py`

**Critical Findings**:

1. **Totals: Linear beats Boosting!**
   - OLS Linear Regression: +3.9% ROI
   - XGBoost: negative ROI
   - Reason: `total_line` has 0.28 correlation with actual total



**v0.3.0 Optimal Model Configuration**:
```python
OPTIMAL_MODELS = {
    "spread": {
        "model": "XGBoost",
        "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        "performance": {"wr": 0.570, "roi": 8.9}
    },
    "totals": {
        "model": "OLS LinearRegression",
        "params": {},
        "performance": {"wr": 0.544, "roi": 3.9}
    },
    "moneyline": {
        "model": "CatBoost",
        "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        "performance": {"accuracy": 0.689}
    },
}
```

---

### v0.3.1 - ESPN API Integration (Current)

**Goal**: Research alternative data sources after discovering models just follow Vegas

## 🚨 CRITICAL DISCOVERY: ALL MODELS FOLLOW VEGAS

**Week 16 2025 Analysis Results**:

| Model | Vegas Agreement | Correlation with implied_prob |
|-------|-----------------|-------------------------------|
| Logistic | 100% (16/16) | r = 0.980 |
| RandomForest | 100% (16/16) | r = 0.980 |
| CatBoost | 94% (15/16) | r = 0.977 |
| XGBoost | 81% (13/16) | r = 0.914 |

**Conclusion**: Models have NO independent predictive power. They're essentially replicating Vegas implied probabilities.

**Root Cause**: All features are either:
1. Directly market-derived (`spread_line`, `total_line`, `implied_prob`)
2. Already priced into Vegas lines (Elo, injuries, weather, etc.)

---

## 📊 ESPN Unofficial API Research

**Source**: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c

### Verified Working Endpoints (No Auth Required)

**Injuries** (Real-time team injuries):
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/injuries
```

**Depth Charts** (Roster changes):
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{YEAR}/teams/{TEAM_ID}/depthcharts
```

**Current Odds** (Spreads, totals, moneylines):
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/odds
```

**Odds Movement** (Line changes - sharp money indicator):
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/odds/{BET_PROVIDER_ID}/history/0/movement
```

**Player Game Logs**:
```
https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{ATHLETE_ID}/gamelog
```

**Team Statistics** (11 categories):
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{YEAR}/types/{SEASONTYPE}/teams/{TEAM_ID}/statistics
```

**Scoreboard** (Live scores):
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
```

### Team IDs (All 32 NFL teams):
```python
TEAMS = {
    'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
    'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
    'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
    'LA': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
    'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
    'TEN': 10, 'WAS': 28
}
```

### Betting Provider IDs:
- Caesars: 38
- Bet365: 2000
- DraftKings: 41
- Consensus: 1002

---

## 📁 Complete File Structure

```
nfl_betting_model/
├── catboost_info/           # CatBoost training logs
├── data/                    # Data storage
│   ├── 2025/
│   ├── processed/
│   └── raw/
├── rd/                      # Research & Development docs
│   ├── ALTERNATIVE_DATA_SOURCES.md
│   ├── FEATURE_INVENTORY.md
│   ├── INVESTIGATION_FINDINGS.md
│   ├── ROADMAP.md
│   └── *.py                 # Various analysis scripts
├── results/                 # Experiment results
│   ├── backtest_2024.json
│   ├── backtest_2024_all_types.json
│   ├── backtest_2025.json
│   ├── backtest_2025_all_types.json
│   ├── deep_analysis.json
│   ├── linear_experiments.json
│   ├── tier_sa_backtest_results.json
│   ├── tier_sa_predictions_*.parquet
│   ├── tier_sa_weekly_*.csv
│   ├── tree_boosting_experiments.json
│   ├── tuning_results.json
│   └── week16_2025_all_models.csv
├── src/                     # Core source code
│   ├── backtesting.py
│   ├── data_loader.py
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── tier1_features.py
│   └── tier_sa_features.py
├── tests/                   # Test files
│
├── backtest_2025.py         # 2025 backtest runner
├── backtest_2025_all.py     # All bet types backtest
├── config.py                # Configuration settings
├── dashboard.py             # Streamlit app (main UI)
├── deep_analysis.py         # Model behavior analysis
├── espn_api_explorer.py     # ESPN API endpoint tester
├── espn_data_fetcher.py     # Production ESPN data fetcher
├── model_experiments.py     # RF, LightGBM, CatBoost comparison
├── run_tier_sa_backtest.py  # Main backtest runner with TIER S+A
├── tune_hyperparameters.py  # Optuna optimization
├── version.py               # Version info and configs
└── week16_analysis.py       # Week 16 predictions + injuries
```

---

## 🔧 Key Functions and Classes

### `run_tier_sa_backtest.py`
```python
def load_and_prepare_data():
    """Returns (all_games_df, completed_games_df) with TIER S+A features"""
    # Loads schedules 1999-2025
    # Computes Elo ratings
    # Merges TIER S features (CPOE, pressure, injuries)
    # Merges TIER A features (RYOE, separation)
    return games_df, completed_games
```

### `espn_data_fetcher.py`
```python
class ESPNDataFetcher:
    """Fetch data from ESPN's unofficial API"""

    def get_team_injuries(self, team_abbr) -> list
    def get_all_injuries(self) -> pd.DataFrame
    def get_current_odds(self) -> pd.DataFrame
    def get_team_ats_record(self, team_abbr, year) -> dict
```

---

## 📈 Data Sources

### Currently Used:
| Source | Data Type | Years |
|--------|-----------|-------|
| `nfl-data-py` | Schedules, scores, lines | 1999-2025 |
| `nfl-data-py` | NGS Passing (CPOE, TTT) | 2016-2025 |
| `nfl-data-py` | NGS Rushing (RYOE) | 2016-2025 |
| `nfl-data-py` | NGS Receiving (Separation) | 2016-2025 |
| `nfl-data-py` | PFR Passing (Pressure) | 2016-2025 |
| `nfl-data-py` | Injuries | 2016-2025 |
| ESPN API | Real-time injuries | Live |
| ESPN API | Current odds | Live |

### Not Yet Used (Potential Edge):
| Source | Data Type | Potential Edge |
|--------|-----------|----------------|
| ESPN API | Line movement history | Sharp money indicators |
| ESPN API | Depth charts | Roster changes before Vegas adjusts |
| ESPN API | QBR weekly | Advanced QB metrics |
| The Odds API | Multi-book odds | Line shopping, arbitrage |
| Weather APIs | Real-time weather | Actual vs forecasted at kickoff |
| Action Network | Public betting % | Fade the public |
| Pro Football Reference | Historical stats | Advanced metrics |
| nflfastR | Play-by-play, EPA | Situation-specific analysis |

---

## 🎯 Identified Potential Edge Sources

1. **Real-Time Injury Timing** - Vegas sets lines early; late injury news may not be priced in
2. **Line Movement / Sharp Money** - Opening vs closing line differential
3. **Weather at Kickoff** - Actual vs forecasted conditions
4. **Coaching Tendencies** - 4th down aggressiveness, red zone play calling
5. **Rest/Travel Combinations** - Cross-country travel + short week
6. **Referee Tendencies** - Penalty rates per crew
7. **Public Betting Percentages** - Fade when >70% public on one side

---

## 📊 Week 16 2025 Sample Injury Data (from ESPN API)

```
Team Injury Summary:
  ARI : Out:2 | Quest:7  | Key: Xavier Weaver, Marvin Harrison Jr.
  KC  : Quest:8          | Key: Trent McDuffie, Rashee Rice
  PHI : Out:2 | Quest:1  | Key: Lane Johnson, Jalen Carter
  PIT : Quest:10         | Key: Zach Frazier, T.J. Watt
  SEA : Out:7 | Quest:3  | Key: Riq Woolen, Nick Emmanwori
```

---

## 🔬 Key Technical Learnings

### 1. L1 vs L2 Regularization
- **L1 (Lasso)**: Produces sparse models, sets coefficients to exactly zero
- **L2 (Ridge)**: Shrinks all coefficients but doesn't eliminate features
- **ElasticNet**: Combines both, good for correlated features

### 2. Why OLS Beats XGBoost for Totals
- `total_line` has 0.28 correlation with actual game total
- Vegas lines already capture most predictive information
- OLS exploits this simple linear relationship directly

### 3. Why CatBoost Beats Other Classifiers
- **Ordered boosting** reduces prediction shift (overfitting)
- Native handling of categorical features (dome, grass, primetime)
- Better default regularization for tabular data

### 4. The Fundamental Problem
- All "predictive" features are already priced into Vegas lines
- Market efficiency means public information = no edge
- Need features Vegas doesn't have or underweights

---

## 🚀 Recommended Next Steps for v0.4.0

### Experiment 1: Remove Market Features
Train models WITHOUT: `spread_line`, `total_line`, `home_implied_prob`, `away_implied_prob`
- If accuracy drops to ~50%, we have NO independent signal
- If accuracy stays >55%, we have genuine edge

### Experiment 2: Build Independent Feature Set
Use ESPN API data that Vegas may underweight:
- Real-time injury updates (timing advantage)
- Depth chart changes
- Line movement for sharp money signals

### Experiment 3: Track Line Movement
Build historical database of opening/closing lines and correlation with outcomes

### Experiment 4: Fade the Public
Scrape public betting percentages and test "fade when >70% public on one side"

---

## 🖥️ Streamlit Dashboard

**Running at**: http://localhost:8501

**Sections**:
1. 📊 Overview - Win rates, ROI, feature importance
2. 📈 Backtesting - Historical performance by week/season
3. 🎯 Predictions - Current week predictions
4. 🔬 Deep Analysis - Feature Correlations, Multicollinearity, OLS vs XGBoost, CatBoost Analysis

---

## 🛠️ How to Run

```bash
# Activate virtual environment
cd C:\Users\shail\source\Rough
.venv\Scripts\Activate.ps1

# Run dashboard
cd nfl_betting_model
streamlit run dashboard.py --server.headless true

# Run Week 16 analysis
python week16_analysis.py

# Test ESPN API
python espn_api_explorer.py

# Run model experiments
python model_experiments.py
```

---

## 📦 Dependencies

```
nfl-data-py, pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, streamlit, requests, tqdm
```

---

## 🏷️ Git Tags

| Tag | Description |
|-----|-------------|
| v0.1.0 | Foundation - Elo + XGBoost baseline |
| v0.2.0 | Hyperparameter tuning with Optuna |
| v0.3.0 | Optimal Mix - Best model per prediction type |
| v1.0 | Initial release (legacy) |
| v1.1 | Updates (legacy) |

**Note**: v0.3.1 has not been tagged yet - only committed to main.

---

## ⚠️ Known Issues

1. **PowerShell syntax**: Use `;` instead of `&&` for command chaining
2. **NaN handling**: Use training medians for imputation
3. **ESPN API**: Some endpoints return nested `$ref` that need additional fetching

---

**Last Updated**: December 18, 2024
