# NFL Betting Model

> A data-driven, empirically validated NFL betting prediction system built on 25+ years of historical data with advanced TIER S+A features and ESPN API integration.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Version](https://img.shields.io/badge/version-0.3.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Last Updated:** December 23, 2025
**Current Version:** v0.3.1 "ESPN API Integration & Research"

---

## 🏈 Overview

This model predicts NFL game outcomes using a combination of:
- **Elo ratings** (FiveThirtyEight-style)
- **TIER S+A features** (CPOE, Pressure Rate, RYOE, Separation, Injuries)
- **Machine learning** (XGBoost, Logistic Regression, CatBoost, RandomForest)
- **ESPN API integration** for live data and independent features

**Training Data:** 6,991 games from 1999-2024
**Validation Data:** 251 completed games from 2025 Week 1-16

### Current Capabilities (v0.3.1)

| Prediction Type | Status | Accuracy (2025) | Notes |
|----------------|--------|-----------------|-------|
| **Moneyline** | ✅ Production | **68.5%** (172/251) | **90% on high-confidence picks (80%+)** |
| **Spread (ATS)** | ⚠️ Needs Improvement | 52.2% (131/251) | Minimal edge over 50% baseline |
| **Totals (O/U)** | ⚠️ Needs Improvement | 50.6% (127/251) | Essentially random |

---

## 📊 Performance Results

### 2025 Season (Weeks 1-16) - **VALIDATED**

**Moneyline Performance:**
| Model | Overall Accuracy | High-Confidence (80%+) | Vegas Baseline |
|-------|------------------|------------------------|----------------|
| **XGBoost** | **68.5%** (172/251) | **90.0%** (34/38) | 67.3% (169/251) |
| Logistic Regression | 67.7% (170/251) | 88.2% (30/34) | - |

**Key Findings:**
- ✅ **XGBoost beats Vegas by +1.2%** overall
- ✅ **90% accuracy on high-confidence picks** (80%+ confidence)
- ✅ **High-confidence picks = 15% of games** (38/251)
- ⚠️ **Vegas correlation = 0.932** (too high - needs reduction)

**Spread & Totals Performance:**
| Bet Type | Accuracy | Baseline | Edge |
|----------|----------|----------|------|
| Spread (ATS) | 52.2% (131/251) | 50.0% | +2.2% (minimal) |
| Totals (O/U) | 50.6% (127/251) | 50.0% | +0.6% (random) |

**Conclusion:** Focus on **moneyline betting** with high-confidence picks. Spread and totals need improvement.

### Week 16 Validation (Dec 2025)

**XGBoost vs Vegas Head-to-Head:**
- **XGBoost:** 73.3% accuracy (11/15 games)
- **Vegas:** 60.0% accuracy (9/15 games)
- **Edge:** +13.3% overall, +16.7% on disagreements

**High-Confidence Picks (>80%):**
- **6/6 correct (100%)** - Validated the confidence calibration

---

## 🎯 Feature Inventory

### TIER S Features (Highest Value)
- **CPOE** (Completion % Over Expected) - NGS data 2016+
- **Pressure Rate** - PFR data 2018+
- **Injury Impact** - 2009-2024 (⚠️ 2025 imputed from medians)
- **QB Out Status** - Binary flag for backup QB starts
- **Rest Days** - Days since last game

### TIER A Features (High Value)
- **RYOE** (Rush Yards Over Expected) - NGS data 2016+
- **Receiver Separation** - NGS data 2016+
- **Time to Throw** - NGS data 2016+

### TIER 1 Features (Baseline)
- **Elo Ratings** - FiveThirtyEight-style team strength
- **Weather** - Temperature, wind, surface (grass/turf)
- **Primetime** - TNF, MNF, SNF flags
- **Division Games** - Rivalry matchups
- **Short Week** - Thursday games

### Vegas-Dependent Features (⚠️ 13 total - TO BE REDUCED)
- Spread lines, total lines, moneyline odds
- Implied probabilities from odds
- **Problem:** Model correlation with Vegas = 0.932 (too high)

### Missing Independent Features (ESPN API Can Provide)
- Team offensive/defensive stats (points/game, yards/game, red zone %)
- QB/RB/WR performance metrics (rating, completion %, yards/carry)
- Home/away splits (win %, point differential)
- Recent form (last 3-5 games performance)
- **Live 2025 injury data** (replace imputed values)
- Turnover differential, third down conversion %, sack rates

---

## 🔥 ESPN API Integration (v0.3.1)

### Tested & Working Endpoints

**1. Team Statistics** (279 stats per team!)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/teams/{id}/statistics
```
- Offensive: passing yards, rushing yards, points/game
- Defensive: yards allowed, points allowed, sacks
- Efficiency: red zone %, third down %, turnovers

**2. Team Record with Splits** (44 stats per team!)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/teams/{id}/record
```
- Home/away splits, division/conference record
- Win streaks, average points for/against

**3. Live Injuries & Roster**
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={id}
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/roster
```
- Real-time player status (active/out/questionable)
- Position, experience, depth chart

**Total Available:** **323 independent stats per team** (279 + 44)

---

## 🚀 Quick Start

```bash
# Clone and setup
cd nfl_betting_model
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run full 2025 evaluation (Weeks 1-16)
python analyze_2025_accuracy.py

# Audit current features
python audit_current_features.py

# Test ESPN API
python espn_independent_data.py

# Generate predictions for upcoming week
python predict_2025.py
```

---

## 🗺️ Enhancement Roadmap (v0.4.0 - Moneyline Focus)

### Phase 1: Data Collection (Week 1) - **IN PROGRESS**
- [x] ESPN API research and documentation
- [x] Current feature audit (identified 13 Vegas-dependent features)
- [x] Built ESPN API wrapper (`espn_independent_data.py`)
- [x] Tested team stats API (279 stats available)
- [ ] Fetch all 32 teams' stats for 2024 & 2025
- [ ] Fetch live injuries for 2025 Week 1-16
- [ ] Parse and structure data into features

### Phase 2: Feature Engineering (Week 2)
- [ ] Compute offensive/defensive efficiency metrics
- [ ] Compute QB performance rolling averages
- [ ] Integrate live 2025 injury data (replace imputed)
- [ ] Compute home/away splits and recent form
- [ ] Statistical validation for each new feature

### Phase 3: Model Retraining (Week 3)
- [ ] Add new features to TIER S+A pipeline
- [ ] Retrain XGBoost on 1999-2024 with new features
- [ ] Validate on 2025 Week 1-16 (target: 75%+ accuracy)
- [ ] Analyze feature importance (ensure Vegas features drop)
- [ ] Reduce Vegas correlation from 0.932 to <0.85

### Phase 4: Production Deployment (Week 4)
- [ ] Deploy live data fetching for current week
- [ ] Generate predictions with new features
- [ ] Monitor accuracy on Week 17+
- [ ] Create Streamlit dashboard for feature tracking

### Success Metrics
1. **Overall Accuracy:** 75%+ (vs current 68.5%)
2. **High-Confidence Accuracy:** 90%+ on 80%+ picks (maintain)
3. **High-Confidence Volume:** 25% of games (vs current 15%)
4. **Vegas Correlation:** <0.85 (vs current 0.932)

---

## ⚠️ Known Limitations & Data Quality Issues

### Data Gaps
1. **2025 Injury Data:** Currently imputed from 2024 medians (not real-time)
2. **Advanced Metrics (2025):** CPOE, pressure, RYOE imputed for 2025 games
3. **Historical Odds:** nfl-data-py has stale/incorrect odds for 2025
4. **ESPN API:** Only provides current week live data, not historical weeks

### Model Limitations
1. **Vegas Correlation:** 0.932 (too dependent on Vegas lines)
2. **Spread/Totals:** Minimal edge (52.2% and 50.6% accuracy)
3. **High-Confidence Volume:** Only 15% of games (need 25%+)
4. **Feature Independence:** 13 Vegas-dependent features need reduction

### Validation Findings
1. **Week 16 Anomaly:** 46.7% moneyline accuracy (worst week)
2. **Best Weeks:** 6, 8, 11, 12 (all above 75% accuracy)
3. **Confidence Calibration:** Works well (90% on 80%+ picks)
4. **Data Quality:** SF @ IND game had backwards odds (8-point swing)

---

## 📁 Project Structure

```
nfl_betting_model/
├── README.md                          # This file
├── config.py                          # Model parameters
├── requirements.txt                   # Python dependencies
├── version.py                         # Version tracking (v0.3.1)
│
├── src/                               # Core modules
│   ├── data_pipeline.py               # Data loading from nfl-data-py
│   ├── data_loader.py                 # Data preparation and splitting
│   ├── feature_engineering.py         # Elo rating system
│   ├── tier_sa_features.py            # TIER S+A feature computation
│   ├── models.py                      # XGBoost, LogReg, CatBoost, RF
│   └── backtesting.py                 # Kelly criterion betting simulation
│
├── data/
│   ├── processed/                     # Cleaned parquet files (1999-2024)
│   └── 2025/                          # Current season data
│
├── results/                           # Evaluation results
│   ├── 2025_week1_16_evaluation.csv   # Full 2025 predictions & results
│   ├── tier_sa_backtest_results.json  # Historical backtest
│   └── xgboost_feature_importance.csv # Feature importance analysis
│
├── espn_independent_data.py           # ESPN API wrapper (NEW)
├── audit_current_features.py          # Feature dependency analysis (NEW)
├── analyze_2025_accuracy.py           # 2025 evaluation script (NEW)
├── prepare_2025_full_dataset.py       # Full data pipeline (NEW)
│
├── ESPN_API_RESEARCH.md               # ESPN API documentation (NEW)
├── MONEYLINE_ENHANCEMENT_PLAN.md      # 4-week roadmap (NEW)
├── 2025_ACCURACY_REPORT.md            # Detailed accuracy analysis (NEW)
│
└── rd/                                # Research & development
    ├── FEATURE_INVENTORY.md
    ├── INVESTIGATION_FINDINGS.md
    └── ROADMAP.md
```

---

## 🧠 Methodology & Strategic Philosophy

### Core Principle: Independent Edge Discovery

**The goal is NOT to ignore Vegas lines, but to:**
1. **Identify genuine independent edges** that Vegas doesn't capture or underweights
2. **Leverage Vegas information strategically** - use as baseline/anchor, not as prediction feature
3. **Build feature-rich, scientifically-grounded dataset** with rigorous validation
4. **Scrutinize every feature addition** with statistical testing

### Feature Selection Criteria

Each new feature must be justified by:
- ✅ **Statistical significance testing** (p-values, correlation analysis)
- ✅ **Feature importance ranking** in the model
- ✅ **Backtesting on historical data** (1999-2024)
- ✅ **Ablation testing** (model performance with/without feature)
- ✅ **Independence from Vegas** (low correlation with betting lines)

### Elo Rating System

Based on FiveThirtyEight's NFL Elo with enhancements:

| Parameter | Value | Description |
|-----------|-------|-------------|
| K-factor | 20 | Rating adjustment speed |
| Home Advantage | 48 Elo points | ~2.8 point spread equivalent |
| Playoff Multiplier | 1.2x | Higher stakes adjustment |
| Mean Reversion | 33% | Preseason regression to 1505 |

### Model Architecture

**Moneyline (Primary Focus):**
- XGBoost (n_estimators=100, max_depth=4, learning_rate=0.1)
- Logistic Regression (calibrated with isotonic regression)
- Ensemble: 0.40 × XGBoost + 0.40 × LogReg + 0.20 × Elo

**Spread & Totals (Secondary):**
- XGBoost margin model (predicts point differential)
- XGBoost total model (predicts total points)
- Normal CDF for probability conversion

### Betting Strategy

- **Kelly Criterion** with 0.25 fractional sizing
- **Minimum Confidence**: 80% for high-confidence bets
- **Maximum Stake**: 5% of bankroll
- **Focus**: Moneyline only (spread/totals need improvement)

---

## 📚 Literature Foundation

| Source | Key Finding |
|--------|-------------|
| Walsh & Joshi (2024) | Calibration > Accuracy: 69.86% higher returns |
| Patel (2023) | XGBoost with Elo features: 58.5% ATS accuracy |
| FiveThirtyEight/nfelo | Market-aware Elo adjustments |
| Szalkowski & Nelson (2012) | Home underdogs beat spread 53.5% |

---

## ⚠️ Disclaimer

This model is for educational and research purposes only. Sports betting involves significant financial risk. Past performance does not guarantee future results. Always gamble responsibly.

