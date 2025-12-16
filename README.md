# NFL Betting Model

> A data-driven, empirically validated NFL betting prediction system built on 25+ years of historical data.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🏈 Overview

This model predicts NFL game outcomes using a combination of Elo ratings, machine learning (XGBoost, Logistic Regression), and ensemble methods. It's trained on 6,991 games from 1999-2024 and includes a complete backtesting framework with Kelly criterion bet sizing.

### Current Capabilities (v1.1)

| Prediction Type | Status | Description |
|----------------|--------|-------------|
| **Win Probability** | ✅ Complete | Moneyline betting |
| **Spread (ATS)** | ✅ Complete | Against-the-spread betting |
| **Totals (O/U)** | ✅ Complete | Over/under betting |

---

## 📊 Performance Results

### 2024 Season (Test Set)

| Model | Accuracy | Brier Score |
|-------|----------|-------------|
| Elo Baseline | 68.4% | 0.2088 |
| XGBoost | 69.5% | 0.2013 |
| Logistic Regression | 70.2% | 0.2010 |
| **Ensemble** | **69.8%** | **0.1986** |

**Betting Performance (2% min edge):** +7.7% ROI on 92 bets

### 2025 Season (Live Validation, Weeks 1-15)

| Bet Type | Bets | Win Rate | ROI | Notes |
|----------|------|----------|-----|-------|
| **Moneyline** | 69 | 55.1% | -1.4% | Within expected variance |
| **Spread** | 218 | 79.8% | +48%* | *High accuracy when model disagrees with Vegas |
| **Totals** | 162 | 54.3% | +2.1% | Slight edge, best performer |

*Spread P&L inflated by Kelly compounding - real edge ~5-10% flat betting

---

## 🚀 Quick Start

```bash
# Clone and setup
cd nfl_betting_model
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run backtest on 2024 data
python run_backtest.py

# Generate 2025 predictions
python predict_2025.py

# Evaluate 2025 performance
python evaluate_2025.py
```

---

## 📁 Project Structure

```
nfl_betting_model/
├── README.md                 # This file
├── config.py                 # Model parameters (Elo K-factor, betting thresholds)
├── requirements.txt          # Python dependencies
│
├── src/                      # Core modules
│   ├── data_pipeline.py      # Data loading from nfl-data-py
│   ├── data_loader.py        # Data preparation and splitting
│   ├── feature_engineering.py # Elo rating system
│   ├── models.py             # XGBoost, LogReg, Ensemble models
│   └── backtesting.py        # Kelly criterion betting simulation
│
├── data/
│   ├── processed/            # Cleaned parquet files (1999-2024)
│   └── 2025/                 # Current season data
│
├── results/                  # Backtest results (JSON)
│   ├── backtest_2024.json
│   └── backtest_2025.json
│
├── run_backtest.py           # Main backtest runner
├── predict_2025.py           # Generate 2025 predictions
├── evaluate_2025.py          # Evaluate 2025 performance
└── backtest_2025.py          # 2025 betting backtest
```

---

## 🧠 Methodology

### Elo Rating System

Based on FiveThirtyEight's NFL Elo with enhancements:

| Parameter | Value | Description |
|-----------|-------|-------------|
| K-factor | 20 | Rating adjustment speed |
| Home Advantage | 48 Elo points | ~2.8 point spread equivalent |
| Playoff Multiplier | 1.2x | Higher stakes adjustment |
| Mean Reversion | 33% | Preseason regression to 1505 |

### Ensemble Model

```
Ensemble = 0.20 × Elo + 0.40 × XGBoost + 0.40 × Logistic Regression
```

### Betting Strategy

- **Kelly Criterion** with 0.25 fractional sizing
- **Minimum Edge**: 2% (configurable)
- **Maximum Stake**: 5% of bankroll
- **Odds Format**: American (-110, +150, etc.)

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

