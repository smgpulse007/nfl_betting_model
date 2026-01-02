# Phase 8D: Interactive Dashboard - README

## 🏈 NFL Betting Model Dashboard

An interactive Streamlit dashboard for exploring NFL betting model performance, feature analysis, and betting strategies.

---

## 📁 Files Created

### Main Dashboard
- **`task_8d1_dashboard_structure.py`** - Main entry point with navigation and home page

### Page Modules
- **`task_8d2_model_performance.py`** - Model performance metrics and visualizations
- **`task_8d3_feature_analysis.py`** - SHAP, permutation importance, and correlation analysis
- **`task_8d4_betting_simulator.py`** - Interactive betting strategy simulator

---

## 🚀 How to Run

### Option 1: Run from command line
```bash
cd nfl_betting_model/phase8_production_models
streamlit run task_8d1_dashboard_structure.py
```

### Option 2: Run with custom port
```bash
streamlit run task_8d1_dashboard_structure.py --server.port 8501
```

### Option 3: Run in browser mode
```bash
streamlit run task_8d1_dashboard_structure.py --server.headless false
```

---

## 📊 Dashboard Features

### 🏠 Home Page
- **Model Overview**: Best model, total games, features, models trained
- **Dataset Splits**: Train (1999-2019), Validation (2020-2023), Test (2024)
- **Performance Summary**: Accuracy, precision, recall, F1, ROC-AUC, PR-AUC for all models
- **Top 10 Features**: Most important features from SHAP analysis
- **Quick Statistics**: Calibration, cross-validation, feature reduction metrics
- **Navigation Guide**: Overview of all dashboard sections

### 📊 Model Performance Page
**Tab 1: Comprehensive Metrics**
- Metrics comparison table with color gradient
- Interactive metric selection and visualization
- Confusion matrices for all models
- ROC curves comparison
- Precision-Recall curves comparison

**Tab 2: Calibration Analysis**
- Brier scores and ECE comparison
- Calibration curves for all models
- Confidence level analysis (50%, 60%, 70%, 80%, 90%)
- Accuracy by confidence level charts

**Tab 3: Cross-Validation**
- 5-fold time series CV results
- Mean accuracy, std dev, min, max for each model
- CV results visualization with error bars
- Key insights: best model, most stable model

**Tab 4: Learning Curves**
- Learning curves for all models
- Train vs validation performance
- Observations on overfitting/underfitting
- Recommendations for training set size

### 🔍 Feature Analysis Page
**Tab 1: SHAP Analysis**
- Model selector for SHAP values
- Top 20 features table with importance scores
- SHAP summary plots (beeswarm plots)
- Global importance comparison across models
- SHAP dependence plots for top 3 features
- Force plots for example predictions (high conf home/away, uncertain)

**Tab 2: Permutation Importance**
- Model selector for permutation importance
- Top 20 features with mean and std dev
- Permutation importance visualization for all models
- SHAP vs Permutation comparison scatter plots
- Rank correlation analysis

**Tab 3: Feature Correlation**
- Summary metrics: total, redundant, recommended features
- Correlation heatmap (top 50 features)
- Highly correlated pairs visualization
- List of redundant features to remove
- Feature importance distribution
- Recommendations for feature reduction

### 💰 Betting Simulator Page
**Betting Strategies:**
1. **Kelly Criterion**
   - Optimal bet sizing based on edge
   - Kelly fraction adjustment (0.1 - 1.0)
   - Minimum edge threshold

2. **Fixed Stake**
   - Fixed percentage of bankroll per bet
   - Minimum confidence threshold
   - Simple and conservative

3. **Confidence Threshold**
   - Only bet when confidence exceeds threshold
   - Fixed stake size
   - Risk management focused

4. **Proportional Betting**
   - Bet size proportional to confidence
   - Maximum stake limit
   - Scales with conviction

**Configuration Options:**
- Model selection (XGBoost, LightGBM, CatBoost, RandomForest)
- Initial bankroll ($100 - $100,000)
- Strategy-specific parameters
- Odds settings (Fair, American -110, Custom)

**Simulation Results:**
- Final bankroll and total profit
- ROI (Return on Investment)
- Total bets placed and win rate
- Bankroll over time chart
- Bet history table (last 20 bets)
- Download full bet history as CSV

---

## 🎨 Dashboard Design

### Color Scheme
- **Primary**: Blue (#1f77b4)
- **Secondary**: Orange (#ff7f0e)
- **Success**: Green (#2ca02c)
- **Danger**: Red (#d62728)

### Layout
- **Wide layout** for maximum screen utilization
- **Sidebar navigation** for easy page switching
- **Tabbed interface** for organized content
- **Responsive columns** for metrics and visualizations

### Custom Styling
- Main headers with custom CSS
- Metric cards with colored borders
- Gradient backgrounds for tables
- Interactive charts with matplotlib/seaborn

---

## 📦 Dependencies

All dependencies are already installed in the virtual environment:
- `streamlit` (1.52.2)
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `PIL` (Pillow)
- `joblib`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit
```

### Port already in use
```bash
# Use a different port
streamlit run task_8d1_dashboard_structure.py --server.port 8502
```

### Images not loading
- Ensure all Phase 8B and 8C tasks have been run
- Check that visualization files exist in `../results/phase8_results/`

### Models not loading
- Ensure all Phase 8A tasks have been run
- Check that model files exist in `../models/`

---

## 📝 Notes

- Dashboard uses cached data loading for performance
- All visualizations are pre-generated from Phase 8B and 8C
- Betting simulator runs in real-time based on user configuration
- Data is loaded from parquet files for fast access

---

## 🚀 Next Steps

After running the dashboard:
1. Explore model performance across different metrics
2. Analyze feature importance using SHAP and permutation
3. Identify redundant features for model optimization
4. Test different betting strategies and compare results
5. Proceed to Phase 8E: 2025 Season Predictions

---

**Dashboard Status:** ✅ Complete and ready to run
**Estimated Load Time:** 2-3 seconds
**Browser Compatibility:** Chrome, Firefox, Safari, Edge

