"""
Task 8D.1: Dashboard Structure and Navigation

Create the main Streamlit dashboard with navigation and page structure.
This is the entry point for the interactive NFL betting model dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NFL Betting Model Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("# 🏈 NFL Betting Model")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Model Performance", "🔍 Feature Analysis", "💰 Betting Simulator", "📅 Weekly Performance", "🏈 2025 Actual Performance"],
    label_visibility="collapsed"
)

# Load data function (cached)
@st.cache_data
def load_data():
    """Load all necessary data for the dashboard"""
    data = {}
    
    # Load game data
    data['games'] = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
    
    # Load feature categorization
    with open('../results/phase8_results/feature_categorization.json', 'r') as f:
        data['feature_cat'] = json.load(f)
    
    # Load model results
    with open('../results/phase8_results/comprehensive_metrics.json', 'r') as f:
        data['metrics'] = json.load(f)
    
    with open('../results/phase8_results/calibration_results.json', 'r') as f:
        cal_data = json.load(f)
        data['calibration'] = cal_data.get('calibration_metrics', cal_data)
        data['confidence_analysis'] = cal_data.get('confidence_analysis', {})
    
    with open('../results/phase8_results/cross_validation_results.json', 'r') as f:
        data['cv_results'] = json.load(f)
    
    # Load feature importance
    with open('../results/phase8_results/shap_analysis/global_feature_importance.json', 'r') as f:
        data['shap_importance'] = json.load(f)
    
    with open('../results/phase8_results/permutation_importance/permutation_importance.json', 'r') as f:
        data['perm_importance'] = json.load(f)
    
    with open('../results/phase8_results/feature_correlation/redundancy_recommendations.json', 'r') as f:
        data['redundancy'] = json.load(f)
    
    return data

# Load models function (cached)
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {
        'XGBoost': joblib.load('../models/xgboost_tuned.pkl'),
        'LightGBM': joblib.load('../models/lightgbm_tuned.pkl'),
        'CatBoost': joblib.load('../models/catboost_tuned.pkl'),
        'RandomForest': joblib.load('../models/randomforest_tuned.pkl')
    }
    return models

# Load data
try:
    data = load_data()
    models = load_models()
    st.sidebar.success("✅ Data loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Display selected page
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏈 NFL Betting Model Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Version 0.4.0 - True Prediction")
    
    st.markdown("---")
    
    # Overview metrics
    st.markdown('<div class="sub-header">📊 Model Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Best Model",
            value="PyTorch NN / TabNet",
            delta="67.72% Accuracy"
        )
    
    with col2:
        st.metric(
            label="Total Games",
            value=f"{len(data['games']):,}",
            delta="1999-2024"
        )
    
    with col3:
        st.metric(
            label="Features",
            value="102",
            delta="Pre-game only"
        )
    
    with col4:
        st.metric(
            label="Models Trained",
            value="6",
            delta="Tree-based + Neural"
        )
    
    st.markdown("---")
    
    # Dataset split information
    st.markdown('<div class="sub-header">📅 Dataset Splits</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    train_games = len(data['games'][data['games']['season'] <= 2019])
    val_games = len(data['games'][(data['games']['season'] >= 2020) & (data['games']['season'] <= 2023)])
    test_games = len(data['games'][data['games']['season'] == 2024])
    
    with col1:
        st.info(f"**Training Set**\n\n1999-2019\n\n{train_games:,} games")
    
    with col2:
        st.info(f"**Validation Set**\n\n2020-2023\n\n{val_games:,} games")
    
    with col3:
        st.info(f"**Test Set**\n\n2024\n\n{test_games:,} games")

    st.markdown("---")

    # Model performance summary
    st.markdown('<div class="sub-header">🎯 Model Performance Summary (2024 Test Set)</div>', unsafe_allow_html=True)

    # Create performance table
    perf_data = []
    for model_name, metrics in data['metrics'].items():
        perf_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']*100:.2f}%",
            'Recall': f"{metrics['recall']*100:.2f}%",
            'F1-Score': f"{metrics['f1_score']*100:.2f}%",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'PR-AUC': f"{metrics['pr_auc']:.4f}"
        })

    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, width='stretch', hide_index=True)

    st.markdown("---")

    # Feature importance preview
    st.markdown('<div class="sub-header">⭐ Top 10 Most Important Features</div>', unsafe_allow_html=True)

    # Get top features from SHAP (using LightGBM as example)
    if 'LightGBM' in data['shap_importance']:
        top_features = pd.DataFrame(data['shap_importance']['LightGBM'][:10])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(
                top_features[['feature', 'importance']].rename(columns={
                    'feature': 'Feature',
                    'importance': 'SHAP Importance'
                }),
                width='stretch',
                hide_index=True
            )

        with col2:
            st.markdown("**Key Insights:**")
            st.markdown("- Point differential features dominate")
            st.markdown("- Trend features are highly predictive")
            st.markdown("- Rolling averages capture recent form")
            st.markdown("- Opponent-adjusted metrics matter")

    st.markdown("---")

    # Quick stats
    st.markdown('<div class="sub-header">📈 Quick Statistics</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Calibration**")
        best_brier = min([cal['brier_score'] for cal in data['calibration'].values()])
        st.metric("Best Brier Score", f"{best_brier:.4f}", "RandomForest")

    with col2:
        st.markdown("**Cross-Validation**")
        best_cv = max([cv['mean_accuracy'] for cv in data['cv_results'].values()])
        st.metric("Best CV Accuracy", f"{best_cv*100:.2f}%", "CatBoost")

    with col3:
        st.markdown("**Feature Reduction**")
        reduction_pct = data['redundancy']['redundant_features_count'] / data['redundancy']['total_features'] * 100
        st.metric("Redundant Features", f"{reduction_pct:.1f}%", f"{data['redundancy']['redundant_features_count']} features")

    st.markdown("---")

    # Navigation guide
    st.markdown('<div class="sub-header">🧭 Navigation Guide</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📊 Model Performance**
        - Comprehensive metrics comparison
        - Calibration analysis
        - Cross-validation results
        - Learning curves
        """)

        st.markdown("""
        **🔍 Feature Analysis**
        - SHAP value analysis
        - Permutation importance
        - Feature correlations
        - Redundancy recommendations
        """)

    with col2:
        st.markdown("""
        **💰 Betting Simulator**
        - Kelly Criterion strategy
        - Fixed stake strategy
        - Confidence threshold strategy
        - Performance comparison
        """)

        st.markdown("""
        **📝 About**
        - Model version: 0.4.0 "True Prediction"
        - Data leakage fixed
        - Pre-game features only
        - Realistic performance metrics
        """)

elif page == "📊 Model Performance":
    from task_8d2_model_performance import show_model_performance
    show_model_performance(data, models)

elif page == "🔍 Feature Analysis":
    from task_8d3_feature_analysis import show_feature_analysis
    show_feature_analysis(data)

elif page == "💰 Betting Simulator":
    from task_8d4_betting_simulator import show_betting_simulator
    show_betting_simulator(data, models)

elif page == "📅 Weekly Performance":
    from task_8d5_weekly_performance import show_weekly_performance
    show_weekly_performance()

elif page == "🏈 2025 Actual Performance":
    from task_8d6_2025_actual_performance import show_2025_performance
    show_2025_performance()
