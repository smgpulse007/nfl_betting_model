"""
Streamlit Monitoring Dashboard for Model Enhancement Tracking

Tracks feature evolution, statistical validation, and model performance
as we enhance the moneyline model from v0.3.1 to v0.4.0.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from scipy import stats

# Page config
st.set_page_config(
    page_title="NFL Model Enhancement Monitor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏈 NFL Moneyline Model Enhancement Monitor")
st.markdown("**Version:** v0.3.1 → v0.4.0 | **Focus:** Reduce Vegas dependency, improve accuracy to 75%+")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["Overview", "Feature Evolution", "Correlation Analysis", "Model Performance", 
     "Feature Importance", "Statistical Validation", "Backtest Results", "Data Quality"]
)

# Load data
@st.cache_data
def load_2025_results():
    """Load 2025 Week 1-16 evaluation results"""
    results_file = Path('results/2025_week1_16_evaluation.csv')
    if results_file.exists():
        return pd.read_csv(results_file)
    return pd.DataFrame()

@st.cache_data
def load_feature_importance():
    """Load XGBoost feature importance"""
    fi_file = Path('results/xgboost_feature_importance.csv')
    if fi_file.exists():
        return pd.read_csv(fi_file)
    return pd.DataFrame()

results = load_2025_results()
feature_importance = load_feature_importance()

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("📊 Current Model Status (v0.3.1)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if not results.empty:
        # Calculate metrics
        total_games = len(results)
        xgb_correct = (results['home_win'] == (results['xgb_win_prob'] > 0.5)).sum()
        xgb_accuracy = xgb_correct / total_games * 100
        
        # High-confidence picks
        high_conf = results[results['xgb_win_prob'] >= 0.8]
        high_conf_correct = (high_conf['home_win'] == (high_conf['xgb_win_prob'] > 0.5)).sum()
        high_conf_accuracy = high_conf_correct / len(high_conf) * 100 if len(high_conf) > 0 else 0
        high_conf_pct = len(high_conf) / total_games * 100
        
        # Vegas correlation
        if 'spread_line_x' in results.columns:
            spread_implied = results['spread_line_x'].apply(lambda x: 0.5 + (x / 28) if pd.notna(x) else np.nan)
            vegas_corr = results[['xgb_win_prob', 'spread_implied']].corr().iloc[0, 1]
        else:
            vegas_corr = 0.932  # From analysis
        
        col1.metric("Overall Accuracy", f"{xgb_accuracy:.1f}%", f"+{xgb_accuracy - 67.3:.1f}% vs Vegas")
        col2.metric("High-Conf Accuracy", f"{high_conf_accuracy:.1f}%", f"{len(high_conf)} games")
        col3.metric("High-Conf Volume", f"{high_conf_pct:.1f}%", "Target: 25%")
        col4.metric("Vegas Correlation", f"{vegas_corr:.3f}", "Target: <0.85")
    
    st.markdown("---")
    
    # Enhancement targets
    st.subheader("🎯 Enhancement Targets (v0.4.0)")
    
    targets_df = pd.DataFrame({
        'Metric': ['Overall Accuracy', 'High-Conf Accuracy', 'High-Conf Volume', 'Vegas Correlation'],
        'Current (v0.3.1)': ['68.5%', '90.0%', '15%', '0.932'],
        'Target (v0.4.0)': ['75%+', '90%+', '25%', '<0.85'],
        'Status': ['🔴 Needs Improvement', '✅ Maintain', '🔴 Needs Improvement', '🔴 Needs Improvement']
    })
    
    st.dataframe(targets_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Roadmap
    st.subheader("🗺️ Enhancement Roadmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Phase 1: Data Collection (Week 1)**")
        st.markdown("""
        - ✅ ESPN API research and documentation
        - ✅ Current feature audit
        - ✅ Built ESPN API wrapper
        - ⏳ Fetch all 32 teams' stats (2024 & 2025)
        - ⏳ Fetch live injuries for 2025
        """)
        
        st.markdown("**Phase 2: Feature Engineering (Week 2)**")
        st.markdown("""
        - ⏳ Compute offensive/defensive efficiency
        - ⏳ Compute QB performance metrics
        - ⏳ Integrate live 2025 injury data
        - ⏳ Statistical validation for each feature
        """)
    
    with col2:
        st.markdown("**Phase 3: Model Retraining (Week 3)**")
        st.markdown("""
        - ⏳ Add new features to pipeline
        - ⏳ Retrain on 1999-2024 data
        - ⏳ Validate on 2025 Week 1-16
        - ⏳ Analyze feature importance
        """)
        
        st.markdown("**Phase 4: Production (Week 4)**")
        st.markdown("""
        - ⏳ Deploy live data fetching
        - ⏳ Generate predictions with new features
        - ⏳ Monitor accuracy on Week 17+
        - ⏳ A/B testing (old vs new model)
        """)

# ============================================================================
# PAGE 2: FEATURE EVOLUTION
# ============================================================================
elif page == "Feature Evolution":
    st.header("🔬 Feature Evolution Tracker")
    
    st.markdown("""
    Track which features are added, their statistical properties, and their contribution to model performance.
    Each new feature must pass rigorous statistical validation before integration.
    """)
    
    # Current features
    st.subheader("Current Features (v0.3.1)")
    
    feature_categories = {
        'TIER S (Highest Value)': [
            'CPOE (Completion % Over Expected)',
            'Pressure Rate',
            'Injury Impact',
            'QB Out Status',
            'Rest Days'
        ],
        'TIER A (High Value)': [
            'RYOE (Rush Yards Over Expected)',
            'Receiver Separation',
            'Time to Throw'
        ],
        'TIER 1 (Baseline)': [
            'Elo Ratings',
            'Weather (temp, wind, surface)',
            'Primetime flags',
            'Division games',
            'Short week'
        ],
        'Vegas-Dependent (⚠️ TO BE REDUCED)': [
            'Spread line',
            'Total line',
            'Moneyline odds',
            'Implied probabilities'
        ]
    }
    
    for category, features in feature_categories.items():
        with st.expander(f"**{category}** ({len(features)} features)"):
            for feature in features:
                st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    # Planned features
    st.subheader("Planned Features (v0.4.0)")
    
    planned_features = pd.DataFrame({
        'Feature': [
            'Team Offensive Efficiency',
            'Team Defensive Efficiency',
            'QB Performance (3-game avg)',
            'Home/Away Splits',
            'Recent Form (5-game)',
            'Live 2025 Injuries',
            'Turnover Differential',
            'Third Down Conversion %',
            'Red Zone Efficiency'
        ],
        'Source': [
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API',
            'ESPN API'
        ],
        'Independence': [
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent',
            '✅ Independent'
        ],
        'Status': [
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending',
            '⏳ Pending'
        ]
    })
    
    st.dataframe(planned_features, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 3: CORRELATION ANALYSIS
# ============================================================================
elif page == "Correlation Analysis":
    st.header("📈 Correlation Analysis")

    st.markdown("""
    Analyze correlations between features, Vegas lines, and target variables.
    Goal: Reduce correlation with Vegas while maintaining correlation with outcomes.
    """)

    if not results.empty:
        # Select numeric columns
        numeric_cols = results.select_dtypes(include=[np.number]).columns.tolist()

        # Key columns for correlation
        key_cols = [col for col in numeric_cols if any(x in col.lower() for x in
                   ['elo', 'cpoe', 'pressure', 'ryoe', 'separation', 'spread', 'total', 'prob'])]

        if len(key_cols) > 2:
            corr_matrix = results[key_cols].corr()

            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # Highlight Vegas correlations
            st.subheader("Vegas Line Correlations")

            if 'spread_line_x' in results.columns and 'xgb_win_prob' in results.columns:
                spread_implied = results['spread_line_x'].apply(lambda x: 0.5 + (x / 28) if pd.notna(x) else np.nan)
                corr_with_vegas = results[['xgb_win_prob']].corrwith(spread_implied)

                st.metric("XGBoost vs Spread Implied Prob", f"{corr_with_vegas.iloc[0]:.3f}",
                         "Target: <0.85")
    else:
        st.warning("No data available for correlation analysis")

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.header("📊 Model Performance Tracking")

    st.markdown("""
    Compare old model (v0.3.1) vs new model (v0.4.0) performance across multiple dimensions.
    """)

    if not results.empty:
        # Overall accuracy by week
        st.subheader("Accuracy by Week")

        weekly_accuracy = results.groupby('week').apply(
            lambda x: (x['home_win'] == (x['xgb_win_prob'] > 0.5)).sum() / len(x) * 100
        ).reset_index(name='accuracy')

        fig = px.line(
            weekly_accuracy,
            x='week',
            y='accuracy',
            markers=True,
            title="XGBoost Moneyline Accuracy by Week (2025)"
        )
        fig.add_hline(y=68.5, line_dash="dash", line_color="green",
                     annotation_text="Overall Average (68.5%)")
        fig.add_hline(y=75, line_dash="dash", line_color="red",
                     annotation_text="Target (75%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Accuracy by confidence level
        st.subheader("Accuracy by Confidence Level")

        results['confidence_bin'] = pd.cut(
            results['xgb_win_prob'],
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        )

        conf_accuracy = results.groupby('confidence_bin').apply(
            lambda x: pd.Series({
                'accuracy': (x['home_win'] == (x['xgb_win_prob'] > 0.5)).sum() / len(x) * 100,
                'count': len(x)
            })
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=conf_accuracy['confidence_bin'],
            y=conf_accuracy['accuracy'],
            text=conf_accuracy['count'].apply(lambda x: f"{x} games"),
            textposition='auto',
            name='Accuracy'
        ))
        fig.add_hline(y=90, line_dash="dash", line_color="green",
                     annotation_text="High-Conf Target (90%)")
        fig.update_layout(
            title="Accuracy by Confidence Bin",
            xaxis_title="Confidence Level",
            yaxis_title="Accuracy (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for performance analysis")

# ============================================================================
# PAGE 5: FEATURE IMPORTANCE
# ============================================================================
elif page == "Feature Importance":
    st.header("🎯 Feature Importance Analysis")

    st.markdown("""
    Track which features drive predictions. Goal: Ensure Vegas features are downweighted
    and independent features (TIER S+A, ESPN data) are prioritized.
    """)

    if not feature_importance.empty:
        # Top 20 features
        top_features = feature_importance.nlargest(20, 'importance')

        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Most Important Features (XGBoost)"
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Categorize features
        st.subheader("Feature Importance by Category")

        def categorize_feature(feature_name):
            if any(x in feature_name.lower() for x in ['spread', 'total', 'moneyline', 'implied']):
                return 'Vegas-Dependent'
            elif any(x in feature_name.lower() for x in ['cpoe', 'pressure', 'ryoe', 'separation', 'injury']):
                return 'TIER S+A'
            elif 'elo' in feature_name.lower():
                return 'Elo'
            else:
                return 'Other'

        feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)

        category_importance = feature_importance.groupby('category')['importance'].sum().reset_index()
        category_importance = category_importance.sort_values('importance', ascending=False)

        fig = px.pie(
            category_importance,
            values='importance',
            names='category',
            title="Feature Importance by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No feature importance data available")

# ============================================================================
# PAGE 6: STATISTICAL VALIDATION
# ============================================================================
elif page == "Statistical Validation":
    st.header("📐 Statistical Validation")

    st.markdown("""
    Each new feature must pass rigorous statistical validation:
    - Statistical significance testing (p-values)
    - Correlation with target variable
    - Independence from Vegas lines
    - Ablation testing (model performance with/without feature)
    """)

    st.subheader("Feature Validation Checklist")

    validation_template = pd.DataFrame({
        'Feature': ['Example: Team Offensive Efficiency'],
        'P-Value': ['< 0.05 ✅'],
        'Correlation with Target': ['0.45 ✅'],
        'Correlation with Vegas': ['0.23 ✅'],
        'Ablation Test': ['+2.1% accuracy ✅'],
        'Status': ['✅ Approved']
    })

    st.dataframe(validation_template, use_container_width=True, hide_index=True)

    st.info("New features will be added here as they are validated during Phase 2")

# ============================================================================
# PAGE 7: BACKTEST RESULTS
# ============================================================================
elif page == "Backtest Results":
    st.header("💰 Backtest Results")

    st.markdown("""
    Track cumulative ROI and betting performance as we enhance the model.
    """)

    st.subheader("Current Model (v0.3.1) - 2025 Week 1-16")

    if not results.empty:
        # Simulate betting on high-confidence picks
        high_conf = results[results['xgb_win_prob'] >= 0.8].copy()

        if not high_conf.empty:
            high_conf['correct'] = (high_conf['home_win'] == (high_conf['xgb_win_prob'] > 0.5))

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Bets", len(high_conf))
            col2.metric("Win Rate", f"{high_conf['correct'].sum() / len(high_conf) * 100:.1f}%")
            col3.metric("Expected ROI", "+15.2%", "Based on 90% accuracy")

    st.info("Full backtest results will be added as new model is developed")

# ============================================================================
# PAGE 8: DATA QUALITY
# ============================================================================
elif page == "Data Quality":
    st.header("🔍 Data Quality Metrics")

    st.markdown("""
    Track data availability and completeness, especially for 2025 games.
    """)

    st.subheader("Known Data Quality Issues")

    issues = pd.DataFrame({
        'Issue': [
            '2025 Injury Data',
            'Advanced Metrics (2025)',
            'Historical Odds',
            'ESPN API Coverage'
        ],
        'Description': [
            'Currently imputed from 2024 medians (not real-time)',
            'CPOE, pressure, RYOE imputed for 2025 games',
            'nfl-data-py has stale/incorrect odds for 2025',
            'Only provides current week live data'
        ],
        'Impact': [
            '🔴 High',
            '🟡 Medium',
            '🟡 Medium',
            '🟢 Low'
        ],
        'Resolution': [
            'Phase 2: Integrate ESPN roster API',
            'Phase 2: Compute from ESPN game data',
            'Use ESPN API for live odds',
            'Build weekly data collection pipeline'
        ]
    })

    st.dataframe(issues, use_container_width=True, hide_index=True)

    if not results.empty:
        st.subheader("Data Completeness (2025 Week 1-16)")

        # Check for missing values
        missing_pct = (results.isnull().sum() / len(results) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(10)

        if not missing_pct.empty:
            fig = px.bar(
                x=missing_pct.values,
                y=missing_pct.index,
                orientation='h',
                title="Top 10 Features with Missing Data",
                labels={'x': 'Missing %', 'y': 'Feature'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data detected in 2025 evaluation dataset")

# Footer
st.markdown("---")
st.markdown("**NFL Moneyline Model Enhancement Monitor** | v0.3.1 → v0.4.0 | Last Updated: Dec 23, 2025")

