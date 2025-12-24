"""
Phase 6: Feature Engineering Dashboard
Interactive visualizations for engineered features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


@st.cache_data
def load_engineered_data():
    """Load engineered dataset."""
    # Try to load the complete dataset with TIER S+A features first
    tier_sa_file = Path('results/game_level_features_complete_with_tier_sa.parquet')
    data_file = Path('results/game_level_features_with_opponents.parquet')

    if tier_sa_file.exists():
        df = pd.read_parquet(tier_sa_file)
    elif data_file.exists():
        df = pd.read_parquet(data_file)
    else:
        return None

    # Add win indicator if not present
    if 'win' not in df.columns:
        df['win'] = (df['total_pointsFor'] > df['total_pointsAgainst']).astype(int)
    return df


@st.cache_data
def load_phase6_eda_results():
    """Load Phase 6 EDA results."""
    results = {}
    
    # Predictive power
    pred_file = Path('results/phase6_predictive_power.csv')
    if pred_file.exists():
        results['predictive'] = pd.read_csv(pred_file)
    
    # Category stats
    cat_file = Path('results/phase6_category_stats.csv')
    if cat_file.exists():
        results['category_stats'] = pd.read_csv(cat_file, index_col=0)
    
    # Missing values
    missing_file = Path('results/phase6_missing_values.csv')
    if missing_file.exists():
        results['missing'] = pd.read_csv(missing_file)
    
    # Summary
    summary_file = Path('results/phase6_eda_summary.json')
    if summary_file.exists():
        with open(summary_file) as f:
            results['summary'] = json.load(f)
    
    # Top by category
    top_file = Path('results/phase6_top_by_category.json')
    if top_file.exists():
        with open(top_file) as f:
            results['top_by_category'] = json.load(f)
    
    return results


def show_phase6_feature_engineering():
    """Main Phase 6 Feature Engineering dashboard section."""
    st.header("🔧 Phase 6: Feature Engineering (1999-2024)")
    
    # Load data
    df = load_engineered_data()
    eda_results = load_phase6_eda_results()
    
    if df is None:
        st.error("Engineered dataset not found. Run Phase 6 feature engineering first.")
        return
    
    # Overview metrics
    st.subheader("📈 Feature Engineering Overview")
    
    if 'summary' in eda_results:
        summary = eda_results['summary']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Features", summary['total_features'])
        with col2:
            st.metric("Base Features", summary['feature_categories']['base'])
        with col3:
            st.metric("Rolling Features", summary['feature_categories']['rolling'])
        with col4:
            st.metric("Opponent Features", summary['feature_categories']['opponent'])
        with col5:
            st.metric("Differential Features", summary['feature_categories']['differential'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Significant Features", f"{summary['predictive_power']['significant_features']}/{summary['predictive_power']['total_analyzed']}")
        with col2:
            st.metric("Mean |r|", f"{summary['predictive_power']['mean_abs_correlation']:.4f}")
        with col3:
            pct_complete = summary['missing_values']['pct_complete']
            st.metric("Complete Features", f"{pct_complete:.1f}%")
    
    # Info box
    st.info("""
    **🎯 Feature Engineering Phases:**
    - **Phase 6B:** Rolling averages (3-game, 5-game, season-to-date) for top 50 features
    - **Phase 6C:** Streak features (win streaks, scoring streaks, momentum)
    - **Phase 6D:** Opponent-adjusted metrics (opponent features, differentials, matchups)
    - **Phase 6A:** TIER S+A features (pending - requires NGS/PFR data integration)
    """)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Categories",
        "🎯 Predictive Power",
        "🔍 Missing Values",
        "🏆 Top Features",
        "💡 Insights & Next Steps"
    ])
    
    with tab1:
        show_feature_categories(df, eda_results)
    
    with tab2:
        show_predictive_power(df, eda_results)
    
    with tab3:
        show_missing_values(df, eda_results)
    
    with tab4:
        show_top_features(df, eda_results)
    
    with tab5:
        show_insights_next_steps(eda_results)


def show_feature_categories(df, eda_results):
    """Display feature category breakdown."""
    st.subheader("📊 Feature Categories")
    
    if 'category_stats' in eda_results:
        cat_stats = eda_results['category_stats']
        
        st.markdown("**Category Statistics**")
        st.dataframe(cat_stats, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            cat_stats.reset_index(),
            x='category',
            y='mean_abs_r',
            title="Mean Absolute Correlation by Category",
            labels={'mean_abs_r': 'Mean |r|', 'category': 'Feature Category'},
            color='mean_abs_r',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature count by category
        if 'summary' in eda_results:
            cat_counts = eda_results['summary']['feature_categories']
            fig2 = px.pie(
                values=list(cat_counts.values()),
                names=list(cat_counts.keys()),
                title="Feature Distribution by Category"
            )
            st.plotly_chart(fig2, use_container_width=True)


def show_predictive_power(df, eda_results):
    """Display predictive power analysis."""
    st.subheader("🎯 Predictive Power")
    
    if 'predictive' not in eda_results:
        st.warning("Predictive power data not available.")
        return
    
    pred_df = eda_results['predictive']
    
    # Top features
    st.markdown("**Top 30 Most Predictive Features**")
    top_30 = pred_df.head(30)[['feature', 'category', 'correlation', 'p_value', 'significant']]
    st.dataframe(top_30, use_container_width=True)
    
    # Visualization
    fig = px.bar(
        pred_df.head(50),
        x='correlation',
        y='feature',
        orientation='h',
        title="Top 50 Features by Correlation with Winning",
        color='category',
        hover_data=['p_value']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    st.markdown("**Significant Features by Category**")
    cat_breakdown = pred_df.groupby('category')['significant'].agg(['sum', 'count'])
    cat_breakdown['pct'] = (cat_breakdown['sum'] / cat_breakdown['count'] * 100).round(1)
    st.dataframe(cat_breakdown, use_container_width=True)


def show_missing_values(df, eda_results):
    """Display missing value analysis."""
    st.subheader("🔍 Missing Values Analysis")
    
    if 'missing' not in eda_results:
        st.warning("Missing value data not available.")
        return
    
    missing_df = eda_results['missing']
    
    st.markdown(f"**Features with Missing Values:** {len(missing_df)}/{df.shape[1]}")
    
    # Show top features with missing values
    st.dataframe(missing_df.head(20), use_container_width=True)
    
    # Visualization
    fig = px.bar(
        missing_df.head(30),
        x='missing_pct',
        y='feature',
        orientation='h',
        title="Top 30 Features by Missing Value Percentage",
        labels={'missing_pct': 'Missing %'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_top_features(df, eda_results):
    """Display top features by category."""
    st.subheader("🏆 Top Features by Category")
    
    if 'top_by_category' not in eda_results:
        st.warning("Top features data not available.")
        return
    
    top_by_cat = eda_results['top_by_category']
    
    for category, features in top_by_cat.items():
        with st.expander(f"**{category.upper()}** - Top 10 Features"):
            df_cat = pd.DataFrame(features)
            st.dataframe(df_cat, use_container_width=True)


def show_insights_next_steps(eda_results):
    """Display insights and next steps."""
    st.subheader("💡 Key Insights & Next Steps")
    
    st.markdown("""
    ### 🎯 Key Findings
    
    **Feature Engineering Success:**
    - ✅ 564 total features (up from 191 base features)
    - ✅ 368/440 features significantly correlated with winning (83.6%)
    - ✅ Differential features show highest predictive power (mean |r| = 0.51)
    - ✅ Opponent features also highly predictive (mean |r| = 0.43)
    
    **Category Performance:**
    1. **Differential Features** (mean |r| = 0.51) - Best performers
    2. **Opponent Features** (mean |r| = 0.43) - Strong predictors
    3. **Base Features** (mean |r| = 0.22) - Solid foundation
    4. **Streak Features** (mean |r| = 0.14) - Moderate value
    5. **Rolling Features** (mean |r| = 0.12) - Contextual value
    
    **Data Quality:**
    - 185/564 features have some missing values
    - Most missing values are in rolling/opponent features (early season games)
    - 98.9% of features have <10% missing values
    
    ### 🚀 Next Steps
    
    **Phase 6A: TIER S+A Integration (Pending)**
    - Integrate NGS/PFR features (CPOE, pressure rate, RYOE, separation)
    - Expected: +14 high-value features
    - Availability: 2016-2024 only
    
    **Model Training:**
    - Feature selection (top 100-200 features)
    - Handle missing values (imputation or exclusion)
    - Train XGBoost/LightGBM models
    - Cross-validation on temporal splits
    
    **Expected Performance:**
    - Baseline accuracy: ~60% (Vegas lines)
    - With engineered features: ~65-70% (expected)
    - ROI improvement: +5-10% over baseline
    """)
    
    # Download button
    if 'predictive' in eda_results:
        csv = eda_results['predictive'].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictive Power Analysis",
            data=csv,
            file_name="phase6_predictive_power.csv",
            mime="text/csv"
        )

