"""
Phase 5D: Game-Level EDA Dashboard
Interactive visualizations for game-level feature analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


@st.cache_data
def load_game_level_data():
    """Load game-level dataset."""
    data_file = Path('results/game_level_features_1999_2024_complete.parquet')
    if data_file.exists():
        df = pd.read_parquet(data_file)
        df['year'] = df['game_id'].str[:4].astype(int)
        # Add win indicator
        if 'total_wins' in df.columns:
            df['win'] = df['total_wins']
        else:
            df['win'] = (df['total_pointsFor'] > df['total_pointsAgainst']).astype(int)
        return df
    return None


@st.cache_data
def load_game_level_eda_results():
    """Load pre-computed game-level EDA results."""
    results = {}
    
    # Summary statistics
    summary_file = Path('results/game_level_eda_summary_statistics.csv')
    if summary_file.exists():
        results['summary'] = pd.read_csv(summary_file)
    
    # Temporal trends
    temporal_file = Path('results/game_level_eda_temporal_trends.csv')
    if temporal_file.exists():
        results['temporal'] = pd.read_csv(temporal_file)
    
    # Correlation matrix
    corr_file = Path('results/game_level_eda_correlation_matrix.csv')
    if corr_file.exists():
        results['correlation'] = pd.read_csv(corr_file, index_col=0)
    
    # High correlations
    high_corr_file = Path('results/game_level_eda_high_correlations.csv')
    if high_corr_file.exists():
        results['high_corr'] = pd.read_csv(high_corr_file)
    
    # Predictive power
    pred_file = Path('results/game_level_eda_predictive_power.csv')
    if pred_file.exists():
        results['predictive'] = pd.read_csv(pred_file)
    
    # Home/away analysis
    home_away_file = Path('results/game_level_eda_home_away.csv')
    if home_away_file.exists():
        results['home_away'] = pd.read_csv(home_away_file)
    
    return results


def show_phase5d_game_level_eda():
    """Main Phase 5D Game-Level EDA dashboard section."""
    st.header("🎮 Phase 5D: Game-Level EDA (1999-2024)")
    
    # Load data
    df = load_game_level_data()
    eda_results = load_game_level_eda_results()
    
    if df is None:
        st.error("Game-level dataset not found. Run game-level derivation first.")
        return
    
    # Overview metrics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Team-Games", f"{len(df):,}")
    with col2:
        st.metric("Total Games", f"{len(df)//2:,}")
    with col3:
        st.metric("Years", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        st.metric("Features", len([c for c in df.columns if c not in ['team', 'game_id', 'year', 'win']]))
    with col5:
        st.metric("Seasons", df['year'].nunique())
    
    # Comparison with season-level
    st.info("""
    **🎯 Game-Level vs Season-Level:**
    - **16.4x more data** (13,564 vs 829 rows)
    - **Game-by-game granularity** (perfect for moneyline betting)
    - **Expected accuracy improvement:** +4-9%
    """)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary Statistics",
        "📈 Temporal Trends",
        "🔗 Correlations",
        "🎯 Predictive Power",
        "🏠 Home/Away Analysis",
        "💡 Insights & Next Steps"
    ])
    
    with tab1:
        show_summary_statistics(df, eda_results)
    
    with tab2:
        show_temporal_trends(df, eda_results)
    
    with tab3:
        show_correlations(df, eda_results)
    
    with tab4:
        show_predictive_power(df, eda_results)
    
    with tab5:
        show_home_away_analysis(df, eda_results)
    
    with tab6:
        show_insights_next_steps(eda_results)


def show_summary_statistics(df, eda_results):
    """Display summary statistics."""
    st.subheader("📊 Summary Statistics")
    
    if 'summary' not in eda_results:
        st.warning("Summary statistics not available. Run eda_comprehensive_analysis.py first.")
        return
    
    summary_df = eda_results['summary']
    
    # Feature selector
    feature = st.selectbox(
        "Select feature to analyze:",
        summary_df['feature'].tolist(),
        key='summary_feature'
    )
    
    # Get feature stats
    feature_stats = summary_df[summary_df['feature'] == feature].iloc[0]
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{feature_stats['mean']:.2f}")
        st.metric("Std Dev", f"{feature_stats['std']:.2f}")
    with col2:
        st.metric("Median", f"{feature_stats['median']:.2f}")
        st.metric("Min", f"{feature_stats['min']:.2f}")
    with col3:
        st.metric("Q25", f"{feature_stats['q25']:.2f}")
        st.metric("Q75", f"{feature_stats['q75']:.2f}")
    with col4:
        st.metric("Max", f"{feature_stats['max']:.2f}")
        st.metric("Skewness", f"{feature_stats['skewness']:.2f}")
    
    # Distribution plot
    fig = px.histogram(
        df, 
        x=feature,
        nbins=50,
        title=f"Distribution of {feature}",
        marginal="box"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Most skewed features
    st.subheader("Most Skewed Features")
    skewed = summary_df.nlargest(10, 'skewness')[['feature', 'skewness', 'mean', 'std']]
    st.dataframe(skewed, use_container_width=True)


def show_temporal_trends(df, eda_results):
    """Display temporal trends."""
    st.subheader("📈 Temporal Trends (1999-2024)")

    if 'temporal' not in eda_results:
        st.warning("Temporal trends not available. Run eda_comprehensive_analysis.py first.")
        return

    temporal_df = eda_results['temporal']

    # Top increasing/decreasing features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Increasing Features**")
        increasing = temporal_df.nlargest(10, 'pct_change')[['feature', 'pct_change', 'p_value']]
        st.dataframe(increasing, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Decreasing Features**")
        decreasing = temporal_df.nsmallest(10, 'pct_change')[['feature', 'pct_change', 'p_value']]
        st.dataframe(decreasing, use_container_width=True)

    # Feature selector for trend visualization
    feature = st.selectbox(
        "Select feature to visualize trend:",
        temporal_df['feature'].tolist(),
        key='temporal_feature'
    )

    # Plot trend
    yearly_means = df.groupby('year')[feature].mean().reset_index()

    fig = px.line(
        yearly_means,
        x='year',
        y=feature,
        title=f"Temporal Trend: {feature}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show stats
    feature_trend = temporal_df[temporal_df['feature'] == feature].iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("% Change (1999-2024)", f"{feature_trend['pct_change']:.1f}%")
    with col2:
        st.metric("R²", f"{feature_trend['r_squared']:.4f}")
    with col3:
        st.metric("P-value", f"{feature_trend['p_value']:.4f}")


def show_correlations(df, eda_results):
    """Display correlation analysis."""
    st.subheader("🔗 Correlation Analysis")

    if 'high_corr' not in eda_results:
        st.warning("Correlation data not available. Run eda_comprehensive_analysis.py first.")
        return

    high_corr_df = eda_results['high_corr']

    # Correlation threshold filter
    threshold = st.slider("Correlation threshold:", 0.7, 1.0, 0.8, 0.05)

    filtered_corr = high_corr_df[high_corr_df['abs_correlation'] >= threshold]

    st.markdown(f"**High Correlation Pairs (|r| >= {threshold}):** {len(filtered_corr)}")
    st.dataframe(filtered_corr.head(20), use_container_width=True)

    # Correlation heatmap
    if 'correlation' in eda_results:
        st.subheader("Correlation Heatmap (Top 30 Features)")
        corr_matrix = eda_results['correlation'].iloc[:30, :30]

        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_predictive_power(df, eda_results):
    """Display predictive power analysis."""
    st.subheader("🎯 Predictive Power (Correlation with Winning)")

    if 'predictive' not in eda_results:
        st.warning("Predictive power data not available. Run eda_comprehensive_analysis.py first.")
        return

    pred_df = eda_results['predictive']

    # Top predictive features
    st.markdown("**Top 20 Most Predictive Features**")
    top_pred = pred_df.head(20)[['feature', 'correlation', 'p_value', 'significant']]
    st.dataframe(top_pred, use_container_width=True)

    # Visualization
    fig = px.bar(
        pred_df.head(30),
        x='correlation',
        y='feature',
        orientation='h',
        title="Top 30 Features by Correlation with Winning",
        color='correlation',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(pred_df))
    with col2:
        st.metric("Significant (p<0.05)", pred_df['significant'].sum())
    with col3:
        st.metric("Mean |r|", f"{pred_df['abs_correlation'].mean():.4f}")


def show_home_away_analysis(df, eda_results):
    """Display home/away analysis."""
    st.subheader("🏠 Home vs Away Performance")

    if 'home_away' not in eda_results:
        st.warning("Home/away data not available. Run eda_comprehensive_analysis.py first.")
        return

    home_away_df = eda_results['home_away']

    # Show differences
    st.dataframe(home_away_df, use_container_width=True)

    # Visualization
    fig = px.bar(
        home_away_df,
        x='feature',
        y=['home_mean', 'away_mean'],
        barmode='group',
        title="Home vs Away Feature Means"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_insights_next_steps(eda_results):
    """Display insights and next steps."""
    st.subheader("💡 Key Insights & Next Steps")

    st.markdown("""
    ### 🎯 Key Findings

    **Data Quality:**
    - ✅ 13,564 team-games (16.4x more than season-level)
    - ✅ 100% completeness (zero missing values)
    - ✅ 177/191 features significantly correlated with winning

    **Temporal Trends:**
    - 170/191 features show significant trends over time
    - Game has evolved significantly since 1999
    - Passing game has increased dramatically

    **Predictive Power:**
    - Strong correlations with winning outcomes
    - Multiple feature categories contribute to predictions
    - Ready for feature engineering

    ### 🚀 Next Steps: Phase 6 - Feature Engineering

    **Phase 6A: TIER S+A Integration**
    - Integrate 32 TIER S+A features into game-level dataset
    - Validate feature quality and correlations

    **Phase 6B: Rolling Averages**
    - 3-game rolling averages (expected +3-5% accuracy)
    - 5-game rolling averages (expected +2-4% accuracy)
    - Season-to-date averages

    **Phase 6C: Streak Features**
    - Win/loss streak length (expected +2-3% accuracy)
    - Scoring streak features
    - Performance momentum indicators

    **Phase 6D: Opponent-Adjusted Metrics**
    - Opponent strength adjustments
    - Division game indicators
    - Rest advantage features

    ### 📊 Expected Final Dataset

    - **Base Features:** 191 (ESPN approved)
    - **TIER S+A Features:** 32
    - **Engineered Features:** ~200-300
    - **Total Features:** ~400-500
    - **Total Rows:** 13,564 team-games
    - **Ready for:** Model training and moneyline predictions
    """)

    # Download button for summary
    if 'predictive' in eda_results:
        csv = eda_results['predictive'].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictive Power Analysis",
            data=csv,
            file_name="game_level_predictive_power.csv",
            mime="text/csv"
        )

