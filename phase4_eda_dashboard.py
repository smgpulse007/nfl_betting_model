"""
Phase 4: Historical EDA Dashboard Section (1999-2024)

Comprehensive exploratory data analysis visualizations for the historical dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json


@st.cache_data
def load_historical_data():
    """Load the complete historical dataset (1999-2024)."""
    data_file = Path('data/derived_features/espn_derived_1999_2024_complete.parquet')
    if data_file.exists():
        return pd.read_parquet(data_file)
    return None


@st.cache_data
def load_eda_results():
    """Load pre-computed EDA results."""
    results = {}
    
    # Summary statistics
    summary_file = Path('results/eda_summary_statistics.csv')
    if summary_file.exists():
        results['summary'] = pd.read_csv(summary_file, index_col=0)
    
    # Temporal trends
    temporal_file = Path('results/eda_temporal_trends.csv')
    if temporal_file.exists():
        results['temporal'] = pd.read_csv(temporal_file, index_col=0)
    
    # Correlation matrix
    corr_file = Path('results/eda_correlation_matrix.csv')
    if corr_file.exists():
        results['correlation'] = pd.read_csv(corr_file, index_col=0)
    
    # High correlations
    high_corr_file = Path('results/eda_high_correlations.csv')
    if high_corr_file.exists():
        results['high_corr'] = pd.read_csv(high_corr_file)
    
    # Predictive power
    pred_power_file = Path('results/eda_predictive_power.csv')
    if pred_power_file.exists():
        results['predictive_power'] = pd.read_csv(pred_power_file)
    
    # Metadata
    metadata_file = Path('results/eda_metadata.json')
    if metadata_file.exists():
        with open(metadata_file) as f:
            results['metadata'] = json.load(f)
    
    return results


def show_phase4_eda():
    """Main Phase 4 EDA dashboard section."""
    st.header("📊 Phase 4: Historical Data EDA (1999-2024)")
    
    # Load data
    df = load_historical_data()
    eda_results = load_eda_results()
    
    if df is None:
        st.error("Historical dataset not found. Run historical derivation first.")
        return
    
    # Overview metrics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Years", f"{df['year'].min()}-{df['year'].max()}")
    with col3:
        st.metric("Unique Teams", df['team'].nunique())
    with col4:
        st.metric("Features", len(df.columns) - 2)
    with col5:
        st.metric("Data Points", f"{len(df) * (len(df.columns)-2):,}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary Statistics",
        "📈 Temporal Trends",
        "🔗 Correlations",
        "🎯 Predictive Power",
        "🏈 Team Analysis",
        "💡 Insights & Recommendations"
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
        show_team_analysis(df)
    
    with tab6:
        show_insights_recommendations(eda_results)


def show_summary_statistics(df, eda_results):
    """Display summary statistics."""
    st.subheader("📊 Summary Statistics")
    
    if 'summary' not in eda_results:
        st.warning("Summary statistics not available. Run comprehensive_eda_analysis.py first.")
        return
    
    summary = eda_results['summary']
    
    # Feature selection
    feature_cols = [col for col in df.columns if col not in ['team', 'year']]
    selected_feature = st.selectbox("Select Feature", feature_cols, key='summary_feature')
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig = px.histogram(
                df, x=selected_feature,
                title=f"Distribution of {selected_feature}",
                nbins=50,
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics table
            if selected_feature in summary.index:
                stats = summary.loc[selected_feature]
                st.markdown("**Statistics:**")
                st.write(f"- Mean: {stats['mean']:.2f}")
                st.write(f"- Median: {stats['50%']:.2f}")
                st.write(f"- Std Dev: {stats['std']:.2f}")
                st.write(f"- Min: {stats['min']:.2f}")
                st.write(f"- Max: {stats['max']:.2f}")
                st.write(f"- Skewness: {stats['skewness']:.2f}")
                st.write(f"- Kurtosis: {stats['kurtosis']:.2f}")
    
    # Top skewed features
    st.markdown("### Most Skewed Features (|skewness| > 2)")
    highly_skewed = summary[abs(summary['skewness']) > 2].sort_values('skewness', ascending=False, key=abs)
    st.dataframe(highly_skewed[['mean', 'std', 'skewness', 'kurtosis']].head(20), use_container_width=True)


def show_temporal_trends(df, eda_results):
    """Display temporal trends analysis."""
    st.subheader("📈 Temporal Trends (1999-2024)")
    
    if 'temporal' not in eda_results:
        st.warning("Temporal trends not available. Run comprehensive_eda_analysis.py first.")
        return
    
    temporal = eda_results['temporal']
    
    # Top increasing/decreasing features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 Increasing Features")
        increasing = temporal.nlargest(10, 'pct_change_1999_2024')
        st.dataframe(increasing[['pct_change_1999_2024', 'r_squared', 'p_value']], use_container_width=True)
    
    with col2:
        st.markdown("### Top 10 Decreasing Features")
        decreasing = temporal.nsmallest(10, 'pct_change_1999_2024')
        st.dataframe(decreasing[['pct_change_1999_2024', 'r_squared', 'p_value']], use_container_width=True)
    
    # Feature selection for trend visualization
    feature_cols = [col for col in df.columns if col not in ['team', 'year']]
    selected_feature = st.selectbox("Select Feature for Trend Visualization", feature_cols, key='temporal_feature')
    
    if selected_feature:
        # Calculate yearly mean
        yearly_mean = df.groupby('year')[selected_feature].mean()
        
        # Create trend plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_mean.index,
            y=yearly_mean.values,
            mode='lines+markers',
            name=selected_feature
        ))
        
        fig.update_layout(
            title=f"Temporal Trend: {selected_feature} (1999-2024)",
            xaxis_title="Year",
            yaxis_title="Mean Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        if selected_feature in temporal.index:
            stats = temporal.loc[selected_feature]
            st.info(f"""
            **Trend Statistics:**
            - % Change (1999-2024): {stats['pct_change_1999_2024']:.1f}%
            - R²: {stats['r_squared']:.4f}
            - p-value: {stats['p_value']:.4f}
            - Significant: {'Yes' if stats['p_value'] < 0.05 else 'No'}
            """)


def show_correlations(df, eda_results):
    """Display correlation analysis."""
    st.subheader("🔗 Feature Correlations")

    if 'high_corr' not in eda_results:
        st.warning("Correlation data not available. Run comprehensive_eda_analysis.py first.")
        return

    high_corr = eda_results['high_corr']

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_corr = st.slider("Minimum |Correlation|", 0.0, 1.0, 0.90, 0.05)
    with col2:
        max_pairs = st.slider("Max Pairs to Display", 10, 100, 50, 10)

    # Filter and display
    filtered = high_corr[abs(high_corr['correlation']) >= min_corr].head(max_pairs)

    st.markdown(f"### Top {len(filtered)} Highly Correlated Feature Pairs (|r| >= {min_corr})")
    st.dataframe(filtered, use_container_width=True)

    # Correlation heatmap for selected features
    st.markdown("### Correlation Heatmap (Top 20 Features by Predictive Power)")

    if 'predictive_power' in eda_results and 'correlation' in eda_results:
        top_features = eda_results['predictive_power'].head(20)['feature'].tolist()

        # Get correlation matrix for top features
        corr_matrix = eda_results['correlation']
        top_corr = corr_matrix.loc[top_features, top_features]

        fig = px.imshow(
            top_corr,
            labels=dict(color="Correlation"),
            x=top_features,
            y=top_features,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Correlation Heatmap: Top 20 Predictive Features"
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)


def show_predictive_power(df, eda_results):
    """Display predictive power analysis."""
    st.subheader("🎯 Predictive Power Analysis")

    if 'predictive_power' not in eda_results:
        st.warning("Predictive power data not available. Run predictive_power_analysis.py first.")
        return

    pred_power = eda_results['predictive_power']

    # Top features by combined score
    st.markdown("### Top 30 Features by Predictive Power")
    st.markdown("*Combined score = 50% correlation with wins + 50% Random Forest importance*")

    top_30 = pred_power.head(30)

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_30['feature'],
        y=top_30['combined_score'],
        marker_color='steelblue',
        text=top_30['combined_score'].round(3),
        textposition='outside'
    ))

    fig.update_layout(
        title="Top 30 Features by Predictive Power",
        xaxis_title="Feature",
        yaxis_title="Combined Score",
        height=600,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown("### Detailed Metrics")
    display_cols = ['feature', 'win_corr', 'win_corr_pvalue', 'rf_importance', 'combined_score']
    st.dataframe(top_30[display_cols], use_container_width=True)

    # Scatter plot: correlation vs importance
    st.markdown("### Correlation vs Random Forest Importance")

    fig = px.scatter(
        pred_power,
        x='win_corr',
        y='rf_importance',
        hover_data=['feature'],
        title="Feature Correlation vs Random Forest Importance",
        labels={'win_corr': 'Correlation with Wins', 'rf_importance': 'RF Importance'}
    )

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[pred_power['win_corr'].min(), pred_power['win_corr'].max()],
        y=[pred_power['win_corr'].min(), pred_power['win_corr'].max()],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Equal Weight Line'
    ))

    st.plotly_chart(fig, use_container_width=True)


def show_team_analysis(df):
    """Display team-level analysis."""
    st.subheader("🏈 Team-Level Analysis")

    # Team selection
    teams = sorted(df['team'].unique())
    selected_team = st.selectbox("Select Team", teams)

    if selected_team:
        team_data = df[df['team'] == selected_team].sort_values('year')

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Seasons in Dataset", len(team_data))
            st.metric("Year Range", f"{team_data['year'].min()}-{team_data['year'].max()}")

        with col2:
            # Calculate win percentage if available
            if 'total_leagueWinPercent' in team_data.columns:
                avg_win_pct = team_data['total_leagueWinPercent'].mean()
                st.metric("Avg Win %", f"{avg_win_pct:.1%}")

        # Feature selection for team trend
        feature_cols = [col for col in df.columns if col not in ['team', 'year']]
        selected_feature = st.selectbox("Select Feature for Team Trend", feature_cols, key='team_feature')

        if selected_feature and selected_feature in team_data.columns:
            # Team trend vs league average
            league_avg = df.groupby('year')[selected_feature].mean()

            fig = go.Figure()

            # Team line
            fig.add_trace(go.Scatter(
                x=team_data['year'],
                y=team_data[selected_feature],
                mode='lines+markers',
                name=selected_team,
                line=dict(width=3)
            ))

            # League average line
            fig.add_trace(go.Scatter(
                x=league_avg.index,
                y=league_avg.values,
                mode='lines',
                name='League Average',
                line=dict(dash='dash', color='gray')
            ))

            fig.update_layout(
                title=f"{selected_team}: {selected_feature} vs League Average",
                xaxis_title="Year",
                yaxis_title=selected_feature,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)


def show_insights_recommendations(eda_results):
    """Display key insights and recommendations."""
    st.subheader("💡 Key Insights & Recommendations")

    # Load feature engineering recommendations
    fe_file = Path('results/feature_engineering_prioritization.csv')
    game_level_file = Path('results/game_level_recommendation.json')

    # Game-level recommendation
    if game_level_file.exists():
        with open(game_level_file) as f:
            game_rec = json.load(f)

        st.markdown("### 🎯 Game-Level vs Season-Level Analysis")
        st.success(f"**RECOMMENDATION: {game_rec['recommendation']}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Volume Increase", f"{game_rec['data_volume_increase']:.1f}x")
        with col2:
            st.metric("Expected Accuracy Gain", game_rec['expected_accuracy_improvement'])
        with col3:
            st.metric("Implementation Complexity", game_rec['implementation_complexity'])

        st.markdown("**Rationale:**")
        for reason in game_rec['rationale']:
            st.write(f"- {reason}")

    # Feature engineering priorities
    if fe_file.exists():
        fe_df = pd.read_csv(fe_file)

        st.markdown("### 🔧 Feature Engineering Priorities")

        # Phase 1 (Immediate)
        phase1 = fe_df[fe_df['phase'] == 1].sort_values('priority_score', ascending=False)

        st.markdown("#### Phase 1: Immediate Implementation")
        for _, row in phase1.iterrows():
            with st.expander(f"**{row['feature_name']}** (Priority: {row['priority_score']})"):
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Expected Impact:** {row['expected_impact']}")
                st.write(f"**Implementation Complexity:** {row['implementation_complexity']}")
                st.write(f"**Data Requirements:** {row['data_requirements']}")
                st.write(f"**Rationale:** {row['rationale']}")

        # Summary table
        st.markdown("#### All Feature Engineering Opportunities")
        display_cols = ['feature_name', 'category', 'expected_impact', 'implementation_complexity', 'priority_score', 'phase']
        st.dataframe(fe_df[display_cols], use_container_width=True)

    # Download section
    st.markdown("### 📥 Download Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'summary' in eda_results:
            csv = eda_results['summary'].to_csv()
            st.download_button(
                label="📊 Summary Statistics (CSV)",
                data=csv,
                file_name="eda_summary_statistics.csv",
                mime="text/csv"
            )

    with col2:
        if 'predictive_power' in eda_results:
            csv = eda_results['predictive_power'].to_csv(index=False)
            st.download_button(
                label="🎯 Predictive Power (CSV)",
                data=csv,
                file_name="eda_predictive_power.csv",
                mime="text/csv"
            )

    with col3:
        if fe_file.exists():
            csv = fe_df.to_csv(index=False)
            st.download_button(
                label="🔧 Feature Engineering (CSV)",
                data=csv,
                file_name="feature_engineering_prioritization.csv",
                mime="text/csv"
            )


