"""
NFL Betting Model Dashboard
============================
Streamlit dashboard for visualizing model performance and backtest results.

Sections:
1. Dataset Overview
2. Feature Profiling
3. Data Quality Metrics
4. Model Performance Across Iterations
5. 2025 Validation Results
6. Backtest Results Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="NFL Betting Model Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")


@st.cache_data
def load_backtest_results():
    """Load backtest results from JSON."""
    results_file = RESULTS_DIR / "tier_sa_backtest_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


@st.cache_data
def load_predictions(year: int):
    """Load prediction data for a given year."""
    pred_file = RESULTS_DIR / f"tier_sa_predictions_{year}.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)
    return None


@st.cache_data
def load_weekly_stats(year: int):
    """Load weekly stats for a given year."""
    weekly_file = RESULTS_DIR / f"tier_sa_weekly_{year}.csv"
    if weekly_file.exists():
        return pd.read_csv(weekly_file)
    return None


def main():
    st.title("🏈 NFL Betting Model Dashboard")
    st.markdown("**TIER S+A Feature Model - 2024 Test & 2025 Validation**")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["📊 Overview", "📈 Feature Profiling", "🎯 Model Performance", 
         "📅 2025 Validation", "💰 Backtest Results"]
    )
    
    # Load data
    results = load_backtest_results()
    pred_2024 = load_predictions(2024)
    pred_2025 = load_predictions(2025)
    weekly_2024 = load_weekly_stats(2024)
    weekly_2025 = load_weekly_stats(2025)
    
    if results is None:
        st.error("No backtest results found. Run `python run_tier_sa_backtest.py` first.")
        return
    
    # Route to page
    if page == "📊 Overview":
        show_overview(results, pred_2024, pred_2025)
    elif page == "📈 Feature Profiling":
        show_feature_profiling(results, pred_2024)
    elif page == "🎯 Model Performance":
        show_model_performance(results, weekly_2024, weekly_2025)
    elif page == "📅 2025 Validation":
        show_2025_validation(pred_2025, weekly_2025)
    elif page == "💰 Backtest Results":
        show_backtest_results(pred_2024, pred_2025, weekly_2024, weekly_2025)


def show_overview(results, pred_2024, pred_2025):
    """Dataset Overview section."""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Version", "TIER S+A")
    with col2:
        st.metric("Features Used", len(results.get('features', [])))
    with col3:
        st.metric("2024 Games", len(pred_2024) if pred_2024 is not None else 0)
    with col4:
        st.metric("2025 Games", len(pred_2025) if pred_2025 is not None else 0)
    
    st.subheader("Key Results Summary")
    
    # Results table
    results_data = []
    if results.get('results_2024'):
        r = results['results_2024']
        results_data.append({
            'Dataset': '2024 Test',
            'Games': r['games'],
            'Spread WR': f"{r['spread_wr']*100:.1f}%",
            'Spread ROI': f"{r['spread_roi']:+.1f}%",
            'Totals WR': f"{r['totals_wr']*100:.1f}%",
            'Totals ROI': f"{r['totals_roi']:+.1f}%",
            'ML WR': f"{r['ml_wr']*100:.1f}%",
            'ML ROI': f"{r.get('ml_roi', 0):+.1f}%",
            'ML Bets': r['ml_bets'],
            'Win Accuracy': f"{r.get('win_pred_accuracy', 0)*100:.1f}%"
        })
    if results.get('results_2025'):
        r = results['results_2025']
        results_data.append({
            'Dataset': '2025 Validation',
            'Games': r['games'],
            'Spread WR': f"{r['spread_wr']*100:.1f}%",
            'Spread ROI': f"{r['spread_roi']:+.1f}%",
            'Totals WR': f"{r['totals_wr']*100:.1f}%",
            'Totals ROI': f"{r['totals_roi']:+.1f}%",
            'ML WR': f"{r['ml_wr']*100:.1f}%",
            'ML ROI': f"{r.get('ml_roi', 0):+.1f}%",
            'ML Bets': r['ml_bets'],
            'Win Accuracy': f"{r.get('win_pred_accuracy', 0)*100:.1f}%"
        })
    
    if results_data:
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Features list
    st.subheader("Features Used")
    features = results.get('features', [])
    cols = st.columns(3)
    for i, feat in enumerate(features):
        cols[i % 3].write(f"• {feat}")


def show_feature_profiling(results, pred_df):
    """Feature Profiling section."""
    st.header("📈 Feature Profiling")
    
    if pred_df is None:
        st.warning("No prediction data available")
        return
    
    features = results.get('features', [])
    numeric_features = [f for f in features if f in pred_df.columns]
    
    # Feature statistics
    st.subheader("Feature Statistics")
    stats = pred_df[numeric_features].describe().T
    st.dataframe(stats, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature", numeric_features)

    fig = px.histogram(pred_df, x=selected_feature, nbins=30,
                       title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig, use_container_width=True)

    # Missing values
    st.subheader("Data Quality - Missing Values")
    missing = pred_df[numeric_features].isnull().sum()
    missing_pct = (missing / len(pred_df) * 100).round(1)
    missing_df = pd.DataFrame({'Feature': missing.index, 'Missing Count': missing.values,
                               'Missing %': missing_pct.values})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values in features!")


def show_model_performance(results, weekly_2024, weekly_2025):
    """Model Performance section."""
    st.header("🎯 Model Performance")

    # Summary metrics at top
    st.subheader("Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Spread Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            st.metric("2024 Win Rate", f"{r['spread_wr']*100:.1f}%",
                     f"{r['spread_roi']:+.1f}% ROI")
        if results.get('results_2025'):
            r = results['results_2025']
            st.metric("2025 Win Rate", f"{r['spread_wr']*100:.1f}%",
                     f"{r['spread_roi']:+.1f}% ROI")

    with col2:
        st.markdown("### Totals Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            st.metric("2024 Win Rate", f"{r['totals_wr']*100:.1f}%",
                     f"{r['totals_roi']:+.1f}% ROI")
        if results.get('results_2025'):
            r = results['results_2025']
            st.metric("2025 Win Rate", f"{r['totals_wr']*100:.1f}%",
                     f"{r['totals_roi']:+.1f}% ROI")

    with col3:
        st.markdown("### Moneyline Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            ml_roi = r.get('ml_roi', 0)
            st.metric("2024 Win Rate", f"{r['ml_wr']*100:.1f}%",
                     f"{ml_roi:+.1f}% ROI ({r['ml_bets']} bets)")
        if results.get('results_2025'):
            r = results['results_2025']
            ml_roi = r.get('ml_roi', 0)
            st.metric("2025 Win Rate", f"{r['ml_wr']*100:.1f}%",
                     f"{ml_roi:+.1f}% ROI ({r['ml_bets']} bets)")

    # Win Prediction Accuracy
    st.subheader("Win Prediction Accuracy (All Games)")
    col1, col2 = st.columns(2)
    with col1:
        if results.get('results_2024'):
            acc = results['results_2024'].get('win_pred_accuracy', 0) * 100
            st.metric("2024 Accuracy", f"{acc:.1f}%",
                     f"{acc - 50:.1f}% vs coin flip")
    with col2:
        if results.get('results_2025'):
            acc = results['results_2025'].get('win_pred_accuracy', 0) * 100
            st.metric("2025 Accuracy", f"{acc:.1f}%",
                     f"{acc - 50:.1f}% vs coin flip")

    # Comparison chart
    st.subheader("Win Rate Comparison")

    comparison_data = {
        'Model': ['Baseline (Elo)', 'TIER S+A (2024)', 'TIER S+A (2025)'],
        'Spread WR': [49.4,
                      results['results_2024']['spread_wr']*100 if results.get('results_2024') else 0,
                      results['results_2025']['spread_wr']*100 if results.get('results_2025') else 0],
        'Totals WR': [50.0,
                      results['results_2024']['totals_wr']*100 if results.get('results_2024') else 0,
                      results['results_2025']['totals_wr']*100 if results.get('results_2025') else 0],
        'ML WR': [55.0,  # Baseline win prediction
                  results['results_2024']['ml_wr']*100 if results.get('results_2024') else 0,
                  results['results_2025']['ml_wr']*100 if results.get('results_2025') else 0]
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Spread WR', x=comparison_data['Model'], y=comparison_data['Spread WR']))
    fig.add_trace(go.Bar(name='Totals WR', x=comparison_data['Model'], y=comparison_data['Totals WR']))
    fig.add_trace(go.Bar(name='Moneyline WR', x=comparison_data['Model'], y=comparison_data['ML WR']))
    fig.add_hline(y=52.38, line_dash="dash", line_color="red", annotation_text="Spread/Totals Breakeven")
    fig.update_layout(barmode='group', title="Win Rate by Bet Type", yaxis_title="Win Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Weekly performance - now includes moneyline
    st.subheader("Weekly Performance Trend")

    col1, col2 = st.columns(2)

    with col1:
        if weekly_2024 is not None:
            st.markdown("**2024 Season**")
            # Include moneyline if column exists
            y_cols = ['spread_wr', 'totals_wr']
            if 'ml_wr' in weekly_2024.columns:
                y_cols.append('ml_wr')
            if 'win_accuracy' in weekly_2024.columns:
                y_cols.append('win_accuracy')
            fig = px.line(weekly_2024, x='week', y=y_cols,
                         title="2024 Weekly Win Rates",
                         labels={'value': 'Win Rate', 'week': 'Week'})
            fig.add_hline(y=0.5238, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if weekly_2025 is not None:
            st.markdown("**2025 Season**")
            y_cols = ['spread_wr', 'totals_wr']
            if 'ml_wr' in weekly_2025.columns:
                y_cols.append('ml_wr')
            if 'win_accuracy' in weekly_2025.columns:
                y_cols.append('win_accuracy')
            fig = px.line(weekly_2025, x='week', y=y_cols,
                         title="2025 Weekly Win Rates",
                         labels={'value': 'Win Rate', 'week': 'Week'})
            fig.add_hline(y=0.5238, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)


def show_2025_validation(pred_2025, weekly_2025):
    """2025 Validation section."""
    st.header("📅 2025 Validation Results")

    if pred_2025 is None:
        st.warning("No 2025 prediction data available")
        return

    # Week selector
    weeks = sorted(pred_2025['week'].unique())
    selected_week = st.selectbox("Select Week", weeks, index=len(weeks)-1)

    week_df = pred_2025[pred_2025['week'] == selected_week].copy()

    st.subheader(f"Week {selected_week} Games ({len(week_df)} games)")

    # Game-by-game table
    display_cols = ['away_team', 'home_team', 'away_score', 'home_score', 'result',
                    'spread_line', 'pred_margin', 'total_line', 'pred_total', 'game_total']

    week_df['spread_correct'] = (week_df['bet_home_cover'] == week_df['home_covered'])
    week_df['totals_correct'] = (week_df['bet_over'] == week_df['went_over'])

    # Format for display
    display_df = week_df[['away_team', 'home_team', 'away_score', 'home_score',
                          'spread_line', 'pred_margin', 'spread_correct',
                          'total_line', 'pred_total', 'game_total', 'totals_correct']].copy()
    display_df.columns = ['Away', 'Home', 'Away Score', 'Home Score',
                          'Spread Line', 'Pred Margin', 'Spread ✓',
                          'Total Line', 'Pred Total', 'Actual Total', 'Totals ✓']

    st.dataframe(display_df, use_container_width=True)

    # Week summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        spread_wr = week_df['spread_correct'].mean()
        st.metric("Spread WR", f"{spread_wr*100:.1f}%",
                  delta=f"{(spread_wr-0.5238)*100:+.1f}%" if spread_wr > 0.5238 else f"{(spread_wr-0.5238)*100:.1f}%")
    with col2:
        totals_wr = week_df['totals_correct'].mean()
        st.metric("Totals WR", f"{totals_wr*100:.1f}%",
                  delta=f"{(totals_wr-0.5238)*100:+.1f}%" if totals_wr > 0.5238 else f"{(totals_wr-0.5238)*100:.1f}%")
    with col3:
        st.metric("Spread Correct", f"{int(week_df['spread_correct'].sum())}/{len(week_df)}")
    with col4:
        st.metric("Totals Correct", f"{int(week_df['totals_correct'].sum())}/{len(week_df)}")


def show_backtest_results(pred_2024, pred_2025, weekly_2024, weekly_2025):
    """Backtest Results Visualization section."""
    st.header("💰 Backtest Results")

    # Cumulative P/L chart
    st.subheader("Cumulative Profit/Loss")

    dataset = st.radio("Select Dataset", ["2024", "2025"], horizontal=True)
    pred_df = pred_2024 if dataset == "2024" else pred_2025

    if pred_df is None:
        st.warning(f"No {dataset} data available")
        return

    # Calculate cumulative P/L (assuming $100 bets at -110)
    pred_df = pred_df.sort_values(['week', 'game_id']).copy()

    # Spread P/L
    pred_df['spread_pl'] = pred_df.apply(
        lambda x: 90.91 if x['bet_home_cover'] == x['home_covered'] else -100, axis=1
    )
    pred_df['spread_cumpl'] = pred_df['spread_pl'].cumsum()

    # Totals P/L
    pred_df['totals_pl'] = pred_df.apply(
        lambda x: 90.91 if x['bet_over'] == x['went_over'] else -100, axis=1
    )
    pred_df['totals_cumpl'] = pred_df['totals_pl'].cumsum()

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(pred_df))), y=pred_df['spread_cumpl'],
                             mode='lines', name='Spread Betting'))
    fig.add_trace(go.Scatter(x=list(range(len(pred_df))), y=pred_df['totals_cumpl'],
                             mode='lines', name='Totals Betting'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Cumulative P/L - {dataset} Season ($100 bets)",
                      xaxis_title="Game #", yaxis_title="Cumulative P/L ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Spread Final P/L", f"${pred_df['spread_cumpl'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Totals Final P/L", f"${pred_df['totals_cumpl'].iloc[-1]:,.0f}")

    # ROI by week
    st.subheader("ROI by Week")
    weekly_df = weekly_2024 if dataset == "2024" else weekly_2025

    if weekly_df is not None:
        weekly_df['spread_roi'] = (weekly_df['spread_wr'] * 1.909 - 1) * 100
        weekly_df['totals_roi'] = (weekly_df['totals_wr'] * 1.909 - 1) * 100

        fig = px.bar(weekly_df, x='week', y=['spread_roi', 'totals_roi'],
                     barmode='group', title=f"ROI by Week - {dataset}",
                     labels={'value': 'ROI (%)', 'week': 'Week'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

