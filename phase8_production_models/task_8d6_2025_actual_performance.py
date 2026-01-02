"""
================================================================================
DASHBOARD PAGE: 2025 ACTUAL PERFORMANCE
================================================================================

Show actual 2025 season performance with backtest results.

Author: NFL Betting Model v0.4.0
Date: 2025-12-27
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def show_2025_performance():
    """Display 2025 actual performance page"""
    
    st.title("🏈 2025 Season - Actual Performance")

    # Warning banner
    st.warning("""
    ⚠️ **Model Limitations Identified:**
    - **NO injury data** in model features (critical gap)
    - Overall accuracy: 53.1% (only 3% above random)
    - High confidence games: 69% accuracy (use these only!)
    - Extreme week-to-week volatility (25% to 73% range)
    """)

    st.markdown("---")
    
    # Load data
    try:
        df_backtest = pd.read_csv('../results/phase8_results/2025_backtest_weeks1_16.csv')
        df_weekly = pd.read_csv('../results/phase8_results/2025_weekly_performance.csv')
        df_schedule = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Run `python backtest_2025_performance.py` to generate backtest data.")
        return
    
    # Overall metrics
    st.header("📊 Overall Performance (Weeks 1-16)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(df_backtest))
    
    with col2:
        correct = df_backtest['correct'].sum()
        st.metric("Correct Predictions", f"{correct}/{len(df_backtest)}")
    
    with col3:
        accuracy = df_backtest['correct'].mean()
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with col4:
        avg_conf = df_backtest['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    st.markdown("---")
    
    # Weekly performance chart
    st.subheader("📈 Weekly Performance Trend")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_weekly['week'],
        y=df_weekly['accuracy'] * 100,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    # Add 50% baseline
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Random Baseline (50%)")
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Accuracy (%)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Performance by confidence level
    st.subheader("🎯 Performance by Confidence Level")
    
    high_conf = df_backtest[df_backtest['confidence'] >= 0.65]
    med_conf = df_backtest[(df_backtest['confidence'] >= 0.60) & (df_backtest['confidence'] < 0.65)]
    low_conf = df_backtest[df_backtest['confidence'] < 0.60]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(high_conf) > 0:
            st.metric(
                "High Confidence (≥65%)",
                f"{high_conf['correct'].mean():.1%}",
                f"{len(high_conf)} games"
            )
        else:
            st.metric("High Confidence (≥65%)", "N/A", "0 games")
    
    with col2:
        if len(med_conf) > 0:
            st.metric(
                "Medium Confidence (60-65%)",
                f"{med_conf['correct'].mean():.1%}",
                f"{len(med_conf)} games"
            )
        else:
            st.metric("Medium Confidence (60-65%)", "N/A", "0 games")
    
    with col3:
        if len(low_conf) > 0:
            st.metric(
                "Low Confidence (<60%)",
                f"{low_conf['correct'].mean():.1%}",
                f"{len(low_conf)} games"
            )
        else:
            st.metric("Low Confidence (<60%)", "N/A", "0 games")
    
    st.markdown("---")
    
    # Week 17 status
    st.subheader("📅 Week 17 Status")
    
    week17 = df_schedule[df_schedule['week'] == 17].copy()
    week17_completed = week17[week17['home_score'].notna()]
    week17_upcoming = week17[week17['home_score'].isna()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Games", len(week17))
    
    with col2:
        st.metric("Completed", len(week17_completed))
    
    with col3:
        st.metric("Upcoming", len(week17_upcoming))
    
    if len(week17_completed) > 0:
        st.write("**Completed Week 17 Games:**")
        for idx, row in week17_completed.iterrows():
            st.write(f"- {row['away_team']} @ {row['home_team']}: {row['away_score']:.0f}-{row['home_score']:.0f}")
    
    if len(week17_upcoming) > 0:
        st.write("**Upcoming Week 17 Games:**")
        for idx, row in week17_upcoming.iterrows():
            gameday = row['gameday'][:10] if pd.notna(row['gameday']) else 'TBD'
            st.write(f"- {row['away_team']} @ {row['home_team']} ({gameday})")
    
    st.markdown("---")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Week selector
    selected_week = st.selectbox(
        "Select Week",
        sorted(df_backtest['week'].unique())
    )
    
    week_data = df_backtest[df_backtest['week'] == selected_week].copy()
    
    # Display table
    display_df = week_data[[
        'gameday', 'away_team', 'home_team', 'away_score_actual', 'home_score_actual',
        'predicted_winner', 'confidence', 'actual_winner', 'correct'
    ]].copy()
    
    display_df['Date'] = pd.to_datetime(display_df['gameday']).dt.strftime('%Y-%m-%d')
    display_df['Score'] = display_df.apply(
        lambda row: f"{row['away_score_actual']:.0f}-{row['home_score_actual']:.0f}", axis=1
    )
    display_df['Result'] = display_df['correct'].apply(lambda x: '✅ Correct' if x == 1 else '❌ Wrong')
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['Date', 'away_team', 'home_team', 'Score', 'predicted_winner', 'Confidence', 'actual_winner', 'Result']],
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    show_2025_performance()

