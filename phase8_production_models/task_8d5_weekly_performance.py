"""
================================================================================
DASHBOARD PAGE: WEEKLY PERFORMANCE ANALYSIS
================================================================================

Interactive dashboard page for analyzing weekly performance and predictions.

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

def show_weekly_performance():
    """Display weekly performance analysis page"""
    
    st.title("📅 Weekly Performance Analysis")
    st.markdown("---")
    
    # Load data
    try:
        df_2024_analysis = pd.read_csv('../results/phase8_results/2024_week16_17_analysis.csv')
        df_2025_predictions = pd.read_csv('../results/phase8_results/2025_predictions.csv')
        df_2025_bets = pd.read_csv('../results/phase8_results/2025_betting_recommendations.csv')
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return
    
    # Sidebar - Week selector
    st.sidebar.header("Week Selection")
    
    # Get available weeks
    weeks_2024 = sorted(df_2024_analysis['week'].unique())
    weeks_2025 = sorted(df_2025_predictions['week'].unique())
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["2024 Retrospective", "2025 Predictions"]
    )
    
    if analysis_type == "2024 Retrospective":
        selected_week = st.sidebar.selectbox("Select Week (2024)", weeks_2024)
        show_2024_week_analysis(df_2024_analysis, selected_week)
    else:
        selected_week = st.sidebar.selectbox("Select Week (2025)", weeks_2025)
        show_2025_week_predictions(df_2025_predictions, df_2025_bets, selected_week)

def show_2024_week_analysis(df, week):
    """Show 2024 week retrospective analysis"""
    
    st.header(f"2024 Week {week} - Retrospective Analysis")
    
    # Filter to selected week
    df_week = df[df['week'] == week].copy()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(df_week))
    
    with col2:
        correct = df_week['correct'].sum()
        st.metric("Correct Predictions", f"{correct}/{len(df_week)}")
    
    with col3:
        accuracy = df_week['correct'].mean()
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with col4:
        avg_conf = df_week['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    st.markdown("---")
    
    # Game-by-game results
    st.subheader("Game-by-Game Results")
    
    # Create display dataframe
    display_df = df_week[[
        'away_team', 'home_team', 'away_score', 'home_score',
        'predicted_winner', 'confidence', 'actual_winner', 'correct'
    ]].copy()
    
    display_df['Score'] = display_df.apply(
        lambda row: f"{row['away_score']:.0f}-{row['home_score']:.0f}", axis=1
    )
    display_df['Result'] = display_df['correct'].apply(lambda x: '✅ Correct' if x == 1 else '❌ Wrong')
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['away_team', 'home_team', 'Score', 'predicted_winner', 'Confidence', 'actual_winner', 'Result']],
        width=1200,
        hide_index=True
    )
    
    # Missed predictions
    misses = df_week[df_week['correct'] == 0]
    if len(misses) > 0:
        st.markdown("---")
        st.subheader(f"❌ Missed Predictions ({len(misses)} games)")
        
        for idx, row in misses.iterrows():
            with st.expander(f"{row['away_team']} @ {row['home_team']} - Predicted {row['predicted_winner']}, Actual {row['actual_winner']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Game Details:**")
                    st.write(f"- Final Score: {row['away_score']:.0f}-{row['home_score']:.0f}")
                    st.write(f"- Predicted: {row['predicted_winner']} ({row['confidence']:.1%})")
                    st.write(f"- Actual: {row['actual_winner']}")
                
                with col2:
                    st.write("**Model Probabilities:**")
                    st.write(f"- XGBoost: {row['XGBoost_prob']:.1%}")
                    st.write(f"- LightGBM: {row['LightGBM_prob']:.1%}")
                    st.write(f"- CatBoost: {row['CatBoost_prob']:.1%}")
                    st.write(f"- RandomForest: {row['RandomForest_prob']:.1%}")
                    st.write(f"- Ensemble: {row['Ensemble_prob']:.1%}")

def show_2025_week_predictions(df_pred, df_bets, week):
    """Show 2025 week predictions"""
    
    st.header(f"2025 Week {week} - Predictions & Betting Opportunities")
    
    # Filter to selected week
    df_week = df_pred[df_pred['week'] == week].copy()
    df_bets_week = df_bets[df_bets['week'] == week].copy()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Games", len(df_week))
    
    with col2:
        avg_conf = df_week['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    with col3:
        high_conf = len(df_week[df_week['confidence'] >= 0.65])
        st.metric("High Confidence (≥65%)", high_conf)
    
    st.markdown("---")
    
    # Predictions table
    st.subheader("All Predictions")
    
    display_df = df_week[[
        'gameday', 'away_team', 'home_team', 'predicted_winner', 'confidence'
    ]].copy()
    
    display_df['Date'] = pd.to_datetime(display_df['gameday']).dt.strftime('%Y-%m-%d')
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['Date', 'away_team', 'home_team', 'predicted_winner', 'Confidence']],
        width=1200,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Top picks
    st.subheader("🎯 Top 5 Most Confident Picks")
    
    top_picks = df_week.nlargest(5, 'confidence')
    
    for i, (idx, row) in enumerate(top_picks.iterrows(), 1):
        opponent = row['away_team'] if row['predicted_winner'] == row['home_team'] else row['home_team']
        st.write(f"{i}. **{row['predicted_winner']}** over {opponent} ({row['confidence']:.1%} confidence)")
    
    st.markdown("---")
    
    # Betting opportunities
    st.subheader("💰 Top 5 Betting Opportunities")
    
    top_bets = df_bets_week.nlargest(5, 'expected_value')
    
    for i, (idx, row) in enumerate(top_bets.iterrows(), 1):
        with st.expander(f"{i}. {row['predicted_winner']} ({row['confidence']:.1%} confidence)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Value Metrics:**")
                st.write(f"- Edge: {row['edge']:.2%}")
                st.write(f"- Expected Value: {row['expected_value']:.2%}")
            
            with col2:
                st.write("**Recommended Bets:**")
                st.write(f"- Kelly: ${row['kelly_bet']:.2f}")
                st.write(f"- Fixed Stake: ${row['fixed_stake_bet']:.2f}")
                st.write(f"- Proportional: ${row['proportional_bet']:.2f}")

if __name__ == "__main__":
    show_weekly_performance()

