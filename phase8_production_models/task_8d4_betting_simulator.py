"""
Task 8D.4: Betting Strategy Simulator

Interactive betting simulator with multiple strategies:
- Kelly Criterion
- Fixed Stake
- Confidence Threshold
- Proportional Betting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def show_betting_simulator(data, models):
    """Display betting simulator page"""
    
    st.markdown('<div class="main-header">💰 Betting Strategy Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Simulate different betting strategies on the 2024 NFL season using trained models.
    Compare performance across strategies and models.
    """)
    
    st.markdown("---")
    
    # Load test data
    df = data['games']
    test_df = df[df['season'] == 2024].copy()
    
    # Prepare features
    with open('../results/phase8_results/feature_categorization.json', 'r') as f:
        cat = json.load(f)
    
    pre_game_dict = cat['pre_game_features']
    pre_game_features = []
    for category, features in pre_game_dict.items():
        pre_game_features.extend(features)
    
    unknown_pregame = [
        'OTLosses', 'losses', 'pointsAgainst', 'pointsFor', 'ties', 'winPercent', 
        'winPercentage', 'wins', 'losses_roll3', 'losses_roll5', 'losses_std',
        'winPercent_roll3', 'winPercent_roll5', 'winPercent_std',
        'wins_roll3', 'wins_roll5', 'wins_std',
        'scored_20plus', 'scored_30plus', 'streak_20plus', 'streak_30plus',
        'vsconf_OTLosses', 'vsconf_leagueWinPercent', 'vsconf_losses', 'vsconf_ties', 'vsconf_wins',
        'vsdiv_OTLosses', 'vsdiv_divisionLosses', 'vsdiv_divisionTies', 
        'vsdiv_divisionWinPercent', 'vsdiv_divisionWins', 'vsdiv_losses', 'vsdiv_ties', 'vsdiv_wins',
        'div_game', 'rest_advantage', 'opponent'
    ]
    pre_game_features.extend(unknown_pregame)
    
    pregame_cols = []
    for feat in pre_game_features:
        home_feat = f'home_{feat}'
        away_feat = f'away_{feat}'
        if home_feat in df.columns:
            pregame_cols.append(home_feat)
        if away_feat in df.columns:
            pregame_cols.append(away_feat)
    
    numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    train = df[df['season'] <= 2019].copy()
    X_test = test_df[numeric_pregame].fillna(train[numeric_pregame].median())
    test_df['home_win'] = (test_df['home_score'] > test_df['away_score']).astype(int)
    y_test = test_df['home_win'].values
    
    # Sidebar: Strategy Configuration
    st.sidebar.markdown("### 🎯 Strategy Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        list(models.keys())
    )
    
    # Get predictions
    model = models[selected_model]
    predictions = model.predict_proba(X_test)[:, 1]  # Probability of home win
    
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Select Betting Strategy:",
        ["Kelly Criterion", "Fixed Stake", "Confidence Threshold", "Proportional Betting"]
    )
    
    # Initial bankroll
    initial_bankroll = st.sidebar.number_input(
        "Initial Bankroll ($):",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )
    
    # Strategy-specific parameters
    if strategy == "Kelly Criterion":
        kelly_fraction = st.sidebar.slider(
            "Kelly Fraction:",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Fraction of Kelly bet size (0.25 = Quarter Kelly)"
        )
        
        min_edge = st.sidebar.slider(
            "Minimum Edge (%):",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Only bet when edge exceeds this threshold"
        )
    
    elif strategy == "Fixed Stake":
        stake_pct = st.sidebar.slider(
            "Stake (% of bankroll):",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        
        min_confidence = st.sidebar.slider(
            "Minimum Confidence (%):",
            min_value=50,
            max_value=90,
            value=60,
            step=5
        )
    
    elif strategy == "Confidence Threshold":
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold (%):",
            min_value=50,
            max_value=95,
            value=70,
            step=5
        )
        
        stake_pct = st.sidebar.slider(
            "Stake (% of bankroll):",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
    
    elif strategy == "Proportional Betting":
        max_stake_pct = st.sidebar.slider(
            "Max Stake (% of bankroll):",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )
        
        min_confidence = st.sidebar.slider(
            "Minimum Confidence (%):",
            min_value=50,
            max_value=90,
            value=55,
            step=5
        )
    
    # Odds assumption
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Odds Settings")
    
    odds_type = st.sidebar.selectbox(
        "Odds Type:",
        ["Fair Odds (No Vig)", "American Odds (-110)", "Custom"]
    )
    
    if odds_type == "Custom":
        custom_odds = st.sidebar.number_input(
            "Decimal Odds:",
            min_value=1.5,
            max_value=3.0,
            value=1.91,
            step=0.01
        )

    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):

        # Calculate odds
        if odds_type == "Fair Odds (No Vig)":
            # Fair odds based on implied probability
            odds = 1 / predictions
        elif odds_type == "American Odds (-110)":
            # Standard -110 odds (1.91 decimal)
            odds = np.full(len(predictions), 1.91)
        else:
            # Custom odds
            odds = np.full(len(predictions), custom_odds)

        # Simulate betting strategy
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        bet_history = []

        for i in range(len(predictions)):
            prob = predictions[i]
            actual = y_test[i]
            game_odds = odds[i]

            # Determine bet size based on strategy
            bet_size = 0
            bet_on_home = True

            if strategy == "Kelly Criterion":
                # Kelly formula: f = (bp - q) / b
                # where b = odds - 1, p = probability, q = 1 - p

                # Bet on home if prob > 0.5, else bet on away
                if prob > 0.5:
                    edge = prob - (1 / game_odds)
                    if edge > min_edge / 100:
                        kelly_bet = (prob * (game_odds - 1) - (1 - prob)) / (game_odds - 1)
                        bet_size = max(0, kelly_bet * kelly_fraction * bankroll)
                    bet_on_home = True
                else:
                    edge = (1 - prob) - (1 / game_odds)
                    if edge > min_edge / 100:
                        kelly_bet = ((1 - prob) * (game_odds - 1) - prob) / (game_odds - 1)
                        bet_size = max(0, kelly_bet * kelly_fraction * bankroll)
                    bet_on_home = False

            elif strategy == "Fixed Stake":
                if prob >= min_confidence / 100:
                    bet_size = bankroll * (stake_pct / 100)
                    bet_on_home = True
                elif prob <= (100 - min_confidence) / 100:
                    bet_size = bankroll * (stake_pct / 100)
                    bet_on_home = False

            elif strategy == "Confidence Threshold":
                if prob >= confidence_threshold / 100:
                    bet_size = bankroll * (stake_pct / 100)
                    bet_on_home = True
                elif prob <= (100 - confidence_threshold) / 100:
                    bet_size = bankroll * (stake_pct / 100)
                    bet_on_home = False

            elif strategy == "Proportional Betting":
                # Bet size proportional to confidence
                if prob >= min_confidence / 100:
                    confidence = prob
                    bet_size = bankroll * (max_stake_pct / 100) * ((confidence - 0.5) / 0.5)
                    bet_on_home = True
                elif prob <= (100 - min_confidence) / 100:
                    confidence = 1 - prob
                    bet_size = bankroll * (max_stake_pct / 100) * ((confidence - 0.5) / 0.5)
                    bet_on_home = False

            # Cap bet size at bankroll
            bet_size = min(bet_size, bankroll)

            # Calculate profit/loss
            if bet_size > 0:
                if (bet_on_home and actual == 1) or (not bet_on_home and actual == 0):
                    # Win
                    profit = bet_size * (game_odds - 1)
                    bankroll += profit
                else:
                    # Loss
                    profit = -bet_size
                    bankroll += profit

                bet_history.append({
                    'game': i + 1,
                    'bet_on': 'Home' if bet_on_home else 'Away',
                    'bet_size': bet_size,
                    'odds': game_odds,
                    'result': 'Win' if profit > 0 else 'Loss',
                    'profit': profit,
                    'bankroll': bankroll
                })

            bankroll_history.append(bankroll)

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        final_bankroll = bankroll_history[-1]
        total_profit = final_bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll) * 100
        num_bets = len(bet_history)

        with col1:
            st.metric("Final Bankroll", f"${final_bankroll:.2f}",
                     f"${total_profit:+.2f}")

        with col2:
            st.metric("ROI", f"{roi:+.2f}%",
                     f"{roi/100*initial_bankroll:+.2f}")

        with col3:
            st.metric("Total Bets", num_bets,
                     f"{num_bets/len(predictions)*100:.1f}% of games")

        with col4:
            if num_bets > 0:
                wins = sum(1 for b in bet_history if b['result'] == 'Win')
                win_rate = wins / num_bets * 100
                st.metric("Win Rate", f"{win_rate:.1f}%",
                         f"{wins}/{num_bets}")
            else:
                st.metric("Win Rate", "N/A", "No bets placed")

        st.markdown("---")

        # Bankroll chart
        st.markdown("### 📈 Bankroll Over Time")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(range(len(bankroll_history)), bankroll_history,
               linewidth=2, color='#1f77b4', label='Bankroll')
        ax.axhline(y=initial_bankroll, color='gray', linestyle='--',
                  alpha=0.5, label='Initial Bankroll')

        ax.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bankroll ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy} - {selected_model}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

        # Color background based on profit/loss
        if final_bankroll > initial_bankroll:
            ax.axhspan(initial_bankroll, max(bankroll_history), alpha=0.1, color='green')
        else:
            ax.axhspan(min(bankroll_history), initial_bankroll, alpha=0.1, color='red')

        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Bet history table
        if num_bets > 0:
            st.markdown("### 📋 Bet History")

            bet_df = pd.DataFrame(bet_history)

            # Show last 20 bets
            st.dataframe(
                bet_df.tail(20).style.format({
                    'bet_size': '${:.2f}',
                    'odds': '{:.2f}',
                    'profit': '${:+.2f}',
                    'bankroll': '${:.2f}'
                }).applymap(
                    lambda x: 'background-color: #d4edda' if x == 'Win' else 'background-color: #f8d7da',
                    subset=['result']
                ),
                width='stretch',
                hide_index=True
            )

            # Download button
            csv = bet_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Bet History (CSV)",
                data=csv,
                file_name=f"bet_history_{strategy.replace(' ', '_')}_{selected_model}.csv",
                mime="text/csv"
            )

    else:
        st.info("👈 Configure your betting strategy in the sidebar and click 'Run Simulation' to begin.")

