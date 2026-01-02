"""
Generate Week 17 predictions using simple team strength approach
Since we don't have full features for Week 17, use team performance from weeks 1-16
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("GENERATING WEEK 17 PREDICTIONS (SIMPLE APPROACH)")
print("=" * 120)

# Load actual schedule
print("\n[1/4] Loading Week 17 schedule...")
df_schedule = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
week17 = df_schedule[df_schedule['week'] == 17].copy()
week17_upcoming = week17[week17['home_score'].isna()].copy()

print(f"  ✅ Total Week 17 games: {len(week17)}")
print(f"  ✅ Completed: {len(week17) - len(week17_upcoming)}")
print(f"  ✅ Upcoming: {len(week17_upcoming)}")

# Load backtest results to calculate team strengths
print("\n[2/4] Calculating team strengths from weeks 1-16...")
df_backtest = pd.read_csv('../results/phase8_results/2025_backtest_weeks1_16.csv')

# Calculate win percentage for each team
team_stats = {}

for team in df_backtest['home_team'].unique():
    # Home games
    home_games = df_backtest[df_backtest['home_team'] == team]
    home_wins = (home_games['home_score_actual'] > home_games['away_score_actual']).sum()
    home_total = len(home_games)
    
    # Away games
    away_games = df_backtest[df_backtest['away_team'] == team]
    away_wins = (away_games['away_score_actual'] > away_games['home_score_actual']).sum()
    away_total = len(away_games)
    
    # Combined
    total_wins = home_wins + away_wins
    total_games = home_total + away_total
    win_pct = total_wins / total_games if total_games > 0 else 0.5
    
    # Average points scored and allowed
    home_scored = home_games['home_score_actual'].mean() if len(home_games) > 0 else 20
    away_scored = away_games['away_score_actual'].mean() if len(away_games) > 0 else 20
    avg_scored = (home_scored + away_scored) / 2
    
    home_allowed = home_games['away_score_actual'].mean() if len(home_games) > 0 else 20
    away_allowed = away_games['home_score_actual'].mean() if len(away_games) > 0 else 20
    avg_allowed = (home_allowed + away_allowed) / 2
    
    team_stats[team] = {
        'win_pct': win_pct,
        'wins': total_wins,
        'games': total_games,
        'avg_scored': avg_scored,
        'avg_allowed': avg_allowed,
        'point_diff': avg_scored - avg_allowed
    }

print(f"  ✅ Calculated stats for {len(team_stats)} teams")

# Generate predictions for upcoming Week 17 games
print("\n[3/4] Generating predictions...")

predictions = []

for idx, row in week17_upcoming.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Get team stats (with defaults for all fields)
    default_stats = {'win_pct': 0.5, 'wins': 0, 'games': 0, 'point_diff': 0, 'avg_scored': 20, 'avg_allowed': 20}
    home_stats = team_stats.get(home_team, default_stats)
    away_stats = team_stats.get(away_team, default_stats)
    
    # Simple prediction model:
    # 1. Home field advantage: +2.5 points
    # 2. Team strength: win_pct and point_diff
    
    home_advantage = 0.025  # 2.5% boost for home team
    
    # Calculate win probability using logistic function
    # Based on win percentage difference and point differential
    win_pct_diff = home_stats['win_pct'] - away_stats['win_pct']
    point_diff = home_stats['point_diff'] - away_stats['point_diff']
    
    # Combine factors (weighted)
    strength_diff = (0.6 * win_pct_diff) + (0.4 * point_diff / 20)  # Normalize point diff
    
    # Add home field advantage
    strength_diff += home_advantage
    
    # Convert to probability using logistic function
    # P(home win) = 1 / (1 + exp(-k * strength_diff))
    k = 5  # Scaling factor
    home_win_prob = 1 / (1 + np.exp(-k * strength_diff))
    
    # Ensure probability is between 0.3 and 0.7 (don't be too confident)
    home_win_prob = np.clip(home_win_prob, 0.35, 0.65)
    
    # Determine winner
    predicted_winner = home_team if home_win_prob > 0.5 else away_team
    confidence = max(home_win_prob, 1 - home_win_prob)
    
    predictions.append({
        'game_id': f"2025_17_{away_team}_{home_team}",
        'season': 2025,
        'week': 17,
        'gameday': row['gameday'],
        'weekday': row['weekday'],
        'home_team': home_team,
        'away_team': away_team,
        'home_win_probability': home_win_prob,
        'away_win_probability': 1 - home_win_prob,
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        'home_record': f"{home_stats['wins']}-{home_stats['games']-home_stats['wins']}",
        'away_record': f"{away_stats['wins']}-{away_stats['games']-away_stats['wins']}",
        'home_point_diff': home_stats['point_diff'],
        'away_point_diff': away_stats['point_diff']
    })

df_pred = pd.DataFrame(predictions)

# Save predictions
print("\n[4/4] Saving predictions...")
output_path = '../results/phase8_results/2025_week17_predictions.csv'
df_pred.to_csv(output_path, index=False)
print(f"  ✅ Saved to: {output_path}")

# Display predictions
print(f"\n{'='*120}")
print("WEEK 17 PREDICTIONS (13 UPCOMING GAMES)")
print("=" * 120)
print(f"{'Date':<12} {'Away (Record)':<20} {'@':<3} {'Home (Record)':<20} {'Predicted Winner':<18} {'Confidence':<12}")
print("-" * 120)

for idx, row in df_pred.iterrows():
    date = row['gameday'][:10] if pd.notna(row['gameday']) else 'TBD'
    away_str = f"{row['away_team']} ({row['away_record']})"
    home_str = f"{row['home_team']} ({row['home_record']})"
    
    print(f"{date:<12} {away_str:<20} @ {home_str:<20} "
          f"{row['predicted_winner']:<18} {row['confidence']:.1%}")

# Summary stats
print(f"\n{'='*120}")
print("PREDICTION SUMMARY")
print("=" * 120)
print(f"Total predictions: {len(df_pred)}")
print(f"Average confidence: {df_pred['confidence'].mean():.1%}")
print(f"High confidence (≥60%): {len(df_pred[df_pred['confidence'] >= 0.60])}")
print(f"Medium confidence (55-60%): {len(df_pred[(df_pred['confidence'] >= 0.55) & (df_pred['confidence'] < 0.60)])}")
print(f"Low confidence (<55%): {len(df_pred[df_pred['confidence'] < 0.55])}")

print(f"\n⚠️ NOTE: These predictions use a simplified model based on team performance in weeks 1-16.")
print(f"   They do NOT include:")
print(f"   • Injury data (critical limitation)")
print(f"   • Playoff implications")
print(f"   • Rest/motivation factors")
print(f"   • Full feature set from trained models")
print(f"\n   Use with caution and only for high confidence picks!")

print(f"\n{'='*120}")
print("WEEK 17 PREDICTIONS COMPLETE")
print("=" * 120)

