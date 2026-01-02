"""
Generate Week 17 predictions using team season averages (Weeks 1-16)
"""

import pandas as pd
import numpy as np
import joblib
import json
import nfl_data_py as nfl

print("="*120)
print("GENERATE 2025 WEEK 17 PREDICTIONS (SIMPLIFIED APPROACH)")
print("="*120)

# Load model
print("\n[1/5] Loading XGBoost model...")
model = joblib.load('../models/xgboost_with_injuries.pkl')

with open('../models/xgboost_with_injuries_features.json', 'r') as f:
    feature_info = json.load(f)
    numeric_features = feature_info['features']

print(f"  ✅ Loaded model with {len(numeric_features)} features")

# Load Week 17 schedule
print("\n[2/5] Loading Week 17 schedule...")
schedules = nfl.import_schedules([2025])
week17 = schedules[schedules['week'] == 17].copy()

# Filter to upcoming games only
week17_upcoming = week17[week17['home_score'].isna()].copy()

print(f"  ✅ Week 17 total games: {len(week17)}")
print(f"  ✅ Week 17 upcoming games: {len(week17_upcoming)}")

# Load 2025 data to calculate team averages
print("\n[3/5] Calculating team season averages (Weeks 1-16)...")
df_2025 = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
df_2025 = df_2025[df_2025['season'] == 2025].copy()

# Calculate team averages for each feature
team_averages = {}

for team in week17_upcoming['home_team'].unique().tolist() + week17_upcoming['away_team'].unique().tolist():
    team_averages[team] = {}
    
    # Get all games for this team (home or away)
    team_home = df_2025[df_2025['home_team'] == team]
    team_away = df_2025[df_2025['away_team'] == team]
    
    # Calculate averages for home features
    for feat in numeric_features:
        if feat.startswith('home_'):
            if len(team_home) > 0 and feat in team_home.columns:
                team_averages[team][feat] = team_home[feat].mean()
            else:
                team_averages[team][feat] = 0
        elif feat.startswith('away_'):
            if len(team_away) > 0 and feat in team_away.columns:
                team_averages[team][feat] = team_away[feat].mean()
            else:
                team_averages[team][feat] = 0
        else:
            # Non-prefixed features (weather, etc.)
            team_averages[team][feat] = 0

print(f"  ✅ Calculated averages for {len(team_averages)} teams")

# Build feature matrix for Week 17 games
print("\n[4/5] Building feature matrix...")

week17_features = []

for idx, game in week17_upcoming.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']
    
    game_features = {'game_id': game['game_id']}
    
    # Use team averages
    for feat in numeric_features:
        if feat.startswith('home_'):
            game_features[feat] = team_averages.get(home_team, {}).get(feat, 0)
        elif feat.startswith('away_'):
            game_features[feat] = team_averages.get(away_team, {}).get(feat, 0)
        else:
            # For non-prefixed features, use average of both teams
            home_val = team_averages.get(home_team, {}).get(feat, 0)
            away_val = team_averages.get(away_team, {}).get(feat, 0)
            game_features[feat] = (home_val + away_val) / 2

    week17_features.append(game_features)

df_week17 = pd.DataFrame(week17_features)

print(f"  ✅ Built feature matrix: {df_week17.shape}")

# Generate predictions
print("\n[5/5] Generating predictions...")

X = df_week17[numeric_features].fillna(0)
y_pred_proba = model.predict_proba(X)[:, 1]

# Add predictions to schedule
week17_upcoming['home_win_prob'] = y_pred_proba
week17_upcoming['predicted_winner'] = week17_upcoming.apply(
    lambda row: row['home_team'] if row['home_win_prob'] >= 0.5 else row['away_team'],
    axis=1
)
week17_upcoming['confidence'] = week17_upcoming['home_win_prob'].apply(
    lambda p: max(p, 1-p)
)

# Create output
output_cols = ['game_id', 'week', 'away_team', 'home_team', 
               'predicted_winner', 'home_win_prob', 'confidence', 'gameday']

df_output = week17_upcoming[output_cols].copy()
df_output = df_output.sort_values('confidence', ascending=False)

# Save
output_file = '../results/phase8_results/2025_week17_predictions_with_injuries.csv'
df_output.to_csv(output_file, index=False)

print(f"  ✅ Saved: {output_file}")

# Display
print(f"\n{'='*120}")
print("WEEK 17 PREDICTIONS (12 UPCOMING GAMES)")
print("="*120)

print(f"\nAverage confidence: {df_output['confidence'].mean():.1%}")
print(f"High confidence (≥65%): {len(df_output[df_output['confidence'] >= 0.65])} games")

print(f"\nAll predictions:")
for idx, row in df_output.iterrows():
    print(f"  • {row['away_team']} @ {row['home_team']}: {row['predicted_winner']} ({row['confidence']:.1%}) - {row['gameday']}")

print(f"\n{'='*120}")

