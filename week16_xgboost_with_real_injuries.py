"""
XGBoost Week 16 Predictions with REAL ESPN Injury Data

The model's injury features use historical data (2009-2024 only).
This script shows:
1. XGBoost predictions
2. ACTUAL ESPN injury data for Week 16
3. Which teams have significant injuries that may not be in the model
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES

# ESPN injury data from week16_analysis.py output
ESPN_INJURIES = {
    'ARI': {'out': 5, 'questionable': 4, 'key': ['Marvin Harrison Jr.', 'Dadrion Taylor-Demerson']},
    'ATL': {'out': 2, 'questionable': 3, 'key': ['Drake London', 'James Pearce Jr.']},
    'BAL': {'out': 0, 'doubtful': 1, 'questionable': 4, 'key': ['Kyle Hamilton', 'Brent Urban']},
    'BUF': {'out': 2, 'questionable': 2, 'key': ['Jordan Phillips', 'Mecole Hardman Jr.']},
    'CAR': {'out': 2, 'questionable': 1, 'key': ['Tershawn Wharton', 'Ikem Ekwonu']},
    'CHI': {'out': 6, 'questionable': 1, 'key': ['C.J. Gardner-Johnson', 'Rome Odunze']},
    'CIN': {'out': 3, 'questionable': 4, 'key': ['Tee Higgins', 'DJ Turner II']},
    'CLE': {'out': 4, 'questionable': 3, 'key': ['Mike Hall Jr.', 'Wyatt Teller']},
    'DAL': {'out': 3, 'questionable': 3, 'key': ['Josh Butler', 'Quinnen Williams']},
    'DEN': {'out': 1, 'questionable': 1, 'key': ['Ben Powers', 'Justin Strnad']},
    'DET': {'out': 0, 'questionable': 5, 'key': ['Christian Mahogany', 'Graham Glasgow']},
    'GB': {'out': 7, 'questionable': 4, 'key': ['John FitzPatrick', 'Romeo Doubs']},
    'HOU': {'out': 0, 'questionable': 5, 'key': ['Derek Stingley Jr.', 'Woody Marks']},
    'IND': {'out': 2, 'questionable': 1, 'key': ['Bernhard Raimann', 'Anthony Gould']},
    'JAX': {'out': 2, 'questionable': 0, 'key': ['Danny Striggow', 'Bhayshul Tuten']},
    'KC': {'out': 5, 'questionable': 0, 'key': ['Jaylon Moore', 'Derrick Nnadi']},
    'LAC': {'out': 2, 'questionable': 5, 'key': ['Trey Pipkins III', 'Teair Tart']},
    'LA': {'out': 0, 'questionable': 0, 'key': []},
    'LV': {'out': 0, 'questionable': 1, 'key': ['Jordan Meredith']},
    'MIA': {'out': 1, 'questionable': 2, 'key': ['Jordyn Brooks', 'Minkah Fitzpatrick']},
    'MIN': {'out': 2, 'questionable': 2, 'key': ['Gavin Bartholomew', "Brian O'Neill"]},
    'NE': {'out': 1, 'questionable': 4, 'key': ['Christian Barmore', 'Carlton Davis III']},
    'NO': {'out': 3, 'questionable': 3, 'key': ['Zaire Mitchell-Paden', 'Asim Richards']},
    'NYG': {'out': 2, 'questionable': 3, 'key': ['Anthony Johnson Jr.', 'Art Green']},
    'NYJ': {'out': 3, 'questionable': 2, 'key': ['Eric Watts', 'Kiko Mauigoa']},
    'PHI': {'out': 7, 'questionable': 1, 'key': ['Jalen Carter', 'Nakobe Dean']},
    'PIT': {'out': 4, 'questionable': 0, 'key': ['T.J. Watt', 'Nick Herbig']},
    'SEA': {'out': 5, 'questionable': 2, 'key': ['Jared Ivey', 'Coby Bryant']},
    'SF': {'out': 4, 'questionable': 0, 'key': ['Kurtis Rourke', 'Renardo Green']},
    'TB': {'out': 0, 'questionable': 0, 'key': []},
    'TEN': {'out': 1, 'questionable': 5, 'key': ['Van Jefferson', 'Gunnar Helm']},
    'WAS': {'out': 0, 'questionable': 0, 'key': []},
}

def main():
    print("="*100)
    print("XGBOOST WEEK 16 PREDICTIONS WITH REAL ESPN INJURY DATA")
    print("="*100)
    
    # Load data and train model
    games_df, _ = load_and_prepare_data()
    week16 = games_df[(games_df['season'] == 2025) & (games_df['week'] == 16)].copy()
    
    available_features = [f for f in FEATURES if f in week16.columns]
    train_df = games_df[
        (games_df['season'] >= 2018) & (games_df['season'] <= 2024)
    ].dropna(subset=available_features + ['home_win'])
    
    X_train = train_df[available_features].copy()
    y_train = train_df['home_win'].astype(int)
    train_medians = X_train.median()
    
    X_week16 = week16[available_features].copy()
    for col in X_week16.columns:
        if X_week16[col].isna().any():
            X_week16[col] = X_week16[col].fillna(train_medians.get(col, 0))
    
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_week16)[:, 1]
    
    # Create results with ESPN injury data
    results = []
    for idx, (i, row) in enumerate(week16.iterrows()):
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_inj = ESPN_INJURIES.get(home_team, {})
        away_inj = ESPN_INJURIES.get(away_team, {})
        
        xgb_prob = xgb_proba[idx]
        xgb_pick = home_team if xgb_prob > 0.5 else away_team
        xgb_conf = max(xgb_prob, 1 - xgb_prob)
        
        results.append({
            'matchup': f"{away_team}@{home_team}",
            'away_team': away_team,
            'home_team': home_team,
            'spread': row['spread_line'],
            'vegas_home_prob': row['home_implied_prob'],
            'xgb_home_prob': xgb_prob,
            'xgb_pick': xgb_pick,
            'xgb_confidence': xgb_conf,
            'home_out': home_inj.get('out', 0),
            'home_q': home_inj.get('questionable', 0),
            'away_out': away_inj.get('out', 0),
            'away_q': away_inj.get('questionable', 0),
            'home_key_injuries': ', '.join(home_inj.get('key', [])[:2]),
            'away_key_injuries': ', '.join(away_inj.get('key', [])[:2]),
        })
    
    df = pd.DataFrame(results).sort_values('xgb_confidence', ascending=False)
    
    print("\n" + "="*100)
    print("XGBOOST PREDICTIONS WITH CURRENT ESPN INJURY DATA")
    print("="*100)
    print(f"\n{'Matchup':<16} {'XGB Pick':<8} {'Conf':>6} {'Vegas':>7} {'XGB':>7} "
          f"{'H-Out':>6} {'H-Q':>5} {'A-Out':>6} {'A-Q':>5}")
    print("-"*100)
    
    for _, row in df.iterrows():
        marker = ""
        if row['home_out'] >= 4 or row['away_out'] >= 4:
            marker = " ⚠️⚠️"
        elif row['home_out'] >= 2 or row['away_out'] >= 2:
            marker = " ⚠️"
        
        print(f"{row['matchup']:<16} {row['xgb_pick']:<8} {row['xgb_confidence']:>5.1%} "
              f"{row['vegas_home_prob']:>6.1%} {row['xgb_home_prob']:>6.1%} "
              f"{row['home_out']:>6} {row['home_q']:>5} {row['away_out']:>6} {row['away_q']:>5}{marker}")
    
    print("\n" + "="*100)
    print("KEY INJURY IMPACTS (Teams with 4+ players OUT)")
    print("="*100)
    
    high_injury = df[(df['home_out'] >= 4) | (df['away_out'] >= 4)]
    for _, row in high_injury.iterrows():
        print(f"\n{row['matchup']}:")
        if row['home_out'] >= 4:
            print(f"  {row['home_team']}: {row['home_out']} OUT - Key: {row['home_key_injuries']}")
        if row['away_out'] >= 4:
            print(f"  {row['away_team']}: {row['away_out']} OUT - Key: {row['away_key_injuries']}")
        print(f"  XGBoost picks: {row['xgb_pick']} ({row['xgb_confidence']:.1%} confidence)")
    
    print("\n" + "="*100)
    print("⚠️  IMPORTANT NOTE")
    print("="*100)
    print("The model's injury features use historical data (2009-2024) only.")
    print("2025 injury data is NOT in the model - it uses imputed values (median ~1.6).")
    print("The ESPN injury data above shows CURRENT injuries that may not be reflected in predictions.")
    print("\nTeams with major injuries (4+ OUT) that XGBoost may be underweighting:")
    
    for team, inj in ESPN_INJURIES.items():
        if inj.get('out', 0) >= 4:
            print(f"  - {team}: {inj['out']} OUT, Key: {', '.join(inj['key'][:2])}")
    
    return df


if __name__ == "__main__":
    predictions = main()

