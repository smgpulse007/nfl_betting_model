"""
Week 16 2025 Analysis - Deep dive into predictions and favorite bias
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier

# Import from our modules
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES

def check_week16_data():
    """Check if Week 16 2025 data is available."""
    import nfl_data_py as nfl
    
    schedules = nfl.import_schedules([2025])
    week16 = schedules[schedules['week'] == 16]
    
    print(f"Week 16 2025 games: {len(week16)}")
    print("\nGames:")
    for _, g in week16.iterrows():
        result = g.get('result', None)
        status = "COMPLETED" if pd.notna(result) else "PENDING"
        home_score = g.get('home_score', 'N/A')
        away_score = g.get('away_score', 'N/A')
        print(f"  {g['away_team']} @ {g['home_team']}: {status} | "
              f"Score: {away_score}-{home_score} | Spread: {g.get('spread_line', 'N/A')}")
    
    return week16

def analyze_favorite_bias():
    """Analyze if models just pick the favorite."""
    print("\n" + "="*60)
    print("FAVORITE BIAS ANALYSIS")
    print("="*60)
    
    # Load predictions
    pred_file = Path("predictions/predictions_2025.csv")
    if not pred_file.exists():
        print("No 2025 predictions found. Running pipeline first...")
        return
    
    df = pd.read_csv(pred_file)
    
    # Define "favorite" as implied_prob > 0.5
    df['home_is_favorite'] = df['home_implied_prob'] > 0.5
    df['model_picks_home'] = df['pred_win_proba'] > 0.5
    df['model_agrees_with_vegas'] = df['home_is_favorite'] == df['model_picks_home']
    
    # Analysis
    agreement_rate = df['model_agrees_with_vegas'].mean()
    print(f"\n📊 Model agrees with Vegas favorite: {agreement_rate:.1%}")
    
    # When model disagrees with Vegas, what happens?
    disagree = df[~df['model_agrees_with_vegas']]
    if len(disagree) > 0 and 'home_win' in disagree.columns:
        # Model went against Vegas - did it work?
        disagree_correct = (disagree['model_picks_home'] == disagree['home_win']).mean()
        vegas_correct = (disagree['home_is_favorite'] == disagree['home_win']).mean()
        print(f"When model disagrees with Vegas ({len(disagree)} games):")
        print(f"  - Model accuracy: {disagree_correct:.1%}")
        print(f"  - Vegas accuracy: {vegas_correct:.1%}")
    
    # Correlation between pred_win_proba and home_implied_prob
    corr = df['pred_win_proba'].corr(df['home_implied_prob'])
    print(f"\n🔗 Correlation(pred_win_proba, home_implied_prob): {corr:.3f}")
    
    if corr > 0.9:
        print("⚠️  HIGH CORRELATION - Model is essentially following Vegas!")
    elif corr > 0.7:
        print("⚠️  MODERATE CORRELATION - Model has some independent signal")
    else:
        print("✅ LOW CORRELATION - Model has independent predictions")
    
    return df

def week16_predictions():
    """Generate Week 16 predictions with all models."""
    print("\n" + "="*60)
    print("WEEK 16 2025 PREDICTIONS")
    print("="*60)

    # Load data with TIER S+A features - returns (all_games, completed_games)
    games_df, _ = load_and_prepare_data()

    # Get Week 16 data
    week16 = games_df[
        (games_df['season'] == 2025) &
        (games_df['week'] == 16)
    ].copy()

    if len(week16) == 0:
        print("No Week 16 data available yet.")
        return None

    print(f"\nFound {len(week16)} Week 16 games")

    # Get features - filter to available ones
    available_features = [f for f in FEATURES if f in week16.columns]

    # Train models on 2018-2024 data
    train_df = games_df[
        (games_df['season'] >= 2018) &
        (games_df['season'] <= 2024)
    ].dropna(subset=available_features + ['home_win'])

    X_train = train_df[available_features].copy()
    y_win = train_df['home_win'].astype(int)

    # Compute medians from training data for imputation
    train_medians = X_train.median()

    # Prepare test data and fill NaN with training medians
    X = week16[available_features].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(train_medians.get(col, 0))
    
    # Train multiple models
    models = {
        'Logistic': LogisticRegression(max_iter=1000, C=1.0),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, verbosity=0),
        'CatBoost': CatBoostClassifier(n_estimators=100, max_depth=4, verbose=0),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6)
    }
    
    predictions = []
    for name, model in models.items():
        model.fit(X_train, y_win)
        proba = model.predict_proba(X)[:, 1]
        predictions.append(pd.Series(proba, name=f'{name}_proba', index=week16.index))
    
    # Combine predictions
    pred_df = pd.concat(predictions, axis=1)
    pred_df['home_team'] = week16['home_team'].values
    pred_df['away_team'] = week16['away_team'].values
    pred_df['spread_line'] = week16['spread_line'].values
    pred_df['home_implied_prob'] = week16['home_implied_prob'].values
    pred_df['home_win'] = week16.get('home_win', pd.NA)
    
    # Compare with Vegas
    pred_df['vegas_pick_home'] = pred_df['home_implied_prob'] > 0.5
    for name in models.keys():
        pred_df[f'{name}_pick_home'] = pred_df[f'{name}_proba'] > 0.5
        pred_df[f'{name}_agrees_vegas'] = (
            pred_df[f'{name}_pick_home'] == pred_df['vegas_pick_home']
        )
    
    print("\nWeek 16 Predictions vs Vegas:")
    print("-" * 80)
    for _, row in pred_df.iterrows():
        vegas_pick = "HOME" if row['vegas_pick_home'] else "AWAY"
        print(f"\n{row['away_team']} @ {row['home_team']} (Vegas: {vegas_pick}, "
              f"implied={row['home_implied_prob']:.1%})")
        for name in models.keys():
            model_pick = "HOME" if row[f'{name}_pick_home'] else "AWAY"
            agrees = "✅" if row[f'{name}_agrees_vegas'] else "❌"
            print(f"  {name:12}: {row[f'{name}_proba']:.1%} → {model_pick} {agrees}")
    
    # Save predictions
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    pred_df.to_csv(results_dir / "week16_2025_all_models.csv", index=False)
    print(f"\n✅ Saved to results/week16_2025_all_models.csv")

    # FAVORITE BIAS SUMMARY
    print("\n" + "="*60)
    print("⚠️  FAVORITE BIAS ANALYSIS")
    print("="*60)

    for name in ['Logistic', 'XGBoost', 'CatBoost', 'RandomForest']:
        agree_count = pred_df[f'{name}_agrees_vegas'].sum()
        agree_pct = agree_count / len(pred_df) * 100
        print(f"{name:12}: {agree_count}/{len(pred_df)} games agree with Vegas ({agree_pct:.0f}%)")

    # Calculate correlation between model predictions and Vegas
    print("\n📊 Correlation with home_implied_prob:")
    for name in ['Logistic', 'XGBoost', 'CatBoost', 'RandomForest']:
        corr = pred_df[f'{name}_proba'].corr(pred_df['home_implied_prob'])
        status = "⚠️  HIGH" if corr > 0.9 else "⚠️  MED" if corr > 0.7 else "✅ LOW"
        print(f"  {name:12}: r = {corr:.3f} {status}")

    print("\n⚠️  CONCLUSION: All models are highly correlated with Vegas implied probabilities!")
    print("   This means we have no independent edge - we're just following the market.")
    print("   To find alpha, we need features that Vegas doesn't incorporate.")

    return pred_df

def get_week16_injuries():
    """Get injury data for Week 16 teams using ESPN API."""
    from espn_data_fetcher import ESPNDataFetcher

    print("\n" + "="*60)
    print("🏥 WEEK 16 INJURY REPORT (ESPN API)")
    print("="*60)

    fetcher = ESPNDataFetcher()

    # Get current odds to know which teams are playing
    odds_df = fetcher.get_current_odds()

    # Get unique teams
    teams = set(odds_df['home_team'].dropna().tolist() +
                odds_df['away_team'].dropna().tolist())

    injury_summary = {}
    for team in sorted(teams):
        if team:
            injuries = fetcher.get_team_injuries(team)
            key_injuries = [i for i in injuries if i['status'] in ['Out', 'Doubtful', 'Questionable']]
            injury_summary[team] = {
                'total': len(injuries),
                'out': len([i for i in injuries if i['status'] == 'Out']),
                'doubtful': len([i for i in injuries if i['status'] == 'Doubtful']),
                'questionable': len([i for i in injuries if i['status'] == 'Questionable']),
                'key_players': [i['player_name'] for i in key_injuries[:3] if i['player_name']]
            }

    # Print summary
    print("\nTeam Injury Summary:")
    print("-" * 60)
    for team, data in sorted(injury_summary.items()):
        out_str = f"Out:{data['out']}" if data['out'] > 0 else ""
        doubt_str = f"Doubt:{data['doubtful']}" if data['doubtful'] > 0 else ""
        quest_str = f"Quest:{data['questionable']}" if data['questionable'] > 0 else ""
        status = " | ".join(filter(None, [out_str, doubt_str, quest_str]))
        key = ", ".join(data['key_players'][:2]) if data['key_players'] else "None"
        print(f"  {team:4}: {status:25} | Key: {key}")

    return injury_summary


if __name__ == "__main__":
    # Check Week 16 data
    week16 = check_week16_data()

    # Get injury data from ESPN
    try:
        injuries = get_week16_injuries()
    except Exception as e:
        print(f"Could not fetch injuries: {e}")

    # Analyze favorite bias
    analyze_favorite_bias()

    # Generate Week 16 predictions
    week16_predictions()

