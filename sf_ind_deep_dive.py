"""
SF vs IND Deep Dive Analysis

Check if this game is in Week 16 or Week 17 and provide detailed analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import nfl_data_py as nfl
from espn_data_fetcher import ESPNDataFetcher

def analyze_sf_ind():
    print("="*100)
    print("SF vs IND GAME ANALYSIS")
    print("="*100)
    
    # Load 2025 schedule
    schedule_2025 = nfl.import_schedules([2025])
    
    # Find SF vs IND game
    sf_ind = schedule_2025[
        ((schedule_2025['home_team'] == 'SF') & (schedule_2025['away_team'] == 'IND')) |
        ((schedule_2025['home_team'] == 'IND') & (schedule_2025['away_team'] == 'SF'))
    ]
    
    if len(sf_ind) == 0:
        print("\n❌ No SF vs IND game found in 2025 schedule")
        return
    
    game = sf_ind.iloc[0]
    print(f"\n📅 Game Found:")
    print(f"  Week: {game['week']}")
    print(f"  Date: {game['gameday']}")
    print(f"  Matchup: {game['away_team']} @ {game['home_team']}")
    print(f"  Spread: {game['home_team']} {game['spread_line']:+.1f}")
    
    # Check if completed
    if pd.notna(game['home_score']):
        print(f"\n✅ GAME COMPLETED")
        print(f"  Final Score: {game['away_team']} {int(game['away_score'])} - {int(game['home_score'])} {game['home_team']}")
        print(f"  Winner: {game['home_team'] if game['home_score'] > game['away_score'] else game['away_team']}")
        print(f"  Margin: {abs(game['home_score'] - game['away_score']):.0f}")
        
        # Load Week 16 predictions if this was Week 16
        if game['week'] == 16:
            try:
                week16_preds = pd.read_csv("results/week16_2025_all_models.csv")
                pred = week16_preds[
                    (week16_preds['home_team'] == game['home_team']) &
                    (week16_preds['away_team'] == game['away_team'])
                ]
                
                if len(pred) > 0:
                    pred = pred.iloc[0]
                    print(f"\n📊 PREDICTION ANALYSIS:")
                    print(f"  XGBoost predicted: {game['home_team']} {pred['XGBoost_proba']:.1%}")
                    print(f"  Vegas implied: {game['home_team']} {pred['home_implied_prob']:.1%}")
                    
                    actual_home_win = game['home_score'] > game['away_score']
                    xgb_correct = (pred['XGBoost_proba'] > 0.5) == actual_home_win
                    vegas_correct = (pred['home_implied_prob'] > 0.5) == actual_home_win
                    
                    print(f"\n  XGBoost: {'✅ CORRECT' if xgb_correct else '❌ WRONG'}")
                    print(f"  Vegas: {'✅ CORRECT' if vegas_correct else '❌ WRONG'}")
                    
            except Exception as e:
                print(f"\n  Could not load predictions: {e}")
    else:
        print(f"\n⏳ GAME UPCOMING")
        
        # Get live data
        fetcher = ESPNDataFetcher()
        
        # Get live odds
        print(f"\n📊 LIVE ODDS:")
        try:
            live_odds = fetcher.get_current_odds()
            game_odds = live_odds[
                (live_odds['home_team'] == game['home_team']) &
                (live_odds['away_team'] == game['away_team'])
            ]
            
            if len(game_odds) > 0:
                odds = game_odds.iloc[0]
                print(f"  Spread: {game['home_team']} {odds['spread']:+.1f}")
                print(f"  Moneyline: {game['home_team']} {odds['home_ml']:+.0f} / {game['away_team']} {odds['away_ml']:+.0f}")
            else:
                print(f"  Spread: {game['home_team']} {game['spread_line']:+.1f}")
        except Exception as e:
            print(f"  Could not fetch live odds: {e}")
        
        # Get injuries
        print(f"\n🏥 INJURY REPORT:")
        try:
            injuries = fetcher.get_all_injuries()
            
            for team in [game['home_team'], game['away_team']]:
                team_inj = injuries[injuries['team'] == team]
                if len(team_inj) > 0:
                    out = team_inj[team_inj['status'] == 'Out']
                    questionable = team_inj[team_inj['status'] == 'Questionable']
                    doubtful = team_inj[team_inj['status'] == 'Doubtful']
                    
                    print(f"\n  {team}:")
                    print(f"    OUT: {len(out)}")
                    if len(out) > 0:
                        for _, player in out.head(5).iterrows():
                            print(f"      - {player['name']} ({player['position']})")
                    
                    print(f"    DOUBTFUL: {len(doubtful)}")
                    print(f"    QUESTIONABLE: {len(questionable)}")
                    if len(questionable) > 0:
                        for _, player in questionable.head(3).iterrows():
                            print(f"      - {player['name']} ({player['position']})")
        except Exception as e:
            print(f"  Could not fetch injuries: {e}")
        
        # Get XGBoost prediction
        week = int(game['week'])
        print(f"\n🤖 XGBOOST PREDICTION:")
        try:
            if week == 16:
                preds = pd.read_csv("results/week16_2025_all_models.csv")
            elif week == 17:
                preds = pd.read_csv("results/week17_predictions.csv")
            else:
                preds = None
            
            if preds is not None:
                pred = preds[
                    (preds['home_team'] == game['home_team']) &
                    (preds['away_team'] == game['away_team'])
                ]
                
                if len(pred) > 0:
                    pred = pred.iloc[0]
                    
                    if 'xgb_home_prob' in pred:
                        xgb_prob = pred['xgb_home_prob']
                        vegas_prob = pred['vegas_home_prob']
                    else:
                        xgb_prob = pred['XGBoost_proba']
                        vegas_prob = pred['home_implied_prob']
                    
                    xgb_pick = game['home_team'] if xgb_prob > 0.5 else game['away_team']
                    vegas_pick = game['home_team'] if vegas_prob > 0.5 else game['away_team']
                    
                    print(f"  XGBoost: {xgb_pick} ({max(xgb_prob, 1-xgb_prob):.1%} confidence)")
                    print(f"  Vegas: {vegas_pick} ({max(vegas_prob, 1-vegas_prob):.1%} confidence)")
                    print(f"  Deviation: {xgb_prob - vegas_prob:+.1%}")
                    
                    if abs(xgb_prob - vegas_prob) > 0.10:
                        print(f"\n  🔥 SIGNIFICANT DISAGREEMENT (>10%)")
                        if xgb_prob > vegas_prob:
                            print(f"     XGBoost is MORE confident in {game['home_team']}")
                        else:
                            print(f"     XGBoost is MORE confident in {game['away_team']}")
        except Exception as e:
            print(f"  Could not load predictions: {e}")


if __name__ == "__main__":
    analyze_sf_ind()

