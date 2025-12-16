"""Predict tonight's game: MIA @ PIT"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels

# Load and train
games = pd.read_parquet(PROCESSED_DATA_DIR / 'games_with_features.parquet')
models = NFLBettingModels()
models.fit(games)

# MIA @ PIT - Tonight's game from screenshot
# PIT -3.5, Total 42.5, PIT 63% implied
game = pd.DataFrame([{
    'game_id': 'MIA_PIT_2025_15',
    'season': 2025, 'week': 15,
    'home_team': 'PIT', 'away_team': 'MIA',
    'spread_line': -3.5,
    'total_line': 42.5,
    'elo_diff': 45,  # PIT slightly better Elo
    'elo_prob': 0.57,
    'home_rest': 7, 'away_rest': 7,
    'rest_advantage': 0,
    'temp': 35, 'wind': 5,
    'is_dome': 0, 'is_cold': 1,
    'div_game': 0,
    'home_implied_prob': 0.63,
    'home_moneyline': -170,
    'away_moneyline': 145,
}])

preds = models.predict(game)

print('=' * 60)
print('MIA @ PIT - Monday Night Football (Week 15)')
print('=' * 60)
print()
print('VEGAS LINES (from your screenshot):')
print('  Spread: PIT -3.5')
print('  Total: 42.5')
print('  Moneyline: PIT 63% | MIA 38%')
print()
print('MODEL PREDICTIONS:')
ens = preds['ensemble_prob'].values[0]
print(f"  Win Prob: PIT {ens:.1%} | MIA {1-ens:.1%}")
margin = preds['pred_margin'].values[0]
print(f"  Pred Margin: PIT by {margin:.1f} pts")
total = preds['pred_total'].values[0]
print(f"  Pred Total: {total:.1f} pts")
print()
print('SPREAD ANALYSIS:')
print('  Vegas: PIT -3.5')
print(f"  Model: PIT by {margin:.1f}")
home_cover = preds['home_cover_prob'].values[0]
print(f"  P(PIT covers -3.5): {home_cover:.1%}")
print(f"  P(MIA covers +3.5): {1-home_cover:.1%}")
edge = abs(home_cover - 0.5)
side = "PIT -3.5" if home_cover > 0.5 else "MIA +3.5"
print(f"  Edge: {edge:.1%} on {side}")
print()
print('TOTALS ANALYSIS:')
print('  Vegas: 42.5')
print(f"  Model: {total:.1f}")
over_prob = preds['over_prob'].values[0]
print(f"  P(Over 42.5): {over_prob:.1%}")
print(f"  P(Under 42.5): {1-over_prob:.1%}")
edge_ou = abs(over_prob - 0.5)
side_ou = "OVER" if over_prob > 0.5 else "UNDER"
print(f"  Edge: {edge_ou:.1%} on {side_ou}")
print()
print('MONEYLINE ANALYSIS:')
vegas_prob = 0.63
ml_edge = ens - vegas_prob
print('  Vegas: PIT 63%')
print(f"  Model: PIT {ens:.1%}")
ml_side = "PIT ML" if ml_edge > 0 else "MIA ML"
print(f"  Edge: {abs(ml_edge):.1%} on {ml_side}")
print()
print('=' * 60)
print('RECOMMENDED BETS (edge >= 2%):')
print('=' * 60)
recs = []
if abs(ml_edge) >= 0.02:
    recs.append(f"  -> {ml_side} ({abs(ml_edge):.1%} edge)")
if edge >= 0.02:
    recs.append(f"  -> {side} ({edge:.1%} edge)")
if edge_ou >= 0.02:
    recs.append(f"  -> {side_ou} 42.5 ({edge_ou:.1%} edge)")
if recs:
    for r in recs:
        print(r)
else:
    print("  No bets meet 2% edge threshold")

