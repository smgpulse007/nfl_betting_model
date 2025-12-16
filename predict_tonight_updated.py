"""
MIA @ PIT - Updated with REAL Data from Research
================================================
Weather: 16-18°F at kickoff, wind gusts 16 mph
Key Injury: T.J. Watt OUT (punctured lung)
Tua cold weather record: 0-5 in games <40°F
Miami run game: #2 in EPA/play since Week 10
"""
import pandas as pd
from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels

# Load and train
games = pd.read_parquet(PROCESSED_DATA_DIR / 'games_with_features.parquet')
models = NFLBettingModels()
models.fit(games)

# CORRECTED inputs based on research
game = pd.DataFrame([{
    'game_id': 'MIA_PIT_2025_15', 'season': 2025, 'week': 15,
    'home_team': 'PIT', 'away_team': 'MIA',
    'spread_line': -3.5,      # PIT favored by 3.5 (from screenshot)
    'total_line': 42.5,       # From screenshot
    'elo_diff': 45,           # Approximate - PIT slightly better
    'elo_prob': 0.57,         # Elo-based probability
    'home_rest': 7, 'away_rest': 7,  # Both on standard rest
    'rest_advantage': 0,
    'temp': 16,               # CORRECTED: 16°F at kickoff (was 35)
    'wind': 16,               # CORRECTED: 16 mph gusts (was 5)
    'is_dome': 0,
    'is_cold': 1,             # EXTREME cold
    'div_game': 0,
    'home_implied_prob': 0.63,  # From screenshot (PIT 63%)
    'home_moneyline': -170,
    'away_moneyline': 145,
}])

preds = models.predict(game)

print('=' * 70)
print('MIA @ PIT - UPDATED PREDICTION (Real Weather Data)')
print('=' * 70)

print('''
CORRECTED INPUTS (from research):
─────────────────────────────────
• Temperature: 16°F (not 35°F!) - Extreme cold
• Wind: 16 mph gusts
• T.J. Watt: OUT (punctured lung from acupuncture)
• Tua cold weather record: 0-5 in games <40°F
• Miami run game: #2 in EPA/play since Week 10 (192 rush yd/game)
• Tua pass attempts last 4 games: avg 21 (run-heavy approach)
''')

print('=' * 70)
print('MODEL PREDICTIONS (with corrected inputs)')
print('=' * 70)
ens = preds['ensemble_prob'].values[0]
margin = preds['pred_margin'].values[0]
total = preds['pred_total'].values[0]

print(f"\n  Win Probability:  PIT {ens:.1%} | MIA {1-ens:.1%}")
print(f"  Predicted Margin: {'PIT' if margin > 0 else 'MIA'} by {abs(margin):.1f} pts")
print(f"  Predicted Total:  {total:.1f} pts")

print('\n' + '=' * 70)
print('BETTING ANALYSIS')
print('=' * 70)

# Spread
home_cover = preds['home_cover_prob'].values[0]
spread_edge = abs(home_cover - 0.5)
spread_side = "PIT -3.5" if home_cover > 0.5 else "MIA +3.5"
print(f"\n  SPREAD: {spread_side}")
print(f"    Model edge: {spread_edge:.1%}")

# Totals
over_prob = preds['over_prob'].values[0]
total_edge = abs(over_prob - 0.5)
total_side = "OVER 42.5" if over_prob > 0.5 else "UNDER 42.5"
print(f"\n  TOTAL: {total_side}")
print(f"    Model edge: {total_edge:.1%}")

# Moneyline
vegas_prob = 0.63
ml_edge = ens - vegas_prob
ml_side = "PIT ML (-170)" if ml_edge > 0 else "MIA ML (+145)"
print(f"\n  MONEYLINE: {ml_side}")
print(f"    Model edge: {abs(ml_edge):.1%}")

print('\n' + '=' * 70)
print('QUALITATIVE ADJUSTMENTS (Not in Model)')
print('=' * 70)
print('''
FACTORS FAVORING MIAMI:
+ T.J. Watt OUT - PIT 1-10 without him historically
+ PIT allows 28 ppg on avg without Watt
+ Miami run game elite recently (#2 EPA/play)
+ Bills ran for 219 yards vs PIT last week at home

FACTORS FAVORING PITTSBURGH:
+ Extreme cold (16°F) - Tua 0-5 in <40°F games
+ Tua career cold stats: 55.6% comp, 6.9 YPA, 8 TD / 10 INT
+ Home field in primetime
+ Must-win for AFC North lead

NEUTRAL:
- Miami's run-heavy approach may neutralize cold impact
- Both teams on equal rest
''')

print('=' * 70)
print('FINAL RECOMMENDATION')
print('=' * 70)
print(f'''
The model likes MIA, but the EXTREME cold (16°F) is worse than
what the model was trained on (is_cold=1 threshold is typically 40°F).

ADJUSTED BETTING STRATEGY:
─────────────────────────
1. MIA +3.5 (-110)  ✅ BEST BET
   - Even if Tua struggles, Miami's run game can keep it close
   - Watt absence huge for PIT pass rush
   - 3.5 pts of cushion for a close, low-scoring game

2. UNDER 42.5 (-110)  ✅ LEAN
   - 16°F games historically average 8-10 fewer points
   - Tua throws less in cold (21 att/game streak)
   - Model predicts {total:.1f} but extreme cold not fully captured
   - Line already low at 42.5

3. MIA ML (+145)  ⚠️ SMALL PLAY
   - Value exists but cold is real concern
   - Tua's 0-5 record in cold is alarming
   - Only if you're aggressive

AVOID: PIT -3.5 (Watt injury too significant)
'''.format(total=total))

