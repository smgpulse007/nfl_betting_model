"""
Audit current features to identify Vegas-dependent vs independent features
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the 2025 evaluation results
results_file = Path('results/2025_week1_16_evaluation.csv')
results = pd.read_csv(results_file)

print("="*100)
print("CURRENT FEATURE AUDIT - VEGAS DEPENDENCY ANALYSIS")
print("="*100)

# Get all columns
all_cols = results.columns.tolist()

print(f"\nTotal columns: {len(all_cols)}")
print(f"Total games: {len(results)}")

# Categorize features
vegas_dependent = []
independent = []
derived = []
target = []
metadata = []

for col in all_cols:
    col_lower = col.lower()
    
    # Vegas-dependent features
    if any(x in col_lower for x in ['spread', 'total_line', 'moneyline', 'implied', 'vegas', 'over_under']):
        vegas_dependent.append(col)
    
    # Target variables
    elif any(x in col_lower for x in ['home_win', 'home_cover', 'over_hit', 'result', 'margin']):
        target.append(col)
    
    # Predictions/probabilities (derived)
    elif any(x in col_lower for x in ['prob', 'pred', 'pick', 'xgb', 'lr_', 'elo_']):
        derived.append(col)
    
    # Metadata
    elif any(x in col_lower for x in ['game_id', 'season', 'week', 'team', 'date', 'gameday']):
        metadata.append(col)
    
    # Independent features
    else:
        independent.append(col)

print("\n" + "="*100)
print("VEGAS-DEPENDENT FEATURES (❌ Remove or reduce weight)")
print("="*100)
for col in sorted(vegas_dependent):
    print(f"  - {col}")
print(f"\nTotal: {len(vegas_dependent)}")

print("\n" + "="*100)
print("INDEPENDENT FEATURES (✅ Keep and expand)")
print("="*100)
for col in sorted(independent):
    # Show sample values
    sample = results[col].dropna().head(3).tolist()
    print(f"  - {col}: {sample}")
print(f"\nTotal: {len(independent)}")

print("\n" + "="*100)
print("TARGET VARIABLES")
print("="*100)
for col in sorted(target):
    print(f"  - {col}")
print(f"\nTotal: {len(target)}")

print("\n" + "="*100)
print("DERIVED/PREDICTION COLUMNS")
print("="*100)
for col in sorted(derived):
    print(f"  - {col}")
print(f"\nTotal: {len(derived)}")

print("\n" + "="*100)
print("METADATA COLUMNS")
print("="*100)
for col in sorted(metadata):
    print(f"  - {col}")
print(f"\nTotal: {len(metadata)}")

# Correlation analysis
print("\n" + "="*100)
print("CORRELATION WITH VEGAS LINES")
print("="*100)

# Check if we have the raw features
if 'spread_line' in results.columns and 'xgb_win_prob' in results.columns:
    # Compute correlation between XGBoost predictions and Vegas lines
    
    # Convert spread to implied home win probability
    # Rough approximation: spread of -3 means ~60% home win prob
    results['spread_implied_prob'] = results['spread_line'].apply(
        lambda x: 0.5 + (x / 28) if pd.notna(x) else np.nan  # Normalize spread to probability
    )
    
    corr = results[['xgb_win_prob', 'spread_implied_prob']].corr().iloc[0, 1]
    print(f"\nXGBoost Win Prob vs Spread Implied Prob: {corr:.3f}")
    
    if 'home_moneyline' in results.columns:
        # Convert moneyline to implied probability
        def ml_to_prob(ml):
            if pd.isna(ml):
                return np.nan
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return -ml / (-ml + 100)
        
        results['ml_implied_prob'] = results['home_moneyline'].apply(ml_to_prob)
        corr_ml = results[['xgb_win_prob', 'ml_implied_prob']].corr().iloc[0, 1]
        print(f"XGBoost Win Prob vs Moneyline Implied Prob: {corr_ml:.3f}")

# Feature importance analysis (if we have the model)
print("\n" + "="*100)
print("MISSING INDEPENDENT FEATURES (ESPN API Can Provide)")
print("="*100)

missing_features = [
    "Team Offensive Stats (points/game, yards/game, red zone %)",
    "Team Defensive Stats (points allowed, yards allowed, sacks)",
    "QB Performance (rating, completion %, yards/attempt)",
    "RB Performance (yards/carry, receptions)",
    "WR Performance (targets, catch rate, YAC)",
    "Home/Away Splits (win %, points scored/allowed)",
    "Recent Form (last 3-5 games performance)",
    "Turnover Differential",
    "Third Down Conversion %",
    "Time of Possession",
    "Sack Rate (offensive and defensive)",
    "Playoff Implications (clinching scenarios)",
    "Strength of Schedule",
    "Live 2025 Injury Data (current roster status)",
    "Depth Chart Changes",
    "Coaching Matchups",
    "Rivalry Game Flag"
]

for i, feature in enumerate(missing_features, 1):
    print(f"  {i}. {feature}")

print("\n" + "="*100)
print("RECOMMENDATIONS")
print("="*100)

print("""
1. **Reduce Vegas Dependency:**
   - Remove or downweight: spread_line, total_line, moneyline odds
   - Keep Elo (independent team strength metric)
   
2. **Add Independent Team Stats (ESPN API):**
   - Offensive efficiency: points/game, yards/game, red zone %
   - Defensive efficiency: points allowed, yards allowed, sacks
   - Turnover differential
   
3. **Add Player-Level Stats (ESPN API):**
   - QB: rating, completion %, yards/attempt, TD/INT ratio
   - RB: yards/carry, receptions
   - WR: targets, catch rate
   
4. **Add Context Features (ESPN API):**
   - Home/away splits
   - Recent form (3-game, 5-game windows)
   - Playoff implications
   
5. **Fix 2025 Data Gap:**
   - Use ESPN roster API for live injury data
   - Replace imputed 2025 injuries with real data
   
6. **Expected Impact:**
   - Reduce XGBoost-Vegas correlation from 0.93 to <0.85
   - Increase high-confidence picks from 15% to 25%
   - Improve overall accuracy from 68.5% to 75%+
""")

