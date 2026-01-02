import pandas as pd

df = pd.read_parquet('../results/phase8_results/game_level_features_2025_weeks1_16_engineered.parquet')

print(f'Shape: {df.shape}')
print(f'\nColumns: {len(df.columns)}')

injury_cols = [c for c in df.columns if 'injury' in c.lower() or 'qb_out' in c.lower()]
print(f'\nInjury columns: {len(injury_cols)}')
print(injury_cols[:20] if len(injury_cols) > 20 else injury_cols)

weather_cols = [c for c in df.columns if c in ['temp', 'wind', 'temp_extreme', 'wind_high', 'is_outdoor', 'roof', 'surface']]
print(f'\nWeather columns: {len(weather_cols)}')
print(weather_cols)

print(f'\nWeeks: {sorted(df["week"].unique())}')
print(f'\nSample columns:')
print(df.columns.tolist()[:30])

