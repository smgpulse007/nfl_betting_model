import pandas as pd

df = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
df_2025 = df[df['season'] == 2025]

print('2025 weeks available:')
print(sorted(df_2025['week'].unique()))

print(f'\nTotal 2025 games: {len(df_2025)}')
print(f'Week 16 games: {len(df_2025[df_2025["week"] == 16])}')
print(f'Week 17 games: {len(df_2025[df_2025["week"] == 17])}')

