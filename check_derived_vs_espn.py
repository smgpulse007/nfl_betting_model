"""Check what was actually derived vs ESPN"""
import pandas as pd
import numpy as np

# Load ESPN
espn_stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
espn = pd.merge(espn_stats, espn_records, on='team', how='outer')
espn = espn.set_index('team').sort_index()

# Load schedules and derive points
schedules = pd.read_parquet('data/cache/schedules_2024.parquet')
schedules_reg = schedules[schedules['week'] <= 18]

teams = sorted(espn.index.unique())
derived_points = []

for team in teams:
    ts = schedules_reg[(schedules_reg['home_team']==team) | (schedules_reg['away_team']==team)].copy()
    ts.loc[:, 'is_home'] = ts['home_team']==team
    ts.loc[:, 'team_score'] = ts.apply(lambda x: x['home_score'] if x['is_home'] else x['away_score'], axis=1)
    derived_points.append({
        'team': team,
        'total_pointsFor': ts['team_score'].sum()
    })

derived = pd.DataFrame(derived_points).set_index('team')

# Compare
print("=" * 80)
print("COMPARISON: total_pointsFor")
print("=" * 80)

comparison = pd.DataFrame({
    'ESPN': espn['total_pointsFor'],
    'Derived': derived['total_pointsFor']
})
comparison['Diff'] = comparison['ESPN'] - comparison['Derived']
comparison['Match'] = comparison['Diff'] == 0

print(comparison)
print(f"\nPerfect matches: {comparison['Match'].sum()}/{len(comparison)}")
print(f"Mean absolute difference: {comparison['Diff'].abs().mean():.2f}")

# Calculate correlation
from scipy.stats import pearsonr
r, p = pearsonr(comparison['ESPN'], comparison['Derived'])
print(f"\nCorrelation: r={r:.4f}, p={p:.2e}")

print("\n" + "=" * 80)
print("ISSUE IDENTIFIED!")
print("=" * 80)

if comparison['Match'].sum() == len(comparison):
    print("✅ ALL teams have perfect matches!")
    print("✅ The derivation logic is CORRECT!")
    print("\n⚠️  But the correlation in comprehensive_validation.py was low (r=0.5056)")
    print("⚠️  This suggests there was a BUG in the comprehensive_validation.py script")
    print("⚠️  Likely issue: The derived dataframe didn't include total_pointsFor column")
else:
    print(f"❌ Only {comparison['Match'].sum()}/{len(comparison)} teams match")
    print("❌ There is a real data discrepancy")

