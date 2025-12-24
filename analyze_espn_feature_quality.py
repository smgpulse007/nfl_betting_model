"""
Analyze ESPN Feature Quality and Suitability for Historical Imputation
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("🔬 ESPN FEATURE QUALITY ANALYSIS")
print("="*80)

# Load ESPN data
espn_2024 = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_2025 = pd.read_parquet('data/espn_raw/team_stats_2025.parquet')
records_2024 = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
records_2025 = pd.read_parquet('data/espn_raw/team_records_2025.parquet')

print(f"\n2024 Stats: {espn_2024.shape}")
print(f"2025 Stats: {espn_2025.shape}")
print(f"2024 Records: {records_2024.shape}")
print(f"2025 Records: {records_2025.shape}")

# Sample some key features
print("\n" + "="*80)
print("📊 SAMPLE ESPN FEATURES (2024 vs 2025)")
print("="*80)

# Select interesting features
sample_features = [
    'passing_completionPct',
    'passing_yardsPerGame',
    'passing_touchdowns',
    'passing_interceptions',
    'rushing_yardsPerGame',
    'rushing_touchdowns',
    'defensive_sacks',
    'defensive_interceptions',
    'general_turnovers',
    'general_totalPenalties',
    'general_totalPenaltyYards',
    'scoring_totalPointsPerGame',
]

# Check which features exist
available_features = [f for f in sample_features if f in espn_2024.columns]

if available_features:
    print("\nFeature Statistics (2024):")
    print("-"*80)
    for feat in available_features[:10]:
        if feat in espn_2024.columns:
            mean_2024 = espn_2024[feat].mean()
            std_2024 = espn_2024[feat].std()
            min_2024 = espn_2024[feat].min()
            max_2024 = espn_2024[feat].max()
            print(f"{feat:40s}: μ={mean_2024:8.2f}, σ={std_2024:7.2f}, range=[{min_2024:7.2f}, {max_2024:7.2f}]")

# Compare 2024 vs 2025 distributions
print("\n" + "="*80)
print("📈 2024 vs 2025 DISTRIBUTION COMPARISON")
print("="*80)

comparison = []
for feat in available_features[:15]:
    if feat in espn_2024.columns and feat in espn_2025.columns:
        mean_2024 = espn_2024[feat].mean()
        mean_2025 = espn_2025[feat].mean()
        std_2024 = espn_2024[feat].std()
        std_2025 = espn_2025[feat].std()
        
        pct_change = ((mean_2025 - mean_2024) / mean_2024 * 100) if mean_2024 != 0 else 0
        
        comparison.append({
            'feature': feat,
            'mean_2024': mean_2024,
            'mean_2025': mean_2025,
            'pct_change': pct_change,
            'std_2024': std_2024,
            'std_2025': std_2025
        })

if comparison:
    print(f"\n{'Feature':<40s} {'2024 Mean':>12s} {'2025 Mean':>12s} {'% Change':>10s}")
    print("-"*80)
    for item in comparison:
        print(f"{item['feature']:<40s} {item['mean_2024']:12.2f} {item['mean_2025']:12.2f} {item['pct_change']:9.1f}%")

# Analyze game evolution: Compare 2024 to what 1999 might have looked like
print("\n" + "="*80)
print("⚠️  IMPUTATION RISK ANALYSIS")
print("="*80)

print("\nKnown NFL Evolution Trends (1999 → 2024):")
print("-"*80)
print("1. Passing Volume:     ↑ ~40% (rule changes favor passing)")
print("2. Completion %:       ↑ ~10% (shorter passes, better QBs)")
print("3. Rushing Attempts:   ↓ ~20% (pass-heavy offenses)")
print("4. Defensive Sacks:    ↑ ~15% (more pass attempts)")
print("5. Penalties:          ↑ ~25% (more rules, more flags)")
print("6. Points Per Game:    ↑ ~15% (offensive-friendly rules)")
print("7. Turnovers:          ↓ ~20% (better QB play)")

print("\n⚠️  HIGH-RISK Features for Historical Imputation:")
print("-"*80)
high_risk = [
    ('passing_*', 'Passing stats have changed dramatically (rule changes)'),
    ('defensive_sacks', 'Sack rates tied to passing volume'),
    ('general_totalPenalties', 'Penalty enforcement has increased'),
    ('scoring_totalPointsPerGame', 'Scoring has increased significantly'),
    ('rushing_yardsPerGame', 'Rushing volume has decreased'),
    ('general_turnovers', 'Turnover rates have decreased'),
]

for feat, reason in high_risk:
    print(f"  ❌ {feat:30s} - {reason}")

print("\n✅ LOW-RISK Features for Historical Imputation:")
print("-"*80)
low_risk = [
    ('home_winPercent', 'Home field advantage relatively stable'),
    ('total_divisionWinPercent', 'Division strength varies but structure same'),
    ('Efficiency ratios', 'Relative metrics less affected by era'),
]

for feat, reason in low_risk:
    print(f"  ✅ {feat:30s} - {reason}")

# Recommendation
print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

print("\n1. ❌ DO NOT IMPUTE ESPN features for 1999-2023")
print("   Reasons:")
print("   - NFL has evolved significantly (passing revolution)")
print("   - 2024 stats are NOT representative of 1999-2015 era")
print("   - Would introduce systematic bias and noise")
print("   - Tree models can't distinguish real vs imputed patterns")

print("\n2. ✅ ALTERNATIVE STRATEGIES:")
print("\n   Option A: ESPN Features for Recent Years Only (2024-2025)")
print("   - Train base model on 1999-2023 WITHOUT ESPN features")
print("   - Add ESPN features ONLY for 2024-2025 predictions")
print("   - Use separate model or ensemble approach")
print("   - Pros: No imputation bias, leverages all historical data")
print("   - Cons: Can't validate ESPN features on historical data")

print("\n   Option B: Collect ESPN Data for 2018-2023 (if available)")
print("   - Attempt to get ESPN data for recent 6 years")
print("   - Train on 2018-2023 with ESPN features (~1,672 games)")
print("   - Pros: Consistent feature set, better validation")
print("   - Cons: Loses 1999-2017 data (~5,034 games)")

print("\n   Option C: Hybrid Two-Stage Model")
print("   - Stage 1: Base model on 1999-2023 (TIER S+A only)")
print("   - Stage 2: ESPN enhancement layer for 2024-2025")
print("   - Combine predictions via ensemble/stacking")
print("   - Pros: Best of both worlds, isolates ESPN contribution")
print("   - Cons: More complex, requires careful validation")

print("\n   Option D: Feature Engineering from nfl-data-py")
print("   - Create ESPN-like features from nfl-data-py for all years")
print("   - E.g., compute team fumbles, penalties from play-by-play")
print("   - Pros: Consistent across all years, no imputation")
print("   - Cons: More work, may not match ESPN exactly")

print("\n3. 🎯 RECOMMENDED: Option C (Hybrid Two-Stage Model)")
print("   - Maximizes historical data usage")
print("   - Avoids imputation bias")
print("   - Allows clean measurement of ESPN feature value")
print("   - Can fall back to base model if ESPN doesn't help")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)

