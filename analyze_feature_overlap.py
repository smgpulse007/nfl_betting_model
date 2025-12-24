"""
Analyze Feature Overlap Between ESPN Data and Existing nfl-data-py Features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("🔍 FEATURE OVERLAP ANALYSIS: ESPN vs nfl-data-py")
print("="*80)

# Load ESPN data to see what features we have
print("\n[1] Loading ESPN Data...")
espn_stats_2024 = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records_2024 = pd.read_parquet('data/espn_raw/team_records_2024.parquet')

print(f"ESPN Team Stats: {len(espn_stats_2024.columns)} columns")
print(f"ESPN Team Records: {len(espn_records_2024.columns)} columns")

# Load existing training data to see what features we already have
print("\n[2] Loading Existing Training Data...")
train_games = pd.read_parquet('data/processed/train_games.parquet')
print(f"Existing features: {len(train_games.columns)} columns")

# Print ESPN stat columns
print("\n[3] ESPN Team Stats Columns (sample):")
print("-"*80)
espn_stat_cols = sorted([col for col in espn_stats_2024.columns if col not in ['team', 'season']])
for i, col in enumerate(espn_stat_cols[:30]):
    print(f"  {i+1:3d}. {col}")
print(f"  ... and {len(espn_stat_cols) - 30} more")

# Print ESPN record columns
print("\n[4] ESPN Team Records Columns:")
print("-"*80)
espn_record_cols = sorted([col for col in espn_records_2024.columns if col not in ['team', 'season']])
for i, col in enumerate(espn_record_cols):
    print(f"  {i+1:3d}. {col}")

# Print existing feature columns
print("\n[5] Existing Training Data Columns:")
print("-"*80)
existing_cols = sorted([col for col in train_games.columns])
for i, col in enumerate(existing_cols[:40]):
    print(f"  {i+1:3d}. {col}")
print(f"  ... and {len(existing_cols) - 40} more")

# Analyze potential overlaps (by name similarity)
print("\n[6] Potential Feature Overlaps (by name):")
print("-"*80)

overlaps = []
for espn_col in espn_stat_cols + espn_record_cols:
    espn_lower = espn_col.lower()
    for existing_col in existing_cols:
        existing_lower = existing_col.lower()
        
        # Check for keyword matches
        keywords = ['pass', 'rush', 'score', 'yard', 'touchdown', 'interception', 
                   'fumble', 'penalty', 'sack', 'completion', 'attempt']
        
        for keyword in keywords:
            if keyword in espn_lower and keyword in existing_lower:
                overlaps.append({
                    'espn': espn_col,
                    'existing': existing_col,
                    'keyword': keyword
                })
                break

print(f"Found {len(overlaps)} potential overlaps")
if overlaps:
    for overlap in overlaps[:20]:
        print(f"  ESPN: {overlap['espn']:40s} <-> Existing: {overlap['existing']:40s} ({overlap['keyword']})")
    if len(overlaps) > 20:
        print(f"  ... and {len(overlaps) - 20} more")

# Categorize ESPN features
print("\n[7] ESPN Feature Categories:")
print("-"*80)

categories = {
    'Passing': [],
    'Rushing': [],
    'Receiving': [],
    'Defense': [],
    'Special Teams': [],
    'Turnovers': [],
    'Penalties': [],
    'Scoring': [],
    'Efficiency': [],
    'Records/Splits': [],
    'Other': []
}

for col in espn_stat_cols:
    col_lower = col.lower()
    if any(x in col_lower for x in ['pass', 'completion', 'qb']):
        categories['Passing'].append(col)
    elif any(x in col_lower for x in ['rush', 'run']):
        categories['Rushing'].append(col)
    elif any(x in col_lower for x in ['receiv', 'catch', 'target']):
        categories['Receiving'].append(col)
    elif any(x in col_lower for x in ['defense', 'defensive', 'tackle', 'sack']):
        categories['Defense'].append(col)
    elif any(x in col_lower for x in ['punt', 'kick', 'return', 'field_goal']):
        categories['Special Teams'].append(col)
    elif any(x in col_lower for x in ['turnover', 'interception', 'fumble']):
        categories['Turnovers'].append(col)
    elif any(x in col_lower for x in ['penalty', 'penalt']):
        categories['Penalties'].append(col)
    elif any(x in col_lower for x in ['score', 'point', 'touchdown']):
        categories['Scoring'].append(col)
    elif any(x in col_lower for x in ['percent', 'rate', 'avg', 'per']):
        categories['Efficiency'].append(col)
    else:
        categories['Other'].append(col)

for col in espn_record_cols:
    categories['Records/Splits'].append(col)

for category, features in categories.items():
    if features:
        print(f"\n{category}: {len(features)} features")
        for feat in features[:5]:
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")

# Save detailed analysis
print("\n[8] Saving Detailed Analysis...")
print("-"*80)

analysis = {
    'espn_stats_count': len(espn_stat_cols),
    'espn_records_count': len(espn_record_cols),
    'existing_features_count': len(existing_cols),
    'espn_stats_columns': espn_stat_cols,
    'espn_records_columns': espn_record_cols,
    'existing_columns': existing_cols,
    'potential_overlaps': overlaps,
    'categories': {k: v for k, v in categories.items() if v}
}

output_path = Path('results/feature_overlap_analysis.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"✅ Detailed analysis saved to: {output_path}")

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"ESPN Team Stats:        {len(espn_stat_cols)} features")
print(f"ESPN Team Records:      {len(espn_record_cols)} features")
print(f"Total ESPN Features:    {len(espn_stat_cols) + len(espn_record_cols)} features")
print(f"Existing Features:      {len(existing_cols)} features")
print(f"Potential Overlaps:     {len(overlaps)} features")
print(f"Likely NEW Features:    {len(espn_stat_cols) + len(espn_record_cols) - len(overlaps)} features")
print("="*80)

