"""
Inspect ESPN features and create detailed feature catalog
"""
import pandas as pd
import json
from pathlib import Path

def inspect_espn_team_stats():
    """Inspect ESPN team stats features"""
    df = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
    
    print("=" * 80)
    print("ESPN TEAM STATS FEATURE CATALOG")
    print("=" * 80)
    print(f"\nTotal Features: {len(df.columns)}")
    print(f"Total Teams: {len(df)}")
    
    # Group by prefix
    categories = {}
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append(col)
        else:
            if 'general' not in categories:
                categories['general'] = []
            categories['general'].append(col)
    
    print(f"\nFeature Categories: {len(categories)}")
    for category, features in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(features)} features):")
        for feat in sorted(features)[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")
    
    # Sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First Team)")
    print("=" * 80)
    sample = df.iloc[0]
    print(f"\nTeam: {sample.get('team', 'N/A')}")
    print(f"Season: {sample.get('season', 'N/A')}")
    
    # Show some key stats
    key_stats = [
        'passing_passingYards', 'passing_passingTouchdowns', 'passing_completionPct',
        'rushing_rushingYards', 'rushing_rushingTouchdowns',
        'receiving_receivingYards', 'receiving_receivingTouchdowns',
        'defense_totalSacks', 'defense_interceptions'
    ]
    
    print("\nKey Stats:")
    for stat in key_stats:
        if stat in df.columns:
            print(f"  {stat}: {sample[stat]}")
    
    return df, categories

def inspect_espn_team_records():
    """Inspect ESPN team records features"""
    df = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
    
    print("\n" + "=" * 80)
    print("ESPN TEAM RECORDS FEATURE CATALOG")
    print("=" * 80)
    print(f"\nTotal Features: {len(df.columns)}")
    print(f"Total Teams: {len(df)}")
    
    # Group by prefix
    categories = {}
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append(col)
        else:
            if 'metadata' not in categories:
                categories['metadata'] = []
            categories['metadata'].append(col)
    
    print(f"\nFeature Categories: {len(categories)}")
    for category, features in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(features)} features):")
        for feat in sorted(features):
            print(f"  - {feat}")
    
    # Sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First Team)")
    print("=" * 80)
    sample = df.iloc[0]
    print(f"\nTeam: {sample.get('team', 'N/A')}")
    print(f"Season: {sample.get('season', 'N/A')}")
    
    print("\nAll Values:")
    for col in df.columns:
        if col not in ['team_id', 'team', 'season', 'season_type']:
            print(f"  {col}: {sample[col]}")
    
    return df, categories

def create_feature_catalog():
    """Create comprehensive feature catalog"""
    stats_df, stats_cats = inspect_espn_team_stats()
    records_df, records_cats = inspect_espn_team_records()
    
    catalog = {
        'team_stats': {
            'total_features': len(stats_df.columns),
            'categories': {cat: len(feats) for cat, feats in stats_cats.items()},
            'features': {cat: feats for cat, feats in stats_cats.items()}
        },
        'team_records': {
            'total_features': len(records_df.columns),
            'categories': {cat: len(feats) for cat, feats in records_cats.items()},
            'features': {cat: feats for cat, feats in records_cats.items()}
        }
    }
    
    # Save to JSON
    output_path = Path('results/espn_feature_catalog.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Feature catalog saved to: {output_path}")
    print("=" * 80)
    
    return catalog

if __name__ == '__main__':
    catalog = create_feature_catalog()

