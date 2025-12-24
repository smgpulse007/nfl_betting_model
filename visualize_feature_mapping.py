"""
Visualize ESPN ↔ nfl-data-py feature mapping results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def create_mapping_visualization():
    """Create comprehensive visualization of feature mapping"""
    
    # Load mapping data
    df = pd.read_csv('results/comprehensive_feature_mapping.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ESPN ↔ nfl-data-py Feature Mapping Analysis', fontsize=16, fontweight='bold')
    
    # 1. Category Distribution (Pie Chart)
    ax1 = axes[0, 0]
    category_counts = df['category'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    explode = (0.05, 0, 0, 0.1)  # Explode EXACT MATCH and CANNOT DERIVE
    
    ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Feature Mapping Categories\n(Total: 323 features)', fontweight='bold')
    
    # Add legend with counts
    legend_labels = [f'{cat}: {count}' for cat, count in category_counts.items()]
    ax1.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    # 2. Feature Type Distribution (Horizontal Bar Chart)
    ax2 = axes[0, 1]
    feature_type_counts = df['feature_type'].value_counts().head(10)
    
    bars = ax2.barh(range(len(feature_type_counts)), feature_type_counts.values, color='#3498db')
    ax2.set_yticks(range(len(feature_type_counts)))
    ax2.set_yticklabels(feature_type_counts.index)
    ax2.set_xlabel('Number of Features')
    ax2.set_title('Top 10 Feature Types', fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, feature_type_counts.values)):
        ax2.text(value + 0.5, i, str(value), va='center', fontsize=9)
    
    # 3. Derivability by Feature Type (Stacked Bar Chart)
    ax3 = axes[1, 0]
    
    # Get top 8 feature types
    top_types = df['feature_type'].value_counts().head(8).index
    df_top = df[df['feature_type'].isin(top_types)]
    
    # Create pivot table
    pivot = pd.crosstab(df_top['feature_type'], df_top['category'])
    
    # Reorder columns
    col_order = ['EXACT MATCH', 'PARTIAL MATCH', 'CLOSE APPROXIMATION', 'CANNOT DERIVE']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns], fill_value=0)
    
    # Plot stacked bar chart
    pivot.plot(kind='bar', stacked=True, ax=ax3, 
               color=['#2ecc71', '#f39c12', '#3498db', '#e74c3c'])
    ax3.set_title('Derivability by Feature Type (Top 8)', fontweight='bold')
    ax3.set_xlabel('Feature Type')
    ax3.set_ylabel('Number of Features')
    ax3.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Data Source Distribution (Pie Chart)
    ax4 = axes[1, 1]
    source_counts = df['nfl_data_py_source'].value_counts()
    colors_source = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    ax4.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
            colors=colors_source, startangle=90, textprops={'fontsize': 9})
    ax4.set_title('nfl-data-py Data Sources', fontweight='bold')
    
    # Add legend with counts
    legend_labels_source = [f'{src}: {count}' for src, count in source_counts.items()]
    ax4.legend(legend_labels_source, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/feature_mapping_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.close()

def create_summary_table():
    """Create summary table"""
    df = pd.read_csv('results/comprehensive_feature_mapping.csv')
    
    print("\n" + "=" * 80)
    print("FEATURE MAPPING SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Total Features: {len(df)}")
    
    print("\n📋 By Category:")
    print(df['category'].value_counts().to_string())
    
    print("\n📋 By Confidence:")
    print(df['confidence'].value_counts().to_string())
    
    print("\n📋 By Feature Type (Top 10):")
    print(df['feature_type'].value_counts().head(10).to_string())
    
    print("\n📋 By Data Source:")
    print(df['nfl_data_py_source'].value_counts().to_string())
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    exact_match = len(df[df['category'] == 'EXACT MATCH'])
    partial_match = len(df[df['category'] == 'PARTIAL MATCH'])
    close_approx = len(df[df['category'] == 'CLOSE APPROXIMATION'])
    cannot_derive = len(df[df['category'] == 'CANNOT DERIVE'])
    
    derivable = exact_match + partial_match + close_approx
    derivable_pct = derivable / len(df) * 100
    
    print(f"\n✅ Derivable Features: {derivable} ({derivable_pct:.1f}%)")
    print(f"   - EXACT MATCH: {exact_match}")
    print(f"   - PARTIAL MATCH: {partial_match}")
    print(f"   - CLOSE APPROXIMATION: {close_approx}")
    
    print(f"\n🆕 Truly NEW Features: {cannot_derive} ({cannot_derive/len(df)*100:.1f}%)")
    
    print("\n🆕 Truly NEW ESPN Features:")
    cannot_derive_features = df[df['category'] == 'CANNOT DERIVE'][['espn_feature', 'notes']]
    for idx, row in cannot_derive_features.iterrows():
        print(f"   - {row['espn_feature']}: {row['notes']}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    print("Creating feature mapping visualization...")
    create_mapping_visualization()
    create_summary_table()
    print("\n✅ Visualization complete!")

