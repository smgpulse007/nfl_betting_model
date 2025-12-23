"""
Visualize Week 16 Results - XGBoost vs Vegas Performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results = pd.read_csv("results/week16_results_analysis.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Week 16 Results: XGBoost vs Vegas Performance', fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
ax1 = axes[0, 0]
models = ['XGBoost', 'Vegas', 'Logistic']
accuracies = [
    results['xgb_correct'].mean(),
    results['vegas_correct'].mean(),
    results['log_correct'].mean()
]
colors = ['#2ecc71', '#e74c3c', '#3498db']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Overall Accuracy (15 Games)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (Random)')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1%}\n({int(acc*15)}/15)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.legend()

# 2. Performance on Disagreements
ax2 = axes[0, 1]
disagreements = results[results['xgb_vegas_deviation'].abs() > 0.05]
if len(disagreements) > 0:
    disagree_acc = [
        disagreements['xgb_correct'].mean(),
        disagreements['vegas_correct'].mean()
    ]
    bars2 = ax2.bar(['XGBoost', 'Vegas'], disagree_acc, 
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(f'Accuracy on Disagreements (>5% deviation)\n{len(disagreements)} games', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar, acc in zip(bars2, disagree_acc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.1%}\n({int(acc*len(disagreements))}/{len(disagreements)})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Calibration Plot
ax3 = axes[1, 0]
results['xgb_confidence'] = results['xgb_home_prob'].apply(lambda x: max(x, 1-x))
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results['conf_bin'] = pd.cut(results['xgb_confidence'], bins=bins)

calibration = results.groupby('conf_bin', observed=True).agg({
    'xgb_correct': ['count', 'mean']
}).reset_index()

if len(calibration) > 0:
    bin_labels = [f'{int(b*100)}-{int((b+0.1)*100)}%' for b in bins[:-1]]
    x_pos = range(len(calibration))
    
    bars3 = ax3.bar(x_pos, calibration[('xgb_correct', 'mean')], 
                    color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_xlabel('Confidence Range', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('XGBoost Calibration', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(b) for b in calibration['conf_bin']], rotation=45, ha='right')
    ax3.set_ylim([0, 1.1])
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    
    # Add count labels
    for i, (bar, row) in enumerate(zip(bars3, calibration.itertuples())):
        height = bar.get_height()
        count = int(row[2])  # count column
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.0%}\n(n={count})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.legend()

# 4. Deviation vs Correctness
ax4 = axes[1, 1]
results_sorted = results.sort_values('xgb_vegas_deviation')
colors_correct = ['#2ecc71' if c else '#e74c3c' for c in results_sorted['xgb_correct']]

ax4.scatter(results_sorted['xgb_vegas_deviation'], 
           range(len(results_sorted)),
           c=colors_correct, s=200, alpha=0.7, edgecolors='black', linewidth=2)
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='Vegas Line')
ax4.set_xlabel('XGBoost Deviation from Vegas', fontsize=12, fontweight='bold')
ax4.set_ylabel('Game Index', fontsize=12, fontweight='bold')
ax4.set_title('XGBoost Deviation vs Correctness\n(Green=Correct, Red=Wrong)', 
              fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add game labels for big deviations
for idx, row in results_sorted.iterrows():
    if abs(row['xgb_vegas_deviation']) > 0.15:
        y_pos = results_sorted.index.get_loc(idx)
        label = f"{row['away_team']}@{row['home_team']}"
        ax4.text(row['xgb_vegas_deviation'], y_pos, label, 
                fontsize=8, ha='left' if row['xgb_vegas_deviation'] > 0 else 'right',
                va='center')

plt.tight_layout()
plt.savefig('results/week16_performance_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to results/week16_performance_analysis.png")
plt.close()

# Create summary stats
print("\n" + "="*80)
print("WEEK 16 PERFORMANCE SUMMARY")
print("="*80)
print(f"\nOverall Accuracy:")
print(f"  XGBoost:  {results['xgb_correct'].mean():.1%} ({results['xgb_correct'].sum()}/15)")
print(f"  Vegas:    {results['vegas_correct'].mean():.1%} ({results['vegas_correct'].sum()}/15)")
print(f"  Edge:     {results['xgb_correct'].mean() - results['vegas_correct'].mean():+.1%}")

if len(disagreements) > 0:
    print(f"\nDisagreements (>5% deviation):")
    print(f"  XGBoost:  {disagreements['xgb_correct'].mean():.1%} ({disagreements['xgb_correct'].sum()}/{len(disagreements)})")
    print(f"  Vegas:    {disagreements['vegas_correct'].mean():.1%} ({disagreements['vegas_correct'].sum()}/{len(disagreements)})")
    print(f"  Edge:     {disagreements['xgb_correct'].mean() - disagreements['vegas_correct'].mean():+.1%}")

high_conf = results[results['xgb_confidence'] > 0.8]
if len(high_conf) > 0:
    print(f"\nHigh Confidence (>80%):")
    print(f"  Accuracy: {high_conf['xgb_correct'].mean():.1%} ({high_conf['xgb_correct'].sum()}/{len(high_conf)})")

print("\n" + "="*80)

