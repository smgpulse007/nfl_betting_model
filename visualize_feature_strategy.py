"""
Visualize Feature Strategy: Imputation vs Derivation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# ============================================================================
# Plot 1: Feature Availability Comparison
# ============================================================================
categories = ['Existing\nTIER S+A', 'Derived from\nnfl-data-py', 'Truly NEW\nESPN', 'TOTAL']
historical_counts = [56, 200, 0, 256]  # 1999-2023
recent_counts = [56, 200, 60, 316]     # 2024-2025

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, historical_counts, width, label='Historical (1999-2023)', 
                color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, recent_counts, width, label='Recent (2024-2025)', 
                color='#9467bd', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_title('Feature Availability: Historical vs Recent Years', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 2: Imputation Risk by Feature Category
# ============================================================================
feature_cats = ['Passing\nStats', 'Rushing\nStats', 'Defense\nStats', 
                'Penalties', 'Scoring', 'Special\nTeams', 'Records/\nSplits']
risk_scores = [9, 8, 7, 8, 8, 6, 3]  # 1-10 scale
colors_risk = ['#d62728' if r >= 7 else '#ff7f0e' if r >= 5 else '#2ca02c' for r in risk_scores]

bars = ax2.barh(feature_cats, risk_scores, color=colors_risk, alpha=0.8, 
                edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Imputation Risk Score (1=Low, 10=High)', fontsize=12, fontweight='bold')
ax2.set_title('Risk of Imputing 2024 Stats for 1999-2023 Games', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, risk_scores)):
    ax2.text(score + 0.2, bar.get_y() + bar.get_height()/2, 
            f'{score}/10', va='center', fontweight='bold', fontsize=10)

# Add risk zones
ax2.axvspan(0, 4, alpha=0.1, color='green', label='Low Risk')
ax2.axvspan(4, 7, alpha=0.1, color='orange', label='Medium Risk')
ax2.axvspan(7, 10, alpha=0.1, color='red', label='High Risk')
ax2.legend(loc='lower right', fontsize=9)

# ============================================================================
# Plot 3: NFL Evolution Impact (1999 vs 2024)
# ============================================================================
metrics = ['Passing\nYards/Game', 'Completion\n%', 'Rushing\nAttempts', 
           'Sacks', 'Penalties', 'Points/Game', 'Turnovers']
pct_change = [40, 10, -20, 15, 25, 15, -20]  # % change from 1999 to 2024

colors_change = ['#2ca02c' if c > 0 else '#d62728' for c in pct_change]
bars = ax3.barh(metrics, pct_change, color=colors_change, alpha=0.8, 
                edgecolor='black', linewidth=1.5)

ax3.set_xlabel('% Change (1999 → 2024)', fontsize=12, fontweight='bold')
ax3.set_title('NFL Evolution: Why 2024 Stats ≠ 1999 Stats', fontsize=14, fontweight='bold')
ax3.axvline(x=0, color='black', linewidth=2)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for bar, change in zip(bars, pct_change):
    x_pos = change + (3 if change > 0 else -3)
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2, 
            f'{change:+d}%', va='center', ha='center', fontweight='bold', 
            fontsize=10, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# ============================================================================
# Plot 4: Strategy Comparison
# ============================================================================
strategies = ['Option A:\nImputation', 'Option B:\n2018-2023\nOnly', 
              'Option C:\nTwo-Stage\nModel', 'Option D+:\nDerivation\n(RECOMMENDED)']

# Metrics: Training Games, Feature Count, Bias Risk, Complexity
training_games = [6706, 1672, 6706, 6706]
feature_count = [383, 383, 383, 316]
bias_risk = [9, 2, 3, 1]  # 1-10 scale (lower is better)
complexity = [3, 2, 8, 5]  # 1-10 scale (lower is better)

x = np.arange(len(strategies))
width = 0.2

# Normalize for visualization
norm_games = [g / max(training_games) * 10 for g in training_games]
norm_features = [f / max(feature_count) * 10 for f in feature_count]
norm_bias = [10 - b for b in bias_risk]  # Invert so higher is better
norm_complexity = [10 - c for c in complexity]  # Invert so higher is better

bars1 = ax4.bar(x - 1.5*width, norm_games, width, label='Training Data', 
                color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax4.bar(x - 0.5*width, norm_features, width, label='Features', 
                color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
bars3 = ax4.bar(x + 0.5*width, norm_bias, width, label='Low Bias', 
                color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)
bars4 = ax4.bar(x + 1.5*width, norm_complexity, width, label='Simplicity', 
                color='#9467bd', alpha=0.8, edgecolor='black', linewidth=1)

ax4.set_ylabel('Score (normalized, higher is better)', fontsize=12, fontweight='bold')
ax4.set_title('Strategy Comparison (Option D+ Recommended)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(strategies, fontsize=9)
ax4.legend(fontsize=9, loc='upper left')
ax4.set_ylim(0, 11)
ax4.grid(axis='y', alpha=0.3)

# Highlight recommended option
ax4.axvspan(2.5, 3.5, alpha=0.2, color='green')
ax4.text(3, 10.5, '⭐ RECOMMENDED', ha='center', fontweight='bold', 
         fontsize=11, color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('Feature Strategy Analysis: Imputation vs Derivation\n' + 
             'Recommendation: Derive ESPN-like features from nfl-data-py (avoid imputation bias)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_path = Path('results/feature_strategy_comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_path}")

plt.show()

# Print summary
print("\n" + "="*80)
print("📊 FEATURE STRATEGY SUMMARY")
print("="*80)
print("\n⚠️  PROBLEM: Imputing 2024 ESPN stats for 1999-2023 introduces bias")
print("   - NFL has evolved significantly (passing +40%, scoring +15%)")
print("   - 2024 distributions don't represent 1999-2015 era")
print("   - Would create systematic bias in model")
print("\n✅ SOLUTION: Derive ESPN-like features from nfl-data-py")
print("   - Compute team stats from play-by-play for ALL years (1999-2024)")
print("   - Add truly NEW ESPN features for 2024-2025 only")
print("   - No imputation bias, consistent definitions across eras")
print("\n📈 EXPECTED RESULTS:")
print("   - Historical (1999-2023): 256 features (56 existing + 200 derived)")
print("   - Recent (2024-2025): 316 features (256 + 60 truly new ESPN)")
print("   - Training data: All 6,706 games (no data loss)")
print("   - Bias risk: Minimal (using real historical data)")
print("="*80)

