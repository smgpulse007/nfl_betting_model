"""
Visualize Feature Availability Across Years
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set up the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Year ranges
years = list(range(1999, 2026))
year_labels = [str(y) if y % 2 == 1 else '' for y in years]

# Feature availability
features = {
    'Basic Features\n(Elo, Weather, Rest)': (1999, 2025),
    'Vegas Lines\n(Spread, Total, ML)': (1999, 2025),
    'NGS Data\n(CPOE, RYOE, Separation)': (2016, 2025),
    'PFR Data\n(Pressure Rate)': (2018, 2025),
    'ESPN Team Stats\n(279 metrics)': (2024, 2025),
    'ESPN Team Records\n(44 metrics)': (2024, 2025),
    'ESPN Live Injuries\n(2,400+ records)': (2025, 2025),
}

# Colors
colors = {
    'Basic Features\n(Elo, Weather, Rest)': '#1f77b4',
    'Vegas Lines\n(Spread, Total, ML)': '#ff7f0e',
    'NGS Data\n(CPOE, RYOE, Separation)': '#2ca02c',
    'PFR Data\n(Pressure Rate)': '#d62728',
    'ESPN Team Stats\n(279 metrics)': '#9467bd',
    'ESPN Team Records\n(44 metrics)': '#8c564b',
    'ESPN Live Injuries\n(2,400+ records)': '#e377c2',
}

# Plot 1: Feature Availability Timeline
ax1.set_xlim(1998, 2026)
ax1.set_ylim(0, len(features))

for i, (feature, (start, end)) in enumerate(features.items()):
    ax1.barh(i, end - start + 1, left=start - 0.5, height=0.8, 
             color=colors[feature], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add text label
    mid_year = (start + end) / 2
    ax1.text(mid_year, i, feature, ha='center', va='center', 
             fontweight='bold', fontsize=9, color='white')

ax1.set_yticks(range(len(features)))
ax1.set_yticklabels([])
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_title('Feature Availability Timeline (1999-2025)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xticks(years)
ax1.set_xticklabels(year_labels, rotation=45)

# Add vertical lines for key years
ax1.axvline(x=2016, color='green', linestyle='--', linewidth=2, alpha=0.5, label='NGS Data Starts')
ax1.axvline(x=2018, color='red', linestyle='--', linewidth=2, alpha=0.5, label='PFR Data Starts')
ax1.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='ESPN Data Starts')
ax1.legend(loc='upper left', fontsize=10)

# Plot 2: Games per Year with Feature Availability
games_per_year = {
    1999: 259, 2000: 259, 2001: 259, 2002: 267, 2003: 267, 2004: 267,
    2005: 267, 2006: 267, 2007: 267, 2008: 267, 2009: 267, 2010: 267,
    2011: 267, 2012: 267, 2013: 267, 2014: 267, 2015: 267, 2016: 267,
    2017: 267, 2018: 267, 2019: 267, 2020: 269, 2021: 285, 2022: 284,
    2023: 285, 2024: 285, 2025: 251
}

# Color code by feature availability
bar_colors = []
for year in years:
    if year >= 2024:
        bar_colors.append('#9467bd')  # ESPN era
    elif year >= 2018:
        bar_colors.append('#d62728')  # PFR era
    elif year >= 2016:
        bar_colors.append('#2ca02c')  # NGS era
    else:
        bar_colors.append('#1f77b4')  # Basic era

bars = ax2.bar(years, [games_per_year.get(y, 0) for y in years], 
               color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
ax2.set_title('Games per Year by Feature Era', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(years)
ax2.set_xticklabels(year_labels, rotation=45)

# Add legend
basic_patch = mpatches.Patch(color='#1f77b4', label='Basic Era (1999-2015): 4,500 games')
ngs_patch = mpatches.Patch(color='#2ca02c', label='NGS Era (2016-2017): 534 games')
pfr_patch = mpatches.Patch(color='#d62728', label='PFR Era (2018-2023): 1,672 games')
espn_patch = mpatches.Patch(color='#9467bd', label='ESPN Era (2024-2025): 536 games')
ax2.legend(handles=[basic_patch, ngs_patch, pfr_patch, espn_patch], 
          loc='upper left', fontsize=10)

# Add annotations
ax2.axvline(x=2023.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax2.text(2011, 280, 'Training Data\n(1999-2023)\n6,706 games', 
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(2024.5, 280, 'Test/Validation\n(2024-2025)\n536 games', 
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('NFL Betting Model: Historical Data & Feature Availability\nTotal: 6,991 games (1999-2024)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_path = Path('results/data_availability_timeline.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_path}")

plt.show()

# Print summary
print("\n" + "="*80)
print("📊 DATA AVAILABILITY SUMMARY")
print("="*80)
print("\nFeature Eras:")
print("  1999-2015 (Basic):     4,500 games - Elo, Weather, Rest, Vegas Lines")
print("  2016-2017 (NGS):         534 games - + CPOE, RYOE, Separation")
print("  2018-2023 (PFR):       1,672 games - + Pressure Rate")
print("  2024-2025 (ESPN):        536 games - + Team Stats, Records, Live Injuries")
print("\nTotal Training Data: 6,706 games (1999-2023)")
print("Total Test Data:       285 games (2024)")
print("Total Validation:      251 games (2025)")
print("\n✅ ESPN data collection complete for 2024-2025")
print("❌ ESPN data NOT needed for 1999-2023 (use nfl-data-py features)")
print("="*80)

