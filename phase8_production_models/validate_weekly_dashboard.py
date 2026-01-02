"""
Validate that all required files exist for the weekly performance dashboard
"""

from pathlib import Path
import pandas as pd

print("=" * 100)
print("VALIDATING WEEKLY PERFORMANCE DASHBOARD")
print("=" * 100)

# Check required files
required_files = {
    'Dashboard main': 'task_8d1_dashboard_structure.py',
    'Weekly performance page': 'task_8d5_weekly_performance.py',
    '2024 analysis data': '../results/phase8_results/2024_week16_17_analysis.csv',
    '2025 predictions': '../results/phase8_results/2025_predictions.csv',
    '2025 betting recs': '../results/phase8_results/2025_betting_recommendations.csv'
}

print("\n[1/3] Checking required files...")
all_exist = True
for name, path in required_files.items():
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some required files are missing!")
    exit(1)

print("\n✅ All required files exist!")

# Check data integrity
print("\n[2/3] Checking data integrity...")

# Load 2024 analysis
df_2024 = pd.read_csv('../results/phase8_results/2024_week16_17_analysis.csv')
print(f"  ✅ 2024 analysis: {len(df_2024)} games, {df_2024.shape[1]} columns")
print(f"     - Weeks: {sorted(df_2024['week'].unique())}")
print(f"     - Accuracy: {df_2024['correct'].mean():.1%}")

# Load 2025 predictions
df_2025 = pd.read_csv('../results/phase8_results/2025_predictions.csv')
print(f"  ✅ 2025 predictions: {len(df_2025)} games, {df_2025.shape[1]} columns")
print(f"     - Weeks: {sorted(df_2025['week'].unique())}")
print(f"     - Avg confidence: {df_2025['confidence'].mean():.1%}")

# Load 2025 betting recommendations
df_bets = pd.read_csv('../results/phase8_results/2025_betting_recommendations.csv')
print(f"  ✅ 2025 betting recs: {len(df_bets)} games, {df_bets.shape[1]} columns")
print(f"     - Avg EV: {df_bets['expected_value'].mean():.2%}")

# Check imports
print("\n[3/3] Checking imports...")
try:
    import streamlit as st
    print("  ✅ streamlit")
except ImportError:
    print("  ❌ streamlit - not installed")
    all_exist = False

try:
    import plotly.graph_objects as go
    print("  ✅ plotly")
except ImportError:
    print("  ❌ plotly - not installed")
    all_exist = False

try:
    import plotly.express as px
    print("  ✅ plotly.express")
except ImportError:
    print("  ❌ plotly.express - not installed")
    all_exist = False

if not all_exist:
    print("\n❌ Some required packages are missing!")
    exit(1)

print("\n" + "=" * 100)
print("✅ VALIDATION COMPLETE - DASHBOARD READY TO LAUNCH")
print("=" * 100)
print("\nTo launch the dashboard:")
print("  streamlit run task_8d1_dashboard_structure.py")
print("\nThen navigate to: 📅 Weekly Performance")

