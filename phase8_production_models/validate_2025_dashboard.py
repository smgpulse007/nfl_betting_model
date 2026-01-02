"""
Validate 2025 actual performance dashboard
"""

from pathlib import Path
import pandas as pd

print("=" * 100)
print("VALIDATING 2025 ACTUAL PERFORMANCE DASHBOARD")
print("=" * 100)

# Check required files
required_files = {
    'Dashboard main': 'task_8d1_dashboard_structure.py',
    '2025 performance page': 'task_8d6_2025_actual_performance.py',
    '2025 schedule': '../results/phase8_results/2025_schedule_actual.csv',
    '2025 backtest': '../results/phase8_results/2025_backtest_weeks1_16.csv',
    '2025 weekly stats': '../results/phase8_results/2025_weekly_performance.csv'
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
    print("\nTo generate missing files:")
    print("  1. python fetch_2025_actual_data.py")
    print("  2. python backtest_2025_performance.py")
    exit(1)

print("\n✅ All required files exist!")

# Check data integrity
print("\n[2/3] Checking data integrity...")

# Load 2025 schedule
df_schedule = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
completed = df_schedule[df_schedule['home_score'].notna()]
print(f"  ✅ 2025 schedule: {len(df_schedule)} total games, {len(completed)} completed")

# Load backtest
df_backtest = pd.read_csv('../results/phase8_results/2025_backtest_weeks1_16.csv')
accuracy = df_backtest['correct'].mean()
print(f"  ✅ 2025 backtest: {len(df_backtest)} games, {accuracy:.1%} accuracy")

# Load weekly stats
df_weekly = pd.read_csv('../results/phase8_results/2025_weekly_performance.csv')
print(f"  ✅ 2025 weekly stats: {len(df_weekly)} weeks")

# Week 17 status
week17 = df_schedule[df_schedule['week'] == 17]
week17_completed = week17[week17['home_score'].notna()]
week17_upcoming = week17[week17['home_score'].isna()]
print(f"  ✅ Week 17: {len(week17_completed)}/{len(week17)} completed, {len(week17_upcoming)} upcoming")

# Performance by confidence
high_conf = df_backtest[df_backtest['confidence'] >= 0.65]
if len(high_conf) > 0:
    high_acc = high_conf['correct'].mean()
    print(f"  ✅ High confidence (≥65%): {high_acc:.1%} accuracy ({len(high_conf)} games)")

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

if not all_exist:
    print("\n❌ Some required packages are missing!")
    exit(1)

print("\n" + "=" * 100)
print("✅ VALIDATION COMPLETE - 2025 DASHBOARD READY")
print("=" * 100)

print("\n📊 2025 SEASON SUMMARY:")
print(f"  • Total games analyzed: {len(df_backtest)}")
print(f"  • Overall accuracy: {accuracy:.1%}")
print(f"  • High confidence accuracy: {high_acc:.1%}" if len(high_conf) > 0 else "  • No high confidence games")
print(f"  • Week 17 status: {len(week17_completed)}/{len(week17)} completed")

print("\n🚀 To launch the dashboard:")
print("  streamlit run task_8d1_dashboard_structure.py")
print("\n📈 Then navigate to: 🏈 2025 Actual Performance")

