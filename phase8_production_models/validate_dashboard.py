"""
Validate Dashboard Components

Quick validation script to ensure all dashboard components can be imported
and required data files exist.
"""

import sys
from pathlib import Path

print("="*80)
print("DASHBOARD VALIDATION")
print("="*80)

# Check imports
print("\n[1/4] Checking imports...")

try:
    import streamlit as st
    print("  ✅ streamlit")
except ImportError as e:
    print(f"  ❌ streamlit: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("  ✅ pandas")
except ImportError as e:
    print(f"  ❌ pandas: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("  ✅ matplotlib")
except ImportError as e:
    print(f"  ❌ matplotlib: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print("  ✅ seaborn")
except ImportError as e:
    print(f"  ❌ seaborn: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("  ✅ PIL")
except ImportError as e:
    print(f"  ❌ PIL: {e}")
    sys.exit(1)

try:
    import joblib
    print("  ✅ joblib")
except ImportError as e:
    print(f"  ❌ joblib: {e}")
    sys.exit(1)

# Check dashboard files
print("\n[2/4] Checking dashboard files...")

dashboard_files = [
    'task_8d1_dashboard_structure.py',
    'task_8d2_model_performance.py',
    'task_8d3_feature_analysis.py',
    'task_8d4_betting_simulator.py'
]

for file in dashboard_files:
    if Path(file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - NOT FOUND")

# Check required data files
print("\n[3/4] Checking required data files...")

data_files = [
    '../results/phase8_results/phase6_game_level_1999_2024.parquet',
    '../results/phase8_results/feature_categorization.json',
    '../results/phase8_results/comprehensive_metrics.json',
    '../results/phase8_results/calibration_results.json',
    '../results/phase8_results/cross_validation_results.json',
    '../results/phase8_results/shap_analysis/global_feature_importance.json',
    '../results/phase8_results/permutation_importance/permutation_importance.json',
    '../results/phase8_results/feature_correlation/redundancy_recommendations.json'
]

missing_files = []
for file in data_files:
    if Path(file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - NOT FOUND")
        missing_files.append(file)

# Check model files
print("\n[4/4] Checking model files...")

model_files = [
    '../models/xgboost_tuned.pkl',
    '../models/lightgbm_tuned.pkl',
    '../models/catboost_tuned.pkl',
    '../models/randomforest_tuned.pkl'
]

for file in model_files:
    if Path(file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - NOT FOUND")
        missing_files.append(file)

# Summary
print("\n" + "="*80)
if len(missing_files) == 0:
    print("✅ VALIDATION PASSED")
    print("="*80)
    print("\nDashboard is ready to run!")
    print("\nTo start the dashboard, run:")
    print("  streamlit run task_8d1_dashboard_structure.py")
else:
    print("❌ VALIDATION FAILED")
    print("="*80)
    print(f"\n{len(missing_files)} required files are missing:")
    for file in missing_files:
        print(f"  - {file}")
    print("\nPlease run the missing Phase 8 tasks before starting the dashboard.")

print("\n" + "="*80)

