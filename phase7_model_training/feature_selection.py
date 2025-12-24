"""
Feature Selection Module
=========================

Reduce from 1,160 features to top 100-200 most predictive features using:
1. Correlation-based selection
2. Mutual information
3. Feature importance from tree models
4. Variance threshold
5. Multicollinearity removal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import json
from datetime import datetime

print("="*120)
print("FEATURE SELECTION: 1,160 → 100-200 FEATURES")
print("="*120)

# Load data
print("\n[1/7] Loading dataset...")
df = pd.read_parquet('../results/game_level_predictions_dataset.parquet')
print(f"  ✅ Loaded {len(df):,} games with {df.shape[1]} features")

# Separate features and target
metadata_cols = ['game_id', 'game_date', 'year', 'week', 'home_team', 'away_team', 'home_win']
feature_cols = [c for c in df.columns if c not in metadata_cols]

# Filter to numeric columns only
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"  ✅ {len(feature_cols)} feature columns identified ({len(numeric_cols)} numeric)")

# Split data
train_df = df[df['year'] <= 2022].copy()
val_df = df[df['year'] == 2023].copy()
test_df = df[df['year'] == 2024].copy()

X_train = train_df[numeric_cols]
y_train = train_df['home_win']
X_val = val_df[numeric_cols]
y_val = val_df['home_win']

print(f"\n  Train: {len(train_df):,} games (1999-2022)")
print(f"  Val:   {len(val_df):,} games (2023)")
print(f"  Test:  {len(test_df):,} games (2024)")

# Method 1: Variance Threshold
print("\n[2/7] Variance threshold (remove low-variance features)...")
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train.fillna(0))
high_var_features = X_train.columns[selector.get_support()].tolist()
print(f"  ✅ {len(high_var_features)} features with variance > 0.01")

# Method 2: Correlation with target
print("\n[3/7] Computing correlation with target...")
correlations = {}
for col in high_var_features:
    valid_data = train_df[[col, 'home_win']].dropna()
    if len(valid_data) > 100:
        corr = valid_data[col].corr(valid_data['home_win'])
        correlations[col] = abs(corr)

corr_df = pd.DataFrame(list(correlations.items()), columns=['feature', 'abs_corr'])
corr_df = corr_df.sort_values('abs_corr', ascending=False)
top_corr_features = corr_df.head(300)['feature'].tolist()
print(f"  ✅ Top 300 features by correlation (mean |r| = {corr_df.head(300)['abs_corr'].mean():.4f})")

# Method 3: Mutual Information
print("\n[4/7] Computing mutual information...")
X_train_filled = X_train[top_corr_features].fillna(0)
mi_scores = mutual_info_classif(X_train_filled, y_train, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': top_corr_features, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)
top_mi_features = mi_df.head(250)['feature'].tolist()
print(f"  ✅ Top 250 features by mutual information (mean MI = {mi_df.head(250)['mi_score'].mean():.4f})")

# Method 4: Random Forest Feature Importance
print("\n[5/7] Training Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train[top_mi_features].fillna(0), y_train)
rf_importance = pd.DataFrame({
    'feature': top_mi_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
top_rf_features = rf_importance.head(200)['feature'].tolist()
print(f"  ✅ Top 200 features by RF importance (mean = {rf_importance.head(200)['importance'].mean():.4f})")

# Method 5: XGBoost Feature Importance
print("\n[6/7] Training XGBoost for feature importance...")
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb.fit(X_train[top_rf_features].fillna(0), y_train)
xgb_importance = pd.DataFrame({
    'feature': top_rf_features,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)
top_xgb_features = xgb_importance.head(150)['feature'].tolist()
print(f"  ✅ Top 150 features by XGBoost importance (mean = {xgb_importance.head(150)['importance'].mean():.4f})")

# Combine all methods (union of top features)
print("\n[7/7] Combining feature selection methods...")
selected_features = list(set(top_xgb_features))
print(f"  ✅ Final selected features: {len(selected_features)}")

# Save results
output_dir = Path('../results/phase7_feature_selection')
output_dir.mkdir(exist_ok=True)

# Save selected features
with open(output_dir / 'selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

# Save feature importance rankings
corr_df.to_csv(output_dir / 'correlation_rankings.csv', index=False)
mi_df.to_csv(output_dir / 'mutual_info_rankings.csv', index=False)
rf_importance.to_csv(output_dir / 'rf_importance_rankings.csv', index=False)
xgb_importance.to_csv(output_dir / 'xgb_importance_rankings.csv', index=False)

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_features': len(numeric_cols),
    'selected_features': len(selected_features),
    'selection_methods': {
        'variance_threshold': len(high_var_features),
        'top_correlation': len(top_corr_features),
        'top_mutual_info': len(top_mi_features),
        'top_rf_importance': len(top_rf_features),
        'top_xgb_importance': len(top_xgb_features)
    },
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df)
}

with open(output_dir / 'feature_selection_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*120}")
print("✅ FEATURE SELECTION COMPLETE!")
print(f"{'='*120}")
print(f"✅ Reduced from {len(numeric_cols):,} to {len(selected_features)} features")
print(f"✅ Saved to: {output_dir}")
print(f"\nSelected features saved to: selected_features.json")

