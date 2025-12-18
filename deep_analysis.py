"""
Deep Analysis - Understanding Model Behavior

This module provides deep analysis into:
1. Why OLS beats boosting for Totals prediction
2. Why CatBoost beats other models for Moneyline
3. Feature correlations and multicollinearity
4. Week 15 2025 prediction analysis
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# Boosting
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Data loading
from run_tier_sa_backtest import load_and_prepare_data, get_feature_columns


def analyze_feature_correlations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute feature correlations with targets and between features."""
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)

    # Compute correlations with targets
    corr_data = []
    for feat in features:
        if feat in df.columns:
            corr_spread = df[feat].corr(df['result'])
            corr_total = df[feat].corr(df['game_total'])
            corr_win = df[feat].corr(df['home_win'].astype(float))
            corr_data.append({
                'feature': feat,
                'corr_spread': corr_spread,
                'corr_total': corr_total,
                'corr_win': corr_win,
                'abs_corr_spread': abs(corr_spread),
                'abs_corr_total': abs(corr_total),
                'abs_corr_win': abs(corr_win)
            })

    corr_df = pd.DataFrame(corr_data)

    # Print top correlations for each target
    print("\n📊 Top 10 Features by Correlation with SPREAD (result):")
    for _, row in corr_df.nlargest(10, 'abs_corr_spread').iterrows():
        print(f"  {row['feature']:30} | r = {row['corr_spread']:+.3f}")

    print("\n📊 Top 10 Features by Correlation with TOTAL (game_total):")
    for _, row in corr_df.nlargest(10, 'abs_corr_total').iterrows():
        print(f"  {row['feature']:30} | r = {row['corr_total']:+.3f}")

    print("\n📊 Top 10 Features by Correlation with WIN (home_win):")
    for _, row in corr_df.nlargest(10, 'abs_corr_win').iterrows():
        print(f"  {row['feature']:30} | r = {row['corr_win']:+.3f}")

    return corr_df


def analyze_feature_multicollinearity(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Find highly correlated feature pairs (multicollinearity)."""
    print("\n" + "="*60)
    print("MULTICOLLINEARITY ANALYSIS")
    print("="*60)

    X = df[features].dropna()
    corr_matrix = X.corr()

    # Find pairs with |correlation| > 0.7
    high_corr_pairs = []
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j and feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                corr = corr_matrix.loc[feat1, feat2]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': corr
                    })

    pairs_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)

    print(f"\n⚠️ Found {len(pairs_df)} highly correlated pairs (|r| > 0.7):")
    for _, row in pairs_df.iterrows():
        print(f"  {row['feature1']:25} ↔ {row['feature2']:25} | r = {row['correlation']:+.3f}")

    return pairs_df


def compare_feature_importance_totals(df: pd.DataFrame, features: List[str]) -> Dict:
    """Compare feature importance between OLS and XGBoost for totals."""
    print("\n" + "="*60)
    print("WHY OLS BEATS XGBOOST FOR TOTALS")
    print("="*60)

    train_df = df[(df['season'] >= 2018) & (df['season'] <= 2023)].copy()
    val_df = df[df['season'] == 2025].copy()

    X_train = train_df[features].fillna(train_df[features].median())
    X_val = val_df[features].fillna(train_df[features].median())
    y_train = train_df['game_total']
    y_val = val_df['game_total']

    # Scale for OLS
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train models
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)

    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    # Predictions
    ols_pred = ols.predict(X_val_scaled)
    xgb_pred = xgb.predict(X_val)

    # Compare
    ols_rmse = np.sqrt(mean_squared_error(y_val, ols_pred))
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))

    print(f"\n📊 RMSE Comparison:")
    print(f"  OLS:     {ols_rmse:.2f}")
    print(f"  XGBoost: {xgb_rmse:.2f}")

    # Feature importance comparison
    ols_coef = pd.DataFrame({
        'feature': features,
        'ols_importance': np.abs(ols.coef_),
        'ols_coef': ols.coef_
    }).sort_values('ols_importance', ascending=False)

    xgb_imp = pd.DataFrame({
        'feature': features,
        'xgb_importance': xgb.feature_importances_
    }).sort_values('xgb_importance', ascending=False)

    # Merge
    importance_df = ols_coef.merge(xgb_imp, on='feature')

    print(f"\n📊 Feature Importance Comparison (Top 10):")
    print(f"{'Feature':30} | {'OLS Coef':>10} | {'XGB Imp':>10}")
    print("-" * 60)
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30} | {row['ols_coef']:+10.3f} | {row['xgb_importance']:10.3f}")

    # Key insight: Check correlation between total_line and game_total
    total_line_corr = val_df['total_line'].corr(val_df['game_total'])
    print(f"\n💡 KEY INSIGHT:")
    print(f"  Correlation between total_line and game_total: {total_line_corr:.3f}")
    print(f"  This high correlation suggests Vegas lines are already very accurate.")
    print(f"  Linear models can exploit this simple relationship more directly.")
    print(f"  XGBoost may be overcomplicating with non-linear interactions.")

    return {
        'ols_rmse': ols_rmse,
        'xgb_rmse': xgb_rmse,
        'importance': importance_df.to_dict('records'),
        'total_line_correlation': total_line_corr,
        'ols_model': ols,
        'xgb_model': xgb,
        'scaler': scaler
    }


def compare_classifiers_moneyline(df: pd.DataFrame, features: List[str]) -> Dict:
    """Compare why CatBoost beats other classifiers for moneyline."""
    print("\n" + "="*60)
    print("WHY CATBOOST BEATS OTHER CLASSIFIERS FOR MONEYLINE")
    print("="*60)

    train_df = df[(df['season'] >= 2018) & (df['season'] <= 2023)].copy()
    val_df = df[df['season'] == 2025].copy()

    X_train = train_df[features].fillna(train_df[features].median())
    X_val = val_df[features].fillna(train_df[features].median())
    y_train = train_df['home_win']
    y_val = val_df['home_win']

    # Scale for logistic
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        'Logistic_L1': LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                  random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                    random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        random_state=42, verbose=0)
    }

    results = {}
    feature_importance = {}

    for name, model in models.items():
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            # Get coefficients
            coef_df = pd.DataFrame({
                'feature': features,
                'importance': np.abs(model.coef_[0]),
                'coefficient': model.coef_[0],
                'odds_ratio': np.exp(model.coef_[0])
            }).sort_values('importance', ascending=False)
            feature_importance[name] = coef_df
        else:
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_val)[:, 1]
            # Get feature importance
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance[name] = imp_df

        predictions = (pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val, predictions)

        # Per-game analysis
        val_copy = val_df.copy()
        val_copy['pred_proba'] = pred_proba
        val_copy['pred_win'] = predictions
        val_copy['correct'] = (val_copy['pred_win'] == val_copy['home_win'])

        results[name] = {
            'accuracy': accuracy,
            'predictions': val_copy[['game_id', 'home_team', 'away_team', 'home_win',
                                      'pred_proba', 'pred_win', 'correct']].to_dict('records')
        }

        print(f"\n{name}: {accuracy:.1%} accuracy")
        print(f"  Top 5 Features:")
        for _, row in feature_importance[name].head(5).iterrows():
            if 'odds_ratio' in row:
                print(f"    {row['feature']:25} | Coef: {row['coefficient']:+.3f} | OR: {row['odds_ratio']:.3f}")
            else:
                print(f"    {row['feature']:25} | Importance: {row['importance']:.3f}")

    # Analyze where CatBoost differs from others
    print("\n" + "-"*60)
    print("ANALYZING CATBOOST'S UNIQUE PREDICTIONS")
    print("-"*60)

    cat_preds = results['CatBoost']['predictions']
    xgb_preds = results['XGBoost']['predictions']

    cat_df = pd.DataFrame(cat_preds)
    xgb_df = pd.DataFrame(xgb_preds)

    # Games where CatBoost was right but XGBoost was wrong
    merged = cat_df.merge(xgb_df, on='game_id', suffixes=('_cat', '_xgb'))
    cat_wins = merged[(merged['correct_cat']) & (~merged['correct_xgb'])]
    xgb_wins = merged[(~merged['correct_cat']) & (merged['correct_xgb'])]

    print(f"\nGames where CatBoost correct, XGBoost wrong: {len(cat_wins)}")
    print(f"Games where XGBoost correct, CatBoost wrong: {len(xgb_wins)}")
    print(f"Net advantage for CatBoost: {len(cat_wins) - len(xgb_wins)} games")

    return {
        'results': results,
        'feature_importance': {k: v.to_dict('records') for k, v in feature_importance.items()},
        'catboost_advantage': len(cat_wins) - len(xgb_wins)
    }


def analyze_week_15_2025(df: pd.DataFrame, features: List[str]) -> Dict:
    """Deep dive into Week 15 2025 predictions."""
    print("\n" + "="*60)
    print("WEEK 15 2025 PREDICTION ANALYSIS")
    print("="*60)

    train_df = df[(df['season'] >= 2018) & (df['season'] <= 2023)].copy()
    week15 = df[(df['season'] == 2025) & (df['week'] == 15)].copy()

    if len(week15) == 0:
        print("No Week 15 2025 data found!")
        return {}

    print(f"\nFound {len(week15)} games in Week 15 2025")

    X_train = train_df[features].fillna(train_df[features].median())
    X_week15 = week15[features].fillna(train_df[features].median())

    # Train best models
    # Spread: XGBoost baseline
    spread_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    spread_model.fit(X_train, train_df['result'])

    # Totals: OLS
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_week15_scaled = scaler.transform(X_week15)

    totals_model = LinearRegression()
    totals_model.fit(X_train_scaled, train_df['game_total'])

    # ML: CatBoost
    ml_model = CatBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   random_state=42, verbose=0)
    ml_model.fit(X_train, train_df['home_win'])

    # Predictions
    week15['pred_spread'] = spread_model.predict(X_week15)
    week15['pred_total'] = totals_model.predict(X_week15_scaled)
    week15['pred_win_proba'] = ml_model.predict_proba(X_week15)[:, 1]
    week15['pred_home_win'] = (week15['pred_win_proba'] > 0.5).astype(int)

    # Evaluate
    week15['spread_bet_home'] = week15['pred_spread'] > week15['spread_line']
    week15['spread_correct'] = (
        ((week15['result'] > week15['spread_line']) & week15['spread_bet_home']) |
        ((week15['result'] < week15['spread_line']) & ~week15['spread_bet_home'])
    )

    week15['totals_bet_over'] = week15['pred_total'] > week15['total_line']
    week15['totals_correct'] = (
        ((week15['game_total'] > week15['total_line']) & week15['totals_bet_over']) |
        ((week15['game_total'] < week15['total_line']) & ~week15['totals_bet_over'])
    )

    week15['ml_correct'] = week15['pred_home_win'] == week15['home_win']

    print(f"\n📊 WEEK 15 2025 RESULTS:")
    print(f"  Spread: {week15['spread_correct'].sum()}/{len(week15)} ({week15['spread_correct'].mean():.1%})")
    print(f"  Totals: {week15['totals_correct'].sum()}/{len(week15)} ({week15['totals_correct'].mean():.1%})")
    print(f"  ML:     {week15['ml_correct'].sum()}/{len(week15)} ({week15['ml_correct'].mean():.1%})")

    # Detailed game-by-game
    print(f"\n📋 GAME-BY-GAME ANALYSIS:")
    print("-" * 100)

    results = []
    for _, game in week15.iterrows():
        result = {
            'game_id': game['game_id'],
            'matchup': f"{game['away_team']} @ {game['home_team']}",
            'actual_result': game['result'],
            'spread_line': game['spread_line'],
            'pred_spread': game['pred_spread'],
            'spread_correct': game['spread_correct'],
            'actual_total': game['game_total'],
            'total_line': game['total_line'],
            'pred_total': game['pred_total'],
            'totals_correct': game['totals_correct'],
            'home_won': game['home_win'],
            'pred_win_proba': game['pred_win_proba'],
            'ml_correct': game['ml_correct']
        }
        results.append(result)

        spread_icon = "✅" if game['spread_correct'] else "❌"
        totals_icon = "✅" if game['totals_correct'] else "❌"
        ml_icon = "✅" if game['ml_correct'] else "❌"

        print(f"{game['away_team']:4} @ {game['home_team']:4} | "
              f"Result: {game['result']:+5.1f} (line {game['spread_line']:+5.1f}) {spread_icon} | "
              f"Total: {game['game_total']:5.1f} (line {game['total_line']:5.1f}) {totals_icon} | "
              f"Win: {game['pred_win_proba']:.0%} {ml_icon}")

    # Analyze misses
    print(f"\n❌ ANALYZING MISSES:")
    spread_misses = week15[~week15['spread_correct']]
    totals_misses = week15[~week15['totals_correct']]
    ml_misses = week15[~week15['ml_correct']]

    if len(spread_misses) > 0:
        print(f"\n  SPREAD MISSES ({len(spread_misses)}):")
        for _, game in spread_misses.iterrows():
            error = game['pred_spread'] - game['result']
            print(f"    {game['away_team']} @ {game['home_team']}: "
                  f"Pred {game['pred_spread']:+.1f}, Actual {game['result']:+.1f}, Error {error:+.1f}")

    if len(ml_misses) > 0:
        print(f"\n  ML MISSES ({len(ml_misses)}):")
        for _, game in ml_misses.iterrows():
            actual = "Home" if game['home_win'] else "Away"
            pred = "Home" if game['pred_home_win'] else "Away"
            print(f"    {game['away_team']} @ {game['home_team']}: "
                  f"Predicted {pred} ({game['pred_win_proba']:.0%}), Actual {actual}")

    return {
        'summary': {
            'spread_wr': week15['spread_correct'].mean(),
            'totals_wr': week15['totals_correct'].mean(),
            'ml_accuracy': week15['ml_correct'].mean()
        },
        'games': results,
        'spread_misses': spread_misses[['game_id', 'home_team', 'away_team', 'result',
                                         'spread_line', 'pred_spread']].to_dict('records'),
        'ml_misses': ml_misses[['game_id', 'home_team', 'away_team', 'home_win',
                                 'pred_win_proba']].to_dict('records')
    }


def run_deep_analysis():
    """Run complete deep analysis."""
    print("="*60)
    print("DEEP MODEL ANALYSIS")
    print("="*60)

    # Load data
    _, df = load_and_prepare_data()
    features = get_feature_columns(df)

    # 1. Feature correlations
    corr_df = analyze_feature_correlations(df, features)

    # 2. Multicollinearity
    multi_df = analyze_feature_multicollinearity(df, features)

    # 3. Totals: OLS vs XGBoost
    totals_analysis = compare_feature_importance_totals(df, features)

    # 4. Moneyline: CatBoost vs others
    ml_analysis = compare_classifiers_moneyline(df, features)

    # 5. Week 15 2025 analysis
    week15_analysis = analyze_week_15_2025(df, features)

    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'correlations': corr_df.to_dict('records'),
        'multicollinearity': multi_df.to_dict('records') if len(multi_df) > 0 else [],
        'totals_analysis': {
            'ols_rmse': totals_analysis['ols_rmse'],
            'xgb_rmse': totals_analysis['xgb_rmse'],
            'importance': totals_analysis['importance']
        },
        'ml_analysis': {
            'feature_importance': ml_analysis['feature_importance'],
            'catboost_advantage': ml_analysis['catboost_advantage']
        },
        'week15_2025': week15_analysis
    }

    output_path = Path("results/deep_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Deep analysis saved to {output_path}")

    return results


if __name__ == "__main__":
    run_deep_analysis()

