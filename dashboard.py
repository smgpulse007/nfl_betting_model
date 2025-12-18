"""
NFL Betting Model Dashboard
============================
Streamlit dashboard for visualizing model performance and backtest results.

Sections:
1. Dataset Overview
2. Feature Profiling
3. Feature Importance
4. Model Performance Across Iterations
5. 2025 Validation Results
6. Backtest Results Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import version info
try:
    from version import VERSION, VERSION_NAME, DESCRIPTION, BASELINE_METRICS
except ImportError:
    VERSION = "0.1.0"
    VERSION_NAME = "Baseline"
    DESCRIPTION = "NFL Betting Model with TIER S+A features"
    BASELINE_METRICS = None

# Page config
st.set_page_config(
    page_title=f"NFL Betting Model v{VERSION}",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")


@st.cache_data
def load_backtest_results():
    """Load backtest results from JSON."""
    results_file = RESULTS_DIR / "tier_sa_backtest_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


@st.cache_data
def load_predictions(year: int):
    """Load prediction data for a given year."""
    pred_file = RESULTS_DIR / f"tier_sa_predictions_{year}.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)
    return None


@st.cache_data
def load_weekly_stats(year: int):
    """Load weekly stats for a given year."""
    weekly_file = RESULTS_DIR / f"tier_sa_weekly_{year}.csv"
    if weekly_file.exists():
        return pd.read_csv(weekly_file)
    return None


def main():
    st.title("🏈 NFL Betting Model Dashboard")
    st.markdown(f"**Version {VERSION} ({VERSION_NAME})** — TIER S+A Feature Model")

    # Sidebar - Version info
    st.sidebar.markdown(f"### 📌 v{VERSION} - {VERSION_NAME}")
    st.sidebar.markdown("---")

    # Sidebar - Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["📊 Overview", "📈 Feature Profiling", "🔥 Feature Importance",
         "🎯 Model Performance", "📅 2025 Validation", "💰 Backtest Results"]
    )

    # Sidebar - Baseline metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Baseline Metrics (2025)")
    if BASELINE_METRICS:
        metrics = BASELINE_METRICS.get('2025_validation', {})
        st.sidebar.metric("Spread ROI", f"{metrics.get('spread_roi', 0):+.1f}%")
        st.sidebar.metric("ML ROI", f"{metrics.get('ml_roi', 0):+.1f}%")
        st.sidebar.metric("Win Accuracy", f"{metrics.get('win_accuracy', 0)*100:.1f}%")
    
    # Load data
    results = load_backtest_results()
    pred_2024 = load_predictions(2024)
    pred_2025 = load_predictions(2025)
    weekly_2024 = load_weekly_stats(2024)
    weekly_2025 = load_weekly_stats(2025)
    
    if results is None:
        st.error("No backtest results found. Run `python run_tier_sa_backtest.py` first.")
        return
    
    # Route to page
    if page == "📊 Overview":
        show_overview(results, pred_2024, pred_2025)
    elif page == "📈 Feature Profiling":
        show_feature_profiling(results, pred_2024)
    elif page == "🔥 Feature Importance":
        show_feature_importance(results, pred_2024)
    elif page == "🎯 Model Performance":
        show_model_performance(results, weekly_2024, weekly_2025)
    elif page == "📅 2025 Validation":
        show_2025_validation(pred_2025, weekly_2025)
    elif page == "💰 Backtest Results":
        show_backtest_results(pred_2024, pred_2025, weekly_2024, weekly_2025)


def show_overview(results, pred_2024, pred_2025):
    """Dataset Overview section."""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Version", f"v{VERSION}")
    with col2:
        st.metric("Features Used", len(results.get('features', [])))
    with col3:
        st.metric("2024 Games", len(pred_2024) if pred_2024 is not None else 0)
    with col4:
        st.metric("2025 Games", len(pred_2025) if pred_2025 is not None else 0)

    # Version description
    st.info(f"""
    **🏷️ {VERSION_NAME} Model (v{VERSION})**
    {DESCRIPTION}
    """)
    
    st.subheader("Key Results Summary")
    
    # Results table
    results_data = []
    if results.get('results_2024'):
        r = results['results_2024']
        results_data.append({
            'Dataset': '2024 Test',
            'Games': r['games'],
            'Spread WR': f"{r['spread_wr']*100:.1f}%",
            'Spread ROI': f"{r['spread_roi']:+.1f}%",
            'Totals WR': f"{r['totals_wr']*100:.1f}%",
            'Totals ROI': f"{r['totals_roi']:+.1f}%",
            'ML WR': f"{r['ml_wr']*100:.1f}%",
            'ML ROI': f"{r.get('ml_roi', 0):+.1f}%",
            'ML Bets': r['ml_bets'],
            'Win Accuracy': f"{r.get('win_pred_accuracy', 0)*100:.1f}%"
        })
    if results.get('results_2025'):
        r = results['results_2025']
        results_data.append({
            'Dataset': '2025 Validation',
            'Games': r['games'],
            'Spread WR': f"{r['spread_wr']*100:.1f}%",
            'Spread ROI': f"{r['spread_roi']:+.1f}%",
            'Totals WR': f"{r['totals_wr']*100:.1f}%",
            'Totals ROI': f"{r['totals_roi']:+.1f}%",
            'ML WR': f"{r['ml_wr']*100:.1f}%",
            'ML ROI': f"{r.get('ml_roi', 0):+.1f}%",
            'ML Bets': r['ml_bets'],
            'Win Accuracy': f"{r.get('win_pred_accuracy', 0)*100:.1f}%"
        })
    
    if results_data:
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Features list
    st.subheader("Features Used")
    features = results.get('features', [])
    cols = st.columns(3)
    for i, feat in enumerate(features):
        cols[i % 3].write(f"• {feat}")


def show_feature_importance(results, pred_df):
    """Feature Importance Analysis from trained models."""
    st.header("🔥 Feature Importance Analysis")

    feature_importance = results.get('feature_importance', {})

    if not feature_importance:
        st.warning("No feature importance data available. Re-run the backtest to generate.")
        st.info("Run: `python run_tier_sa_backtest.py`")
        return

    # Model selector
    st.subheader("Model-Specific Feature Importance")

    model_names = list(feature_importance.keys())
    model_display = {
        'spread': '📊 Spread Model (Margin Prediction)',
        'totals': '📈 Totals Model (Total Points)',
        'moneyline': '🎯 Moneyline Model (Win Probability)'
    }

    tabs = st.tabs([model_display.get(m, m) for m in model_names])

    for tab, model_name in zip(tabs, model_names):
        with tab:
            importance = feature_importance[model_name]

            # Sort by importance
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            imp_df = pd.DataFrame(sorted_imp, columns=['Feature', 'Importance'])

            # Color by category
            def get_category(feat):
                if any(x in feat for x in ['elo', 'spread_line', 'total_line', 'rest', 'div', 'implied']):
                    return 'Base/Elo'
                elif any(x in feat for x in ['dome', 'cold', 'wind', 'prime', 'grass', 'weather', 'short_week']):
                    return 'Venue/Weather'
                elif any(x in feat for x in ['cpoe', 'pressure', 'time_to_throw']):
                    return 'Passing'
                elif any(x in feat for x in ['injury', 'qb_out']):
                    return 'Injuries'
                elif any(x in feat for x in ['ryoe', 'separation']):
                    return 'Rush/Rec'
                return 'Other'

            imp_df['Category'] = imp_df['Feature'].apply(get_category)

            # Bar chart
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                        color='Category',
                        title=f"Feature Importance - {model_display.get(model_name, model_name)}",
                        color_discrete_map={
                            'Base/Elo': '#1f77b4',
                            'Venue/Weather': '#ff7f0e',
                            'Passing': '#2ca02c',
                            'Injuries': '#d62728',
                            'Rush/Rec': '#9467bd',
                            'Other': '#8c564b'
                        })
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Top 10 table
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top 10 Features")
                top10 = imp_df.head(10).copy()
                top10['Importance'] = (top10['Importance'] * 100).round(2).astype(str) + '%'
                st.dataframe(top10, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### Bottom 10 Features (Least Important)")
                bottom10 = imp_df.tail(10).copy()
                bottom10['Importance'] = (bottom10['Importance'] * 100).round(2).astype(str) + '%'
                st.dataframe(bottom10, use_container_width=True, hide_index=True)

    # Cross-model comparison
    st.subheader("Cross-Model Comparison")
    st.markdown("How does feature importance differ across models?")

    # Create comparison dataframe
    comparison_data = []
    for feat in results.get('features', []):
        row = {'Feature': feat}
        for model_name in model_names:
            row[model_name] = feature_importance.get(model_name, {}).get(feat, 0)
        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)

    # Heatmap
    if len(model_names) > 0:
        features_for_heatmap = comp_df.set_index('Feature')[model_names]

        # Sort by average importance
        features_for_heatmap['avg'] = features_for_heatmap.mean(axis=1)
        features_for_heatmap = features_for_heatmap.sort_values('avg', ascending=True).drop('avg', axis=1)

        fig = px.imshow(features_for_heatmap.T,
                       aspect='auto',
                       color_continuous_scale='RdYlGn',
                       title="Feature Importance Heatmap Across Models")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.subheader("Key Insights")

    # Find consistently important features
    avg_importance = {}
    for feat in results.get('features', []):
        avg = np.mean([feature_importance.get(m, {}).get(feat, 0) for m in model_names])
        avg_importance[feat] = avg

    top_overall = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom_overall = sorted(avg_importance.items(), key=lambda x: x[1])[:5]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Most Important (All Models)")
        for feat, imp in top_overall:
            st.write(f"• **{feat}**: {imp*100:.2f}%")

    with col2:
        st.markdown("### ⚠️ Least Important (Consider Removing)")
        for feat, imp in bottom_overall:
            st.write(f"• {feat}: {imp*100:.2f}%")

    # Injury features analysis
    st.subheader("🏥 Injury Features Analysis")
    injury_feats = ['home_injury_impact', 'away_injury_impact', 'injury_diff', 'home_qb_out', 'away_qb_out']
    injury_importance = {m: sum(feature_importance.get(m, {}).get(f, 0) for f in injury_feats)
                         for m in model_names}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Combined Injury Feature Importance:**")
        for model, imp in injury_importance.items():
            st.metric(model_display.get(model, model).split('(')[0].strip(), f"{imp*100:.2f}%")

    with col2:
        st.markdown("**Individual Injury Features:**")
        if pred_df is not None:
            for feat in injury_feats:
                if feat in pred_df.columns:
                    non_zero_pct = (pred_df[feat] != 0).mean() * 100
                    avg_imp = avg_importance.get(feat, 0) * 100
                    st.write(f"• {feat}: {avg_imp:.2f}% importance, {non_zero_pct:.1f}% non-zero")


def show_feature_profiling(results, pred_df):
    """Comprehensive Feature EDA section."""
    st.header("📈 Feature Deep Dive EDA")

    if pred_df is None:
        st.warning("No prediction data available")
        return

    features = results.get('features', [])
    numeric_features = [f for f in features if f in pred_df.columns]

    # Feature categories for better organization
    feature_categories = {
        'Base/Elo': ['elo_diff', 'elo_prob', 'spread_line', 'total_line', 'rest_advantage',
                     'div_game', 'home_implied_prob', 'away_implied_prob'],
        'Venue/Weather': ['is_dome', 'is_cold', 'is_windy', 'is_primetime', 'is_grass',
                          'bad_weather', 'home_short_week', 'away_short_week'],
        'Passing (CPOE/Pressure)': ['home_cpoe_3wk', 'away_cpoe_3wk', 'cpoe_diff',
                                     'home_pressure_rate_3wk', 'away_pressure_rate_3wk', 'pressure_diff',
                                     'home_time_to_throw_3wk', 'away_time_to_throw_3wk'],
        'Injuries': ['home_injury_impact', 'away_injury_impact', 'injury_diff',
                     'home_qb_out', 'away_qb_out'],
        'Rushing/Receiving': ['home_ryoe_3wk', 'away_ryoe_3wk', 'ryoe_diff',
                               'home_separation_3wk', 'away_separation_3wk', 'separation_diff'],
    }

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Feature Details", "📈 Correlations",
        "⚠️ Data Quality", "🧮 Feature Engineering"
    ])

    with tab1:
        show_feature_overview(pred_df, numeric_features, feature_categories)

    with tab2:
        show_feature_details(pred_df, numeric_features, feature_categories)

    with tab3:
        show_feature_correlations(pred_df, numeric_features)

    with tab4:
        show_data_quality(pred_df, numeric_features)

    with tab5:
        show_feature_engineering_docs()


def show_feature_overview(pred_df, numeric_features, feature_categories):
    """Overview of all features."""
    st.subheader("Feature Summary by Category")

    for category, feats in feature_categories.items():
        available_feats = [f for f in feats if f in pred_df.columns]
        if not available_feats:
            continue

        with st.expander(f"**{category}** ({len(available_feats)} features)", expanded=True):
            # Create summary table
            summary_data = []
            for feat in available_feats:
                col = pred_df[feat]
                non_null = col.dropna()
                variance = non_null.var() if len(non_null) > 1 else 0

                # Correlation with home_win
                if 'home_win' in pred_df.columns and len(non_null) > 0:
                    corr = pred_df[[feat, 'home_win']].dropna().corr().iloc[0, 1]
                else:
                    corr = np.nan

                summary_data.append({
                    'Feature': feat,
                    'Mean': non_null.mean() if len(non_null) > 0 else np.nan,
                    'Std': non_null.std() if len(non_null) > 0 else np.nan,
                    'Min': non_null.min() if len(non_null) > 0 else np.nan,
                    'Max': non_null.max() if len(non_null) > 0 else np.nan,
                    'Variance': variance,
                    'Unique': col.nunique(),
                    'Missing %': (col.isnull().sum() / len(col) * 100),
                    'Corr w/ Win': corr
                })

            summary_df = pd.DataFrame(summary_data)

            # Highlight low variance features
            def highlight_issues(row):
                styles = [''] * len(row)
                if row['Variance'] < 0.01 and row['Variance'] >= 0:
                    styles[5] = 'background-color: #ffcccc'  # Low variance
                if row['Missing %'] > 10:
                    styles[7] = 'background-color: #ffcccc'  # High missing
                if abs(row['Corr w/ Win']) > 0.15:
                    styles[8] = 'background-color: #ccffcc'  # Good correlation
                return styles

            st.dataframe(
                summary_df.style.apply(highlight_issues, axis=1).format({
                    'Mean': '{:.3f}', 'Std': '{:.3f}', 'Min': '{:.3f}', 'Max': '{:.3f}',
                    'Variance': '{:.4f}', 'Missing %': '{:.1f}%', 'Corr w/ Win': '{:.3f}'
                }),
                use_container_width=True
            )


def show_feature_details(pred_df, numeric_features, feature_categories):
    """Detailed view of individual features."""
    st.subheader("Feature Deep Dive")

    # Category selector
    category = st.selectbox("Select Category", list(feature_categories.keys()))
    available_feats = [f for f in feature_categories[category] if f in pred_df.columns]

    if not available_feats:
        st.warning("No features available in this category")
        return

    selected_feature = st.selectbox("Select Feature", available_feats)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution
        fig = px.histogram(pred_df, x=selected_feature, nbins=30,
                          color='home_win' if 'home_win' in pred_df.columns else None,
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot by outcome
        if 'home_win' in pred_df.columns:
            fig = px.box(pred_df, x='home_win', y=selected_feature,
                        title=f"{selected_feature} by Home Win",
                        labels={'home_win': 'Home Win (0=No, 1=Yes)'})
            st.plotly_chart(fig, use_container_width=True)

    # Value counts for binary/categorical features
    col = pred_df[selected_feature].dropna()
    if col.nunique() <= 10:
        st.subheader("Value Distribution")
        value_counts = col.value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / len(col) * 100).round(1)

        # Add win rate by value
        if 'home_win' in pred_df.columns:
            win_rates = pred_df.groupby(selected_feature)['home_win'].mean().reset_index()
            win_rates.columns = ['Value', 'Home Win Rate']
            value_counts = value_counts.merge(win_rates, on='Value', how='left')
            value_counts['Home Win Rate'] = (value_counts['Home Win Rate'] * 100).round(1)

        st.dataframe(value_counts, use_container_width=True)

    # Statistics
    st.subheader("Detailed Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Mean", f"{col.mean():.4f}")
    with stats_col2:
        st.metric("Std Dev", f"{col.std():.4f}")
    with stats_col3:
        st.metric("Variance", f"{col.var():.6f}")
    with stats_col4:
        st.metric("Non-Zero %", f"{(col != 0).mean() * 100:.1f}%")


def show_feature_correlations(pred_df, numeric_features):
    """Feature correlations with outcomes."""
    st.subheader("Feature Correlations")

    # Target selection
    targets = ['home_win', 'result', 'home_covered', 'went_over']
    available_targets = [t for t in targets if t in pred_df.columns]

    selected_target = st.selectbox("Select Target Variable", available_targets)

    # Compute correlations
    valid_features = [f for f in numeric_features if f in pred_df.columns]
    corr_data = []

    for feat in valid_features:
        df_clean = pred_df[[feat, selected_target]].dropna()
        if len(df_clean) > 10:
            corr = df_clean.corr().iloc[0, 1]
            corr_data.append({'Feature': feat, 'Correlation': corr, 'Abs Correlation': abs(corr)})

    corr_df = pd.DataFrame(corr_data).sort_values('Abs Correlation', ascending=False)

    # Bar chart
    fig = px.bar(corr_df.head(20), x='Correlation', y='Feature', orientation='h',
                 title=f"Top 20 Feature Correlations with {selected_target}",
                 color='Correlation', color_continuous_scale='RdYlGn',
                 color_continuous_midpoint=0)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix for top features
    st.subheader("Feature-to-Feature Correlation Matrix")
    top_features = corr_df.head(10)['Feature'].tolist()
    if selected_target not in top_features:
        top_features.append(selected_target)

    corr_matrix = pred_df[top_features].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                    title="Correlation Matrix (Top Features)")
    st.plotly_chart(fig, use_container_width=True)


def show_data_quality(pred_df, numeric_features):
    """Data quality analysis."""
    st.subheader("Data Quality Report")

    # Missing values
    missing = pred_df[numeric_features].isnull().sum()
    missing_pct = (missing / len(pred_df) * 100).round(1)
    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Missing Values")
        if missing_df[missing_df['Missing Count'] > 0].empty:
            st.success("✅ No missing values!")
        else:
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

    with col2:
        st.markdown("### Low Variance Features")
        variance_data = []
        for feat in numeric_features:
            if feat in pred_df.columns:
                var = pred_df[feat].var()
                unique = pred_df[feat].nunique()
                variance_data.append({'Feature': feat, 'Variance': var, 'Unique Values': unique})

        var_df = pd.DataFrame(variance_data).sort_values('Variance')
        low_var = var_df[var_df['Variance'] < 0.1]

        if len(low_var) > 0:
            st.warning(f"⚠️ {len(low_var)} features with variance < 0.1")
            st.dataframe(low_var, use_container_width=True)
        else:
            st.success("✅ All features have sufficient variance!")

    # Zero-dominated features
    st.markdown("### Zero-Dominated Features")
    zero_data = []
    for feat in numeric_features:
        if feat in pred_df.columns:
            zero_pct = (pred_df[feat] == 0).mean() * 100
            if zero_pct > 50:
                zero_data.append({'Feature': feat, 'Zero %': zero_pct, 'Non-Zero %': 100 - zero_pct})

    if zero_data:
        st.warning(f"⚠️ {len(zero_data)} features with >50% zeros")
        st.dataframe(pd.DataFrame(zero_data).sort_values('Zero %', ascending=False), use_container_width=True)
    else:
        st.success("✅ No features dominated by zeros!")


def show_feature_engineering_docs():
    """Documentation of how features are created."""
    st.subheader("Feature Engineering Documentation")

    st.markdown("""
    ### How Each Feature is Created

    ---

    #### **Base Features**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `elo_diff` | Computed | Home Elo - Away Elo |
    | `elo_prob` | Computed | 1 / (1 + 10^(-elo_diff/400)) |
    | `spread_line` | Schedule | Vegas spread (+ = home underdog) |
    | `total_line` | Schedule | Vegas over/under line |
    | `rest_advantage` | Schedule | home_rest - away_rest |
    | `div_game` | Schedule | 1 if divisional game |
    | `home_implied_prob` | Schedule | Derived from home_moneyline |
    | `away_implied_prob` | Schedule | Derived from away_moneyline |

    ---

    #### **Venue/Weather Features**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `is_dome` | Schedule | roof == 'dome' or 'closed' |
    | `is_cold` | Schedule | temp < 40°F |
    | `is_windy` | Schedule | wind > 15 mph |
    | `is_primetime` | Schedule | TNF, SNF, or MNF game |
    | `is_grass` | Schedule | surface == 'grass' |
    | `bad_weather` | Computed | is_freezing OR is_very_windy |
    | `home_short_week` | Schedule | home_rest < 6 days |
    | `away_short_week` | Schedule | away_rest < 6 days |

    ---

    #### **Passing Features (TIER S)**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `home_cpoe_3wk` | NGS Passing | 3-week rolling avg of Completion % Over Expected |
    | `away_cpoe_3wk` | NGS Passing | 3-week rolling avg of CPOE |
    | `cpoe_diff` | Computed | home_cpoe_3wk - away_cpoe_3wk |
    | `home_pressure_rate_3wk` | PFR Passing | 3-week rolling avg of times pressured % |
    | `away_pressure_rate_3wk` | PFR Passing | 3-week rolling avg of pressure % |
    | `pressure_diff` | Computed | home_pressure - away_pressure |
    | `home_time_to_throw_3wk` | NGS Passing | 3-week rolling avg time to throw |
    | `away_time_to_throw_3wk` | NGS Passing | 3-week rolling avg |

    **⚠️ Note:** CPOE is shifted by 1 week to prevent data leakage.

    ---

    #### **Injury Features (TIER S)**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `home_injury_impact` | Injuries | Sum of (position_weight × status_prob) for all injured players |
    | `away_injury_impact` | Injuries | Same calculation for away team |
    | `injury_diff` | Computed | home_injury_impact - away_injury_impact |
    | `home_qb_out` | Injuries | 1 if any QB is Out/Doubtful/IR |
    | `away_qb_out` | Injuries | 1 if any QB is Out/Doubtful/IR |

    **Position Weights:**
    ```
    QB: 1.0, RB: 0.7, WR: 0.6, TE: 0.5, OL: 0.5, DL: 0.5, LB: 0.5, CB: 0.6, S: 0.5, K/P: 0.3
    ```

    **Status Probabilities:**
    ```
    Out: 1.0, Doubtful: 0.75, Questionable: 0.5, Probable: 0.25
    ```

    **⚠️ Issues:**
    - `qb_out` only captures backup QBs on report, not starters resting
    - Doesn't account for player quality (Pro Bowl vs. depth player)
    - 2025 injury data not yet available in nflverse

    ---

    #### **Rushing Features (TIER A)**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `home_ryoe_3wk` | NGS Rushing | 3-week rolling avg of Rush Yards Over Expected per attempt |
    | `away_ryoe_3wk` | NGS Rushing | Same for away team |
    | `ryoe_diff` | Computed | home_ryoe - away_ryoe |

    ---

    #### **Receiving Features (TIER A)**
    | Feature | Source | Calculation |
    |---------|--------|-------------|
    | `home_separation_3wk` | NGS Receiving | 3-week rolling avg of avg_separation |
    | `away_separation_3wk` | NGS Receiving | Same for away team |
    | `separation_diff` | Computed | home_separation - away_separation |

    ---

    ### Known Issues & Limitations

    1. **QB Out Flag**: Only captures QBs explicitly listed as Out/Doubtful/IR. Many injured starters show as "Questionable" or "None" on the report.

    2. **Injury Impact**: Simple position weighting doesn't account for:
       - Player skill level (Patrick Mahomes vs. backup QB)
       - Scheme fit importance
       - Depth at position

    3. **Data Availability**:
       - NGS data starts in 2016
       - PFR data starts in 2018
       - 2025 injuries not yet released

    4. **Rolling Windows**: 3-week windows may be too short for reliable signal; 5-week tested but not better.
    """)


def show_model_performance(results, weekly_2024, weekly_2025):
    """Model Performance section."""
    st.header("🎯 Model Performance")

    # Summary metrics at top
    st.subheader("Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Spread Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            st.metric("2024 Win Rate", f"{r['spread_wr']*100:.1f}%",
                     f"{r['spread_roi']:+.1f}% ROI")
        if results.get('results_2025'):
            r = results['results_2025']
            st.metric("2025 Win Rate", f"{r['spread_wr']*100:.1f}%",
                     f"{r['spread_roi']:+.1f}% ROI")

    with col2:
        st.markdown("### Totals Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            st.metric("2024 Win Rate", f"{r['totals_wr']*100:.1f}%",
                     f"{r['totals_roi']:+.1f}% ROI")
        if results.get('results_2025'):
            r = results['results_2025']
            st.metric("2025 Win Rate", f"{r['totals_wr']*100:.1f}%",
                     f"{r['totals_roi']:+.1f}% ROI")

    with col3:
        st.markdown("### Moneyline Betting")
        if results.get('results_2024'):
            r = results['results_2024']
            ml_roi = r.get('ml_roi', 0)
            st.metric("2024 Win Rate", f"{r['ml_wr']*100:.1f}%",
                     f"{ml_roi:+.1f}% ROI ({r['ml_bets']} bets)")
        if results.get('results_2025'):
            r = results['results_2025']
            ml_roi = r.get('ml_roi', 0)
            st.metric("2025 Win Rate", f"{r['ml_wr']*100:.1f}%",
                     f"{ml_roi:+.1f}% ROI ({r['ml_bets']} bets)")

    # Win Prediction Accuracy
    st.subheader("Win Prediction Accuracy (All Games)")
    col1, col2 = st.columns(2)
    with col1:
        if results.get('results_2024'):
            acc = results['results_2024'].get('win_pred_accuracy', 0) * 100
            st.metric("2024 Accuracy", f"{acc:.1f}%",
                     f"{acc - 50:.1f}% vs coin flip")
    with col2:
        if results.get('results_2025'):
            acc = results['results_2025'].get('win_pred_accuracy', 0) * 100
            st.metric("2025 Accuracy", f"{acc:.1f}%",
                     f"{acc - 50:.1f}% vs coin flip")

    # Comparison chart
    st.subheader("Win Rate Comparison")

    comparison_data = {
        'Model': ['Baseline (Elo)', 'TIER S+A (2024)', 'TIER S+A (2025)'],
        'Spread WR': [49.4,
                      results['results_2024']['spread_wr']*100 if results.get('results_2024') else 0,
                      results['results_2025']['spread_wr']*100 if results.get('results_2025') else 0],
        'Totals WR': [50.0,
                      results['results_2024']['totals_wr']*100 if results.get('results_2024') else 0,
                      results['results_2025']['totals_wr']*100 if results.get('results_2025') else 0],
        'ML WR': [55.0,  # Baseline win prediction
                  results['results_2024']['ml_wr']*100 if results.get('results_2024') else 0,
                  results['results_2025']['ml_wr']*100 if results.get('results_2025') else 0]
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Spread WR', x=comparison_data['Model'], y=comparison_data['Spread WR']))
    fig.add_trace(go.Bar(name='Totals WR', x=comparison_data['Model'], y=comparison_data['Totals WR']))
    fig.add_trace(go.Bar(name='Moneyline WR', x=comparison_data['Model'], y=comparison_data['ML WR']))
    fig.add_hline(y=52.38, line_dash="dash", line_color="red", annotation_text="Spread/Totals Breakeven")
    fig.update_layout(barmode='group', title="Win Rate by Bet Type", yaxis_title="Win Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Weekly performance - now includes moneyline
    st.subheader("Weekly Performance Trend")

    col1, col2 = st.columns(2)

    with col1:
        if weekly_2024 is not None:
            st.markdown("**2024 Season**")
            # Include moneyline if column exists
            y_cols = ['spread_wr', 'totals_wr']
            if 'ml_wr' in weekly_2024.columns:
                y_cols.append('ml_wr')
            if 'win_accuracy' in weekly_2024.columns:
                y_cols.append('win_accuracy')
            fig = px.line(weekly_2024, x='week', y=y_cols,
                         title="2024 Weekly Win Rates",
                         labels={'value': 'Win Rate', 'week': 'Week'})
            fig.add_hline(y=0.5238, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if weekly_2025 is not None:
            st.markdown("**2025 Season**")
            y_cols = ['spread_wr', 'totals_wr']
            if 'ml_wr' in weekly_2025.columns:
                y_cols.append('ml_wr')
            if 'win_accuracy' in weekly_2025.columns:
                y_cols.append('win_accuracy')
            fig = px.line(weekly_2025, x='week', y=y_cols,
                         title="2025 Weekly Win Rates",
                         labels={'value': 'Win Rate', 'week': 'Week'})
            fig.add_hline(y=0.5238, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)


def show_2025_validation(pred_2025, weekly_2025):
    """2025 Validation section."""
    st.header("📅 2025 Validation Results")

    if pred_2025 is None:
        st.warning("No 2025 prediction data available")
        return

    # Week selector
    weeks = sorted(pred_2025['week'].unique())
    selected_week = st.selectbox("Select Week", weeks, index=len(weeks)-1)

    week_df = pred_2025[pred_2025['week'] == selected_week].copy()

    st.subheader(f"Week {selected_week} Games ({len(week_df)} games)")

    # Game-by-game table with moneyline
    week_df['spread_correct'] = (week_df['bet_home_cover'] == week_df['home_covered'])
    week_df['totals_correct'] = (week_df['bet_over'] == week_df['went_over'])

    # Build display columns dynamically
    display_cols = ['away_team', 'home_team', 'away_score', 'home_score',
                    'spread_line', 'pred_margin', 'spread_correct',
                    'total_line', 'pred_total', 'game_total', 'totals_correct']
    col_names = ['Away', 'Home', 'Away Score', 'Home Score',
                 'Spread Line', 'Pred Margin', 'Spread ✓',
                 'Total Line', 'Pred Total', 'Actual Total', 'Totals ✓']

    # Add moneyline columns if available
    if 'pred_win_prob' in week_df.columns:
        display_cols.extend(['pred_win_prob', 'ml_bet', 'ml_correct'])
        col_names.extend(['Win Prob', 'ML Bet', 'ML ✓'])

    display_df = week_df[[c for c in display_cols if c in week_df.columns]].copy()
    display_df.columns = col_names[:len(display_df.columns)]

    st.dataframe(display_df, use_container_width=True)

    # Week summary - 6 columns now
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        spread_wr = week_df['spread_correct'].mean()
        st.metric("Spread WR", f"{spread_wr*100:.1f}%",
                  delta=f"{(spread_wr-0.5238)*100:+.1f}%" if spread_wr > 0.5238 else f"{(spread_wr-0.5238)*100:.1f}%")
    with col2:
        totals_wr = week_df['totals_correct'].mean()
        st.metric("Totals WR", f"{totals_wr*100:.1f}%",
                  delta=f"{(totals_wr-0.5238)*100:+.1f}%" if totals_wr > 0.5238 else f"{(totals_wr-0.5238)*100:.1f}%")
    with col3:
        # Moneyline WR
        if 'ml_bet' in week_df.columns and 'ml_correct' in week_df.columns:
            ml_bets = week_df[week_df['ml_bet']]
            if len(ml_bets) > 0:
                ml_wr = ml_bets['ml_correct'].mean()
                st.metric("ML WR", f"{ml_wr*100:.1f}%", f"{len(ml_bets)} bets")
            else:
                st.metric("ML WR", "N/A", "0 bets")
        else:
            st.metric("ML WR", "N/A")
    with col4:
        # Win prediction accuracy
        if 'pred_home_win' in week_df.columns and 'actual_home_win' in week_df.columns:
            win_acc = (week_df['pred_home_win'] == week_df['actual_home_win']).mean()
            st.metric("Win Accuracy", f"{win_acc*100:.1f}%")
        else:
            st.metric("Win Accuracy", "N/A")
    with col5:
        st.metric("Spread", f"{int(week_df['spread_correct'].sum())}/{len(week_df)}")
    with col6:
        st.metric("Totals", f"{int(week_df['totals_correct'].sum())}/{len(week_df)}")


def show_backtest_results(pred_2024, pred_2025, weekly_2024, weekly_2025):
    """Backtest Results Visualization section."""
    st.header("💰 Backtest Results")

    # Cumulative P/L chart
    st.subheader("Cumulative Profit/Loss")

    dataset = st.radio("Select Dataset", ["2024", "2025"], horizontal=True)
    pred_df = pred_2024 if dataset == "2024" else pred_2025

    if pred_df is None:
        st.warning(f"No {dataset} data available")
        return

    # Calculate cumulative P/L (assuming $100 bets at -110)
    pred_df = pred_df.sort_values(['week', 'game_id']).copy()

    # Spread P/L
    pred_df['spread_pl'] = pred_df.apply(
        lambda x: 90.91 if x['bet_home_cover'] == x['home_covered'] else -100, axis=1
    )
    pred_df['spread_cumpl'] = pred_df['spread_pl'].cumsum()

    # Totals P/L
    pred_df['totals_pl'] = pred_df.apply(
        lambda x: 90.91 if x['bet_over'] == x['went_over'] else -100, axis=1
    )
    pred_df['totals_cumpl'] = pred_df['totals_pl'].cumsum()

    # Moneyline P/L (only on games where we bet)
    has_ml = 'ml_bet' in pred_df.columns and 'ml_correct' in pred_df.columns
    if has_ml:
        def calc_ml_pl(row):
            if not row.get('ml_bet', False):
                return 0  # No bet placed
            if row.get('ml_correct', False):
                return 90.91  # Win at -110 odds (simplified)
            else:
                return -100  # Loss
        pred_df['ml_pl'] = pred_df.apply(calc_ml_pl, axis=1)
        pred_df['ml_cumpl'] = pred_df['ml_pl'].cumsum()

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(pred_df))), y=pred_df['spread_cumpl'],
                             mode='lines', name='Spread Betting'))
    fig.add_trace(go.Scatter(x=list(range(len(pred_df))), y=pred_df['totals_cumpl'],
                             mode='lines', name='Totals Betting'))
    if has_ml:
        fig.add_trace(go.Scatter(x=list(range(len(pred_df))), y=pred_df['ml_cumpl'],
                                 mode='lines', name='Moneyline Betting'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Cumulative P/L - {dataset} Season ($100 bets)",
                      xaxis_title="Game #", yaxis_title="Cumulative P/L ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spread Final P/L", f"${pred_df['spread_cumpl'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Totals Final P/L", f"${pred_df['totals_cumpl'].iloc[-1]:,.0f}")
    with col3:
        if has_ml:
            st.metric("Moneyline Final P/L", f"${pred_df['ml_cumpl'].iloc[-1]:,.0f}")
        else:
            st.metric("Moneyline Final P/L", "N/A")

    # Moneyline stats
    if has_ml:
        st.subheader("Moneyline Betting Details")
        ml_bets_df = pred_df[pred_df['ml_bet']]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total ML Bets", len(ml_bets_df))
        with col2:
            ml_wins = ml_bets_df['ml_correct'].sum()
            st.metric("ML Wins", int(ml_wins))
        with col3:
            ml_wr = ml_bets_df['ml_correct'].mean() * 100 if len(ml_bets_df) > 0 else 0
            st.metric("ML Win Rate", f"{ml_wr:.1f}%")
        with col4:
            ml_roi = (ml_bets_df['ml_pl'].sum() / (len(ml_bets_df) * 100)) * 100 if len(ml_bets_df) > 0 else 0
            st.metric("ML ROI", f"{ml_roi:+.1f}%")

    # ROI by week
    st.subheader("ROI by Week")
    weekly_df = weekly_2024 if dataset == "2024" else weekly_2025

    if weekly_df is not None:
        weekly_df = weekly_df.copy()
        weekly_df['spread_roi'] = (weekly_df['spread_wr'] * 1.909 - 1) * 100
        weekly_df['totals_roi'] = (weekly_df['totals_wr'] * 1.909 - 1) * 100

        y_cols = ['spread_roi', 'totals_roi']
        if 'ml_wr' in weekly_df.columns:
            weekly_df['ml_roi'] = (weekly_df['ml_wr'] * 1.909 - 1) * 100
            y_cols.append('ml_roi')

        fig = px.bar(weekly_df, x='week', y=y_cols,
                     barmode='group', title=f"ROI by Week - {dataset}",
                     labels={'value': 'ROI (%)', 'week': 'Week'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

