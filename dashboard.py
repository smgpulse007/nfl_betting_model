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
         "🎯 Model Performance", "📅 2025 Validation", "💰 Backtest Results",
         "🧪 Model Experiments", "🔬 Deep Analysis"]
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
    elif page == "🧪 Model Experiments":
        show_model_experiments()
    elif page == "🔬 Deep Analysis":
        show_deep_analysis()


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


def show_model_experiments():
    """Model Experiments section - Compare different ML/DL architectures."""
    st.header("🧪 Model Experiments")
    st.markdown("""
    This section tracks experiments with different model architectures,
    comparing them against the **v0.2.0 XGBoost baseline**.

    **Educational Progression:**
    1. **Linear Models** - Interpretable baselines (OLS, Ridge, Lasso, Logistic)
    2. **Regularized Models** - ElasticNet, L1/L2 penalties
    3. **Tree-Based** - Decision Trees, Random Forests
    4. **Boosting** - XGBoost (current), LightGBM, CatBoost
    5. **Deep Learning** - PyTorch MLP, Residual, Attention
    """)

    # Load linear experiment results
    linear_results_path = RESULTS_DIR / "linear_experiments.json"
    if linear_results_path.exists():
        with open(linear_results_path) as f:
            linear_results = json.load(f)
    else:
        st.warning("No linear experiment results found. Run `python model_experiments.py` first.")
        linear_results = None

    # Tabs for different model types
    tab1, tab2, tab3 = st.tabs(["📉 Linear Models", "📊 Odds Ratios", "🔄 Model Comparison"])

    with tab1:
        show_linear_models_tab(linear_results)

    with tab2:
        show_odds_ratios_tab(linear_results)

    with tab3:
        show_model_comparison_tab(linear_results)


def show_linear_models_tab(linear_results):
    """Show linear model results."""
    st.subheader("Linear Model Performance")

    if linear_results is None:
        st.info("Run experiments first: `python model_experiments.py`")
        return

    # Baseline comparison
    st.markdown("### v0.2.0 Baseline (XGBoost)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Spread ROI", "+11.4%", help="Tuned XGBoost")
    col2.metric("Totals ROI", "+5.8%", help="Tuned XGBoost")
    col3.metric("ML Accuracy", "64.5%", help="Baseline params")

    st.markdown("---")
    st.markdown("### Linear Model Results (2025 Validation)")

    # Build comparison dataframe
    models_data = []
    for name, metrics in linear_results.get('models', {}).items():
        model_type = name.split('_')[0]
        reg = name.split('_')[1] if '_' in name else 'base'
        models_data.append({
            'Model': name,
            'Type': model_type.title(),
            'Regularization': reg.upper(),
            'RMSE': metrics.get('rmse', None),
            'Win Rate': metrics.get('wr', None),
            'ROI (%)': metrics.get('roi', None),
            'LogLoss': metrics.get('logloss', None),
            'Accuracy': metrics.get('accuracy', None)
        })

    if models_data:
        df = pd.DataFrame(models_data)

        # Spread/Totals table
        spread_totals = df[df['Type'].isin(['Spread', 'Totals'])].copy()
        if not spread_totals.empty:
            spread_totals['ROI (%)'] = spread_totals['ROI (%)'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
            spread_totals['Win Rate'] = spread_totals['Win Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            st.dataframe(
                spread_totals[['Model', 'Type', 'Regularization', 'RMSE', 'Win Rate', 'ROI (%)']],
                use_container_width=True, hide_index=True
            )

        # Moneyline table
        ml = df[df['Type'] == 'Moneyline'].copy()
        if not ml.empty:
            st.markdown("### Moneyline Classification")
            ml['Accuracy'] = ml['Accuracy'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            st.dataframe(
                ml[['Model', 'Regularization', 'LogLoss', 'Accuracy']],
                use_container_width=True, hide_index=True
            )

        # Visualization
        st.markdown("### ROI Comparison")
        roi_df = df[df['ROI (%)'].notna()].copy()
        if not roi_df.empty:
            fig = px.bar(roi_df, x='Model', y='ROI (%)', color='Type',
                        title="ROI by Model", text='ROI (%)')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            # Add XGBoost baseline reference
            fig.add_hline(y=11.4, line_dash="dot", line_color="green",
                         annotation_text="v0.2.0 Spread Baseline")
            st.plotly_chart(fig, use_container_width=True)


def show_odds_ratios_tab(linear_results):
    """Show odds ratios from logistic regression."""
    st.subheader("Logistic Regression Odds Ratios")

    st.markdown("""
    **Interpretation Guide:**
    - **Odds Ratio > 1**: Increases probability of home win
    - **Odds Ratio < 1**: Decreases probability of home win
    - **Odds Ratio = 1**: No effect

    Example: OR=1.5 means a 1-unit increase in that feature increases home win odds by 50%.
    """)

    if linear_results is None:
        st.info("Run experiments first")
        return

    coefficients = linear_results.get('coefficients', {})

    for model_name in ['moneyline_l2', 'moneyline_l1']:
        if model_name in coefficients:
            st.markdown(f"### {model_name.replace('_', ' ').title()}")
            coef_df = pd.DataFrame(coefficients[model_name])

            if 'odds_ratio' in coef_df.columns:
                coef_df = coef_df.sort_values('abs_coef', ascending=False)

                # Top positive and negative
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top 10 Positive (Favor Home)**")
                    pos = coef_df[coef_df['coefficient'] > 0].head(10)
                    for _, row in pos.iterrows():
                        st.write(f"• **{row['feature']}**: OR={row['odds_ratio']:.3f}")

                with col2:
                    st.markdown("**Top 10 Negative (Favor Away)**")
                    neg = coef_df[coef_df['coefficient'] < 0].head(10)
                    for _, row in neg.iterrows():
                        st.write(f"• **{row['feature']}**: OR={row['odds_ratio']:.3f}")

                # Visualization
                top_n = coef_df.head(15).copy()
                fig = px.bar(top_n, x='odds_ratio', y='feature', orientation='h',
                            title=f"Odds Ratios - {model_name.replace('_', ' ').title()}",
                            color='coefficient', color_continuous_scale='RdBu_r')
                fig.add_vline(x=1, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)


def show_model_comparison_tab(linear_results):
    """Compare all models side by side."""
    st.subheader("Model Architecture Comparison")

    # Hardcoded baseline metrics
    comparison_data = [
        {'Model': 'XGBoost v0.2.0', 'Type': 'Spread', 'WR': 52.2, 'ROI': -0.4, 'Architecture': 'Gradient Boosting'},
        {'Model': 'XGBoost v0.2.0', 'Type': 'Totals', 'WR': 50.0, 'ROI': -4.5, 'Architecture': 'Gradient Boosting'},
        {'Model': 'XGBoost v0.1.0', 'Type': 'ML', 'WR': 49.3, 'ROI': 53.0, 'Architecture': 'Gradient Boosting'},
    ]

    # Add linear results
    if linear_results:
        for name, metrics in linear_results.get('models', {}).items():
            model_type = name.split('_')[0].title()
            comparison_data.append({
                'Model': name,
                'Type': model_type,
                'WR': (metrics.get('wr', 0) or 0) * 100,
                'ROI': metrics.get('roi', 0) or 0,
                'Architecture': 'Linear'
            })

    df = pd.DataFrame(comparison_data)

    # Summary table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Scatter plot: WR vs ROI
    fig = px.scatter(df, x='WR', y='ROI', color='Architecture',
                    hover_data=['Model', 'Type'],
                    title="Win Rate vs ROI by Architecture",
                    size=[10]*len(df))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=52.4, line_dash="dash", line_color="gray",
                  annotation_text="Break-even (52.4%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Key Insights

    | Architecture | Pros | Cons |
    |--------------|------|------|
    | **Linear** | Interpretable, fast, odds ratios | Limited to linear relationships |
    | **XGBoost** | Captures non-linear patterns | Black box, can overfit |
    | **Deep Learning** | *Coming soon* | Needs more data, complex tuning |
    """)


def show_deep_analysis():
    """Deep Analysis section - Understanding model behavior."""
    st.header("🔬 Deep Model Analysis")
    st.markdown("""
    This section provides deep insights into **why different models perform differently**
    and helps understand the underlying patterns in NFL betting data.
    """)

    # Load deep analysis results
    deep_analysis_file = RESULTS_DIR / "deep_analysis.json"
    if not deep_analysis_file.exists():
        st.warning("No deep analysis results found. Run `python deep_analysis.py` first.")
        return

    with open(deep_analysis_file) as f:
        analysis = json.load(f)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Correlations",
        "⚠️ Multicollinearity",
        "🎯 Totals: OLS vs XGBoost",
        "🏆 ML: CatBoost Analysis",
        "📅 Week 15 Deep Dive"
    ])

    with tab1:
        show_correlation_analysis(analysis)

    with tab2:
        show_multicollinearity_analysis(analysis)

    with tab3:
        show_totals_analysis(analysis)

    with tab4:
        show_catboost_analysis(analysis)

    with tab5:
        show_week15_analysis(analysis)


def show_correlation_analysis(analysis):
    """Show feature correlations with targets."""
    st.subheader("Feature Correlations with Targets")

    correlations = analysis.get('correlations', [])
    if not correlations:
        st.warning("No correlation data available")
        return

    corr_df = pd.DataFrame(correlations)

    # Create three columns for each target
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Spread (Result)")
        top_spread = corr_df.nlargest(10, 'abs_corr_spread')[['feature', 'corr_spread']]
        fig = px.bar(top_spread, x='corr_spread', y='feature', orientation='h',
                     color='corr_spread', color_continuous_scale='RdBu_r',
                     range_color=[-0.5, 0.5])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Totals (Game Total)")
        top_total = corr_df.nlargest(10, 'abs_corr_total')[['feature', 'corr_total']]
        fig = px.bar(top_total, x='corr_total', y='feature', orientation='h',
                     color='corr_total', color_continuous_scale='RdBu_r',
                     range_color=[-0.5, 0.5])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("#### Moneyline (Home Win)")
        top_win = corr_df.nlargest(10, 'abs_corr_win')[['feature', 'corr_win']]
        fig = px.bar(top_win, x='corr_win', y='feature', orientation='h',
                     color='corr_win', color_continuous_scale='RdBu_r',
                     range_color=[-0.5, 0.5])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key Insight**: `spread_line`, `home_implied_prob`, and `away_implied_prob` have the
    strongest correlations with outcomes. These are **market-derived features** that already
    incorporate wisdom of the crowd.
    """)


def show_multicollinearity_analysis(analysis):
    """Show multicollinearity between features."""
    st.subheader("Multicollinearity Analysis")

    multi = analysis.get('multicollinearity', [])
    if not multi:
        st.success("No highly correlated feature pairs found (|r| > 0.7)")
        return

    st.warning(f"Found **{len(multi)} highly correlated pairs** (|r| > 0.7)")

    multi_df = pd.DataFrame(multi)
    multi_df['abs_corr'] = multi_df['correlation'].abs()
    multi_df = multi_df.sort_values('abs_corr', ascending=False)

    # Display as table
    st.dataframe(
        multi_df[['feature1', 'feature2', 'correlation']].head(20),
        use_container_width=True,
        hide_index=True
    )

    # Heatmap of top pairs
    fig = px.bar(multi_df.head(10), x='correlation', y=multi_df.head(10).apply(
        lambda r: f"{r['feature1']} ↔ {r['feature2']}", axis=1),
        orientation='h', color='correlation', color_continuous_scale='RdBu_r',
        range_color=[-1, 1], title="Top 10 Correlated Feature Pairs")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Why This Matters:**
    - `home_implied_prob` and `away_implied_prob` are **-0.999** correlated (they sum to 1)
    - `spread_line` and `implied_prob` are **0.989** correlated (same market information)
    - `elo_diff` and `elo_prob` are **0.986** correlated (one derives from other)

    **Recommendation**: Consider removing redundant features to reduce noise.
    """)


def show_totals_analysis(analysis):
    """Explain why OLS beats XGBoost for totals."""
    st.subheader("Why OLS Beats XGBoost for Totals Prediction")

    totals = analysis.get('totals_analysis', {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("OLS RMSE", f"{totals.get('ols_rmse', 0):.2f}")
    with col2:
        st.metric("XGBoost RMSE", f"{totals.get('xgb_rmse', 0):.2f}")

    st.success(f"OLS wins by **{totals.get('xgb_rmse', 0) - totals.get('ols_rmse', 0):.2f}** RMSE points")

    # Feature importance comparison
    importance = totals.get('importance', [])
    if importance:
        imp_df = pd.DataFrame(importance)

        fig = make_subplots(rows=1, cols=2, subplot_titles=['OLS Coefficients', 'XGBoost Importance'])

        top_10 = imp_df.head(10)
        fig.add_trace(
            go.Bar(x=top_10['ols_coef'], y=top_10['feature'], orientation='h', name='OLS'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=top_10['xgb_importance'], y=top_10['feature'], orientation='h', name='XGB'),
            row=1, col=2
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    ### 💡 Key Insight: Linear Relationship

    The correlation between `total_line` and actual `game_total` is **{totals.get('total_line_correlation', 0):.3f}**.

    **Why OLS wins:**
    1. **Simple is better**: Game totals have a mostly linear relationship with Vegas lines
    2. **XGBoost overcomplicates**: Tries to find non-linear patterns that don't exist
    3. **Regularization**: OLS naturally regularizes by using fewer complex interactions
    4. **Vegas efficiency**: The total line already incorporates most relevant factors

    **Recommendation**: Use **OLS/Linear Regression** for totals prediction in v0.3.0
    """)


def show_catboost_analysis(analysis):
    """Explain why CatBoost beats other classifiers."""
    st.subheader("Why CatBoost Wins for Moneyline Prediction")

    ml_analysis = analysis.get('ml_analysis', {})
    feature_importance = ml_analysis.get('feature_importance', {})

    st.metric("CatBoost Net Advantage", f"{ml_analysis.get('catboost_advantage', 0)} games over XGBoost")

    # Compare feature importance across models
    if feature_importance:
        st.markdown("#### Feature Importance by Model")

        model_tabs = st.tabs(list(feature_importance.keys()))

        for i, (model_name, importance) in enumerate(feature_importance.items()):
            with model_tabs[i]:
                imp_df = pd.DataFrame(importance).head(10)

                if 'odds_ratio' in imp_df.columns:
                    # Logistic model - show odds ratios
                    fig = px.bar(imp_df, x='odds_ratio', y='feature', orientation='h',
                                 color='coefficient', color_continuous_scale='RdBu_r',
                                 title=f"{model_name} - Top 10 Features (Odds Ratios)")
                    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
                else:
                    fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                                 title=f"{model_name} - Top 10 Features")

                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 💡 Why CatBoost Wins

    1. **Ordered Boosting**: CatBoost uses ordered boosting to reduce prediction shift
    2. **Categorical Handling**: Natively handles `is_dome`, `is_grass`, `div_game` without encoding
    3. **Better Regularization**: Default regularization is well-tuned for tabular data
    4. **Symmetric Trees**: Uses oblivious decision trees that are more robust

    **Net Advantage**: CatBoost correctly predicted **2 more games** than XGBoost in 2025

    **Recommendation**: Use **CatBoost** for moneyline prediction in v0.3.0
    """)


def show_week15_analysis(analysis):
    """Deep dive into Week 15 2025 predictions."""
    st.subheader("Week 15 2025 - Game-by-Game Analysis")

    week15 = analysis.get('week15_2025', {})

    if not week15:
        st.warning("No Week 15 2025 data available")
        return

    summary = week15.get('summary', {})

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        wr = summary.get('spread_wr', 0)
        st.metric("Spread", f"{wr:.1%}", delta=f"{(wr - 0.5)*100:+.1f}% vs coin flip")
    with col2:
        wr = summary.get('totals_wr', 0)
        st.metric("Totals", f"{wr:.1%}", delta=f"{(wr - 0.5)*100:+.1f}% vs coin flip")
    with col3:
        acc = summary.get('ml_accuracy', 0)
        st.metric("Moneyline", f"{acc:.1%}", delta=f"{(acc - 0.5)*100:+.1f}% vs coin flip")

    # Game-by-game table
    games = week15.get('games', [])
    if games:
        games_df = pd.DataFrame(games)

        # Format for display
        display_df = games_df[['matchup', 'actual_result', 'spread_line', 'pred_spread',
                               'spread_correct', 'actual_total', 'total_line', 'pred_total',
                               'totals_correct', 'pred_win_proba', 'ml_correct']].copy()

        # Add icons
        display_df['Spread'] = display_df['spread_correct'].map({True: '✅', False: '❌'})
        display_df['Totals'] = display_df['totals_correct'].map({True: '✅', False: '❌'})
        display_df['ML'] = display_df['ml_correct'].map({True: '✅', False: '❌'})

        st.dataframe(
            display_df[['matchup', 'actual_result', 'spread_line', 'pred_spread', 'Spread',
                        'actual_total', 'total_line', 'pred_total', 'Totals',
                        'pred_win_proba', 'ML']],
            use_container_width=True,
            hide_index=True
        )

    # Analyze misses
    st.markdown("### ❌ Miss Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Spread Misses")
        spread_misses = week15.get('spread_misses', [])
        if spread_misses:
            for miss in spread_misses:
                error = miss.get('pred_spread', 0) - miss.get('result', 0)
                st.write(f"**{miss.get('away_team')} @ {miss.get('home_team')}**: "
                        f"Pred {miss.get('pred_spread', 0):+.1f}, "
                        f"Actual {miss.get('result', 0):+.1f} (Error: {error:+.1f})")

    with col2:
        st.markdown("#### ML Misses")
        ml_misses = week15.get('ml_misses', [])
        if ml_misses:
            for miss in ml_misses:
                prob = miss.get('pred_win_proba', 0)
                actual = "Home" if miss.get('home_win') else "Away"
                st.write(f"**{miss.get('away_team')} @ {miss.get('home_team')}**: "
                        f"Predicted Home ({prob:.0%}), Actual {actual}")

    st.info("""
    **Key Observations:**
    - Large spread misses often involve **unexpected blowouts** (NYJ@JAX, CLE@CHI)
    - ML misses include **upsets** where underdogs won (LAC@KC, ATL@TB)
    - Model performs best when games go according to expectations
    """)


if __name__ == "__main__":
    main()

