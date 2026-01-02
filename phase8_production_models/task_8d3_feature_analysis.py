"""
Task 8D.3: Feature Analysis Visualizations

Display SHAP values, permutation importance, and feature correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def show_feature_analysis(data):
    """Display feature analysis page"""
    
    st.markdown('<div class="main-header">🔍 Feature Analysis</div>', unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "⭐ SHAP Analysis",
        "🔄 Permutation Importance",
        "🔗 Feature Correlation"
    ])
    
    # Tab 1: SHAP Analysis
    with tab1:
        st.markdown("### SHAP (SHapley Additive exPlanations) Analysis")
        
        st.markdown("""
        SHAP values explain individual predictions by showing how each feature contributes.
        - **Positive SHAP**: Feature pushes prediction toward home win
        - **Negative SHAP**: Feature pushes prediction toward away win
        - **Magnitude**: Strength of feature's impact
        """)
        
        st.markdown("---")
        
        # Model selector
        available_models = [m for m in data['shap_importance'].keys()]
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            key='shap_model_select'
        )
        
        # Top features table
        st.markdown(f"#### Top 20 Features - {selected_model}")
        
        shap_df = pd.DataFrame(data['shap_importance'][selected_model][:20])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                shap_df[['feature', 'importance']].style.format({
                    'importance': '{:.4f}'
                }).background_gradient(cmap='YlOrRd', subset=['importance']),
                width='stretch',
                hide_index=True
            )
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(6, 8))
            
            top_10 = shap_df.head(10)
            ax.barh(range(len(top_10)), top_10['importance'], color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels([f.replace('home_', '').replace('away_', '')[:30] 
                               for f in top_10['feature']], fontsize=9)
            ax.set_xlabel('SHAP Importance', fontsize=10)
            ax.set_title(f'Top 10 Features', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # SHAP summary plot
        st.markdown(f"#### SHAP Summary Plot - {selected_model}")
        
        try:
            img_path = f'../results/phase8_results/shap_analysis/summary_plot_{selected_model.lower().replace(" ", "_")}.png'
            img = Image.open(img_path)
            st.image(img, width='stretch')
        except:
            st.warning(f"SHAP summary plot not found for {selected_model}")
        
        st.markdown("---")
        
        # Global importance comparison
        st.markdown("#### Global Feature Importance Comparison")
        
        try:
            img = Image.open('../results/phase8_results/shap_analysis/global_importance_comparison.png')
            st.image(img, width='stretch')
        except:
            st.warning("Global importance comparison not found")
        
        st.markdown("---")
        
        # Dependence plots
        st.markdown("#### SHAP Dependence Plots (Top 3 Features)")
        
        st.markdown("""
        Dependence plots show how feature values affect predictions:
        - **X-axis**: Feature value
        - **Y-axis**: SHAP value (impact on prediction)
        - **Color**: Interaction with another feature
        """)
        
        col1, col2, col3 = st.columns(3)
        
        top_3_features = shap_df.head(3)['feature'].values
        
        for idx, (col, feat) in enumerate(zip([col1, col2, col3], top_3_features)):
            with col:
                try:
                    clean_feat = feat.replace('/', '_').replace(' ', '_')
                    img = Image.open(f'../results/phase8_results/shap_analysis/dependence_{clean_feat}.png')
                    st.image(img, width='stretch')
                except:
                    st.warning(f"Dependence plot not found for {feat}")
        
        st.markdown("---")
        
        # Force plots
        st.markdown("#### SHAP Force Plots (Example Predictions)")
        
        st.markdown("""
        Force plots show how features push predictions for individual games:
        - **Red**: Features pushing toward home win
        - **Blue**: Features pushing toward away win
        - **Base value**: Average prediction
        """)
        
        col1, col2, col3 = st.columns(3)
        
        force_plot_types = [
            ('high_confidence_home_win', 'High Conf Home Win'),
            ('high_confidence_away_win', 'High Conf Away Win'),
            ('uncertain_prediction', 'Uncertain Prediction')
        ]
        
        for col, (plot_type, title) in zip([col1, col2, col3], force_plot_types):
            with col:
                st.markdown(f"**{title}**")
                try:
                    img = Image.open(f'../results/phase8_results/shap_analysis/force_plot_{plot_type}.png')
                    st.image(img, width='stretch')
                except:
                    st.warning(f"Force plot not found: {plot_type}")
    
    # Tab 2: Permutation Importance
    with tab2:
        st.markdown("### Permutation Importance Analysis")
        
        st.markdown("""
        Permutation importance measures the decrease in model performance when a feature's values are randomly shuffled.
        - **Higher value**: More important feature
        - **Error bars**: Variability across permutations
        """)
        
        st.markdown("---")
        
        # Model selector
        selected_model_perm = st.selectbox(
            "Select Model:",
            list(data['perm_importance'].keys()),
            key='perm_model_select'
        )
        
        # Top features table
        st.markdown(f"#### Top 20 Features - {selected_model_perm}")
        
        perm_df = pd.DataFrame(data['perm_importance'][selected_model_perm][:20])
        
        st.dataframe(
            perm_df[['feature', 'importance_mean', 'importance_std']].rename(columns={
                'feature': 'Feature',
                'importance_mean': 'Importance',
                'importance_std': 'Std Dev'
            }).style.format({
                'Importance': '{:.4f}',
                'Std Dev': '{:.4f}'
            }).background_gradient(cmap='YlOrRd', subset=['Importance']),
            width='stretch',
            hide_index=True
        )

        st.markdown("---")

        # Permutation importance visualization
        st.markdown("#### Permutation Importance - All Models")

        try:
            img = Image.open('../results/phase8_results/permutation_importance/permutation_importance_all_models.png')
            st.image(img, width='stretch')
        except:
            st.warning("Permutation importance visualization not found")

        st.markdown("---")

        # SHAP vs Permutation comparison
        st.markdown("#### SHAP vs Permutation Importance Comparison")

        st.markdown("""
        Comparing SHAP and permutation importance helps validate feature importance:
        - **High correlation**: Consistent feature rankings
        - **Low correlation**: Different aspects of importance captured
        """)

        try:
            img = Image.open('../results/phase8_results/permutation_importance/shap_vs_permutation_comparison.png')
            st.image(img, width='stretch')
        except:
            st.warning("SHAP vs Permutation comparison not found")

    # Tab 3: Feature Correlation
    with tab3:
        st.markdown("### Feature Correlation & Redundancy Analysis")

        st.markdown("""
        Identifying highly correlated features helps reduce dimensionality and improve model efficiency.
        """)

        st.markdown("---")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Features", data['redundancy']['total_features'])

        with col2:
            st.metric("Redundant Features", data['redundancy']['redundant_features_count'])

        with col3:
            st.metric("Recommended Features", data['redundancy']['recommended_feature_count'])

        with col4:
            reduction_pct = (data['redundancy']['redundant_features_count'] /
                           data['redundancy']['total_features'] * 100)
            st.metric("Reduction", f"{reduction_pct:.1f}%")

        st.markdown("---")

        # Correlation heatmap
        st.markdown("#### Correlation Heatmap (Top 50 Features)")

        try:
            img = Image.open('../results/phase8_results/feature_correlation/correlation_heatmap_top50.png')
            st.image(img, width='stretch')
        except:
            st.warning("Correlation heatmap not found")

        st.markdown("---")

        # Highly correlated pairs
        st.markdown("#### Highly Correlated Feature Pairs (|r| > 0.9)")

        try:
            img = Image.open('../results/phase8_results/feature_correlation/highly_correlated_pairs.png')
            st.image(img, width='stretch')
        except:
            st.warning("Highly correlated pairs visualization not found")

        st.markdown("---")

        # Redundant features list
        st.markdown("#### Redundant Features to Remove")

        redundant_features = data['redundancy']['redundant_features']

        # Display in columns
        n_cols = 3
        cols = st.columns(n_cols)

        for idx, feat in enumerate(redundant_features):
            with cols[idx % n_cols]:
                st.markdown(f"- `{feat}`")

        st.markdown("---")

        # Feature importance distribution
        st.markdown("#### Feature Importance Distribution")

        try:
            img = Image.open('../results/phase8_results/feature_correlation/feature_importance_distribution.png')
            st.image(img, width='stretch')
        except:
            st.warning("Feature importance distribution not found")

        st.markdown("---")

        # Recommendations
        st.markdown("#### Recommendations")

        st.success(f"""
        **Feature Reduction Strategy:**

        1. **Remove {data['redundancy']['redundant_features_count']} redundant features**
           (highly correlated with more important features)

        2. **Retain {data['redundancy']['recommended_feature_count']} features**
           for optimal model performance

        3. **Benefits:**
           - Faster training and inference
           - Reduced overfitting risk
           - Improved model interpretability
           - Lower computational cost

        4. **Next Steps:**
           - Retrain models with reduced feature set
           - Compare performance with full feature set
           - Validate on 2024 test set
        """)

