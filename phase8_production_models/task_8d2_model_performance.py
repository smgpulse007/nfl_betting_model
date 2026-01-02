"""
Task 8D.2: Model Performance Visualizations

Display comprehensive model performance metrics, calibration, and cross-validation results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def show_model_performance(data, models):
    """Display model performance page"""
    
    st.markdown('<div class="main-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Comprehensive Metrics",
        "🎯 Calibration Analysis", 
        "🔄 Cross-Validation",
        "📉 Learning Curves"
    ])
    
    # Tab 1: Comprehensive Metrics
    with tab1:
        st.markdown("### Comprehensive Metrics Comparison (2024 Test Set)")
        
        # Metrics table
        metrics_data = []
        for model_name, metrics in data['metrics'].items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display table with formatting
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'ROC-AUC': '{:.4f}',
                'PR-AUC': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score', 'ROC-AUC']),
            width='stretch',
            hide_index=True
        )
        
        st.markdown("---")
        
        # Metric selection for visualization
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_metric = st.selectbox(
                "Select Metric to Visualize:",
                ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
            )
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            metric_col = selected_metric.lower().replace('-', '_')
            values = [m[metric_col] for m in data['metrics'].values()]
            model_names = list(data['metrics'].keys())
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            bars = ax.barh(model_names, values, color=colors[:len(model_names)], alpha=0.7)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
            
            ax.set_xlabel(selected_metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{selected_metric} Comparison Across Models', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Confusion matrices
        st.markdown("### Confusion Matrices")
        
        # Load confusion matrix image
        try:
            img = Image.open('../results/phase8_results/visualizations/confusion_matrices.png')
            st.image(img, width='stretch')
        except:
            st.warning("Confusion matrix visualization not found")
        
        st.markdown("---")
        
        # ROC and PR curves
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ROC Curves")
            try:
                img = Image.open('../results/phase8_results/visualizations/roc_curves.png')
                st.image(img, width='stretch')
            except:
                st.warning("ROC curves visualization not found")
        
        with col2:
            st.markdown("### Precision-Recall Curves")
            try:
                img = Image.open('../results/phase8_results/visualizations/precision_recall_curves.png')
                st.image(img, width='stretch')
            except:
                st.warning("PR curves visualization not found")
    
    # Tab 2: Calibration Analysis
    with tab2:
        st.markdown("### Calibration Analysis")
        
        # Brier scores
        st.markdown("#### Brier Scores (Lower is Better)")
        
        brier_data = []
        for model_name, cal in data['calibration'].items():
            brier_data.append({
                'Model': model_name,
                'Brier Score': cal['brier_score'],
                'ECE': cal['ece']
            })
        
        brier_df = pd.DataFrame(brier_data).sort_values('Brier Score')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                brier_df.style.format({
                    'Brier Score': '{:.4f}',
                    'ECE': '{:.4f}'
                }).background_gradient(cmap='RdYlGn_r', subset=['Brier Score', 'ECE']),
                width='stretch',
                hide_index=True
            )
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            bars = ax.barh(brier_df['Model'], brier_df['Brier Score'], 
                          color=colors[:len(brier_df)], alpha=0.7)
            
            for i, (bar, val) in enumerate(zip(bars, brier_df['Brier Score'])):
                ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
            
            ax.set_xlabel('Brier Score', fontsize=12, fontweight='bold')
            ax.set_title('Brier Score Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")

        # Calibration curves
        st.markdown("#### Calibration Curves")
        try:
            img = Image.open('../results/phase8_results/visualizations/calibration_curves.png')
            st.image(img, width='stretch')
        except:
            st.warning("Calibration curves visualization not found")

        st.markdown("---")

        # Confidence level analysis
        st.markdown("#### Confidence Level Analysis")

        # Model selector
        selected_model = st.selectbox(
            "Select Model:",
            list(data['calibration'].keys()),
            key='cal_model_select'
        )

        # Check if confidence analysis exists in the data
        if 'confidence_analysis' in data and selected_model in data['confidence_analysis']:
            conf_analysis = data['confidence_analysis'][selected_model]
        elif 'confidence_analysis' in data['calibration'][selected_model]:
            conf_analysis = data['calibration'][selected_model]['confidence_analysis']
        else:
            conf_analysis = None

        if conf_analysis:

            conf_data = []
            for conf_level, stats in conf_analysis.items():
                conf_data.append({
                    'Confidence Level': conf_level,
                    'Games': stats['n_games'],
                    'Accuracy': stats['accuracy'],
                    '% of Total': stats['pct_of_total']
                })

            conf_df = pd.DataFrame(conf_data)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(
                    conf_df.style.format({
                        'Accuracy': '{:.2%}',
                        '% of Total': '{:.2%}'
                    }).background_gradient(cmap='RdYlGn', subset=['Accuracy']),
                    width='stretch',
                    hide_index=True
                )

            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))

                x = range(len(conf_df))
                ax.plot(x, conf_df['Accuracy'], marker='o', linewidth=2, markersize=8,
                       color='#1f77b4', label='Accuracy')

                ax.set_xticks(x)
                ax.set_xticklabels(conf_df['Confidence Level'])
                ax.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
                ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax.set_title(f'{selected_model} - Accuracy by Confidence Level',
                           fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.legend()

                st.pyplot(fig)
                plt.close()

    # Tab 3: Cross-Validation
    with tab3:
        st.markdown("### Cross-Validation Results (5-Fold Time Series CV)")

        # CV results table
        cv_data = []
        for model_name, cv_result in data['cv_results'].items():
            cv_data.append({
                'Model': model_name,
                'Mean Accuracy': cv_result['mean_accuracy'],
                'Std Dev': cv_result['std_accuracy'],
                'Min': cv_result['min_accuracy'],
                'Max': cv_result['max_accuracy']
            })

        cv_df = pd.DataFrame(cv_data).sort_values('Mean Accuracy', ascending=False)

        st.dataframe(
            cv_df.style.format({
                'Mean Accuracy': '{:.2%}',
                'Std Dev': '{:.2%}',
                'Min': '{:.2%}',
                'Max': '{:.2%}'
            }).background_gradient(cmap='RdYlGn', subset=['Mean Accuracy']),
            width='stretch',
            hide_index=True
        )

        st.markdown("---")

        # CV visualization
        try:
            img = Image.open('../results/phase8_results/visualizations/cross_validation_results.png')
            st.image(img, width='stretch')
        except:
            st.warning("Cross-validation visualization not found")

        st.markdown("---")

        # Insights
        st.markdown("#### Key Insights")

        best_model = cv_df.iloc[0]['Model']
        best_acc = cv_df.iloc[0]['Mean Accuracy']
        most_stable = cv_df.sort_values('Std Dev').iloc[0]['Model']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best CV Accuracy", f"{best_acc:.2%}", best_model)

        with col2:
            st.metric("Most Stable", most_stable,
                     f"σ = {cv_df[cv_df['Model']==most_stable]['Std Dev'].values[0]:.2%}")

        with col3:
            avg_cv = cv_df['Mean Accuracy'].mean()
            st.metric("Average CV Accuracy", f"{avg_cv:.2%}", "All Models")

    # Tab 4: Learning Curves
    with tab4:
        st.markdown("### Learning Curves")

        st.markdown("""
        Learning curves show how model performance changes with training set size.
        They help identify:
        - **Overfitting**: Large gap between train and validation curves
        - **Underfitting**: Both curves plateau at low performance
        - **Optimal training size**: Point of diminishing returns
        """)

        st.markdown("---")

        try:
            img = Image.open('../results/phase8_results/visualizations/learning_curves.png')
            st.image(img, width='stretch')
        except:
            st.warning("Learning curves visualization not found")

        st.markdown("---")

        # Insights
        st.markdown("#### Observations")

        st.markdown("""
        - **No severe overfitting detected**: Train and validation curves are close
        - **All models benefit from full training set**: Performance improves with more data
        - **Convergence**: Models reach stable performance with 80-100% of training data
        - **Recommendation**: Use full training set (1999-2023) for final models
        """)

