"""
Visualization Module for Hospital Readmission Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReadmissionVisualizer:
    """Handles visualizations for the hospital readmission project."""
    
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        """Initialize the visualizer."""
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_target_distribution(self, df, target_col='readmitted'):
        """
        Plot target variable distribution.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Name of target column
            
        Returns:
            go.Figure: Plotly figure
        """
        counts = df[target_col].value_counts()
        
        fig = make_subplots(rows=1, cols=2,
                           specs=[[{"type": "pie"}, {"type": "bar"}]],
                           subplot_titles=['Distribution', 'Counts'])
        
        fig.add_trace(go.Pie(labels=counts.index, values=counts.values,
                           hole=0.4, textinfo='label+percent'), row=1, col=1)
        
        fig.add_trace(go.Bar(x=counts.index, y=counts.values,
                           text=counts.values, textposition='outside'), row=1, col=2)
        
        fig.update_layout(title_text='Readmission Distribution', showlegend=False)
        return fig
    
    def plot_age_readmission(self, df):
        """
        Plot readmission rates by age group.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            go.Figure: Plotly figure
        """
        age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        
        age_readmission = pd.crosstab(df['age'], df['readmitted'], normalize='index') * 100
        age_readmission = age_readmission.reindex(age_order)
        
        fig = px.bar(age_readmission, x=age_readmission.index, y=['<30', '>30', 'NO'],
                    title='Readmission Rate by Age Group', barmode='group',
                    labels={'value': 'Percentage (%)'}, color_discrete_sequence=px.colors.qualitative.Set2)
        
        return fig
    
    def plot_correlation_heatmap(self, df, numerical_cols):
        """
        Plot correlation heatmap.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
            
        Returns:
            go.Figure: Plotly figure
        """
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap')
        
        return fig
    
    def plot_feature_importance(self, feature_importance, top_n=20):
        """
        Plot feature importance.
        
        Args:
            feature_importance (pd.DataFrame): DataFrame with feature and importance columns
            top_n (int): Number of top features to show
            
        Returns:
            go.Figure: Plotly figure
        """
        top_features = feature_importance.head(top_n)
        
        fig = px.bar(top_features, x='importance', y='feature',
                    orientation='h', title=f'Top {top_n} Feature Importances',
                    color='importance', color_continuous_scale='Viridis')
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels (list): Label names
            
        Returns:
            go.Figure: Plotly figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'),
                       x=labels, y=labels, color_continuous_scale='Blues',
                       title='Confusion Matrix')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            go.Figure: Plotly figure
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'ROC (AUC = {auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                name='Random', line=dict(dash='dash')))
        
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate',
                         xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        
        return fig
    
    def plot_model_comparison(self, results_df):
        """
        Plot model comparison.
        
        Args:
            results_df (pd.DataFrame): DataFrame with model results
            
        Returns:
            go.Figure: Plotly figure
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(name=metric, x=results_df['Model'], y=results_df[metric]))
        
        fig.update_layout(title='Model Performance Comparison', barmode='group',
                         yaxis=dict(range=[0, 1]))
        
        return fig
    
    def plot_risk_factors(self, df, feature_col, target_col='readmitted'):
        """
        Plot readmission rate by a specific feature.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_col (str): Feature column name
            target_col (str): Target column name
            
        Returns:
            go.Figure: Plotly figure
        """
        feature_readmission = pd.crosstab(df[feature_col], df[target_col], 
                                         normalize='index') * 100
        
        fig = px.bar(feature_readmission, x=feature_readmission.index, y=['<30', '>30', 'NO'],
                    title=f'Readmission Rate by {feature_col}', barmode='group')
        
        return fig


def create_dashboard_summary(df, model_metrics, feature_importance):
    """
    Create a summary dashboard with key visualizations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        model_metrics (dict): Model performance metrics
        feature_importance (pd.DataFrame): Feature importance DataFrame
        
    Returns:
        go.Figure: Combined dashboard figure
    """
    visualizer = ReadmissionVisualizer()
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=['Target Distribution', 'Feature Importance (Top 10)',
                                      'Model Performance', 'Key Metrics'],
                       specs=[[{"type": "pie"}, {"type": "bar"}],
                             [{"type": "bar", "colspan": 2}, None]])
    
    counts = df['readmitted'].value_counts()
    fig.add_trace(go.Pie(labels=counts.index, values=counts.values, hole=0.4), 
                  row=1, col=1)
    
    top_10 = feature_importance.head(10)
    fig.add_trace(go.Bar(x=top_10['importance'], y=top_10['feature'], 
                        orientation='h', marker_color='steelblue'), row=1, col=2)
    
    metrics_df = pd.DataFrame({'Metric': list(model_metrics.keys()), 
                              'Value': list(model_metrics.values())})
    fig.add_trace(go.Bar(x=metrics_df['Metric'], y=metrics_df['Value'],
                        marker_color='coral'), row=2, col=1)
    
    fig.update_layout(height=700, title_text='Hospital Readmission Prediction - Summary Dashboard',
                     showlegend=False)
    
    return fig
