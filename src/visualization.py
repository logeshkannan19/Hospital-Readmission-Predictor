"""
Visualization Module for Hospital Readmission Prediction.

This module provides utilities for creating visualizations and charts.
"""

from typing import Any, Optional
import pandas as pd
import numpy as np


class ReadmissionVisualizer:
    """
    Handles visualizations for the hospital readmission project.
    
    Attributes:
        style: Matplotlib style to use.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid') -> None:
        """Initialize the visualizer."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            self.plt = plt
            self.sns = sns
            self.px = px
            self.go = go
            self.make_subplots = make_subplots
            self.plt.style.use(style)
            self.sns.set_palette("husl")
        except ImportError as e:
            raise ImportError("Required packages not installed: matplotlib, seaborn, plotly") from e
        
    def plot_target_distribution(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'readmitted'
    ) -> Any:
        """
        Plot target variable distribution.
        
        Args:
            df: Input DataFrame.
            target_col: Name of target column.
            
        Returns:
            Plotly figure.
        """
        counts = df[target_col].value_counts()
        
        fig = self.make_subplots(
            rows=1, 
            cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=['Distribution', 'Counts']
        )
        
        fig.add_trace(
            self.go.Pie(
                labels=counts.index, 
                values=counts.values,
                hole=0.4, 
                textinfo='label+percent'
            ), 
            row=1, 
            col=1
        )
        
        fig.add_trace(
            self.go.Bar(
                x=counts.index, 
                y=counts.values,
                text=counts.values, 
                textposition='outside'
            ), 
            row=1, 
            col=2
        )
        
        fig.update_layout(
            title_text='Readmission Distribution', 
            showlegend=False
        )
        return fig
    
    def plot_feature_importance(
        self, 
        feature_importance: pd.DataFrame, 
        top_n: int = 20
    ) -> Any:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature and importance columns.
            top_n: Number of top features to show.
            
        Returns:
            Plotly figure.
        """
        top_features = feature_importance.head(top_n)
        
        fig = self.px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h', 
            title=f'Top {top_n} Feature Importances',
            color='importance', 
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        labels: Optional[list] = None
    ) -> Any:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: Label names.
            
        Returns:
            Plotly figure.
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = self.px.imshow(
            cm, 
            text_auto=True, 
            labels=dict(x='Predicted', y='Actual'),
            x=labels, 
            y=labels, 
            color_continuous_scale='Blues',
            title='Confusion Matrix'
        )
        
        return fig
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Any:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            
        Returns:
            Plotly figure.
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig = self.go.Figure()
        
        fig.add_trace(
            self.go.Scatter(
                x=fpr, 
                y=tpr, 
                mode='lines', 
                name=f'ROC (AUC = {auc_score:.3f})'
            )
        )
        fig.add_trace(
            self.go.Scatter(
                x=[0, 1], 
                y=[0, 1], 
                mode='lines',
                name='Random', 
                line=dict(dash='dash')
            )
        )
        
        fig.update_layout(
            title='ROC Curve', 
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]), 
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame) -> Any:
        """
        Plot model comparison.
        
        Args:
            results_df: DataFrame with model results.
            
        Returns:
            Plotly figure.
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = self.go.Figure()
        
        for metric in metrics:
            fig.add_trace(
                self.go.Bar(
                    name=metric, 
                    x=results_df['Model'], 
                    y=results_df[metric]
                )
            )
        
        fig.update_layout(
            title='Model Performance Comparison', 
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig


def create_dashboard_summary(
    df: pd.DataFrame, 
    model_metrics: dict, 
    feature_importance: pd.DataFrame
) -> Any:
    """
    Create a summary dashboard with key visualizations.
    
    Args:
        df: Input DataFrame.
        model_metrics: Model performance metrics.
        feature_importance: Feature importance DataFrame.
        
    Returns:
        Combined dashboard figure.
    """
    visualizer = ReadmissionVisualizer()
    
    fig = visualizer.make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[
            'Target Distribution', 
            'Feature Importance (Top 10)',
            'Model Performance', 
            'Key Metrics'
        ],
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar", "colspan": 2}, None]
        ]
    )
    
    counts = df['readmitted'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=counts.index, 
            values=counts.values, 
            hole=0.4
        ), 
        row=1, 
        col=1
    )
    
    top_10 = feature_importance.head(10)
    fig.add_trace(
        go.Bar(
            x=top_10['importance'], 
            y=top_10['feature'], 
            orientation='h', 
            marker_color='steelblue'
        ), 
        row=1, 
        col=2
    )
    
    metrics_df = pd.DataFrame({
        'Metric': list(model_metrics.keys()), 
        'Value': list(model_metrics.values())
    })
    fig.add_trace(
        go.Bar(
            x=metrics_df['Metric'], 
            y=metrics_df['Value'],
            marker_color='coral'
        ), 
        row=2, 
        col=1
    )
    
    fig.update_layout(
        height=700, 
        title_text='Hospital Readmission Prediction - Summary Dashboard',
        showlegend=False
    )
    
    return fig
