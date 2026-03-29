"""
evaluate.py
Evaluation utilities — metrics, plots, and prediction helpers.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
    accuracy_score,
)


def get_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a dict of evaluation metrics."""
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'f1_score':  round(f1_score(y_true, y_pred), 4),
        'roc_auc':   round(roc_auc_score(y_true, y_prob), 4),
        'recall_1':  round(float(pd.Series(y_true).eq(1).sum() / len(y_true)), 4),
    }


def plot_roc_curve(results: dict) -> go.Figure:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    results : dict  {model_name: {'y_test': ..., 'y_prob': ...}}
    """
    fig = go.Figure()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        auc_val = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_val:.3f})',
            line=dict(color=color, width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline',
        line=dict(color='gray', width=1.5, dash='dash'),
    ))
    fig.update_layout(
        title='ROC Curves — All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.05),
        height=500,
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name: str = '') -> go.Figure:
    """Plot a single confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['No Readmit', 'Early Readmit']

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels, y=labels,
        color_continuous_scale='Blues',
        labels=dict(x='Predicted', y='Actual'),
        title=f'Confusion Matrix — {model_name}',
    )
    fig.update_layout(height=400)
    return fig


def plot_feature_importance(feature_cols: list, importances: np.ndarray, top_n: int = 20) -> go.Figure:
    """Bar chart of top N feature importances."""
    feat_df = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=feat_df.values,
        y=feat_df.index,
        orientation='h',
        marker_color='#009688',
    ))
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance Score',
        height=600,
    )
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing models on key metrics."""
    metrics = ['Accuracy', 'F1 Score', 'ROC AUC', 'Recall (class 1)']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                marker_color=color,
            ))

    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison',
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-30,
        height=450,
        legend=dict(orientation='h', y=1.05),
    )
    return fig


def predict_patient(model, scaler, feature_cols: list, patient_dict: dict) -> dict:
    """
    Run inference on a single patient record.

    Parameters
    ----------
    model        : trained sklearn-compatible model
    scaler       : fitted StandardScaler
    feature_cols : list of expected feature column names
    patient_dict : dict of {feature: value}

    Returns
    -------
    dict with 'prediction' (0/1) and 'probability'
    """
    df = pd.DataFrame([patient_dict])
    # Align to expected features, fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    X = scaler.transform(df)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {'prediction': int(pred), 'probability': round(float(prob), 4)}
