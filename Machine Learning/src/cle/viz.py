"""
Visualization utilities for CLE.

Functions for plotting CLI timelines, feature distributions, ROC curves, etc.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def plot_cli_timeline(
    df: pd.DataFrame,
    save_path: str,
    cli_col: str = "cli",
    conf_col: str = "confidence",
    time_col: str = "t_start_s",
    label_col: Optional[str] = "label",
) -> None:
    """
    Plot CLI timeline with confidence bands.

    Args:
        df: DataFrame with CLI predictions
        save_path: Path to save plot
        cli_col: Column name for CLI values
        conf_col: Column name for confidence values
        time_col: Column name for time
        label_col: Optional column name for ground truth labels
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot CLI
    ax1 = axes[0]
    time = df[time_col].values
    cli = df[cli_col].values
    conf = df[conf_col].values if conf_col in df.columns else np.ones_like(cli)

    ax1.plot(time, cli, 'b-', linewidth=2, label='CLI')
    ax1.fill_between(
        time,
        np.clip(cli - (1 - conf) * 0.5, 0, 1),
        np.clip(cli + (1 - conf) * 0.5, 0, 1),
        alpha=0.3,
        label='Confidence band'
    )

    # Add ground truth if available
    if label_col and label_col in df.columns:
        label_map = {"low": 0.2, "high": 0.8, "none": 0.5}
        gt_values = df[label_col].map(lambda x: label_map.get(x, 0.5) if isinstance(x, str) else x)
        ax1.scatter(time, gt_values, c='red', s=20, alpha=0.5, label='Ground truth', zorder=5)

    ax1.set_ylabel('CLI', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cognitive Load Index Timeline', fontsize=14, fontweight='bold')

    # Plot confidence
    ax2 = axes[1]
    ax2.plot(time, conf, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Prediction Confidence', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved CLI timeline plot to {save_path}")


def plot_feature_distributions(
    df: pd.DataFrame,
    save_path: str,
    feature_names: list,
    label_col: str = "label",
    max_features: int = 12,
) -> None:
    """
    Plot feature distributions by label.

    Args:
        df: Features DataFrame
        save_path: Path to save plot
        feature_names: List of feature names to plot
        label_col: Column name for labels
        max_features: Maximum number of features to plot
    """
    # Limit number of features
    plot_features = feature_names[:max_features]

    # Calculate grid size
    n_features = len(plot_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    # Get unique labels
    labels = df[label_col].unique()
    colors = ['blue', 'red', 'green', 'orange']

    for i, feature in enumerate(plot_features):
        ax = axes[i]

        if feature not in df.columns:
            ax.text(0.5, 0.5, f'{feature}\nNot found', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        for j, label in enumerate(labels):
            data = df[df[label_col] == label][feature].values
            # Remove NaN and inf
            data = data[np.isfinite(data)]

            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, color=colors[j % len(colors)], label=str(label), density=True)

        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature Distributions by Label', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature distributions plot to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
) -> None:
    """
    Plot ROC curve with AUC.

    Args:
        y_true: True binary labels
        y_score: Predicted probabilities
        save_path: Path to save plot
        title: Plot title
    """
    from sklearn.metrics import auc

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curve plot to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    labels: list = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        save_path: Path to save plot
        labels: Class labels
        title: Plot title
    """
    if labels is None:
        labels = ['Low', 'High']

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted',
        ylabel='True',
        title=title
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold'
            )

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved confusion matrix plot to {save_path}")

