"""
Shared metrics computation for classification.

Provides unified metrics calculation across training and evaluation pipelines.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_proba: Predicted probabilities for positive class (optional, for AUC)

    Returns:
        Dictionary with metrics:
            - accuracy: Overall accuracy
            - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
            - precision: Precision for positive class
            - recall: Recall for positive class
            - f1: F1 score for positive class
            - confusion_matrix: 2x2 confusion matrix as list
            - n_samples: Total number of samples
            - n_positive: Number of positive samples (if applicable)
            - n_negative: Number of negative samples (if applicable)
            - auc: ROC AUC score (only if y_proba provided)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }

    if y_proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            # AUC undefined if only one class present
            metrics["auc"] = None

    return metrics


def format_confusion_matrix(cm: list, labels: tuple = ("LOW", "HIGH")) -> str:
    """
    Format confusion matrix as a readable string.

    Args:
        cm: 2x2 confusion matrix as list
        labels: Class label names

    Returns:
        Formatted string representation
    """
    return (
        f"          Pred: {labels[0]:>5}  {labels[1]:>5}\n"
        f"  True {labels[0]}:  {cm[0][0]:5d}  {cm[0][1]:5d}\n"
        f"  True {labels[1]}: {cm[1][0]:5d}  {cm[1][1]:5d}"
    )
