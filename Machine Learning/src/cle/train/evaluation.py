"""
Shared evaluation utilities for regression model assessment.

Provides:
  - compute_spearman: Safe Spearman rank correlation
  - compute_regression_metrics: MAE, RMSE, R², Spearman
  - aggregate_to_session: Window → session aggregation
  - compute_physio_validation: Independent physiological validation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation, returning NaN if fewer than 3 samples."""
    if len(y_true) < 3:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if r is not None else float("nan")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, rmse, r2, spearman_r, spearman_p, n_samples.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) >= 3:
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
    else:
        spearman_r, spearman_p = np.nan, np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
        "n_samples": len(y_true),
    }


def aggregate_to_session(
    predictions_df: pd.DataFrame,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate window-level predictions to session level.

    Args:
        predictions_df: DataFrame with columns user_id, task, y_true, y_pred.
        agg_method: Aggregation method ("mean" or "median").

    Returns:
        Session-level DataFrame with aggregated predictions.
    """
    agg_func = "mean" if agg_method == "mean" else "median"

    session_df = (
        predictions_df.groupby(["user_id", "task"])
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", agg_func),
            n_windows=("y_pred", "count"),
        )
        .reset_index()
    )

    return session_df


def compute_physio_validation(
    predictions_df: pd.DataFrame,
    physio_labels_path: str,
    round_dp: int = 1,
) -> Dict:
    """
    Validate model predictions against physiological stress scores.

    Computes within-subject Spearman correlation between model predictions
    and physiological arousal as an independent validation metric.

    This is the key defensibility metric: the model was trained on
    self-report labels (load_0_1), and we independently check whether
    its predictions also track objective physiological arousal.

    Args:
        predictions_df: DataFrame with columns user_id, task, t_start_s, y_pred.
        physio_labels_path: Path to physio_stress_labels.csv.
        round_dp: Decimal places for rounding t_start_s join key.

    Returns:
        Dictionary with validation metrics.
    """
    physio_path = Path(physio_labels_path)
    if not physio_path.exists():
        logger.warning(f"Physio labels not found: {physio_path}")
        return {"status": "file_not_found", "path": str(physio_path)}

    physio_df = pd.read_csv(physio_path)

    # Ensure consistent types
    preds = predictions_df.copy()
    preds["user_id"] = preds["user_id"].astype(str)
    physio_df["user_id"] = physio_df["user_id"].astype(str)
    if "task" in preds.columns:
        preds["task"] = preds["task"].astype(str)
    physio_df["task"] = physio_df["task"].astype(str)

    # Round for joining
    preds["_t_join"] = preds["t_start_s"].round(round_dp)
    physio_df["_t_join"] = physio_df["t_start_s"].round(round_dp)

    merged = preds.merge(
        physio_df[["user_id", "task", "_t_join", "physio_stress_score"]],
        on=["user_id", "task", "_t_join"],
        how="inner",
    )
    merged = merged.dropna(subset=["physio_stress_score", "y_pred"])

    if len(merged) == 0:
        logger.warning("No matches between predictions and physio labels")
        return {"status": "no_matches", "n_matched": 0}

    logger.info(f"Physio validation: matched {len(merged)} windows")

    # Within-subject Spearman correlations
    per_user: List[Dict] = []
    for user_id, user_df in merged.groupby("user_id"):
        if len(user_df) < 3:
            continue

        # Within-subject z-score the physio score to remove between-subject
        # baseline differences and focus on *within-person* variation.
        physio_vals = user_df["physio_stress_score"].values.astype(float)
        mu = np.nanmean(physio_vals)
        sigma = np.nanstd(physio_vals)
        if sigma < 1e-6:
            continue
        physio_z = (physio_vals - mu) / sigma

        r, p = spearmanr(user_df["y_pred"].values, physio_z)
        if not np.isnan(r):
            per_user.append({
                "user_id": str(user_id),
                "spearman_r": float(r),
                "spearman_p": float(p),
                "n_windows": len(user_df),
            })

    # Overall (pooled) correlation
    overall_r: Optional[float] = None
    overall_p: Optional[float] = None
    if len(merged) >= 3:
        r, p = spearmanr(merged["y_pred"].values, merged["physio_stress_score"].values)
        if not np.isnan(r):
            overall_r = float(r)
            overall_p = float(p)

    mean_within = (
        float(np.mean([u["spearman_r"] for u in per_user]))
        if per_user
        else None
    )
    std_within = (
        float(np.std([u["spearman_r"] for u in per_user]))
        if per_user
        else None
    )

    result = {
        "status": "computed",
        "n_matched_windows": len(merged),
        "n_users_with_correlation": len(per_user),
        "overall_spearman_r": overall_r,
        "overall_spearman_p": overall_p,
        "mean_within_subject_spearman": mean_within,
        "std_within_subject_spearman": std_within,
        "per_user": per_user,
    }

    logger.info(
        f"Physio validation results: "
        f"overall r={overall_r}, "
        f"mean within-subject r={mean_within} +/- {std_within}, "
        f"({len(per_user)} users)"
    )
    return result
