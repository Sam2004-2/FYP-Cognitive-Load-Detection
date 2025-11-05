"""
Model calibration utilities.

Implements probability calibration using Platt scaling (sigmoid) or isotonic regression.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def calibrate_classifier(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Calibrate classifier using Platt scaling or isotonic regression.

    Args:
        estimator: Base estimator (must implement predict_proba)
        X_train: Training features
        y_train: Training labels
        method: Calibration method ("sigmoid" for Platt, "isotonic" for isotonic regression)
        cv: Number of cross-validation folds

    Returns:
        Calibrated classifier
    """
    logger.info(f"Calibrating classifier using {method} method (cv={cv})")

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
    )

    calibrated.fit(X_train, y_train)

    logger.info("Calibration complete")
    return calibrated


def compute_user_baseline_stats(
    features_df,
    user_id: str,
    feature_names: list,
    role: str = "calibration",
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Compute per-user baseline statistics from calibration videos.

    Args:
        features_df: Features DataFrame
        user_id: User ID
        feature_names: List of feature names
        role: Role to filter by (default: "calibration")

    Returns:
        Dictionary with mean and std for each feature, or None if no data
    """
    # Filter calibration data for this user
    user_calib = features_df[
        (features_df["user_id"] == user_id) & (features_df["role"] == role)
    ]

    if len(user_calib) == 0:
        logger.warning(f"No calibration data found for user {user_id}")
        return None

    stats = {}
    for feature in feature_names:
        if feature in user_calib.columns:
            values = user_calib[feature].values
            stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    logger.info(f"Computed baseline stats for user {user_id} from {len(user_calib)} windows")
    return stats

