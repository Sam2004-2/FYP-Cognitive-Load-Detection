"""
Advanced feature engineering for cognitive load classification.

Implements derived features, interaction terms, and transformations
to improve model performance beyond raw eye-tracking metrics.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from base eye-tracking metrics.

    Adds interaction terms, ratios, and non-linear transformations
    that may capture cognitive load patterns better than raw features.

    Args:
        df: DataFrame with base features (blink_rate, blink_count, etc.)

    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()

    # Blink-related derived features
    if "blink_rate" in df.columns and "mean_blink_duration" in df.columns:
        # Total blink time ratio (time spent blinking per minute)
        df["blink_time_ratio"] = (
            df["blink_rate"] * df["mean_blink_duration"] / 60000
        ).fillna(0)

    if "blink_rate" in df.columns:
        # Log-transformed blink rate (handles skewed distribution)
        df["log_blink_rate"] = np.log1p(df["blink_rate"])

        # Blink rate squared (captures non-linear effects)
        df["blink_rate_sq"] = df["blink_rate"] ** 2

    # EAR variability features
    if "ear_std" in df.columns:
        # Log-transformed EAR std
        df["log_ear_std"] = np.log1p(df["ear_std"])

        # EAR stability (inverse of variability, capped)
        df["ear_stability"] = 1 / (df["ear_std"] + 0.01)
        df["ear_stability"] = df["ear_stability"].clip(upper=100)

    # PERCLOS derived features
    if "perclos" in df.columns:
        # Logit transform of PERCLOS (more linear relationship)
        # Clip to avoid log(0) or log(1)
        perclos_clipped = df["perclos"].clip(0.001, 0.999)
        df["perclos_logit"] = np.log(perclos_clipped / (1 - perclos_clipped))

        # PERCLOS squared (emphasize extreme values)
        df["perclos_sq"] = df["perclos"] ** 2

    # Brightness features
    if "mean_brightness" in df.columns and "std_brightness" in df.columns:
        # Coefficient of variation (normalized variability)
        df["brightness_cv"] = (
            df["std_brightness"] / (df["mean_brightness"] + 1)
        ).fillna(0)

        # Brightness stability
        df["brightness_stability"] = 1 / (df["std_brightness"] + 1)

    # Interaction terms between key features
    if "blink_rate" in df.columns and "perclos" in df.columns:
        # Blink-PERCLOS interaction (both increase with fatigue/load)
        df["blink_perclos_interaction"] = df["blink_rate"] * df["perclos"]

    if "blink_rate" in df.columns and "ear_std" in df.columns:
        # Blink rate × eye variability interaction
        df["blink_ear_interaction"] = df["blink_rate"] * df["ear_std"]

    if "perclos" in df.columns and "ear_std" in df.columns:
        # PERCLOS × EAR variability interaction
        df["perclos_ear_interaction"] = df["perclos"] * df["ear_std"]

    # Quality-adjusted features
    if "mean_quality" in df.columns and "blink_rate" in df.columns:
        # Quality-weighted blink rate
        df["quality_weighted_blink"] = df["blink_rate"] * df["mean_quality"]

    if "valid_frame_ratio" in df.columns and "perclos" in df.columns:
        # Reliability-weighted PERCLOS
        df["reliable_perclos"] = df["perclos"] * df["valid_frame_ratio"]

    # Composite alertness/fatigue indicators
    if all(col in df.columns for col in ["blink_rate", "perclos", "ear_std"]):
        # Fatigue index: combines multiple fatigue indicators
        # Higher values = more fatigue = potentially higher cognitive load
        df["fatigue_index"] = (
            0.4 * _normalize_series(df["blink_rate"]) +
            0.4 * _normalize_series(df["perclos"]) +
            0.2 * _normalize_series(df["ear_std"])
        )

    # Blink regularity (low std in blink duration = more regular)
    if "mean_blink_duration" in df.columns and "blink_count" in df.columns:
        # Approximate regularity from available features
        df["blink_regularity"] = df["blink_count"] / (df["mean_blink_duration"] + 1)

    logger.info(f"Added {len(df.columns) - 9} derived features")

    return df


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    min_val = s.min()
    max_val = s.max()
    if max_val - min_val == 0:
        return pd.Series(0.5, index=s.index)
    return (s - min_val) / (max_val - min_val)


def compute_z_scores(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Compute z-score normalized features.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to z-score normalize

    Returns:
        DataFrame with additional z-score columns
    """
    df = df.copy()

    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_zscore"] = (df[col] - mean) / std
            else:
                df[f"{col}_zscore"] = 0

    return df


def compute_rank_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Compute rank-transformed features (robust to outliers).

    Args:
        df: DataFrame with features
        feature_cols: List of columns to rank-transform

    Returns:
        DataFrame with additional rank columns
    """
    df = df.copy()

    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(pct=True)

    return df


def compute_user_relative_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    user_col: str = "user_id"
) -> pd.DataFrame:
    """
    Compute user-relative features (deviation from personal baseline).

    This helps account for individual differences in eye metrics.

    Args:
        df: DataFrame with features and user_id
        feature_cols: List of columns to compute user-relative values
        user_col: Column name for user identifier

    Returns:
        DataFrame with additional user-relative columns
    """
    df = df.copy()

    if user_col not in df.columns:
        logger.warning(f"User column '{user_col}' not found, skipping user-relative features")
        return df

    for col in feature_cols:
        if col in df.columns:
            # User mean
            user_means = df.groupby(user_col)[col].transform("mean")
            user_stds = df.groupby(user_col)[col].transform("std")

            # Deviation from user's mean
            df[f"{col}_user_dev"] = df[col] - user_means

            # User-normalized (z-score within user)
            df[f"{col}_user_zscore"] = np.where(
                user_stds > 0,
                (df[col] - user_means) / user_stds,
                0
            )

    logger.info(f"Added user-relative features for {len(feature_cols)} columns")

    return df


def compute_task_relative_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_col: str = "task"
) -> pd.DataFrame:
    """
    Compute task-relative features (deviation from task baseline).

    Different tasks may have different baseline eye metrics.

    Args:
        df: DataFrame with features and task column
        feature_cols: List of columns to compute task-relative values
        task_col: Column name for task identifier

    Returns:
        DataFrame with additional task-relative columns
    """
    df = df.copy()

    if task_col not in df.columns:
        logger.warning(f"Task column '{task_col}' not found, skipping task-relative features")
        return df

    for col in feature_cols:
        if col in df.columns:
            # Task mean
            task_means = df.groupby(task_col)[col].transform("mean")

            # Deviation from task's mean
            df[f"{col}_task_dev"] = df[col] - task_means

    logger.info(f"Added task-relative features for {len(feature_cols)} columns")

    return df


def select_features_by_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "mutual_info",
    top_k: Optional[int] = None,
    threshold: float = 0.01
) -> List[str]:
    """
    Select features based on importance scores.

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        method: "mutual_info" or "f_classif"
        top_k: Select top k features (if None, use threshold)
        threshold: Minimum importance score threshold

    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif

    if method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == "f_classif":
        scores, _ = f_classif(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle NaN scores
    scores = np.nan_to_num(scores, nan=0)

    # Create feature importance ranking
    feature_importance = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )

    logger.info("Feature importance ranking:")
    for name, score in feature_importance[:10]:
        logger.info(f"  {name}: {score:.4f}")

    if top_k is not None:
        selected = [name for name, _ in feature_importance[:top_k]]
    else:
        selected = [name for name, score in feature_importance if score >= threshold]

    logger.info(f"Selected {len(selected)} features")

    return selected


def get_all_feature_names(base_features: List[str], include_derived: bool = True) -> List[str]:
    """
    Get complete list of feature names including derived features.

    Args:
        base_features: List of base feature names
        include_derived: Whether to include derived features

    Returns:
        Complete list of feature names
    """
    all_features = list(base_features)

    if not include_derived:
        return all_features

    # Derived features mapping (base_feature -> derived_names)
    derived_mapping = {
        "blink_rate": ["log_blink_rate", "blink_rate_sq"],
        "ear_std": ["log_ear_std", "ear_stability"],
        "perclos": ["perclos_logit", "perclos_sq"],
    }

    # Add derived features
    for base in base_features:
        if base in derived_mapping:
            all_features.extend(derived_mapping[base])

    # Interaction terms (if both base features present)
    if "blink_rate" in base_features and "mean_blink_duration" in base_features:
        all_features.append("blink_time_ratio")

    if "blink_rate" in base_features and "perclos" in base_features:
        all_features.append("blink_perclos_interaction")

    if "blink_rate" in base_features and "ear_std" in base_features:
        all_features.append("blink_ear_interaction")

    if "perclos" in base_features and "ear_std" in base_features:
        all_features.append("perclos_ear_interaction")

    if "mean_brightness" in base_features and "std_brightness" in base_features:
        all_features.extend(["brightness_cv", "brightness_stability"])

    if "mean_quality" in base_features and "blink_rate" in base_features:
        all_features.append("quality_weighted_blink")

    if "valid_frame_ratio" in base_features and "perclos" in base_features:
        all_features.append("reliable_perclos")

    # Composite features
    if all(f in base_features for f in ["blink_rate", "perclos", "ear_std"]):
        all_features.append("fatigue_index")

    if "mean_blink_duration" in base_features and "blink_count" in base_features:
        all_features.append("blink_regularity")

    return all_features
