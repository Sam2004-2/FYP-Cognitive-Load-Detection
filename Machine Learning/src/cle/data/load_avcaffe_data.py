"""
AVCAffe dataset adapter for regression training.

Adapts AVCAffe feature schema to be compatible with the existing
train_regression.py pipeline which expects StressID-style column names.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def prepare_avcaffe_for_regression(
    features_path: str,
    feature_names: List[str],
    quality_filters: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Load AVCAffe features and adapt schema for regression training.

    The main branch's train_regression.py expects:
    - 'user_id' column (AVCAffe has 'participant_id')
    - 'load_0_1' label column (AVCAffe has 'cognitive_load')
    - Optional 'task' column for task filtering

    This function adapts the AVCAffe schema to match these expectations.

    Args:
        features_path: Path to avcaffe_labeled_features.csv
        feature_names: List of feature column names to use
        quality_filters: Optional dict of {column: min_value} for filtering
                        e.g., {'min_valid_frame_ratio': 0.80}

    Returns:
        DataFrame with adapted schema compatible with train_regression.py

    Raises:
        FileNotFoundError: If features file doesn't exist
        ValueError: If required columns are missing
    """
    features_file = Path(features_path)
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    logger.info(f"Loading AVCAffe features from: {features_path}")
    df = pd.read_csv(features_path)

    original_count = len(df)
    logger.info(f"Loaded {original_count:,} feature windows")

    # Check for required columns
    required_cols = ["participant_id", "cognitive_load"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Schema adaptation for compatibility with train_regression.py
    logger.info("Adapting schema for regression training...")

    # Rename columns to match expected schema
    df = df.rename(columns={
        "participant_id": "user_id",
        "cognitive_load": "load_0_1",
    })

    # Ensure task column exists (required for grouping)
    if "task" not in df.columns:
        raise ValueError("Missing 'task' column in features")

    # Apply quality filters if provided
    if quality_filters:
        logger.info("Applying quality filters:")
        for col, min_val in quality_filters.items():
            if col in df.columns:
                before = len(df)
                df = df[df[col] >= min_val]
                after = len(df)
                retained_pct = (after / before) * 100 if before > 0 else 0
                logger.info(f"  {col} >= {min_val}: {after:,}/{before:,} "
                           f"windows retained ({retained_pct:.1f}%)")
            else:
                logger.warning(f"  Quality filter column '{col}' not found, skipping")

    # Validate that we still have data
    if len(df) == 0:
        raise ValueError("No data remaining after quality filtering")

    # Validate load_0_1 range
    if not ((df["load_0_1"] >= 0) & (df["load_0_1"] <= 1)).all():
        invalid_count = ((df["load_0_1"] < 0) | (df["load_0_1"] > 1)).sum()
        logger.warning(f"Found {invalid_count} windows with load_0_1 outside [0,1] range")
        df = df[(df["load_0_1"] >= 0) & (df["load_0_1"] <= 1)]
        logger.info(f"Filtered to {len(df):,} windows with valid labels")

    # Check for missing labels
    missing_labels = df["load_0_1"].isna().sum()
    if missing_labels > 0:
        logger.warning(f"Found {missing_labels} windows without labels")
        df = df.dropna(subset=["load_0_1"])
        logger.info(f"Dropped unlabeled windows, {len(df):,} remaining")

    # Validate feature columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Log final statistics
    final_count = len(df)
    retained_pct = (final_count / original_count) * 100
    logger.info(f"Final dataset: {final_count:,}/{original_count:,} "
               f"windows ({retained_pct:.1f}% retained)")
    logger.info(f"Unique participants: {df['user_id'].nunique()}")
    logger.info(f"Unique tasks: {df['task'].nunique()}")
    logger.info(f"Label range: [{df['load_0_1'].min():.3f}, {df['load_0_1'].max():.3f}]")
    logger.info(f"Label mean: {df['load_0_1'].mean():.3f} ± {df['load_0_1'].std():.3f}")

    return df


def load_avcaffe_features(
    features_path: str,
    config: dict,
) -> pd.DataFrame:
    """
    High-level loader for AVCAffe features with config-based setup.

    Args:
        features_path: Path to avcaffe_labeled_features.csv
        config: Configuration dict with feature names and quality filters

    Returns:
        DataFrame ready for regression training
    """
    # Extract feature names from config
    from src.cle.extract.features import get_feature_names
    feature_names = get_feature_names(config)

    # Extract quality filters from config
    quality_filters = {}
    if config.get("quality.min_valid_frame_ratio"):
        quality_filters["valid_frame_ratio"] = config.get("quality.min_valid_frame_ratio")
    if config.get("quality.min_mean_quality"):
        quality_filters["mean_quality"] = config.get("quality.min_mean_quality")

    # Load and prepare data
    df = prepare_avcaffe_for_regression(
        features_path=features_path,
        feature_names=feature_names,
        quality_filters=quality_filters if quality_filters else None,
    )

    return df


if __name__ == "__main__":
    # Test the data adapter
    import sys
    from src.cle.logging_setup import setup_logging

    setup_logging(level="INFO")

    features_path = "data/processed/avcaffe_labeled_features.csv"
    if len(sys.argv) > 1:
        features_path = sys.argv[1]

    feature_names = [
        "blink_count",
        "blink_rate",
        "ear_std",
        "mean_blink_duration",
        "mean_brightness",
        "mean_quality",
        "perclos",
        "std_brightness",
        "valid_frame_ratio",
    ]

    quality_filters = {
        "valid_frame_ratio": 0.80,
        "mean_quality": 0.85,
    }

    df = prepare_avcaffe_for_regression(
        features_path=features_path,
        feature_names=feature_names,
        quality_filters=quality_filters,
    )

    print(f"\nAdapted DataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLabel statistics:")
    print(df["load_0_1"].describe())
