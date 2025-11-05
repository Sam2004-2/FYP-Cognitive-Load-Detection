"""
Model training pipeline.

Trains and calibrates cognitive load estimation models.
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.cle.config import load_config
from src.cle.extract.features import get_feature_names
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.train.calibrate import calibrate_classifier
from src.cle.utils.io import (
    load_features_csv,
    load_json,
    save_json,
    save_model_artifact,
)

logger = get_logger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def prepare_data(
    features_df,
    feature_names: list,
    config: dict,
) -> Tuple:
    """
    Prepare data for training.

    Args:
        features_df: Features DataFrame
        feature_names: List of feature names
        config: Configuration dictionary

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_idx, test_idx)
    """
    # Filter out calibration and non-train/test data
    if config.get("eval.split_by_role", True) and "role" in features_df.columns:
        # Split by role column
        train_data = features_df[features_df["role"] == "train"]
        test_data = features_df[features_df["role"] == "test"]

        if len(test_data) == 0:
            logger.warning("No test data found with role='test', using validation split")
            train_data = features_df[features_df["role"].isin(["train", "test"])]
            test_data = None
    else:
        train_data = features_df
        test_data = None

    # Remove calibration role
    if "role" in train_data.columns:
        train_data = train_data[train_data["role"] != "calibration"]

    # Map labels to binary
    label_map = {"low": 0, "high": 1, "none": 0}
    train_data = train_data.copy()
    train_data["label_binary"] = train_data["label"].map(
        lambda x: label_map.get(x, 0) if isinstance(x, str) else x
    )

    # Extract features and labels
    X_train = train_data[feature_names].values
    y_train = train_data["label_binary"].values

    if test_data is not None and len(test_data) > 0:
        test_data = test_data.copy()
        test_data["label_binary"] = test_data["label"].map(
            lambda x: label_map.get(x, 0) if isinstance(x, str) else x
        )
        X_test = test_data[feature_names].values
        y_test = test_data["label_binary"].values
        train_idx = train_data.index
        test_idx = test_data.index
    else:
        # Split using stratified sampling
        stratify = y_train if config.get("eval.stratify", True) else None
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_train,
            y_train,
            train_data.index,
            test_size=config.get("eval.test_size", 0.2),
            stratify=stratify,
            random_state=config.get("seed", 42),
        )

    logger.info(
        f"Data split: train={len(X_train)} (low={np.sum(y_train==0)}, high={np.sum(y_train==1)}), "
        f"test={len(X_test)} (low={np.sum(y_test==0)}, high={np.sum(y_test==1)})"
    )

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def create_model(config: dict):
    """
    Create model based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Scikit-learn estimator
    """
    model_type = config.get("model.type", "logreg")

    if model_type == "logreg":
        params = config.get("model.logreg_params", {})
        model = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            random_state=config.get("seed", 42),
        )
        logger.info(f"Created LogisticRegression model with params: {params}")

    elif model_type == "gbt":
        params = config.get("model.gbt_params", {})
        model = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=config.get("seed", 42),
        )
        logger.info(f"Created GradientBoostingClassifier model with params: {params}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train cognitive load estimation model")
    parser.add_argument(
        "--in",
        dest="input",
        type=str,
        required=True,
        help="Input features CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="train.log")
    logger.info("=" * 80)
    logger.info("Starting model training pipeline")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration (hash: {config.hash()[:8]})")

    # Set random seed
    set_random_seed(config.get("seed", 42))

    # Load features
    features_df = load_features_csv(args.input)

    # Get feature names
    feature_names = get_feature_names(config.to_dict())
    logger.info(f"Using {len(feature_names)} features: {feature_names}")

    # Prepare data
    X_train, X_test, y_train, y_test, train_idx, test_idx = prepare_data(
        features_df, feature_names, config.to_dict()
    )

    # Handle NaN values
    if np.any(np.isnan(X_train)):
        logger.warning("Found NaN values in training data, replacing with zeros")
        X_train = np.nan_to_num(X_train, nan=0.0)
    if np.any(np.isnan(X_test)):
        logger.warning("Found NaN values in test data, replacing with zeros")
        X_test = np.nan_to_num(X_test, nan=0.0)

    # Create and fit scaler
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model
    base_model = create_model(config.to_dict())

    # Train base model
    logger.info("Training base model...")
    base_model.fit(X_train_scaled, y_train)

    # Compute training accuracy
    train_acc = base_model.score(X_train_scaled, y_train)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    # Calibrate model
    calibration_method = config.get("model.calibration", "platt")
    if calibration_method == "platt":
        method = "sigmoid"
    elif calibration_method == "isotonic":
        method = "isotonic"
    else:
        method = "sigmoid"

    calibrated_model = calibrate_classifier(
        base_model, X_train_scaled, y_train, method=method, cv=5
    )

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    logger.info("Saving model artifacts...")

    # Save scaler
    save_model_artifact(scaler, output_dir / "scaler.bin")

    # Save calibrated model
    save_model_artifact(calibrated_model, output_dir / "model.bin")

    # Save feature spec
    feature_spec = {
        "features": feature_names,
        "n_features": len(feature_names),
    }
    save_json(feature_spec, output_dir / "feature_spec.json")

    # Save calibration metadata
    calibration_meta = {
        "training_date": datetime.now().isoformat(),
        "config_hash": config.hash(),
        "model_type": config.get("model.type", "logreg"),
        "calibration_method": calibration_method,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "train_accuracy": train_acc,
        "feature_names": feature_names,
    }
    save_json(calibration_meta, output_dir / "calibration.json")

    logger.info("=" * 80)
    logger.info("Model training complete!")
    logger.info(f"Artifacts saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

