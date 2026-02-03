"""
Model evaluation pipeline.

Evaluates trained models and generates reports.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.cle.config import load_config
from src.cle.extract.features import get_feature_names
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.train.metrics import compute_classification_metrics
from src.cle.utils.io import (
    load_features_csv,
    load_json,
    load_model_artifact,
    save_json,
)

logger = get_logger(__name__)


def evaluate_model(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained (calibrated) model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        threshold: Decision threshold

    Returns:
        Dictionary with evaluation metrics
    """
    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics using shared function
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    logger.info(
        f"Evaluation metrics: AUC={metrics.get('auc', 'N/A'):.4f}, "
        f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}"
    )
    logger.info(f"Confusion matrix:\n{np.array(metrics['confusion_matrix'])}")

    return metrics


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate cognitive load estimation model")
    parser.add_argument(
        "--in",
        dest="input",
        type=str,
        required=True,
        help="Input features CSV file",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Output path for evaluation report (JSON)",
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="eval.log")
    logger.info("=" * 80)
    logger.info("Starting model evaluation pipeline")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration (hash: {config.hash()[:8]})")

    # Load model artifacts
    models_dir = Path(args.models)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    try:
        model = load_model_artifact(models_dir / "model.bin")
        scaler = load_model_artifact(models_dir / "scaler.bin")
        feature_spec = load_json(models_dir / "feature_spec.json")
        calibration_meta = load_json(models_dir / "calibration.json")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        sys.exit(1)

    feature_names = feature_spec["features"]
    logger.info(f"Loaded model with {len(feature_names)} features")

    # Load features
    features_df = load_features_csv(args.input)

    # Prepare test data
    # Use 'test' role if available, otherwise use all non-calibration data
    if "role" in features_df.columns:
        test_data = features_df[features_df["role"] == "test"]
        if len(test_data) == 0:
            logger.warning("No test data found with role='test', using all non-calibration data")
            test_data = features_df[features_df["role"] != "calibration"]
    else:
        test_data = features_df

    # Map labels
    label_map = {"low": 0, "high": 1, "none": 0}
    test_data = test_data.copy()
    test_data["label_binary"] = test_data["label"].map(
        lambda x: label_map.get(x, 0) if isinstance(x, str) else x
    )

    X_test = test_data[feature_names].values
    y_test = test_data["label_binary"].values

    # Handle NaN values
    if np.any(np.isnan(X_test)):
        logger.warning("Found NaN values in test data, replacing with zeros")
        X_test = np.nan_to_num(X_test, nan=0.0)

    logger.info(f"Test set: {len(X_test)} samples (low={np.sum(y_test==0)}, high={np.sum(y_test==1)})")
    logger.info(f"Using decision threshold: {args.threshold}")

    # Evaluate model
    metrics = evaluate_model(model, scaler, X_test, y_test, threshold=args.threshold)

    # Create evaluation report
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "config_hash": config.hash(),
        "model_info": calibration_meta,
        "metrics": metrics,
    }

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, report_path)

    # Print summary
    logger.info("=" * 80)
    logger.info("Model evaluation complete!")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Report saved to: {report_path}")
    logger.info("=" * 80)

    # Check if AUC meets baseline requirement
    if metrics["auc"] >= 0.70:
        logger.info("✓ Model meets baseline AUC requirement (≥0.70)")
    else:
        logger.warning(f"✗ Model does not meet baseline AUC requirement (got {metrics['auc']:.4f}, need ≥0.70)")


if __name__ == "__main__":
    main()

