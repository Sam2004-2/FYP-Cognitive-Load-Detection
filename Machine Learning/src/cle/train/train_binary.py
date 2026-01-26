"""
Binary classification training for cognitive load detection.

Classifies cognitive load as HIGH or LOW based on threshold.
- HIGH: load_0_1 >= 0.5
- LOW:  load_0_1 < 0.5

Uses subject-wise GroupKFold cross-validation to prevent data leakage.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.cle.logging_setup import get_logger, setup_logging
from src.cle.utils.io import load_features_csv, save_json, save_model_artifact

logger = get_logger(__name__)

# Binary class definitions
CLASS_LABELS = ["LOW", "HIGH"]
BINARY_THRESHOLD = 0.5  # >= 0.5 is HIGH

# 9 base features (no derived features)
FEATURE_NAMES = [
    "blink_rate",
    "blink_count",
    "mean_blink_duration",
    "ear_std",
    "perclos",
    "mean_brightness",
    "std_brightness",
    "mean_quality",
    "valid_frame_ratio",
]

# Tasks to include
TASK_FILTER = {
    "task_1", "task_2", "task_3", "task_4", "task_5",
    "task_6", "task_7", "task_8", "task_9"
}


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def continuous_to_binary(y: np.ndarray, threshold: float = BINARY_THRESHOLD) -> np.ndarray:
    """
    Convert continuous load [0,1] to binary classes.

    Args:
        y: Continuous load values
        threshold: Classification threshold (default 0.5)

    Returns:
        Binary labels: 0=LOW, 1=HIGH
    """
    return (y >= threshold).astype(int)


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare data for training.

    Returns:
        X, y_continuous, y_binary, groups, metadata_df
    """
    # Extract features
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    if len(available_features) < len(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(available_features)
        logger.warning(f"Missing features: {missing}")

    X = df[available_features].values
    y_continuous = df["load_0_1"].values
    y_binary = continuous_to_binary(y_continuous)

    # Encode subjects for grouping
    groups = LabelEncoder().fit_transform(df["user_id"].values)

    # Keep metadata
    metadata_cols = ["user_id", "task", "load_0_1"]
    metadata_df = df[[c for c in metadata_cols if c in df.columns]].copy()

    # Log stats
    n_high = (y_binary == 1).sum()
    n_low = (y_binary == 0).sum()
    logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
    logger.info(f"Class distribution: LOW={n_low}, HIGH={n_high}")
    logger.info(f"Unique subjects: {len(np.unique(groups))}")

    return X, y_continuous, y_binary, groups, metadata_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
    }


def aggregate_to_session(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions to session level using majority voting."""
    def majority_vote(x):
        return int(np.round(x.mean()))  # >50% HIGH -> HIGH

    session_df = (
        preds_df.groupby(["user_id", "task"])
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", majority_vote),
            n_windows=("y_pred", "count"),
        )
        .reset_index()
    )
    return session_df


def run_cross_validation(
    X: np.ndarray,
    y_binary: np.ndarray,
    groups: np.ndarray,
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run subject-wise cross-validation.

    Returns:
        Dictionary with CV results
    """
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=actual_splits)

    logger.info(f"Running {actual_splits}-fold GroupKFold CV")

    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_binary, groups)):
        logger.info(f"--- Fold {fold_idx + 1}/{actual_splits} ---")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # Impute and scale
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train XGBoost
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Window-level metrics
        window_metrics = compute_metrics(y_test, y_pred)
        logger.info(f"Window: Acc={window_metrics['accuracy']:.3f}, F1={window_metrics['f1']:.3f}")

        # Build predictions DataFrame
        test_metadata = metadata_df.iloc[test_idx].copy()
        test_metadata["y_true"] = y_test
        test_metadata["y_pred"] = y_pred
        test_metadata["fold"] = fold_idx

        # Session-level metrics
        session_df = aggregate_to_session(test_metadata)
        session_metrics = compute_metrics(
            session_df["y_true"].values,
            session_df["y_pred"].values,
        )
        logger.info(f"Session: Acc={session_metrics['accuracy']:.3f}, F1={session_metrics['f1']:.3f}")

        fold_results.append({
            "fold": fold_idx,
            "window_metrics": window_metrics,
            "session_metrics": session_metrics,
        })
        all_predictions.append(test_metadata)

    # Aggregate results
    all_preds_df = pd.concat(all_predictions, ignore_index=True)

    overall_window = compute_metrics(
        all_preds_df["y_true"].values,
        all_preds_df["y_pred"].values,
    )

    overall_session_df = aggregate_to_session(all_preds_df)
    overall_session = compute_metrics(
        overall_session_df["y_true"].values,
        overall_session_df["y_pred"].values,
    )

    return {
        "n_splits": actual_splits,
        "fold_results": fold_results,
        "overall_window_metrics": overall_window,
        "overall_session_metrics": overall_session,
        "predictions_df": all_preds_df,
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Tuple:
    """Train final model on all data."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_scaled, y)

    return model, scaler, imputer


def main():
    parser = argparse.ArgumentParser(
        description="Train binary classifier for cognitive load (HIGH/LOW)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input features CSV file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/binary_classifier",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup
    setup_logging(level=args.log_level, log_dir="logs", log_file="train_binary.log")
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("BINARY CLASSIFICATION TRAINING")
    logger.info(f"Classes: {CLASS_LABELS}")
    logger.info(f"Threshold: >= {BINARY_THRESHOLD} is HIGH")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading features from {args.input}")
    df = load_features_csv(args.input)

    # Filter tasks if needed
    if "task" in df.columns:
        df = df[df["task"].isin(TASK_FILTER)]
        logger.info(f"Filtered to {len(df)} rows for tasks: {TASK_FILTER}")

    # Prepare data
    X, y_continuous, y_binary, groups, metadata_df = prepare_data(df)

    # Run CV
    logger.info("\nRunning cross-validation...")
    cv_results = run_cross_validation(
        X, y_binary, groups, metadata_df,
        n_splits=args.cv_folds,
        seed=args.seed,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Window-level accuracy: {cv_results['overall_window_metrics']['accuracy']:.4f}")
    logger.info(f"Window-level F1: {cv_results['overall_window_metrics']['f1']:.4f}")
    logger.info(f"Session-level accuracy: {cv_results['overall_session_metrics']['accuracy']:.4f}")
    logger.info(f"Session-level F1: {cv_results['overall_session_metrics']['f1']:.4f}")

    cm = cv_results['overall_session_metrics']['confusion_matrix']
    logger.info(f"\nConfusion Matrix (session):")
    logger.info(f"          Pred: LOW  HIGH")
    logger.info(f"  True LOW:    {cm[0][0]:4d}  {cm[0][1]:4d}")
    logger.info(f"  True HIGH:   {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Train final model
    logger.info("\nTraining final model on all data...")
    model, scaler, imputer = train_final_model(X, y_binary, args.seed)

    # Save artifacts
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model_artifact(model, output_dir / "model.bin")
    save_model_artifact(scaler, output_dir / "scaler.bin")
    save_model_artifact(imputer, output_dir / "imputer.bin")

    # Save feature spec
    feature_spec = {
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "task_mode": "binary_classification",
        "classes": CLASS_LABELS,
        "threshold": BINARY_THRESHOLD,
    }
    save_json(feature_spec, output_dir / "feature_spec.json")

    # Save metrics
    metrics = {
        "training_date": datetime.now().isoformat(),
        "seed": args.seed,
        "n_samples": len(X),
        "n_subjects": len(np.unique(groups)),
        "cv_folds": cv_results["n_splits"],
        "window_metrics": cv_results["overall_window_metrics"],
        "session_metrics": cv_results["overall_session_metrics"],
    }
    save_json(metrics, output_dir / "metrics.json")

    logger.info("=" * 60)
    logger.info(f"Training complete! Artifacts saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
