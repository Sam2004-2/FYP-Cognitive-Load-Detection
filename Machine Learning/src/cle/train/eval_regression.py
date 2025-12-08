"""
Evaluation pipeline for regression models.

Evaluates trained regression models and generates reports with
both window-level and session-level metrics.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cle.config import load_config
from src.cle.data.load_data import TASK_FILTER, join_features_with_load, load_load_index
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.utils.io import (
    load_features_csv,
    load_json,
    load_model_artifact,
    save_json,
)

logger = get_logger(__name__)


def aggregate_to_session(
    predictions_df: pd.DataFrame,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate window-level predictions to session level.

    Args:
        predictions_df: DataFrame with columns user_id, task, y_true, y_pred
        agg_method: Aggregation method ("mean" or "median")

    Returns:
        Session-level DataFrame with aggregated predictions
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


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Spearman correlation
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


def evaluate_regression_model(
    model,
    scaler,
    imputer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata_df: pd.DataFrame,
    agg_method: str = "mean",
) -> Dict:
    """
    Evaluate regression model at both window and session level.

    Args:
        model: Trained regression model
        scaler: Fitted StandardScaler
        imputer: Fitted SimpleImputer
        X_test: Test features
        y_test: Test labels (load_0_1)
        metadata_df: DataFrame with user_id, task for session grouping
        agg_method: Session aggregation method

    Returns:
        Dictionary with evaluation results
    """
    # Impute and scale features
    X_imputed = imputer.transform(X_test)
    X_scaled = scaler.transform(X_imputed)

    # Predict
    y_pred = model.predict(X_scaled)

    # Clip to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)

    # Window-level metrics
    window_metrics = compute_regression_metrics(y_test, y_pred)

    # Build predictions DataFrame
    predictions_df = metadata_df.copy()
    predictions_df["y_true"] = y_test
    predictions_df["y_pred"] = y_pred

    # Session-level metrics
    session_df = aggregate_to_session(predictions_df, agg_method)
    session_metrics = compute_regression_metrics(
        session_df["y_true"].values,
        session_df["y_pred"].values,
    )

    # Log results
    logger.info("Window-level metrics:")
    logger.info(f"  MAE: {window_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {window_metrics['rmse']:.4f}")
    logger.info(f"  R2: {window_metrics['r2']:.4f}")
    logger.info(f"  Spearman r: {window_metrics['spearman_r']}")

    logger.info("\nSession-level metrics (PRIMARY):")
    logger.info(f"  MAE: {session_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {session_metrics['rmse']:.4f}")
    logger.info(f"  R2: {session_metrics['r2']:.4f}")
    logger.info(f"  Spearman r: {session_metrics['spearman_r']}")
    logger.info(f"  Spearman p: {session_metrics['spearman_p']}")

    return {
        "window_level": window_metrics,
        "session_level": session_metrics,
        "predictions_df": predictions_df,
        "session_predictions_df": session_df,
    }


def main():
    """Main entry point for regression evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate regression model for cognitive load estimation"
    )
    parser.add_argument(
        "--in",
        dest="input",
        type=str,
        required=True,
        help="Input features CSV file",
    )
    parser.add_argument(
        "--load-index",
        type=str,
        required=True,
        help="Path to self_assessments_loadindex.csv",
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
        help="Path to config YAML (default: configs/regression.yaml)",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Session aggregation method",
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
    setup_logging(level=args.log_level, log_dir="logs", log_file="eval_regression.log")
    logger.info("=" * 80)
    logger.info("Starting REGRESSION evaluation pipeline")
    logger.info("=" * 80)

    # Load configuration
    config_path = args.config or "configs/regression.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Load model artifacts
    models_dir = Path(args.models)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    try:
        model = load_model_artifact(models_dir / "model_regression.bin")
        scaler = load_model_artifact(models_dir / "scaler.bin")
        imputer = load_model_artifact(models_dir / "imputer.bin")
        feature_spec = load_json(models_dir / "feature_spec.json")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        sys.exit(1)

    feature_names = feature_spec["features"]
    logger.info(f"Loaded model with {len(feature_names)} features")

    # Load features
    logger.info(f"Loading features from {args.input}")
    features_df = load_features_csv(args.input)

    # Check if features already contain load_0_1 (from pipeline_loadindex)
    if "load_0_1" in features_df.columns and "task" in features_df.columns:
        logger.info("Features already contain 'load_0_1' and 'task' - skipping load index join")
        merged_df = features_df[features_df["task"].isin(TASK_FILTER)].copy()
        logger.info(f"Filtered to {len(merged_df)} rows for tasks: {TASK_FILTER}")
    else:
        # Load load index and join
        logger.info(f"Loading load index from {args.load_index}")
        load_df = load_load_index(args.load_index)

        # Join features with load labels
        logger.info("Joining features with load labels...")
        merged_df = join_features_with_load(features_df, load_df, tasks=TASK_FILTER)

    # Extract features and labels
    X_test = merged_df[feature_names].values
    y_test = merged_df["load_0_1"].values

    # Metadata for session grouping
    metadata_df = merged_df[["user_id", "task"]].copy()
    if "t_start_s" in merged_df.columns:
        metadata_df["t_start_s"] = merged_df["t_start_s"]
    if "t_end_s" in merged_df.columns:
        metadata_df["t_end_s"] = merged_df["t_end_s"]

    logger.info(f"Evaluation set: {len(X_test)} windows")

    # Evaluate
    results = evaluate_regression_model(
        model=model,
        scaler=scaler,
        imputer=imputer,
        X_test=X_test,
        y_test=y_test,
        metadata_df=metadata_df,
        agg_method=args.agg_method,
    )

    # Create evaluation report
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "task_mode": "regression",
        "agg_method": args.agg_method,
        "n_windows": len(X_test),
        "n_sessions": len(results["session_predictions_df"]),
        "n_subjects": merged_df["user_id"].nunique(),
        "tasks": list(TASK_FILTER),
        "metrics": {
            "window_level": results["window_level"],
            "session_level": results["session_level"],
        },
    }

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, report_path)

    # Save predictions
    predictions_path = report_path.parent / "eval_predictions.csv"
    results["predictions_df"].to_csv(predictions_path, index=False)

    session_predictions_path = report_path.parent / "eval_session_predictions.csv"
    results["session_predictions_df"].to_csv(session_predictions_path, index=False)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Windows evaluated: {report['n_windows']}")
    logger.info(f"Sessions evaluated: {report['n_sessions']}")
    logger.info(f"Subjects: {report['n_subjects']}")
    logger.info("")
    logger.info("Session-level metrics (PRIMARY):")
    logger.info(f"  MAE: {results['session_level']['mae']:.4f}")
    logger.info(f"  Spearman r: {results['session_level']['spearman_r']}")
    logger.info("")
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Predictions saved to: {predictions_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
