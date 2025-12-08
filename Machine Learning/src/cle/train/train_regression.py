"""
Regression training pipeline for continuous cognitive load prediction.

Trains models to predict load_0_1 (continuous 0-1 target) using subject-wise
cross-validation to prevent data leakage.
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.cle.config import load_config
from src.cle.data.load_data import TASK_FILTER, join_features_with_load, load_load_index
from src.cle.extract.features import get_feature_names
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.utils.io import load_features_csv, save_json, save_model_artifact

logger = get_logger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def create_regression_model(model_type: str, config: dict, seed: int = 42):
    """
    Create a regression model based on configuration.

    Args:
        model_type: One of "ridge", "rf", "xgb"
        config: Configuration dictionary
        seed: Random seed

    Returns:
        Scikit-learn regressor instance
    """
    if model_type == "ridge":
        params = config.get("regression.ridge_params", {})
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=seed,
        )
        logger.info(f"Created Ridge regression model with alpha={params.get('alpha', 1.0)}")

    elif model_type == "rf":
        params = config.get("regression.rf_params", {})
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
        logger.info(f"Created RandomForestRegressor with {params.get('n_estimators', 100)} trees")

    elif model_type == "xgb":
        params = config.get("regression.xgb_params", {})
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=seed,
        )
        logger.info(f"Created GradientBoostingRegressor with {params.get('n_estimators', 100)} estimators")

    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ridge', 'rf', or 'xgb'")

    return model


def prepare_regression_data(
    merged_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare data for regression training.

    Args:
        merged_df: DataFrame with features and load_0_1 label
        feature_names: List of feature column names

    Returns:
        Tuple of (X, y, groups, metadata_df)
        - X: Feature matrix
        - y: Target values (load_0_1)
        - groups: Encoded subject IDs for GroupKFold
        - metadata_df: DataFrame with user_id, task, window info for session aggregation
    """
    # Extract features
    X = merged_df[feature_names].values

    # Extract target
    y = merged_df["load_0_1"].values

    # Encode subjects for grouping
    label_encoder = LabelEncoder()
    groups = label_encoder.fit_transform(merged_df["user_id"].values)

    # Keep metadata for session aggregation
    metadata_cols = ["user_id", "task", "t_start_s", "t_end_s", "load_0_1"]
    available_cols = [c for c in metadata_cols if c in merged_df.columns]
    metadata_df = merged_df[available_cols].copy()

    logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    logger.info(f"Unique subjects (groups): {len(np.unique(groups))}")

    return X, y, groups, metadata_df


def validate_no_subject_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
) -> None:
    """
    Validate that no subject appears in both train and test sets.

    Raises:
        AssertionError if subject leakage is detected
    """
    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    overlap = train_subjects & test_subjects

    if overlap:
        raise AssertionError(
            f"SUBJECT LEAKAGE DETECTED! Subjects in both train and test: {overlap}"
        )


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
            y_true=("y_true", "first"),  # Same for all windows in session
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


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata_df: pd.DataFrame,
    model_factory: Callable,
    n_splits: int = 5,
    use_loso: bool = False,
    agg_method: str = "mean",
) -> Dict:
    """
    Run subject-wise cross-validation.

    Args:
        X: Feature matrix
        y: Target values
        groups: Subject group labels
        metadata_df: Metadata for session aggregation
        model_factory: Callable that returns a new model instance
        n_splits: Number of folds for GroupKFold
        use_loso: If True, use Leave-One-Subject-Out instead of GroupKFold
        agg_method: Session aggregation method

    Returns:
        Dictionary with CV results
    """
    # Choose CV strategy
    if use_loso:
        cv = LeaveOneGroupOut()
        n_splits = len(np.unique(groups))
        logger.info(f"Using Leave-One-Subject-Out CV ({n_splits} folds)")
    else:
        n_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_groups)
        if actual_splits < n_splits:
            logger.warning(
                f"Only {n_groups} subjects, reducing folds from {n_splits} to {actual_splits}"
            )
        cv = GroupKFold(n_splits=actual_splits)
        n_splits = actual_splits
        logger.info(f"Using GroupKFold CV with {n_splits} folds")

    # Imputer for NaN handling
    imputer = SimpleImputer(strategy="median")

    # Storage for results
    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        # Validate no leakage
        validate_no_subject_leakage(train_idx, test_idx, groups)

        # Get train/test subjects for logging
        train_subjects = sorted(set(groups[train_idx]))
        test_subjects = sorted(set(groups[test_idx]))
        logger.info(f"Train subjects: {train_subjects} ({len(train_idx)} windows)")
        logger.info(f"Test subjects: {test_subjects} ({len(test_idx)} windows)")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle NaN values
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = model_factory()
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Clip predictions to [0, 1]
        y_pred = np.clip(y_pred, 0, 1)

        # Window-level metrics
        window_metrics = compute_regression_metrics(y_test, y_pred)
        logger.info(f"Window-level MAE: {window_metrics['mae']:.4f}")

        # Build predictions DataFrame for session aggregation
        test_metadata = metadata_df.iloc[test_idx].copy()
        test_metadata["y_pred"] = y_pred
        test_metadata["y_true"] = y_test

        # Session-level metrics
        session_df = aggregate_to_session(test_metadata, agg_method)
        session_metrics = compute_regression_metrics(
            session_df["y_true"].values,
            session_df["y_pred"].values,
        )
        logger.info(
            f"Session-level MAE: {session_metrics['mae']:.4f}, "
            f"Spearman r: {session_metrics['spearman_r']}"
        )

        # Store results
        fold_results.append(
            {
                "fold": fold_idx,
                "train_subjects": [int(s) for s in train_subjects],
                "test_subjects": [int(s) for s in test_subjects],
                "n_train_windows": len(train_idx),
                "n_test_windows": len(test_idx),
                "n_test_sessions": len(session_df),
                "window_metrics": window_metrics,
                "session_metrics": session_metrics,
            }
        )

        # Store predictions for final aggregation
        test_metadata["fold"] = fold_idx
        all_predictions.append(test_metadata)

    # Aggregate across folds
    all_preds_df = pd.concat(all_predictions, ignore_index=True)

    # Overall window-level metrics
    overall_window_metrics = compute_regression_metrics(
        all_preds_df["y_true"].values,
        all_preds_df["y_pred"].values,
    )

    # Overall session-level metrics
    overall_session_df = aggregate_to_session(all_preds_df, agg_method)
    overall_session_metrics = compute_regression_metrics(
        overall_session_df["y_true"].values,
        overall_session_df["y_pred"].values,
    )

    # Compute fold statistics
    window_maes = [f["window_metrics"]["mae"] for f in fold_results]
    session_maes = [f["session_metrics"]["mae"] for f in fold_results]
    session_spearmans = [
        f["session_metrics"]["spearman_r"]
        for f in fold_results
        if f["session_metrics"]["spearman_r"] is not None
    ]

    cv_summary = {
        "window_mae_mean": float(np.mean(window_maes)),
        "window_mae_std": float(np.std(window_maes)),
        "session_mae_mean": float(np.mean(session_maes)),
        "session_mae_std": float(np.std(session_maes)),
        "session_spearman_mean": float(np.mean(session_spearmans)) if session_spearmans else None,
        "session_spearman_std": float(np.std(session_spearmans)) if session_spearmans else None,
    }

    return {
        "cv_method": "loso" if use_loso else "group_kfold",
        "n_splits": n_splits,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "overall_window_metrics": overall_window_metrics,
        "overall_session_metrics": overall_session_metrics,
        "predictions_df": all_preds_df,
        "session_predictions_df": overall_session_df,
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
) -> Tuple:
    """
    Train final model on all data.

    Args:
        X: Feature matrix
        y: Target values
        model_factory: Callable that returns a new model instance

    Returns:
        Tuple of (model, scaler, imputer)
    """
    # Impute NaN values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train model
    model = model_factory()
    model.fit(X_scaled, y)

    return model, scaler, imputer


def main():
    """Main entry point for regression training."""
    parser = argparse.ArgumentParser(
        description="Train regression model for cognitive load estimation"
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
        required=False,
        help="Path to self_assessments_loadindex.csv",
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
        help="Path to config YAML (default: configs/regression.yaml)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["ridge", "rf", "xgb"],
        help="Model type (overrides config)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of CV folds (overrides config)",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Use Leave-One-Subject-Out CV instead of GroupKFold",
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
    setup_logging(level=args.log_level, log_dir="logs", log_file="train_regression.log")
    logger.info("=" * 80)
    logger.info("Starting REGRESSION training pipeline")
    logger.info("=" * 80)

    # Load configuration
    config_path = args.config or "configs/regression.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Set random seed
    seed = config.get("seed", 42)
    set_random_seed(seed)

    # Determine model type
    model_type = args.model_type or config.get("regression.model_type", "ridge")
    logger.info(f"Model type: {model_type}")

    # Determine CV settings
    n_splits = args.cv_folds or config.get("cv.n_splits", 5)
    use_loso = args.loso or (config.get("cv.method", "group_kfold") == "loso")
    agg_method = config.get("session_agg.method", "mean")

    # Load features
    logger.info(f"Loading features from {args.input}")
    features_df = load_features_csv(args.input)
    logger.info(f"Loaded {len(features_df)} feature rows")

    # Check if features already have load_0_1 - skip load index join if so
    if "load_0_1" in features_df.columns and "task" in features_df.columns:
        logger.info("Features already contain 'load_0_1' and 'task' - skipping load index join")
        # Filter to specified tasks
        merged_df = features_df[features_df["task"].isin(TASK_FILTER)].copy()
        logger.info(f"Filtered to {len(merged_df)} rows for tasks: {TASK_FILTER}")
    else:
        # Need to load and join with load index
        if args.load_index is None:
            raise ValueError(
                "--load-index is required when features CSV doesn't contain 'load_0_1' column"
            )
        logger.info(f"Loading load index from {args.load_index}")
        load_df = load_load_index(args.load_index)

        # Join features with load labels
        logger.info("Joining features with load labels...")
        merged_df = join_features_with_load(features_df, load_df, tasks=TASK_FILTER)

    # Get feature names
    feature_names = get_feature_names(config.to_dict())
    logger.info(f"Using {len(feature_names)} features: {feature_names}")

    # Prepare data
    X, y, groups, metadata_df = prepare_regression_data(merged_df, feature_names)

    # Model factory
    def model_factory():
        return create_regression_model(model_type, config.to_dict(), seed)

    # Run cross-validation
    logger.info("\n" + "=" * 80)
    logger.info("Running cross-validation...")
    logger.info("=" * 80)

    cv_results = run_cross_validation(
        X=X,
        y=y,
        groups=groups,
        metadata_df=metadata_df,
        model_factory=model_factory,
        n_splits=n_splits,
        use_loso=use_loso,
        agg_method=agg_method,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"CV Method: {cv_results['cv_method']}")
    logger.info(f"Folds: {cv_results['n_splits']}")
    logger.info("")
    logger.info("Window-level metrics (aggregated across all folds):")
    logger.info(f"  MAE: {cv_results['overall_window_metrics']['mae']:.4f}")
    logger.info(f"  RMSE: {cv_results['overall_window_metrics']['rmse']:.4f}")
    logger.info("")
    logger.info("Session-level metrics (PRIMARY - aggregated across all folds):")
    logger.info(f"  MAE: {cv_results['overall_session_metrics']['mae']:.4f}")
    logger.info(f"  RMSE: {cv_results['overall_session_metrics']['rmse']:.4f}")
    logger.info(f"  Spearman r: {cv_results['overall_session_metrics']['spearman_r']}")
    logger.info(f"  Spearman p: {cv_results['overall_session_metrics']['spearman_p']}")
    logger.info("")
    logger.info("CV fold statistics:")
    logger.info(f"  Session MAE: {cv_results['cv_summary']['session_mae_mean']:.4f} +/- {cv_results['cv_summary']['session_mae_std']:.4f}")
    if cv_results['cv_summary']['session_spearman_mean'] is not None:
        logger.info(f"  Session Spearman: {cv_results['cv_summary']['session_spearman_mean']:.4f} +/- {cv_results['cv_summary']['session_spearman_std']:.4f}")

    # Train final model on all data
    logger.info("\n" + "=" * 80)
    logger.info("Training final model on all data...")
    logger.info("=" * 80)

    final_model, scaler, imputer = train_final_model(X, y, model_factory)

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    logger.info("Saving model artifacts...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v1_{timestamp}"

    # Save versioned artifacts
    save_model_artifact(final_model, output_dir / f"model_regression_{version}.bin")
    save_model_artifact(scaler, output_dir / f"scaler_{version}.bin")
    save_model_artifact(imputer, output_dir / f"imputer_{version}.bin")

    # Save without version for convenience
    save_model_artifact(final_model, output_dir / "model_regression.bin")
    save_model_artifact(scaler, output_dir / "scaler.bin")
    save_model_artifact(imputer, output_dir / "imputer.bin")

    # Save feature spec
    feature_spec = {
        "features": feature_names,
        "n_features": len(feature_names),
        "task_mode": "regression",
    }
    save_json(feature_spec, output_dir / "feature_spec.json")

    # Save CV results
    cv_results_to_save = {
        "training_date": datetime.now().isoformat(),
        "model_type": model_type,
        "task_mode": "regression",
        "cv_method": cv_results["cv_method"],
        "n_splits": cv_results["n_splits"],
        "n_subjects": len(np.unique(groups)),
        "n_sessions": len(cv_results["session_predictions_df"]),
        "n_windows": len(X),
        "tasks": list(TASK_FILTER),
        "cv_summary": cv_results["cv_summary"],
        "overall_window_metrics": cv_results["overall_window_metrics"],
        "overall_session_metrics": cv_results["overall_session_metrics"],
        "fold_results": cv_results["fold_results"],
        "feature_names": feature_names,
    }
    save_json(cv_results_to_save, output_dir / "cv_results.json")

    # Save predictions
    cv_results["predictions_df"].to_csv(
        output_dir / "window_predictions.csv", index=False
    )
    cv_results["session_predictions_df"].to_csv(
        output_dir / "session_predictions.csv", index=False
    )

    logger.info("=" * 80)
    logger.info("Regression training complete!")
    logger.info(f"Artifacts saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
