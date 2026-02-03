"""
Binary classification training for cognitive load detection.

Classifies cognitive load as HIGH or LOW based on threshold.
- HIGH: load_0_1 >= 0.5
- LOW:  load_0_1 < 0.5

Improvements over baseline:
1. Feature engineering with derived features and interactions
2. RobustScaler for outlier-resistant preprocessing
3. Hyperparameter tuning with RandomizedSearchCV
4. Ensemble methods (VotingClassifier)
5. Probability-based session aggregation

Uses subject-wise GroupKFold cross-validation to prevent data leakage.
"""

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.cle.logging_setup import get_logger, setup_logging
from src.cle.train.metrics import compute_classification_metrics
from src.cle.utils.io import load_features_csv, save_json, save_model_artifact
from src.cle.extract.feature_engineering import compute_derived_features

logger = get_logger(__name__)

# Binary class definitions
CLASS_LABELS = ["LOW", "HIGH"]
BINARY_THRESHOLD = 0.5  # >= 0.5 is HIGH

# 9 base features
BASE_FEATURE_NAMES = [
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

# Derived features added by feature engineering
DERIVED_FEATURE_NAMES = [
    "blink_time_ratio",
    "log_blink_rate",
    "blink_rate_sq",
    "log_ear_std",
    "ear_stability",
    "perclos_logit",
    "perclos_sq",
    "brightness_cv",
    "brightness_stability",
    "blink_perclos_interaction",
    "blink_ear_interaction",
    "perclos_ear_interaction",
    "quality_weighted_blink",
    "reliable_perclos",
    "fatigue_index",
    "blink_regularity",
]

# Tasks to include
TASK_FILTER = {
    "task_1", "task_2", "task_3", "task_4", "task_5",
    "task_6", "task_7", "task_8", "task_9"
}

# Hyperparameter search space
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.05, 0.1, 0.15],
    "min_samples_leaf": [2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def continuous_to_binary(y: np.ndarray, threshold: float = BINARY_THRESHOLD) -> np.ndarray:
    """Convert continuous load [0,1] to binary classes."""
    return (y >= threshold).astype(int)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to add derived features."""
    logger.info("Applying feature engineering...")
    df = compute_derived_features(df)
    return df


def get_feature_names(use_derived: bool = True) -> List[str]:
    """Get list of feature names to use."""
    features = list(BASE_FEATURE_NAMES)
    if use_derived:
        features.extend(DERIVED_FEATURE_NAMES)
    return features


def prepare_data(
    df: pd.DataFrame,
    use_derived_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Prepare data for training.

    Returns:
        X, y_continuous, y_binary, groups, metadata_df, feature_names
    """
    # Apply feature engineering if enabled
    if use_derived_features:
        df = apply_feature_engineering(df)

    # Get feature names
    feature_names = get_feature_names(use_derived_features)
    available_features = [f for f in feature_names if f in df.columns]

    if len(available_features) < len(feature_names):
        missing = set(feature_names) - set(available_features)
        logger.warning(f"Missing features (will be skipped): {missing}")

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

    return X, y_continuous, y_binary, groups, metadata_df, available_features


def aggregate_to_session_probability(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions to session level using probability averaging."""
    if "prob_high" in preds_df.columns:
        session_df = (
            preds_df.groupby(["user_id", "task"])
            .agg(
                y_true=("y_true", "first"),
                prob_high=("prob_high", "mean"),
                n_windows=("y_pred", "count"),
            )
            .reset_index()
        )
        session_df["y_pred"] = (session_df["prob_high"] >= 0.5).astype(int)
    else:
        # Fallback to majority voting
        def majority_vote(x):
            return int(np.round(x.mean()))

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


def create_ensemble_model(seed: int = 42) -> VotingClassifier:
    """Create an ensemble model combining multiple classifiers."""
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
        ("xgb", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=4,
            random_state=seed,
        )),
        ("extra_trees", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
        ("logistic", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_iter: int = 20,
) -> Dict:
    """Tune hyperparameters using RandomizedSearchCV."""
    logger.info(f"Running hyperparameter tuning ({n_iter} iterations)...")

    base_model = GradientBoostingClassifier(random_state=seed)
    search = RandomizedSearchCV(
        base_model,
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=3,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    return search.best_params_


def run_cross_validation(
    X: np.ndarray,
    y_binary: np.ndarray,
    groups: np.ndarray,
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    use_ensemble: bool = True,
    tune_params: bool = True,
) -> Dict:
    """
    Run subject-wise cross-validation with improvements.

    Returns:
        Dictionary with CV results
    """
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=actual_splits)

    logger.info(f"Running {actual_splits}-fold GroupKFold CV")
    logger.info(f"Using ensemble: {use_ensemble}, Tuning params: {tune_params}")

    fold_results = []
    all_predictions = []
    best_params = None

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_binary, groups)):
        logger.info(f"--- Fold {fold_idx + 1}/{actual_splits} ---")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Use RobustScaler (outlier-resistant)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Choose model
        if use_ensemble:
            model = create_ensemble_model(seed)
        elif tune_params and fold_idx == 0:
            # Tune on first fold only to save time
            best_params = tune_hyperparameters(X_train, y_train, seed, n_iter=20)
            model = GradientBoostingClassifier(random_state=seed, **best_params)
        elif best_params is not None:
            model = GradientBoostingClassifier(random_state=seed, **best_params)
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=seed,
            )

        model.fit(X_train, y_train)

        # Predict with probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of HIGH

        # Window-level metrics
        window_metrics = compute_classification_metrics(y_test, y_pred)
        logger.info(f"Window: Acc={window_metrics['accuracy']:.3f}, F1={window_metrics['f1']:.3f}")

        # Build predictions DataFrame with probabilities
        test_metadata = metadata_df.iloc[test_idx].copy()
        test_metadata["y_true"] = y_test
        test_metadata["y_pred"] = y_pred
        test_metadata["prob_high"] = y_proba
        test_metadata["fold"] = fold_idx

        # Session-level metrics using probability aggregation
        session_df = aggregate_to_session_probability(test_metadata)
        session_metrics = compute_classification_metrics(
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

    overall_window = compute_classification_metrics(
        all_preds_df["y_true"].values,
        all_preds_df["y_pred"].values,
    )

    overall_session_df = aggregate_to_session_probability(all_preds_df)
    overall_session = compute_classification_metrics(
        overall_session_df["y_true"].values,
        overall_session_df["y_pred"].values,
    )

    return {
        "n_splits": actual_splits,
        "fold_results": fold_results,
        "overall_window_metrics": overall_window,
        "overall_session_metrics": overall_session,
        "predictions_df": all_preds_df,
        "best_params": best_params,
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    use_ensemble: bool = True,
    best_params: Optional[Dict] = None,
) -> Tuple:
    """Train final model on all data."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if use_ensemble:
        model = create_ensemble_model(seed)
    elif best_params is not None:
        model = GradientBoostingClassifier(random_state=seed, **best_params)
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
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
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble model (use single GradientBoosting)",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning",
    )
    parser.add_argument(
        "--no-derived-features",
        action="store_true",
        help="Disable derived feature engineering",
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

    use_ensemble = not args.no_ensemble
    tune_params = not args.no_tuning
    use_derived = not args.no_derived_features

    logger.info("=" * 60)
    logger.info("IMPROVED BINARY CLASSIFICATION TRAINING")
    logger.info(f"Classes: {CLASS_LABELS}")
    logger.info(f"Threshold: >= {BINARY_THRESHOLD} is HIGH")
    logger.info(f"Feature engineering: {use_derived}")
    logger.info(f"Ensemble model: {use_ensemble}")
    logger.info(f"Hyperparameter tuning: {tune_params}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading features from {args.input}")
    df = load_features_csv(args.input)

    # Filter tasks if needed
    if "task" in df.columns:
        df = df[df["task"].isin(TASK_FILTER)]
        logger.info(f"Filtered to {len(df)} rows for tasks: {TASK_FILTER}")

    # Prepare data with feature engineering
    X, y_continuous, y_binary, groups, metadata_df, feature_names = prepare_data(
        df, use_derived_features=use_derived
    )

    # Run CV
    logger.info("\nRunning cross-validation...")
    cv_results = run_cross_validation(
        X, y_binary, groups, metadata_df,
        n_splits=args.cv_folds,
        seed=args.seed,
        use_ensemble=use_ensemble,
        tune_params=tune_params,
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
    model, scaler, imputer = train_final_model(
        X, y_binary, args.seed,
        use_ensemble=use_ensemble,
        best_params=cv_results.get("best_params"),
    )

    # Save artifacts
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model_artifact(model, output_dir / "model.bin")
    save_model_artifact(scaler, output_dir / "scaler.bin")
    save_model_artifact(imputer, output_dir / "imputer.bin")

    # Save feature spec
    feature_spec = {
        "features": feature_names,
        "base_features": BASE_FEATURE_NAMES,
        "derived_features": DERIVED_FEATURE_NAMES if use_derived else [],
        "n_features": len(feature_names),
        "task_mode": "binary_classification",
        "classes": CLASS_LABELS,
        "threshold": BINARY_THRESHOLD,
        "use_ensemble": use_ensemble,
        "use_derived_features": use_derived,
    }
    save_json(feature_spec, output_dir / "feature_spec.json")

    # Save metrics
    metrics = {
        "training_date": datetime.now().isoformat(),
        "seed": args.seed,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "n_subjects": len(np.unique(groups)),
        "cv_folds": cv_results["n_splits"],
        "use_ensemble": use_ensemble,
        "use_derived_features": use_derived,
        "best_params": cv_results.get("best_params"),
        "window_metrics": cv_results["overall_window_metrics"],
        "session_metrics": cv_results["overall_session_metrics"],
    }
    save_json(metrics, output_dir / "metrics.json")

    logger.info("=" * 60)
    logger.info(f"Training complete! Artifacts saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
