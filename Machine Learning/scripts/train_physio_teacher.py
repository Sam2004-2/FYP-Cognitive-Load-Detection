#!/usr/bin/env python3
"""
Train a physiological teacher model for soft label distillation.

This script:
1. Loads physiological features (HRV, EDA, respiratory metrics)
2. Joins with load_0_1 labels from self-assessments
3. Filters windows with low ECG quality (< 0.5)
4. Trains a GradientBoostingClassifier with StandardScaler
5. Evaluates using GroupKFold (subject-wise) cross-validation
6. Saves the teacher model and scaler

Usage:
    python scripts/train_physio_teacher.py
    python scripts/train_physio_teacher.py --physio-features data/processed/physio_features.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cle.extract.physio_features import get_physio_feature_names

# Quality threshold for filtering
QUALITY_THRESHOLD = 0.5

# Features to use for training (exclude quality metric itself)
PHYSIO_FEATURES = [
    "hr",
    "rmssd",
    "sdnn",
    "scl",
    "scr_count",
    "scr_amplitude_mean",
    "resp_rate",
    "resp_amplitude_mean",
    "resp_variability",
]

# Tasks to include (matching video training)
TASK_FILTER = {"Math", "Counting1", "Counting2", "Speaking"}


def load_physio_features(physio_path: Path) -> pd.DataFrame:
    """Load extracted physiological features."""
    if not physio_path.exists():
        raise FileNotFoundError(
            f"Physiological features not found: {physio_path}\n"
            f"Run extract_all_physio_features.py first."
        )
    
    df = pd.read_csv(physio_path)
    print(f"Loaded {len(df)} windows from {physio_path}")
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load self-assessment labels."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labels from {labels_path}")
    return df


def prepare_teacher_data(
    physio_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    quality_threshold: float = QUALITY_THRESHOLD,
    tasks: set = TASK_FILTER,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for teacher model training.
    
    Args:
        physio_df: Physiological features DataFrame
        labels_df: Labels DataFrame with subject, task, load_0_1
        quality_threshold: Minimum ECG quality to include
        tasks: Set of tasks to include
        
    Returns:
        Tuple of (merged_df, X, y_binary, groups)
    """
    # Filter by quality
    if "ecg_quality" in physio_df.columns:
        before = len(physio_df)
        physio_df = physio_df[physio_df["ecg_quality"] >= quality_threshold].copy()
        after = len(physio_df)
        print(f"Quality filter: {before} -> {after} windows ({100*after/before:.1f}% retained)")
    
    # Filter by tasks
    if tasks:
        physio_df = physio_df[physio_df["task"].isin(tasks)].copy()
        print(f"Task filter: {len(physio_df)} windows for tasks {tasks}")
    
    # Aggregate physio features to task level (mean across windows)
    # This matches the task-level labels
    physio_task = physio_df.groupby(["user_id", "task"])[PHYSIO_FEATURES].mean().reset_index()
    print(f"Aggregated to {len(physio_task)} task-level entries")
    
    # Merge with labels
    merged = physio_task.merge(
        labels_df[["subject", "task", "load_0_1"]],
        left_on=["user_id", "task"],
        right_on=["subject", "task"],
        how="inner",
    )
    print(f"After merge with labels: {len(merged)} entries")
    
    if len(merged) == 0:
        raise ValueError("No data after merging physio features with labels!")
    
    # Extract features
    X = merged[PHYSIO_FEATURES].values
    
    # Create binary labels (low/high stress)
    # Using median split for balanced classes
    threshold = merged["load_0_1"].median()
    y_binary = (merged["load_0_1"] >= threshold).astype(int)
    print(f"Binary label threshold: {threshold:.3f}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    # Groups for cross-validation
    label_encoder = LabelEncoder()
    groups = label_encoder.fit_transform(merged["user_id"].values)
    
    return merged, X, y_binary, groups


def train_teacher_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict, StandardScaler, SimpleImputer, GradientBoostingClassifier]:
    """
    Train and evaluate teacher model with cross-validation.
    
    Returns:
        Tuple of (cv_results, fitted_scaler, fitted_imputer, final_model)
    """
    cv = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    all_probs = np.zeros(len(y))
    all_preds = np.zeros(len(y), dtype=int)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Validate no subject leakage
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert len(train_subjects & test_subjects) == 0, "Subject leakage detected!"
        
        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=random_state,
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
        
        all_probs[test_idx] = probs
        all_preds[test_idx] = preds
        
        # Fold metrics
        fold_acc = accuracy_score(y_test, preds)
        fold_f1 = f1_score(y_test, preds)
        fold_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        
        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "auc": fold_auc,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        
        print(f"Fold {fold_idx + 1}: Acc={fold_acc:.3f}, F1={fold_f1:.3f}, AUC={fold_auc:.3f}")
    
    # Overall metrics
    overall_acc = accuracy_score(y, all_preds)
    overall_f1 = f1_score(y, all_preds)
    overall_auc = roc_auc_score(y, all_probs)
    
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {overall_acc:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    print(f"Overall AUC: {overall_auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, all_preds, target_names=["Low Stress", "High Stress"]))
    
    cv_results = {
        "fold_results": fold_results,
        "overall": {
            "accuracy": overall_acc,
            "f1": overall_f1,
            "auc": overall_auc,
        },
    }
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_imputer = SimpleImputer(strategy="median")
    X_imp = final_imputer.fit_transform(X)
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_imp)
    
    final_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=random_state,
    )
    final_model.fit(X_scaled, y)
    
    # Feature importance
    print("\nFeature Importance:")
    importances = final_model.feature_importances_
    for feat, imp in sorted(zip(PHYSIO_FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    cv_results["feature_importance"] = dict(zip(PHYSIO_FEATURES, importances.tolist()))
    
    return cv_results, final_scaler, final_imputer, final_model


def save_artifacts(
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    cv_results: Dict,
    output_dir: Path,
) -> None:
    """Save trained model, scaler, imputer, and results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "physio_teacher.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Save scaler
    scaler_path = output_dir / "physio_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save imputer
    imputer_path = output_dir / "physio_imputer.joblib"
    joblib.dump(imputer, imputer_path)
    print(f"Saved imputer to {imputer_path}")
    
    # Save results
    results_path = output_dir / "physio_teacher_eval.json"
    cv_results["timestamp"] = datetime.now().isoformat()
    cv_results["quality_threshold"] = QUALITY_THRESHOLD
    cv_results["features"] = PHYSIO_FEATURES
    cv_results["tasks"] = list(TASK_FILTER)
    
    with open(results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"Saved evaluation to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train physiological teacher model for soft label distillation"
    )
    parser.add_argument(
        "--physio-features",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "physio_features.csv",
        help="Path to extracted physiological features CSV",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw" / "assessments" / "self_assessments_loadindex.csv",
        help="Path to self-assessment labels CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=QUALITY_THRESHOLD,
        help=f"Minimum ECG quality threshold (default: {QUALITY_THRESHOLD})",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHYSIOLOGICAL TEACHER MODEL TRAINING")
    print("=" * 60)
    print(f"Physio features: {args.physio_features}")
    print(f"Labels: {args.labels}")
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"CV splits: {args.n_splits}")
    print()
    
    # Load data
    physio_df = load_physio_features(args.physio_features)
    labels_df = load_labels(args.labels)
    
    # Prepare data
    merged_df, X, y, groups = prepare_teacher_data(
        physio_df, labels_df, 
        quality_threshold=args.quality_threshold,
    )
    
    print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Unique subjects: {len(np.unique(groups))}")
    
    # Train with CV
    cv_results, scaler, imputer, model = train_teacher_cv(
        X, y, groups,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    
    # Save artifacts
    save_artifacts(model, scaler, imputer, cv_results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
