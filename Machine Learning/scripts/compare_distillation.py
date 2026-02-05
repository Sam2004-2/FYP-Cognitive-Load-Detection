#!/usr/bin/env python3
"""
Compare baseline (load_0_1 target) vs distillation (soft_label target) models.

This script:
1. Trains a baseline model on video features -> load_0_1
2. Trains a distilled model on video features -> soft_label  
3. Evaluates BOTH models on load_0_1 (the ground truth)
4. Compares performance to measure distillation benefit

The key insight: soft_label comes from physio teacher which has learned
physiological stress patterns. If video features can predict soft_label
better than load_0_1, and soft_label correlates with actual stress,
then the distilled model should generalize better.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Tasks to include (all stress conditions)
HIGH_STRESS_TASKS = {"Math", "Speaking", "Stroop", "Counting1", "Counting2", "Counting3"}
LOW_STRESS_TASKS = {"Relax", "Baseline", "Breathing"}
ALL_TASKS = HIGH_STRESS_TASKS | LOW_STRESS_TASKS

# Video features (excluding metadata)
VIDEO_FEATURES = [
    "blink_rate",
    "blink_count", 
    "mean_blink_duration",
    "ear_std",
    "mean_brightness",
    "std_brightness",
    "perclos",
    "mean_quality",
    "valid_frame_ratio",
]


def load_data(features_path: Path) -> pd.DataFrame:
    """Load features with soft labels."""
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} windows from {features_path}")
    
    # Ensure task column exists
    if "task" not in df.columns and "video" in df.columns:
        df["task"] = df["video"].apply(
            lambda v: Path(v).stem.split("_")[-1] if v else None
        )
    
    # Filter to included tasks
    df = df[df["task"].isin(ALL_TASKS)].copy()
    print(f"Filtered to {len(df)} windows for {len(ALL_TASKS)} tasks")
    
    return df


def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Prepare features and target for training.
    
    Returns:
        X, y, groups (for GroupKFold)
    """
    # Get available video features
    feature_cols = [f for f in VIDEO_FEATURES if f in df.columns]
    
    # Filter rows with valid target
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Groups for subject-wise CV
    le = LabelEncoder()
    groups = le.fit_transform(df["user_id"].values)
    
    return X, y, groups, df


def cross_validate(X, y, groups, y_eval=None, n_splits=5, seed=42):
    """
    Train with CV and return predictions.
    
    Args:
        X: Features
        y: Training targets
        groups: Subject IDs for GroupKFold
        y_eval: Optional different target for evaluation (e.g., train on soft_label, eval on load_0_1)
        n_splits: Number of CV folds
        seed: Random seed
        
    Returns:
        Dictionary of results
    """
    if y_eval is None:
        y_eval = y
    
    cv = GroupKFold(n_splits=n_splits)
    
    all_preds = np.zeros(len(y))
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        y_test_eval = y_eval[test_idx]
        
        # Impute and scale
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Train model
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        all_preds[test_idx] = preds
        
        # Evaluate against y_eval (might be different from training target)
        mae = mean_absolute_error(y_test_eval, preds)
        rmse = np.sqrt(mean_squared_error(y_test_eval, preds))
        r2 = r2_score(y_test_eval, preds)
        spearman_r, spearman_p = spearmanr(y_test_eval, preds)
        
        fold_results.append({
            "fold": fold + 1,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })
    
    # Overall metrics
    overall_mae = mean_absolute_error(y_eval, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(y_eval, all_preds))
    overall_r2 = r2_score(y_eval, all_preds)
    overall_spearman, overall_p = spearmanr(y_eval, all_preds)
    
    return {
        "overall": {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "r2": overall_r2,
            "spearman_r": overall_spearman,
            "spearman_p": overall_p,
        },
        "folds": fold_results,
        "predictions": all_preds,
    }


def main():
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "processed" / "features_with_soft_labels.csv"
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("BASELINE VS DISTILLATION COMPARISON")
    print("=" * 70)
    
    # Load data
    df = load_data(features_path)
    
    # Check we have both targets
    if "soft_label" not in df.columns:
        raise ValueError("soft_label column not found - run generate_soft_labels.py first")
    if "load_0_1" not in df.columns:
        raise ValueError("load_0_1 column not found")
    
    print(f"\nData summary:")
    print(f"  Windows: {len(df)}")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Tasks: {df['task'].unique().tolist()}")
    print(f"  load_0_1 range: [{df['load_0_1'].min():.3f}, {df['load_0_1'].max():.3f}]")
    print(f"  soft_label range: [{df['soft_label'].min():.3f}, {df['soft_label'].max():.3f}]")
    print(f"  Correlation(soft_label, load_0_1): {df['soft_label'].corr(df['load_0_1']):.3f}")
    
    # Prepare data
    X, y_load, groups, df_clean = prepare_data(df, "load_0_1")
    _, y_soft, _, _ = prepare_data(df, "soft_label")
    
    # Align targets (same rows)
    mask = df_clean.index.isin(df.dropna(subset=["soft_label"]).index)
    X = X[mask]
    y_load = y_load[mask]
    y_soft = y_soft[:len(X)]  # Align
    groups = groups[mask]
    
    print(f"\nTraining data: {len(X)} windows, {len(VIDEO_FEATURES)} features")
    
    # ===== BASELINE: Train on load_0_1, eval on load_0_1 =====
    print("\n" + "-" * 70)
    print("BASELINE MODEL: Video features → load_0_1")
    print("-" * 70)
    
    baseline_results = cross_validate(X, y_load, groups, y_eval=y_load)
    
    print(f"  MAE:       {baseline_results['overall']['mae']:.4f}")
    print(f"  RMSE:      {baseline_results['overall']['rmse']:.4f}")
    print(f"  R²:        {baseline_results['overall']['r2']:.4f}")
    print(f"  Spearman:  {baseline_results['overall']['spearman_r']:.4f}")
    
    # ===== DISTILLED: Train on soft_label, eval on load_0_1 =====
    print("\n" + "-" * 70)
    print("DISTILLED MODEL: Video features → soft_label, evaluated on load_0_1")
    print("-" * 70)
    
    distilled_results = cross_validate(X, y_soft, groups, y_eval=y_load)
    
    print(f"  MAE:       {distilled_results['overall']['mae']:.4f}")
    print(f"  RMSE:      {distilled_results['overall']['rmse']:.4f}")
    print(f"  R²:        {distilled_results['overall']['r2']:.4f}")
    print(f"  Spearman:  {distilled_results['overall']['spearman_r']:.4f}")
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON (Distilled vs Baseline)")
    print("=" * 70)
    
    mae_diff = distilled_results['overall']['mae'] - baseline_results['overall']['mae']
    spearman_diff = distilled_results['overall']['spearman_r'] - baseline_results['overall']['spearman_r']
    
    print(f"  MAE change:      {mae_diff:+.4f} ({'worse' if mae_diff > 0 else 'better'})")
    print(f"  Spearman change: {spearman_diff:+.4f} ({'better' if spearman_diff > 0 else 'worse'})")
    
    if spearman_diff > 0.05:
        print("\n✓ Distillation IMPROVED correlation with ground truth!")
    elif spearman_diff < -0.05:
        print("\n✗ Distillation REDUCED correlation with ground truth")
    else:
        print("\n○ Distillation had minimal effect on correlation")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_windows": len(X),
        "n_features": len(VIDEO_FEATURES),
        "tasks": list(ALL_TASKS),
        "baseline": {
            "target": "load_0_1",
            "eval_target": "load_0_1",
            "metrics": baseline_results["overall"],
        },
        "distilled": {
            "target": "soft_label",
            "eval_target": "load_0_1",
            "metrics": distilled_results["overall"],
        },
        "comparison": {
            "mae_diff": float(mae_diff),
            "spearman_diff": float(spearman_diff),
            "distillation_helps": bool(spearman_diff > 0.05),
        }
    }
    
    output_path = output_dir / "distillation_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
