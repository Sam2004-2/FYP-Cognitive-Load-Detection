#!/usr/bin/env python3
"""
Train video model on physiological stress ground truth.

This script:
1. Merges video features with physio stress labels (window-level)
2. Trains video model to predict physio_stress_score
3. Validates against binary_stress from labels.csv (task-level)

Usage:
    python scripts/train_on_physio_labels.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Video features
VIDEO_FEATURES = [
    "blink_rate", "blink_count", "mean_blink_duration",
    "ear_std", "mean_brightness", "std_brightness",
    "perclos", "mean_quality", "valid_frame_ratio",
]

# Models to test
MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, random_state=42,
    ),
}


def load_and_merge_data(
    video_path: Path,
    physio_labels_path: Path,
    time_tolerance: float = 2.5,
) -> pd.DataFrame:
    """
    Load video features and merge with physio stress labels.
    
    Matches windows on (user_id, task) and closest t_start_s within tolerance.
    """
    # Load video features
    video_df = pd.read_csv(video_path)
    print(f"Loaded {len(video_df)} video windows")
    
    # Extract task from video path if needed
    if "task" not in video_df.columns:
        video_df["task"] = video_df["video"].apply(
            lambda v: Path(v).stem.split("_")[-1] if pd.notna(v) else None
        )
    
    # Load physio labels
    physio_df = pd.read_csv(physio_labels_path)
    print(f"Loaded {len(physio_df)} physio labels")
    
    # Filter physio to valid stress scores
    physio_df = physio_df[physio_df["physio_stress_score"].notna()].copy()
    print(f"  Valid physio labels: {len(physio_df)}")
    
    # Merge on (user_id, task, t_start_s) with tolerance
    merged_rows = []
    
    for _, video_row in video_df.iterrows():
        user_id = video_row["user_id"]
        task = video_row["task"]
        t_start = video_row["t_start_s"]
        
        # Find matching physio windows
        mask = (
            (physio_df["user_id"] == user_id) &
            (physio_df["task"] == task) &
            (abs(physio_df["t_start_s"] - t_start) <= time_tolerance)
        )
        matches = physio_df[mask]
        
        if len(matches) > 0:
            # Take closest match
            closest_idx = (matches["t_start_s"] - t_start).abs().idxmin()
            physio_row = physio_df.loc[closest_idx]
            
            merged_row = video_row.to_dict()
            merged_row["physio_stress_score"] = physio_row["physio_stress_score"]
            merged_rows.append(merged_row)
    
    merged_df = pd.DataFrame(merged_rows)
    print(f"Merged: {len(merged_df)} windows with physio labels")
    
    return merged_df


def load_binary_stress_labels(labels_path: Path) -> pd.DataFrame:
    """Load binary-stress labels from labels.csv."""
    df = pd.read_csv(labels_path)
    
    # Split subject/task column
    df[["user_id", "task"]] = df["subject/task"].str.split("_", n=1, expand=True)
    
    return df[["user_id", "task", "binary-stress"]].rename(columns={"binary-stress": "binary_stress"})


def evaluate_model(
    df: pd.DataFrame,
    model_factory,
    target_col: str = "physio_stress_score",
    n_splits: int = 5,
) -> dict:
    """
    Train and evaluate model with GroupKFold CV.
    
    Returns metrics for both physio (training) and binary_stress (validation).
    """
    X = df[VIDEO_FEATURES].values
    y = df[target_col].values
    groups = df["user_id"].values
    
    all_preds = np.zeros(len(df))
    cv = GroupKFold(n_splits=n_splits)
    
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Impute and scale
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Train and predict
        model = model_factory()
        model.fit(X_train_scaled, y_train)
        all_preds[test_idx] = model.predict(X_test_scaled)
    
    df_eval = df.copy()
    df_eval["pred"] = all_preds
    
    # Window-level metrics on physio target
    y_true = df_eval[target_col].values
    y_pred = df_eval["pred"].values
    
    window_mae = mean_absolute_error(y_true, y_pred)
    window_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    window_r2 = r2_score(y_true, y_pred)
    window_spearman, _ = spearmanr(y_true, y_pred)
    
    # Session-level aggregation for binary_stress validation
    session_df = df_eval.groupby(["user_id", "task"]).agg({
        "pred": "mean",
        target_col: "mean",
        "binary_stress": "first",  # Same for all windows in session
    }).reset_index()
    
    session_spearman, _ = spearmanr(session_df[target_col], session_df["pred"])
    
    # Binary stress validation (AUC)
    binary_stress = session_df["binary_stress"].values
    pred_scores = session_df["pred"].values
    
    # AUC: can predictions distinguish binary stress?
    try:
        binary_auc = roc_auc_score(binary_stress, pred_scores)
    except ValueError:
        binary_auc = np.nan
    
    # Accuracy at threshold
    threshold = 0.5
    pred_binary = (pred_scores >= threshold).astype(int)
    binary_accuracy = (pred_binary == binary_stress).mean()
    
    return {
        "window_mae": window_mae,
        "window_rmse": window_rmse,
        "window_r2": window_r2,
        "window_spearman": window_spearman,
        "session_spearman": session_spearman,
        "binary_stress_auc": binary_auc,
        "binary_stress_accuracy": binary_accuracy,
        "n_windows": len(df),
        "n_sessions": len(session_df),
    }


def main():
    """Run training on physio labels with binary stress validation."""
    # Paths
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "processed" / "stress_features.csv"
    physio_labels_path = project_root / "data" / "processed" / "physio_stress_labels.csv"
    binary_labels_path = project_root.parent / "labels.csv"
    output_path = project_root / "reports" / "physio_ground_truth_eval.json"
    
    print("=" * 70)
    print("TRAINING VIDEO MODEL ON PHYSIOLOGICAL STRESS GROUND TRUTH")
    print("=" * 70)
    print()
    
    # Load and merge data
    df = load_and_merge_data(video_path, physio_labels_path)
    
    # Load binary stress labels for validation
    binary_labels = load_binary_stress_labels(binary_labels_path)
    df = df.merge(binary_labels, on=["user_id", "task"], how="left")
    
    # Filter to rows with both physio and binary stress labels
    df = df[df["binary_stress"].notna()].copy()
    print(f"Final dataset: {len(df)} windows with both labels")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Tasks: {df['task'].nunique()}")
    print()
    
    # Evaluate models
    results = {}
    
    for model_name, model_factory in MODELS.items():
        print(f"Evaluating {model_name}...", end=" ", flush=True)
        metrics = evaluate_model(df, model_factory)
        results[model_name] = metrics
        print(f"AUC={metrics['binary_stress_auc']:.3f}, Spearman={metrics['session_spearman']:.3f}")
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Physio Spearman':>16} {'Binary AUC':>12} {'Binary Acc':>12}")
    print("-" * 70)
    
    for model_name, m in results.items():
        print(f"{model_name:<20} {m['session_spearman']:>16.3f} {m['binary_stress_auc']:>12.3f} {m['binary_stress_accuracy']:>12.1%}")
    
    print("-" * 70)
    
    # Best model by binary AUC
    best_model = max(results.keys(), key=lambda m: results[m]["binary_stress_auc"])
    best_auc = results[best_model]["binary_stress_auc"]
    print(f"\nBest model: {best_model} (Binary Stress AUC = {best_auc:.3f})")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  - Video model trained on physio stress achieves AUC={best_auc:.3f}")
    print(f"    for predicting self-reported binary stress")
    if best_auc >= 0.65:
        print("  - This indicates good alignment between physio-derived and self-reported stress")
    elif best_auc >= 0.55:
        print("  - This indicates moderate alignment between physio and self-reported stress")
    else:
        print("  - Weak alignment - physio stress may capture different construct than self-report")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "training_target": "physio_stress_score",
        "validation_target": "binary_stress",
        "results": results,
        "best_model": best_model,
    }
    
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
