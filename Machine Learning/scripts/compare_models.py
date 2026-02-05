#!/usr/bin/env python3
"""
Quick model comparison: Ridge vs RandomForest vs GradientBoosting.

Compares models on soft_label target, evaluates against load_0_1 ground truth.
Uses session-level Spearman correlation as primary metric.

Usage:
    python scripts/compare_models.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Video features for prediction
VIDEO_FEATURES = [
    "blink_rate", "blink_count", "mean_blink_duration",
    "ear_std", "mean_brightness", "std_brightness",
    "perclos", "mean_quality", "valid_frame_ratio",
]

# Models to compare with fixed hyperparameters
MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    ),
}


def load_data(features_path: Path) -> pd.DataFrame:
    """Load features with soft labels."""
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} windows from {features_path.name}")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Tasks: {df['task'].nunique()}")
    return df


def evaluate_model(
    df: pd.DataFrame,
    model_factory,
    train_target: str = "soft_label",
    eval_target: str = "load_0_1",
    n_splits: int = 5,
) -> dict:
    """
    Evaluate model using GroupKFold CV.
    
    Trains on train_target, evaluates predictions against eval_target.
    Returns session-level metrics.
    """
    # Prepare features and targets
    X = df[VIDEO_FEATURES].values
    y_train_target = df[train_target].values
    y_eval_target = df[eval_target].values
    groups = df["user_id"].values
    
    # Store predictions
    all_preds = np.zeros(len(df))
    
    cv = GroupKFold(n_splits=n_splits)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_train_target, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_train_target[train_idx]
        
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
    
    # Add predictions to dataframe for session-level aggregation
    df_eval = df.copy()
    df_eval["pred"] = all_preds
    
    # Aggregate to session level (user, task)
    session_df = df_eval.groupby(["user_id", "task"]).agg({
        "pred": "mean",
        eval_target: "mean",
    }).reset_index()
    
    y_true = session_df[eval_target].values
    y_pred = session_df["pred"].values
    
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n_sessions": len(session_df),
        "n_windows": len(df),
    }


def main():
    """Run model comparison."""
    # Paths
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "processed" / "stress_features_soft_labels.csv"
    output_path = project_root / "reports" / "model_comparison.json"
    
    print("=" * 60)
    print("MODEL COMPARISON: Ridge vs RandomForest vs GradientBoosting")
    print("=" * 60)
    print()
    
    # Load data
    df = load_data(features_path)
    print()
    
    # Run comparison for both targets
    all_results = {}
    
    for train_target in ["soft_label", "load_0_1"]:
        print(f"\n--- Training on: {train_target} ---")
        results = {}
        
        for model_name, model_factory in MODELS.items():
            print(f"  {model_name}...", end=" ", flush=True)
            metrics = evaluate_model(df, model_factory, train_target=train_target)
            results[model_name] = metrics
            print(f"Spearman r = {metrics['spearman_r']:.3f}")
        
        all_results[train_target] = results
    
    # Print combined results table
    print("\n" + "=" * 70)
    print("RESULTS (Session-level, evaluated on load_0_1)")
    print("=" * 70)
    print(f"{'Model':<20} {'Target':<12} {'Spearman r':>12} {'MAE':>10} {'R²':>10}")
    print("-" * 70)
    
    best_combo = None
    best_spearman = -1
    
    for train_target in ["soft_label", "load_0_1"]:
        for model_name in MODELS.keys():
            m = all_results[train_target][model_name]
            print(f"{model_name:<20} {train_target:<12} {m['spearman_r']:>12.3f} {m['mae']:>10.3f} {m['r2']:>10.3f}")
            
            if m['spearman_r'] > best_spearman:
                best_spearman = m['spearman_r']
                best_combo = (model_name, train_target)
        print()
    
    print("-" * 70)
    print(f"\nBest: {best_combo[0]} trained on {best_combo[1]} (Spearman r = {best_spearman:.3f})")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "eval_target": "load_0_1",
        "n_splits": 5,
        "results": all_results,
        "best_model": best_combo[0],
        "best_target": best_combo[1],
        "best_spearman": best_spearman,
    }
    
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
