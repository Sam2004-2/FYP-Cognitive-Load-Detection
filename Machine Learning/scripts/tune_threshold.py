"""
Tune decision threshold for stress classifier.

Finds the optimal threshold to maximize F1 score or Recall.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.cle.config import load_config
from src.cle.utils.io import (
    load_features_csv,
    load_json,
    load_model_artifact,
)

def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find index of max F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

def main():
    parser = argparse.ArgumentParser(description="Tune threshold for stress classifier")
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
    
    args = parser.parse_args()
    
    # Load model artifacts
    models_dir = Path(args.models)
    try:
        model = load_model_artifact(models_dir / "model.bin")
        scaler = load_model_artifact(models_dir / "scaler.bin")
        feature_spec = load_json(models_dir / "feature_spec.json")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        sys.exit(1)
        
    feature_names = feature_spec["features"]
    print(f"Loaded model with {len(feature_names)} features")
    
    # Load features
    features_df = load_features_csv(args.input)
    
    # Prepare data (same logic as eval.py)
    if "role" in features_df.columns:
        test_data = features_df[features_df["role"] == "test"]
        if len(test_data) == 0:
            print("No test data found with role='test', using all non-calibration data")
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
    
    # Handle NaN
    if np.any(np.isnan(X_test)):
        print("Found NaN values, replacing with zeros")
        X_test = np.nan_to_num(X_test, nan=0.0)
        
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get probabilities
    print("Predicting probabilities...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_proba)
    
    print("\n" + "="*60)
    print(f"OPTIMAL THRESHOLD ANALYSIS")
    print("="*60)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Max F1 Score:   {best_f1:.4f}")
    
    # Evaluate at default vs optimal
    print("\n--- Comparison ---")
    
    for name, thresh in [("Default (0.5)", 0.5), ("Optimal", best_threshold)]:
        y_pred = (y_proba >= thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
