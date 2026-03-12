#!/usr/bin/env python3
"""
Generate physiological stress labels for video model training.

This script:
1. Loads extracted physio features (25k+ windows)
2. Computes physio_stress_score using validated features
3. Outputs window-level labels for joining with video features

The physio_stress_score is an objective measure of physiological arousal/stress
derived from HR, HRV (SDNN), EDA (SCR amplitude), and respiratory amplitude.

Usage:
    python scripts/generate_physio_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from cle.extract.physio_features import compute_physio_stress_score, VALIDATED_FEATURES


def main():
    """Generate physio stress labels."""
    # Paths
    project_root = Path(__file__).parent.parent
    physio_path = project_root / "data" / "processed" / "physio_features.csv"
    output_path = project_root / "data" / "processed" / "physio_stress_labels.csv"
    
    print("=" * 60)
    print("GENERATING PHYSIOLOGICAL STRESS LABELS")
    print("=" * 60)
    print(f"Using validated features: {VALIDATED_FEATURES}")
    print()
    
    # Load physio features
    print(f"Loading {physio_path}...")
    df = pd.read_csv(physio_path)
    print(f"  Loaded {len(df)} windows")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Tasks: {df['task'].unique().tolist()}")
    
    # Check feature availability
    available = [f for f in VALIDATED_FEATURES if f in df.columns]
    print(f"\nAvailable validated features: {available}")
    
    # Check for missing values
    for feat in available:
        n_missing = df[feat].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        print(f"  {feat}: {n_missing} missing ({pct_missing:.1f}%)")
    
    # Compute physio stress score
    print("\nComputing physio_stress_score...")
    df["physio_stress_score"] = compute_physio_stress_score(df)
    
    # Statistics
    valid_scores = df["physio_stress_score"].dropna()
    print(f"\nPhysio stress score statistics:")
    print(f"  Valid: {len(valid_scores)} / {len(df)} ({100*len(valid_scores)/len(df):.1f}%)")
    print(f"  Mean: {valid_scores.mean():.3f}")
    print(f"  Std:  {valid_scores.std():.3f}")
    print(f"  Min:  {valid_scores.min():.3f}")
    print(f"  Max:  {valid_scores.max():.3f}")
    print(f"  Median: {valid_scores.median():.3f}")
    
    # Check distribution by task type
    print("\nMean physio_stress_score by task:")
    task_means = df.groupby("task")["physio_stress_score"].mean().sort_values(ascending=False)
    for task, score in task_means.items():
        print(f"  {task}: {score:.3f}")
    
    # Save output
    output_cols = ["user_id", "task", "t_start_s", "t_end_s", "physio_stress_score", "ecg_quality"]
    df_out = df[output_cols].copy()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_out)} labels to {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
