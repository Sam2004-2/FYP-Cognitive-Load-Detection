#!/usr/bin/env python3
"""
Generate soft labels for video features using the trained physiological teacher.

This script:
1. Loads the trained physio teacher model and scaler
2. Loads video features and physiological features
3. For windows with good quality physio data: generates soft_label from teacher
4. For windows with missing/low quality physio: falls back to load_0_1
5. Outputs features_with_soft_labels.csv for video model training

Usage:
    python scripts/generate_soft_labels.py
    python scripts/generate_soft_labels.py --video-features data/processed/features.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Quality threshold (must match teacher training)
QUALITY_THRESHOLD = 0.5

# Physiological features used by teacher
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

# Tasks to include
TASK_FILTER = {"Math", "Counting1", "Counting2", "Speaking"}


def load_teacher_artifacts(model_dir: Path) -> Tuple:
    """Load trained teacher model, scaler, and imputer."""
    model_path = model_dir / "physio_teacher.joblib"
    scaler_path = model_dir / "physio_scaler.joblib"
    imputer_path = model_dir / "physio_imputer.joblib"
    
    for path in [model_path, scaler_path, imputer_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Teacher artifact not found: {path}\n"
                f"Run train_physio_teacher.py first."
            )
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    print(f"Loaded teacher model from {model_dir}")
    return model, scaler, imputer


def load_video_features(features_path: Path) -> pd.DataFrame:
    """Load video features CSV."""
    if not features_path.exists():
        raise FileNotFoundError(f"Video features not found: {features_path}")
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} video feature windows from {features_path}")
    return df


def load_physio_features(physio_path: Path) -> pd.DataFrame:
    """Load physiological features CSV."""
    if not physio_path.exists():
        raise FileNotFoundError(
            f"Physiological features not found: {physio_path}\n"
            f"Run extract_all_physio_features.py first."
        )
    
    df = pd.read_csv(physio_path)
    print(f"Loaded {len(df)} physio feature windows from {physio_path}")
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load self-assessment labels for fallback."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labels from {labels_path}")
    return df


def extract_task_from_video(video_path: str) -> str:
    """Extract task name from video path."""
    if not video_path or not isinstance(video_path, str):
        return None
    filename = Path(video_path).stem
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return None


def generate_soft_labels(
    video_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model,
    scaler,
    imputer,
    quality_threshold: float = QUALITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate soft labels for video features.
    
    For each video window:
    1. Find matching physio data (same user_id, task)
    2. If quality >= threshold: soft_label = teacher.predict_proba()[:, 1]
    3. If quality < threshold or no match: soft_label = load_0_1 (fallback)
    
    Returns:
        Video features DataFrame with added soft_label column
    """
    # Add task column to video features if needed
    if "task" not in video_df.columns and "video" in video_df.columns:
        video_df = video_df.copy()
        video_df["task"] = video_df["video"].apply(extract_task_from_video)
    
    # Filter to relevant tasks
    video_df = video_df[video_df["task"].isin(TASK_FILTER)].copy()
    print(f"Filtered to {len(video_df)} windows for tasks {TASK_FILTER}")
    
    # Create load_0_1 lookup from labels (task-level)
    labels_lookup = labels_df.set_index(["subject", "task"])["load_0_1"].to_dict()
    
    # Aggregate physio features to task level for teacher prediction
    # (Teacher was trained on task-level aggregated features)
    physio_task = physio_df.groupby(["user_id", "task"]).agg({
        **{feat: "mean" for feat in PHYSIO_FEATURES},
        "ecg_quality": "mean",
    }).reset_index()
    
    # Filter by quality
    high_quality = physio_task[physio_task["ecg_quality"] >= quality_threshold].copy()
    print(f"High quality physio data: {len(high_quality)} task entries")
    
    # Generate teacher predictions for high quality physio data
    teacher_preds = {}
    if len(high_quality) > 0:
        X_physio = high_quality[PHYSIO_FEATURES].values
        X_imp = imputer.transform(X_physio)
        X_scaled = scaler.transform(X_imp)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        for idx, row in enumerate(high_quality.itertuples()):
            key = (row.user_id, row.task)
            teacher_preds[key] = probs[idx]
    
    print(f"Generated {len(teacher_preds)} teacher predictions")
    
    # Assign soft labels to video windows
    soft_labels = []
    sources = []  # Track where each label came from
    
    for _, row in video_df.iterrows():
        user_id = row["user_id"]
        task = row["task"]
        key = (user_id, task)
        
        if key in teacher_preds:
            # Use teacher prediction
            soft_labels.append(teacher_preds[key])
            sources.append("teacher")
        elif key in labels_lookup:
            # Fallback to load_0_1
            soft_labels.append(labels_lookup[key])
            sources.append("fallback")
        else:
            # No label available
            soft_labels.append(np.nan)
            sources.append("missing")
    
    video_df["soft_label"] = soft_labels
    video_df["soft_label_source"] = sources
    
    # Also add load_0_1 for comparison
    video_df["load_0_1"] = video_df.apply(
        lambda row: labels_lookup.get((row["user_id"], row["task"]), np.nan),
        axis=1,
    )
    
    # Summary
    source_counts = video_df["soft_label_source"].value_counts()
    print("\nSoft label sources:")
    for source, count in source_counts.items():
        pct = 100 * count / len(video_df)
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    # Drop rows with no label
    before = len(video_df)
    video_df = video_df.dropna(subset=["soft_label"])
    after = len(video_df)
    if before > after:
        print(f"Dropped {before - after} windows with no label")
    
    return video_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate soft labels for video features using physiological teacher"
    )
    parser.add_argument(
        "--video-features",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "features.csv",
        help="Path to video features CSV",
    )
    parser.add_argument(
        "--physio-features",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "physio_features.csv",
        help="Path to physiological features CSV",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw" / "self_assessments_loadindex.csv",
        help="Path to self-assessment labels CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory containing teacher model artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "features_with_soft_labels.csv",
        help="Output path for features with soft labels",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=QUALITY_THRESHOLD,
        help=f"ECG quality threshold for using teacher (default: {QUALITY_THRESHOLD})",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOFT LABEL GENERATION")
    print("=" * 60)
    print(f"Video features: {args.video_features}")
    print(f"Physio features: {args.physio_features}")
    print(f"Model dir: {args.model_dir}")
    print(f"Quality threshold: {args.quality_threshold}")
    print()
    
    # Load artifacts
    model, scaler, imputer = load_teacher_artifacts(args.model_dir)
    
    # Load data
    video_df = load_video_features(args.video_features)
    physio_df = load_physio_features(args.physio_features)
    labels_df = load_labels(args.labels)
    
    # Generate soft labels
    result_df = generate_soft_labels(
        video_df, physio_df, labels_df,
        model, scaler, imputer,
        quality_threshold=args.quality_threshold,
    )
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(result_df)} windows to {args.output}")
    
    # Print correlation between soft_label and load_0_1
    if "soft_label" in result_df.columns and "load_0_1" in result_df.columns:
        corr = result_df["soft_label"].corr(result_df["load_0_1"])
        print(f"\nCorrelation between soft_label and load_0_1: {corr:.3f}")
    
    print("\n" + "=" * 60)
    print("SOFT LABEL GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
