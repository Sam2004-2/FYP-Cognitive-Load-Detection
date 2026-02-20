#!/usr/bin/env python3
"""
Extract physiological features from all participants in the StressID dataset.

Processes ECG, EDA, and RR signals to extract HRV, skin conductance, and
respiratory features. Outputs a CSV aligned to video feature windows.

Usage:
    python scripts/extract_all_physio_features.py
    python scripts/extract_all_physio_features.py --physio-dir ../Physiological --output data/processed/physio_features.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add Machine Learning root to path for imports
ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "src"))

from cle.extract.physio_features import (
    SAMPLING_RATE,
    WINDOW_LENGTH_S,
    WINDOW_STEP_S,
    extract_physio_features_for_participant,
    get_physio_feature_names,
)


def find_participants(physio_dir: Path) -> list[str]:
    """
    Find all participant directories in the physiological data folder.
    
    Args:
        physio_dir: Path to the Physiological data directory
        
    Returns:
        List of participant IDs (directory names)
    """
    participants = []
    for item in physio_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if directory contains physiological data files
            txt_files = list(item.glob("*.txt"))
            if txt_files:
                participants.append(item.name)
    return sorted(participants)


def extract_all_features(
    physio_dir: Path,
    output_path: Path,
    window_length_s: float = WINDOW_LENGTH_S,
    window_step_s: float = WINDOW_STEP_S,
    sampling_rate: int = SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Extract physiological features for all participants.
    
    Args:
        physio_dir: Path to the Physiological data directory
        output_path: Path to save the output CSV
        window_length_s: Window length in seconds
        window_step_s: Window step in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with all extracted features
    """
    participants = find_participants(physio_dir)
    print(f"Found {len(participants)} participants in {physio_dir}")
    
    all_features = []
    
    for participant_id in tqdm(participants, desc="Extracting physiological features"):
        participant_dir = physio_dir / participant_id
        
        try:
            df = extract_physio_features_for_participant(
                participant_dir=participant_dir,
                participant_id=participant_id,
                window_length_s=window_length_s,
                window_step_s=window_step_s,
                sampling_rate=sampling_rate,
            )
            
            if not df.empty:
                all_features.append(df)
                
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
            continue
    
    if not all_features:
        print("No features extracted!")
        return pd.DataFrame()
    
    # Combine all participants
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Reorder columns for clarity
    meta_cols = ["user_id", "task", "t_start_s", "t_end_s"]
    feature_cols = get_physio_feature_names()
    ordered_cols = meta_cols + [c for c in feature_cols if c in combined_df.columns]
    combined_df = combined_df[ordered_cols]
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved {len(combined_df)} windows to {output_path}")
    
    # Print summary statistics
    print_summary(combined_df)
    
    return combined_df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the extracted features."""
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal windows: {len(df)}")
    print(f"Participants: {df['user_id'].nunique()}")
    print(f"Tasks: {df['task'].nunique()}")
    
    # Quality statistics
    if "ecg_quality" in df.columns:
        quality = df["ecg_quality"]
        print(f"\nECG Quality:")
        print(f"  Mean: {quality.mean():.3f}")
        print(f"  Std:  {quality.std():.3f}")
        print(f"  Min:  {quality.min():.3f}")
        print(f"  Max:  {quality.max():.3f}")
        
        high_quality_pct = (quality >= 0.5).mean() * 100
        print(f"  Windows with quality >= 0.5: {high_quality_pct:.1f}%")
    
    # Feature availability
    feature_cols = get_physio_feature_names()
    print("\nFeature availability (non-NaN %):")
    for col in feature_cols:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col}: {pct:.1f}%")
    
    # Task distribution
    print("\nWindows per task:")
    task_counts = df["task"].value_counts()
    for task, count in task_counts.items():
        print(f"  {task}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract physiological features from StressID data"
    )
    parser.add_argument(
        "--physio-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "Physiological",
        help="Path to the Physiological data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "physio_features.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=WINDOW_LENGTH_S,
        help=f"Window length in seconds (default: {WINDOW_LENGTH_S})",
    )
    parser.add_argument(
        "--window-step",
        type=float,
        default=WINDOW_STEP_S,
        help=f"Window step in seconds (default: {WINDOW_STEP_S})",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=SAMPLING_RATE,
        help=f"Sampling rate in Hz (default: {SAMPLING_RATE})",
    )
    
    args = parser.parse_args()
    
    if not args.physio_dir.exists():
        print(f"Error: Physiological data directory not found: {args.physio_dir}")
        sys.exit(1)
    
    extract_all_features(
        physio_dir=args.physio_dir,
        output_path=args.output,
        window_length_s=args.window_length,
        window_step_s=args.window_step,
        sampling_rate=args.sampling_rate,
    )


if __name__ == "__main__":
    main()
