"""
Process CLARE dataset gaze data and align with cognitive load labels.

CLARE dataset provides eye tracker data (not video), so we extract
compatible features from the gaze data that match our feature set.

Usage:
    python scripts/process_clare_data.py --clare_dir data/CLARE --output data/processed/clare_features.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# CLARE sampling rate (approximate, based on 9 min sessions)
CLARE_SAMPLING_RATE = 1000  # Hz (assumed from row counts)
WINDOW_LENGTH_S = 10.0  # Match label frequency (every 10 seconds)


def load_clare_gaze_session(gaze_file: Path) -> pd.DataFrame:
    """
    Load CLARE gaze data file.

    Args:
        gaze_file: Path to gaze CSV file

    Returns:
        DataFrame with gaze data
    """
    logger.info(f"Loading gaze data: {gaze_file}")

    # Load CSV - many columns have mixed types or missing data
    df = pd.read_csv(gaze_file, low_memory=False)

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns[:5])}...")

    return df


def load_clare_labels(label_file: Path) -> pd.DataFrame:
    """
    Load CLARE cognitive load labels.

    Labels are organized as:
    - Rows: 10-second windows (55 rows = 550s = ~9 min)
    - Columns: level_0, level_1, level_2, level_3 (4 sessions)

    Args:
        label_file: Path to label CSV file

    Returns:
        DataFrame with labels for all sessions
    """
    logger.info(f"Loading labels: {label_file}")

    df = pd.read_csv(label_file)

    logger.info(f"Loaded labels for {len(df)} windows across {len(df.columns)} sessions")

    return df


def detect_blinks_from_binary(
    blink_series: pd.Series, timestamps: pd.Series
) -> List[Tuple[float, float]]:
    """
    Detect blink events from binary blink detection signal.

    Args:
        blink_series: Binary blink detection (1 = blink, 0 = no blink)
        timestamps: Timestamps in seconds

    Returns:
        List of (start_time, end_time) tuples for each blink
    """
    # Convert to numpy arrays
    blinks = blink_series.fillna(0).astype(float).values
    times = timestamps.values

    blink_events = []
    in_blink = False
    blink_start = None

    for i, (is_blink, t) in enumerate(zip(blinks, times)):
        if not in_blink and is_blink > 0.5:
            # Blink start
            in_blink = True
            blink_start = t
        elif in_blink and is_blink < 0.5:
            # Blink end
            blink_end = t
            blink_duration_ms = (blink_end - blink_start) * 1000

            # Filter reasonable blinks (100-500ms)
            if 100 <= blink_duration_ms <= 500:
                blink_events.append((blink_start, blink_end))

            in_blink = False
            blink_start = None

    return blink_events


def extract_window_features_clare(
    gaze_data: pd.DataFrame,
    window_start_s: float,
    window_end_s: float,
) -> Dict[str, float]:
    """
    Extract features from CLARE gaze data for a single window.

    Note: CLARE provides eye tracker data, not video/facial landmarks,
    so we can only extract features available from the eye tracker:
    - Blink features (from binary blink detection)
    - Pupil features (if needed, though TEPR disabled)

    We CANNOT extract:
    - EAR (needs facial landmarks)
    - Brightness (needs video frames)
    - PERCLOS (needs EAR)

    Args:
        gaze_data: CLARE gaze DataFrame
        window_start_s: Window start time (seconds)
        window_end_s: Window end time (seconds)

    Returns:
        Dictionary of features (partial - only what's available from eye tracker)
    """
    # Filter data to window
    window_data = gaze_data[
        (gaze_data["Timestamp"] >= window_start_s)
        & (gaze_data["Timestamp"] < window_end_s)
    ]

    if len(window_data) == 0:
        logger.warning(f"No data in window [{window_start_s:.1f}, {window_end_s:.1f}]")
        return get_zero_features_clare()

    # Extract blink events
    blink_events = detect_blinks_from_binary(
        window_data["Blink detected (binary)"], window_data["Timestamp"]
    )

    # Compute blink features
    window_duration_min = (window_end_s - window_start_s) / 60.0
    blink_rate = len(blink_events) / window_duration_min if window_duration_min > 0 else 0.0
    blink_count = len(blink_events)

    if blink_events:
        blink_durations_ms = [
            (end - start) * 1000 for start, end in blink_events
        ]
        mean_blink_duration = float(np.mean(blink_durations_ms))
    else:
        mean_blink_duration = 0.0

    # Pupil features (optional - TEPR disabled, but we could include for completeness)
    pupil_left = pd.to_numeric(window_data["ET_PupilLeft"], errors="coerce")
    pupil_right = pd.to_numeric(window_data["ET_PupilRight"], errors="coerce")

    valid_pupil_left = pupil_left.dropna()
    valid_pupil_right = pupil_right.dropna()

    if len(valid_pupil_left) > 0 and len(valid_pupil_right) > 0:
        pupil_mean = (valid_pupil_left.mean() + valid_pupil_right.mean()) / 2.0
        pupil_std = (valid_pupil_left.std() + valid_pupil_right.std()) / 2.0
    else:
        pupil_mean = 0.0
        pupil_std = 0.0

    # Quality metrics
    total_samples = len(window_data)
    valid_samples = window_data["Blink detected (binary)"].notna().sum()
    valid_frame_ratio = valid_samples / total_samples if total_samples > 0 else 0.0

    # NOTE: We CANNOT compute these from eye tracker data:
    # - ear_std (needs EAR from facial landmarks)
    # - perclos (needs EAR from facial landmarks)
    # - mean_brightness, std_brightness (needs video frames)
    # These will be set to placeholder values

    features = {
        "blink_rate": blink_rate,
        "blink_count": float(blink_count),
        "mean_blink_duration": mean_blink_duration,
        "ear_std": 0.0,  # NOT AVAILABLE from eye tracker
        "mean_brightness": 128.0,  # PLACEHOLDER (neutral brightness)
        "std_brightness": 0.0,  # NOT AVAILABLE from eye tracker
        "perclos": 0.0,  # NOT AVAILABLE from eye tracker
        "mean_quality": valid_frame_ratio,  # Use valid data ratio as quality proxy
        "valid_frame_ratio": valid_frame_ratio,
        # Additional info for debugging
        "pupil_mean": pupil_mean,
        "pupil_std": pupil_std,
    }

    return features


def get_zero_features_clare() -> Dict[str, float]:
    """Get zero-valued features for invalid windows."""
    return {
        "blink_rate": 0.0,
        "blink_count": 0.0,
        "mean_blink_duration": 0.0,
        "ear_std": 0.0,
        "mean_brightness": 128.0,
        "std_brightness": 0.0,
        "perclos": 0.0,
        "mean_quality": 0.0,
        "valid_frame_ratio": 0.0,
        "pupil_mean": 0.0,
        "pupil_std": 0.0,
    }


def process_participant(
    participant_id: str,
    clare_dir: Path,
) -> pd.DataFrame:
    """
    Process all sessions for a participant.

    Args:
        participant_id: Participant ID (e.g., "1026")
        clare_dir: Root CLARE data directory

    Returns:
        DataFrame with features and labels for all windows
    """
    logger.info(f"Processing participant {participant_id}")

    # Load labels
    label_file = clare_dir / "label" / f"{participant_id}.csv"
    labels_df = load_clare_labels(label_file)

    all_windows = []

    # Process each session (level_0, level_1, level_2, level_3)
    for session_idx in range(4):
        session_name = f"level_{session_idx}"
        logger.info(f"  Processing session {session_idx} ({session_name})")

        # Load gaze data for this session
        gaze_file = clare_dir / "gaze" / participant_id / f"gaze_data_experiment_{session_idx}.csv"

        if not gaze_file.exists():
            logger.warning(f"  Gaze file not found: {gaze_file}")
            continue

        gaze_data = load_clare_gaze_session(gaze_file)

        # Get labels for this session
        session_labels = labels_df[session_name].values

        # Process each 10-second window
        num_windows = len(session_labels)

        for window_idx in range(num_windows):
            window_start_s = window_idx * WINDOW_LENGTH_S
            window_end_s = (window_idx + 1) * WINDOW_LENGTH_S

            # Get label for this window
            cognitive_load = session_labels[window_idx]

            # Skip if label is NaN
            if pd.isna(cognitive_load):
                continue

            # Extract features
            features = extract_window_features_clare(
                gaze_data,
                window_start_s,
                window_end_s,
            )

            # Add metadata
            window_data = {
                "user_id": participant_id,
                "session": session_idx,
                "window": window_idx,
                "t_start_s": window_start_s,
                "t_end_s": window_end_s,
                "cognitive_load": cognitive_load,
                "label": 1 if cognitive_load >= 7 else 0,  # Binary: high (>=7) vs low (<7)
                **features,
            }

            all_windows.append(window_data)

        logger.info(f"  Extracted {len(session_labels)} windows from session {session_idx}")

    result_df = pd.DataFrame(all_windows)
    logger.info(f"Participant {participant_id}: {len(result_df)} total windows")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Process CLARE dataset for cognitive load estimation")
    parser.add_argument(
        "--clare_dir",
        type=str,
        default="data/CLARE",
        help="Root directory of CLARE dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clare_features.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--participants",
        type=str,
        nargs="+",
        default=None,
        help="Specific participant IDs to process (default: all)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    clare_dir = Path(args.clare_dir)

    if not clare_dir.exists():
        logger.error(f"CLARE directory not found: {clare_dir}")
        return

    # Find participants
    if args.participants:
        participants = args.participants
    else:
        # Auto-detect from label directory
        label_dir = clare_dir / "label"
        participants = [f.stem for f in label_dir.glob("*.csv")]

    logger.info(f"Found {len(participants)} participants: {participants}")

    # Process all participants
    all_data = []

    for participant_id in participants:
        try:
            participant_df = process_participant(participant_id, clare_dir)
            all_data.append(participant_df)
        except Exception as e:
            logger.error(f"Error processing participant {participant_id}: {e}", exc_info=True)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(combined_df)} windows to {output_path}")

        # Print summary statistics
        logger.info("\n=== CLARE Dataset Summary ===")
        logger.info(f"Total participants: {combined_df['user_id'].nunique()}")
        logger.info(f"Total windows: {len(combined_df)}")
        logger.info(f"Cognitive load range: {combined_df['cognitive_load'].min():.1f} - {combined_df['cognitive_load'].max():.1f}")
        logger.info(f"High load (>=7) samples: {(combined_df['label'] == 1).sum()}")
        logger.info(f"Low load (<7) samples: {(combined_df['label'] == 0).sum()}")
        logger.info(f"Mean blink rate: {combined_df['blink_rate'].mean():.2f} blinks/min")
        logger.info(f"Valid frame ratio: {combined_df['valid_frame_ratio'].mean():.2%}")

    else:
        logger.error("No data processed!")


if __name__ == "__main__":
    main()

