"""
Windowing logic for temporal feature aggregation.

Implements sliding windows for offline and real-time processing.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def sliding_window_indices(
    n_frames: int, fps: float, length_s: float, step_s: float
) -> List[Tuple[int, int, float, float]]:
    """
    Generate sliding window indices for frame sequence.

    Args:
        n_frames: Total number of frames
        fps: Frames per second
        length_s: Window length in seconds
        step_s: Step size in seconds

    Returns:
        List of (start_idx, end_idx, start_time_s, end_time_s) tuples
    """
    length_frames = int(length_s * fps)
    step_frames = int(step_s * fps)

    windows = []
    start_idx = 0

    while start_idx + length_frames <= n_frames:
        end_idx = start_idx + length_frames
        start_time_s = start_idx / fps
        end_time_s = end_idx / fps

        windows.append((start_idx, end_idx, start_time_s, end_time_s))
        start_idx += step_frames

    logger.debug(
        f"Generated {len(windows)} windows "
        f"(length={length_s}s, step={step_s}s, fps={fps})"
    )

    return windows


def validate_window_quality(
    frame_features: List[Dict], max_bad_ratio: float = 0.2
) -> Tuple[bool, float]:
    """
    Validate window quality based on fraction of bad frames.

    Args:
        frame_features: List of per-frame feature dictionaries
        max_bad_ratio: Maximum allowed ratio of bad frames

    Returns:
        Tuple of (is_valid, bad_ratio)
    """
    if not frame_features:
        return False, 1.0

    bad_count = sum(1 for f in frame_features if not f.get("valid", False))
    bad_ratio = bad_count / len(frame_features)

    is_valid = bad_ratio <= max_bad_ratio

    return is_valid, bad_ratio


def interpolate_gaps(frame_features: List[Dict], max_gap: int = 3) -> List[Dict]:
    """
    Interpolate small gaps in frame features.

    Linear interpolation for gaps up to max_gap frames.

    Args:
        frame_features: List of per-frame feature dictionaries
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        List of frame features with gaps interpolated
    """
    if not frame_features:
        return frame_features

    # Find valid frames
    valid_indices = [i for i, f in enumerate(frame_features) if f.get("valid", False)]

    if not valid_indices:
        return frame_features

    # Copy features for modification
    interpolated = [f.copy() for f in frame_features]

    # Interpolate gaps
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]
        gap = idx2 - idx1 - 1

        if 0 < gap <= max_gap:
            # Interpolate between idx1 and idx2
            f1 = frame_features[idx1]
            f2 = frame_features[idx2]

            # Interpolate numeric features
            numeric_keys = [
                "ear_left",
                "ear_right",
                "ear_mean",
                "brightness",
                "eye_center_x",
                "eye_center_y",
                "mouth_mar",
                "roll",
                "pitch",
                "yaw",
            ]

            for gap_idx in range(1, gap + 1):
                interp_idx = idx1 + gap_idx
                alpha = gap_idx / (gap + 1)  # Interpolation weight

                for key in numeric_keys:
                    if key in f1 and key in f2:
                        interpolated[interp_idx][key] = (1 - alpha) * f1[key] + alpha * f2[key]

                interpolated[interp_idx]["valid"] = True
                interpolated[interp_idx]["quality"] = min(f1["quality"], f2["quality"])

    return interpolated
