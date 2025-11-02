"""
Window-level feature computation.

Computes aggregated features over time windows: TEPR, blinks, PERCLOS, etc.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def compute_tepr_features(
    pupil_series: np.ndarray, fps: float, baseline_s: float = 10.0, min_baseline_samples: int = 150
) -> Dict[str, float]:
    """
    Compute Task-Evoked Pupillary Response (TEPR) features.

    TEPR measures pupil dilation relative to a baseline period.

    Args:
        pupil_series: Array of pupil diameter values (normalized)
        fps: Frames per second
        baseline_s: Baseline window duration in seconds
        min_baseline_samples: Minimum samples needed for valid baseline

    Returns:
        Dictionary with TEPR features:
            - tepr_delta_mean: Mean pupil change from baseline
            - tepr_delta_peak: Peak pupil change from baseline
            - tepr_auc: Area under curve (integral of change)
            - tepr_baseline: Baseline pupil value
    """
    if len(pupil_series) == 0:
        return {
            "tepr_delta_mean": 0.0,
            "tepr_delta_peak": 0.0,
            "tepr_auc": 0.0,
            "tepr_baseline": 0.0,
        }

    # Filter out invalid values (zeros)
    valid_pupil = pupil_series[pupil_series > 0]

    if len(valid_pupil) == 0:
        return {
            "tepr_delta_mean": 0.0,
            "tepr_delta_peak": 0.0,
            "tepr_auc": 0.0,
            "tepr_baseline": 0.0,
        }

    # Compute baseline from first baseline_s seconds
    baseline_frames = int(baseline_s * fps)

    if baseline_frames < min_baseline_samples or len(valid_pupil) < baseline_frames:
        # Not enough data for baseline, use median of all data
        baseline = np.median(valid_pupil)
    else:
        # Use median of first baseline_s seconds
        baseline_data = valid_pupil[:baseline_frames]
        baseline = np.median(baseline_data)

    # Compute changes from baseline
    delta = valid_pupil - baseline

    # TEPR features
    tepr_delta_mean = float(np.mean(delta))
    tepr_delta_peak = float(np.max(delta))

    # Area under curve (integral using trapezoidal rule)
    # Normalize by time to make it comparable across different window lengths
    tepr_auc = float(np.trapz(delta) / len(delta))

    return {
        "tepr_delta_mean": tepr_delta_mean,
        "tepr_delta_peak": tepr_delta_peak,
        "tepr_auc": tepr_auc,
        "tepr_baseline": float(baseline),
    }


def detect_blinks(
    ear_series: np.ndarray,
    fps: float,
    ear_threshold: float = 0.21,
    min_blink_ms: int = 120,
    max_blink_ms: int = 400,
) -> List[Tuple[int, int]]:
    """
    Detect blinks from Eye Aspect Ratio (EAR) time series.

    Uses threshold-based state machine to detect blinks.

    Args:
        ear_series: Array of EAR values
        fps: Frames per second
        ear_threshold: EAR threshold for blink detection
        min_blink_ms: Minimum blink duration in milliseconds
        max_blink_ms: Maximum blink duration in milliseconds

    Returns:
        List of (start_idx, end_idx) tuples for each blink
    """
    if len(ear_series) == 0:
        return []

    min_blink_frames = int((min_blink_ms / 1000.0) * fps)
    max_blink_frames = int((max_blink_ms / 1000.0) * fps)

    blinks = []
    in_blink = False
    blink_start = 0

    for i, ear in enumerate(ear_series):
        if not in_blink:
            # Check for blink start
            if ear < ear_threshold:
                in_blink = True
                blink_start = i
        else:
            # Check for blink end
            if ear >= ear_threshold:
                blink_duration = i - blink_start

                # Validate blink duration
                if min_blink_frames <= blink_duration <= max_blink_frames:
                    blinks.append((blink_start, i))

                in_blink = False

    return blinks


def compute_blink_features(
    ear_series: np.ndarray, fps: float, config: Dict
) -> Dict[str, float]:
    """
    Compute blink-related features.

    Args:
        ear_series: Array of EAR values
        fps: Frames per second
        config: Configuration dictionary with blink parameters

    Returns:
        Dictionary with blink features:
            - blink_rate: Blinks per minute
            - blink_count: Total number of blinks
            - mean_blink_duration: Mean blink duration in ms
    """
    if len(ear_series) == 0:
        return {
            "blink_rate": 0.0,
            "blink_count": 0.0,
            "mean_blink_duration": 0.0,
        }

    # Detect blinks
    blinks = detect_blinks(
        ear_series,
        fps,
        ear_threshold=config.get("blink.ear_thresh", 0.21),
        min_blink_ms=config.get("blink.min_blink_ms", 120),
        max_blink_ms=config.get("blink.max_blink_ms", 400),
    )

    # Compute blink rate (blinks per minute)
    window_duration_min = len(ear_series) / fps / 60.0
    blink_rate = len(blinks) / window_duration_min if window_duration_min > 0 else 0.0

    # Compute mean blink duration
    if blinks:
        blink_durations = [(end - start) / fps * 1000 for start, end in blinks]
        mean_blink_duration = float(np.mean(blink_durations))
    else:
        mean_blink_duration = 0.0

    return {
        "blink_rate": blink_rate,
        "blink_count": float(len(blinks)),
        "mean_blink_duration": mean_blink_duration,
    }


def compute_perclos(ear_series: np.ndarray, ear_threshold: float = 0.21) -> float:
    """
    Compute PERCLOS (Percentage of Eye Closure).

    PERCLOS = percentage of time EAR is below threshold.

    Args:
        ear_series: Array of EAR values
        ear_threshold: EAR threshold

    Returns:
        PERCLOS value (0-1)
    """
    if len(ear_series) == 0:
        return 0.0

    closed_frames = np.sum(ear_series < ear_threshold)
    perclos = closed_frames / len(ear_series)

    return float(perclos)


def compute_control_features(
    brightness_series: np.ndarray, ear_series: np.ndarray, config: Dict
) -> Dict[str, float]:
    """
    Compute control features (brightness, PERCLOS).

    Args:
        brightness_series: Array of brightness values
        ear_series: Array of EAR values
        config: Configuration dictionary

    Returns:
        Dictionary with control features:
            - mean_brightness: Mean face region brightness
            - std_brightness: Standard deviation of brightness
            - perclos: Percentage of eye closure
    """
    # Brightness statistics
    if len(brightness_series) > 0:
        mean_brightness = float(np.mean(brightness_series))
        std_brightness = float(np.std(brightness_series))
    else:
        mean_brightness = 0.0
        std_brightness = 0.0

    # PERCLOS
    ear_threshold = config.get("blink.ear_thresh", 0.21)
    perclos = compute_perclos(ear_series, ear_threshold)

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "perclos": perclos,
    }


def compute_window_features(window_data: List[Dict], config: Dict, fps: float) -> Dict[str, float]:
    """
    Compute all window-level features.

    This is the main aggregator that orchestrates feature computation.

    Args:
        window_data: List of per-frame feature dictionaries
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Dictionary with all window features in fixed order
    """
    features = {}

    # Extract time series from window data
    valid_frames = [f for f in window_data if f.get("valid", False)]

    if not valid_frames:
        # Return zero features if no valid frames
        logger.warning("No valid frames in window, returning zero features")
        return get_zero_features(config)

    # Extract arrays
    ear_series = np.array([f["ear_mean"] for f in valid_frames])
    pupil_series = np.array([f["pupil_mean"] for f in valid_frames])
    brightness_series = np.array([f["brightness"] for f in valid_frames])

    # Compute TEPR features (if enabled)
    if config.get("features_enabled.tepr", True):
        tepr_features = compute_tepr_features(
            pupil_series,
            fps,
            baseline_s=config.get("tepr.baseline_s", 10.0),
            min_baseline_samples=config.get("tepr.min_baseline_samples", 150),
        )
        features.update(tepr_features)

    # Compute blink features (if enabled)
    if config.get("features_enabled.blinks", True):
        blink_features = compute_blink_features(ear_series, fps, config)
        features.update(blink_features)

    # Compute control features (if enabled)
    if config.get("features_enabled.perclos", True) or config.get("features_enabled.brightness", True):
        control_features = compute_control_features(brightness_series, ear_series, config)

        if config.get("features_enabled.brightness", True):
            features["mean_brightness"] = control_features["mean_brightness"]
            features["std_brightness"] = control_features["std_brightness"]

        if config.get("features_enabled.perclos", True):
            features["perclos"] = control_features["perclos"]

    # Add quality metrics
    qualities = [f["quality"] for f in valid_frames]
    features["mean_quality"] = float(np.mean(qualities))
    features["valid_frame_ratio"] = len(valid_frames) / len(window_data)

    return features


def get_zero_features(config: Dict) -> Dict[str, float]:
    """
    Get zero-valued features dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with all features set to zero
    """
    features = {}

    if config.get("features_enabled.tepr", True):
        features.update({
            "tepr_delta_mean": 0.0,
            "tepr_delta_peak": 0.0,
            "tepr_auc": 0.0,
            "tepr_baseline": 0.0,
        })

    if config.get("features_enabled.blinks", True):
        features.update({
            "blink_rate": 0.0,
            "blink_count": 0.0,
            "mean_blink_duration": 0.0,
        })

    if config.get("features_enabled.brightness", True):
        features.update({
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
        })

    if config.get("features_enabled.perclos", True):
        features["perclos"] = 0.0

    features.update({
        "mean_quality": 0.0,
        "valid_frame_ratio": 0.0,
    })

    return features


def get_feature_names(config: Dict) -> List[str]:
    """
    Get ordered list of feature names based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Ordered list of feature names
    """
    feature_names = []

    if config.get("features_enabled.tepr", True):
        feature_names.extend([
            "tepr_delta_mean",
            "tepr_delta_peak",
            "tepr_auc",
            "tepr_baseline",
        ])

    if config.get("features_enabled.blinks", True):
        feature_names.extend([
            "blink_rate",
            "blink_count",
            "mean_blink_duration",
        ])

    if config.get("features_enabled.brightness", True):
        feature_names.extend([
            "mean_brightness",
            "std_brightness",
        ])

    if config.get("features_enabled.perclos", True):
        feature_names.append("perclos")

    feature_names.extend([
        "mean_quality",
        "valid_frame_ratio",
    ])

    return feature_names

