"""
Window-level feature computation.

Computes aggregated features over time windows: blinks, PERCLOS, etc.
Note: TEPR (pupil-based) features have been removed - focus on EAR-based features.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def _get_config_value(config: Union[Dict, Any], key: str, default: Any = None) -> Any:
    """
    Get configuration value supporting both Config objects and plain dicts.

    Supports dot notation for nested keys (e.g., 'blink.ear_thresh').

    Args:
        config: Configuration (Config object or dict)
        key: Configuration key (supports dot notation)
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    # Try as plain dict first (handles both dict and Config.to_dict())
    if isinstance(config, dict):
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    # If config has a get method that supports dot notation (Config object)
    if hasattr(config, 'get') and callable(config.get):
        try:
            return config.get(key, default)
        except (AttributeError, TypeError):
            pass
    
    return default


# TEPR functions removed - pupil-based features are no longer used
# Feature set now focuses on EAR-based (Eye Aspect Ratio) measurements which are
# more robust and don't require calibration or special lighting conditions


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
            - ear_std: Standard deviation of EAR (eye openness variability)
    """
    if len(ear_series) == 0:
        return {
            "blink_rate": 0.0,
            "blink_count": 0.0,
            "mean_blink_duration": 0.0,
            "ear_std": 0.0,
        }

    # Detect blinks
    blinks = detect_blinks(
        ear_series,
        fps,
        ear_threshold=_get_config_value(config, "blink.ear_thresh", 0.21),
        min_blink_ms=_get_config_value(config, "blink.min_blink_ms", 120),
        max_blink_ms=_get_config_value(config, "blink.max_blink_ms", 400),
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

    # Compute EAR variability (std dev)
    # Filter out zeros (invalid frames) before computing std
    valid_ear = ear_series[ear_series > 0]
    ear_std = float(np.std(valid_ear)) if len(valid_ear) > 0 else 0.0

    return {
        "blink_rate": blink_rate,
        "blink_count": float(len(blinks)),
        "mean_blink_duration": mean_blink_duration,
        "ear_std": ear_std,
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
    ear_threshold = _get_config_value(config, "blink.ear_thresh", 0.21)
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
        # Return NaN features if no valid frames (distinguishes from real zeros)
        logger.warning("No valid frames in window, returning NaN features")
        return get_zero_features(config)

    # Extract arrays
    ear_series = np.array([f["ear_mean"] for f in valid_frames])
    brightness_series = np.array([f["brightness"] for f in valid_frames])

    # TEPR features removed - no longer computing pupil-based features

    # Compute blink features (if enabled)
    if _get_config_value(config, "features_enabled.blinks", True):
        blink_features = compute_blink_features(ear_series, fps, config)
        features.update(blink_features)

    # Compute control features (if enabled)
    if _get_config_value(config, "features_enabled.perclos", True) or _get_config_value(config, "features_enabled.brightness", True):
        control_features = compute_control_features(brightness_series, ear_series, config)

        if _get_config_value(config, "features_enabled.brightness", True):
            features["mean_brightness"] = control_features["mean_brightness"]
            features["std_brightness"] = control_features["std_brightness"]

        if _get_config_value(config, "features_enabled.perclos", True):
            features["perclos"] = control_features["perclos"]

    # Add quality metrics
    qualities = [f["quality"] for f in valid_frames]
    features["mean_quality"] = float(np.mean(qualities))
    features["valid_frame_ratio"] = len(valid_frames) / len(window_data)

    # Geometry + motion features (if enabled)
    if _get_config_value(config, "features_enabled.geometry", False):
        mouth_vals = np.array([f.get("mouth_mar", np.nan) for f in valid_frames], dtype=float)
        mouth_vals = mouth_vals[np.isfinite(mouth_vals)]
        if len(mouth_vals) > 0:
            features["mouth_open_mean"] = float(np.mean(mouth_vals))
            features["mouth_open_std"] = float(np.std(mouth_vals))
        else:
            features["mouth_open_mean"] = 0.0
            features["mouth_open_std"] = 0.0

        roll_vals = np.array([f.get("roll", np.nan) for f in valid_frames], dtype=float)
        roll_vals = roll_vals[np.isfinite(roll_vals)]
        features["roll_std"] = float(np.std(roll_vals)) if len(roll_vals) > 0 else 0.0

        pitch_vals = np.array([f.get("pitch", np.nan) for f in valid_frames], dtype=float)
        pitch_vals = pitch_vals[np.isfinite(pitch_vals)]
        features["pitch_std"] = float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0

        yaw_vals = np.array([f.get("yaw", np.nan) for f in valid_frames], dtype=float)
        yaw_vals = yaw_vals[np.isfinite(yaw_vals)]
        features["yaw_std"] = float(np.std(yaw_vals)) if len(yaw_vals) > 0 else 0.0

        # Motion: speed of eye-center movement for valid consecutive frames
        speeds: List[float] = []
        for prev, curr in zip(valid_frames, valid_frames[1:]):
            try:
                prev_idx = prev.get("frame_idx")
                curr_idx = curr.get("frame_idx")
                if prev_idx is not None and curr_idx is not None and curr_idx != prev_idx + 1:
                    continue
            except Exception:
                # If frame_idx is missing/non-numeric, fall back to sequential assumption.
                pass

            dx = float(curr.get("eye_center_x", 0.0) - prev.get("eye_center_x", 0.0))
            dy = float(curr.get("eye_center_y", 0.0) - prev.get("eye_center_y", 0.0))
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            speeds.append(float(np.sqrt(dx * dx + dy * dy) * fps))

        if speeds:
            sp = np.array(speeds, dtype=float)
            features["motion_mean"] = float(np.mean(sp))
            features["motion_std"] = float(np.std(sp))
        else:
            features["motion_mean"] = 0.0
            features["motion_std"] = 0.0

    return features


def get_zero_features(config: Union[Dict, Any]) -> Dict[str, float]:
    """
    Get NaN-valued features dictionary for invalid windows.

    Args:
        config: Configuration dictionary or Config object

    Returns:
        Dictionary with all features set to NaN
    """
    features = {}

    # TEPR features removed - no longer part of feature set

    if _get_config_value(config, "features_enabled.blinks", True):
        features.update({
            "blink_rate": np.nan,
            "blink_count": np.nan,
            "mean_blink_duration": np.nan,
            "ear_std": np.nan,
        })

    if _get_config_value(config, "features_enabled.brightness", True):
        features.update({
            "mean_brightness": np.nan,
            "std_brightness": np.nan,
        })

    if _get_config_value(config, "features_enabled.perclos", True):
        features["perclos"] = np.nan

    features.update({
        "mean_quality": np.nan,
        "valid_frame_ratio": np.nan,
    })

    if _get_config_value(config, "features_enabled.geometry", False):
        features.update({
            "mouth_open_mean": np.nan,
            "mouth_open_std": np.nan,
            "roll_std": np.nan,
            "pitch_std": np.nan,
            "yaw_std": np.nan,
            "motion_mean": np.nan,
            "motion_std": np.nan,
        })

    return features


def get_feature_names(config: Union[Dict, Any]) -> List[str]:
    """
    Get ordered list of **model** feature names based on configuration.

    This returns only the features used as model inputs.  Environmental
    confounds (brightness) and quality-control metrics (mean_quality,
    valid_frame_ratio) are excluded here -- they are still *computed* by
    ``compute_window_features`` for monitoring and filtering but are not
    used as predictive features.

    Args:
        config: Configuration dictionary or Config object

    Returns:
        Ordered list of feature names
    """
    feature_names: List[str] = []

    # TEPR features removed - no longer part of feature set

    if _get_config_value(config, "features_enabled.blinks", True):
        feature_names.extend([
            "blink_rate",
            "blink_count",
            "mean_blink_duration",
            "ear_std",
        ])

    # NOTE: brightness features (mean_brightness, std_brightness) are
    # intentionally excluded from the model feature set.  They capture
    # ambient lighting conditions, not cognitive load, and act as a
    # confound.  They are still computed for quality monitoring.

    if _get_config_value(config, "features_enabled.perclos", True):
        feature_names.append("perclos")

    # NOTE: mean_quality and valid_frame_ratio are quality-control
    # metrics, not cognitive load indicators.  Removed from model
    # features to avoid confounding.

    if _get_config_value(config, "features_enabled.geometry", False):
        feature_names.extend([
            "mouth_open_mean",
            "mouth_open_std",
            "roll_std",
            "pitch_std",
            "yaw_std",
            "motion_mean",
            "motion_std",
        ])

    return feature_names
