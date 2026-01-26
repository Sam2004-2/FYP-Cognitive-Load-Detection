"""
Public API for CLE.

High-level functions for loading models and making predictions.
Supports binary classification (HIGH/LOW) with trend detection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.cle.extract.features import compute_window_features
from src.cle.logging_setup import get_logger
from src.cle.predict.trend import TrendDetector, Trend
from src.cle.utils.io import load_json, load_model_artifact

logger = get_logger(__name__)

# Global trend detector for stateful trend tracking
_trend_detector: Optional[TrendDetector] = None


def load_model(models_dir: str) -> Dict:
    """
    Load trained model artifacts.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with:
            - model: Trained model
            - scaler: Fitted scaler
            - imputer: Fitted imputer (optional)
            - feature_spec: Feature specification

    Raises:
        FileNotFoundError: If required artifacts are missing
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Load required artifacts
    model = load_model_artifact(models_dir / "model.bin")
    scaler = load_model_artifact(models_dir / "scaler.bin")
    feature_spec = load_json(models_dir / "feature_spec.json")

    # Load optional artifacts
    imputer = None
    imputer_path = models_dir / "imputer.bin"
    if imputer_path.exists():
        imputer = load_model_artifact(imputer_path)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_spec": feature_spec,
    }

    logger.info(
        f"Loaded model artifacts from {models_dir} "
        f"({len(feature_spec['features'])} features)"
    )

    return artifacts


def init_trend_detector(window: int = 5, threshold: float = 0.1) -> None:
    """Initialize the global trend detector."""
    global _trend_detector
    _trend_detector = TrendDetector(window=window, threshold=threshold)
    logger.info(f"Initialized trend detector (window={window}, threshold={threshold})")


def reset_trend_detector() -> None:
    """Reset the trend detector state."""
    global _trend_detector
    if _trend_detector:
        _trend_detector.reset()
        logger.info("Reset trend detector")


def get_trend_detector() -> Optional[TrendDetector]:
    """Get the global trend detector instance."""
    return _trend_detector


def predict_window(
    features: Union[Dict, np.ndarray, List],
    artifacts: Dict,
    update_trend: bool = True,
) -> Dict:
    """
    Predict cognitive load for a window of features.

    Returns binary classification (HIGH/LOW) with confidence and trend.

    Args:
        features: Window features as:
            - Dict: feature_name -> value mapping
            - np.ndarray: array of feature values
            - List: list of feature values
        artifacts: Model artifacts from load_model()
        update_trend: Whether to update the trend detector

    Returns:
        Dictionary with:
            - level: "HIGH" or "LOW"
            - confidence: Prediction confidence (0-1)
            - trend: "INCREASING", "DECREASING", "STABLE", or "INSUFFICIENT_DATA"
            - raw_score: Continuous prediction [0-1]

    Raises:
        ValueError: If features don't match expected format
    """
    global _trend_detector

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    imputer = artifacts.get("imputer")
    feature_names = artifacts["feature_spec"]["features"]

    # Convert features to array
    if isinstance(features, dict):
        feature_array = np.array([features.get(name, 0.0) for name in feature_names])
    elif isinstance(features, (list, tuple)):
        feature_array = np.array(features)
    elif isinstance(features, np.ndarray):
        feature_array = features
    else:
        raise ValueError(f"Unsupported feature type: {type(features)}")

    # Validate shape
    if feature_array.shape[-1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch: got {feature_array.shape[-1]}, "
            f"expected {len(feature_names)}"
        )

    # Ensure 2D shape
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)

    # Handle NaN values with imputer or fallback
    if np.any(np.isnan(feature_array)):
        if imputer is not None:
            feature_array = imputer.transform(feature_array)
        else:
            logger.warning("Found NaN values in features, replacing with zeros")
            feature_array = np.nan_to_num(feature_array, nan=0.0)

    # Scale features
    features_scaled = scaler.transform(feature_array)

    # Predict probability
    proba = model.predict_proba(features_scaled)[0, 1]  # P(HIGH)
    raw_score = float(proba)

    # Binary classification
    level = "HIGH" if proba >= 0.5 else "LOW"

    # Confidence: distance from decision boundary
    confidence = abs(proba - 0.5) * 2.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Update trend detector
    trend = Trend.INSUFFICIENT_DATA
    if update_trend and _trend_detector is not None:
        _trend_detector.add(raw_score)
        trend = _trend_detector.get_trend()
    elif _trend_detector is None:
        # Auto-initialize if not done
        init_trend_detector()
        if _trend_detector is not None:
            _trend_detector.add(raw_score)
            trend = _trend_detector.get_trend()

    return {
        "level": level,
        "confidence": confidence,
        "trend": trend.value,
        "raw_score": raw_score,
    }


def predict_binary(
    features: Union[Dict, np.ndarray, List],
    artifacts: Dict,
) -> Tuple[str, float]:
    """
    Simple binary prediction without trend tracking.

    Args:
        features: Window features
        artifacts: Model artifacts

    Returns:
        Tuple of (level, confidence) where level is "HIGH" or "LOW"
    """
    result = predict_window(features, artifacts, update_trend=False)
    return result["level"], result["confidence"]


def extract_features_from_window(
    frame_data: List[Dict],
    config: Dict,
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Extract window-level features from list of per-frame features.

    Args:
        frame_data: List of per-frame feature dictionaries
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Dictionary of window-level features
    """
    features = compute_window_features(frame_data, config, fps)
    return features


def predict_from_frame_data(
    frame_data: List[Dict],
    artifacts: Dict,
    config: Dict,
    fps: float = 30.0,
) -> Dict:
    """
    End-to-end prediction from per-frame features.

    Combines feature extraction and prediction in one call.

    Args:
        frame_data: List of per-frame feature dictionaries
        artifacts: Model artifacts from load_model()
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Prediction result dictionary with level, confidence, trend, raw_score
    """
    window_features = extract_features_from_window(frame_data, config, fps)
    return predict_window(window_features, artifacts)
