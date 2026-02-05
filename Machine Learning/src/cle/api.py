"""
Public API for CLE.

High-level functions for loading models and making predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.cle.extract.features import compute_window_features
from src.cle.logging_setup import get_logger
from src.cle.utils.io import load_json, load_model_artifact

logger = get_logger(__name__)


def load_model(models_dir: str) -> Dict:
    """
    Load trained model artifacts.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with:
            - model: Trained (calibrated) model
            - scaler: Fitted scaler
            - imputer: Optional fitted imputer (regression only)
            - feature_spec: Feature specification
            - calibration: Calibration metadata (classification) or model_meta (regression)
            - task_mode: "classification" or "regression"

    Raises:
        FileNotFoundError: If required artifacts are missing
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Load artifacts (classification or regression)
    try:
        model_path = models_dir / "model.bin"
        regression_model_path = models_dir / "model_regression.bin"

        task_mode = "classification"
        if regression_model_path.exists() and not model_path.exists():
            task_mode = "regression"

        if task_mode == "classification":
            model = load_model_artifact(model_path)
        else:
            model = load_model_artifact(regression_model_path)

        scaler = load_model_artifact(models_dir / "scaler.bin")
        feature_spec = load_json(models_dir / "feature_spec.json")
        if task_mode == "classification":
            calibration: Dict[str, Any] = load_json(models_dir / "calibration.json")
            imputer: Any | None = None
        else:
            meta_path = models_dir / "model_meta.json"
            calibration = load_json(meta_path) if meta_path.exists() else {}
            imputer_path = models_dir / "imputer.bin"
            imputer = load_model_artifact(imputer_path) if imputer_path.exists() else None
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model artifact: {e}")

    artifacts = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_spec": feature_spec,
        "calibration": calibration,
        "task_mode": task_mode,
    }

    logger.info(
        f"Loaded model artifacts from {models_dir} "
        f"({len(feature_spec['features'])} features)"
    )

    return artifacts


def predict_window(
    features: Union[Dict, np.ndarray, List],
    artifacts: Dict,
) -> Tuple[float, float]:
    """
    Predict Cognitive Load Index (CLI) for a window of features.

    Args:
        features: Window features as:
            - Dict: feature_name -> value mapping
            - np.ndarray: array of feature values (must match feature_spec order)
            - List: list of feature values (must match feature_spec order)
        artifacts: Model artifacts from load_model()

    Returns:
        Tuple of (cli, confidence):
            - cli: Cognitive Load Index in [0, 1]
            - confidence: Prediction confidence in [0, 1]

    Raises:
        ValueError: If features don't match expected format
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_spec"]["features"]

    # Convert features to array
    valid_frame_ratio = 1.0
    if isinstance(features, dict):
        # Extract features in correct order
        feature_array = np.array([features.get(name, 0.0) for name in feature_names])
        try:
            valid_frame_ratio = float(features.get("valid_frame_ratio", 1.0))
        except (TypeError, ValueError):
            valid_frame_ratio = 1.0
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

    # Handle NaN values
    if np.any(np.isnan(feature_array)):
        logger.warning("Found NaN values in features, replacing with zeros")
        feature_array = np.nan_to_num(feature_array, nan=0.0)

    # Optional imputation (used for regression artifacts)
    imputer = artifacts.get("imputer")
    if imputer is not None:
        feature_array = imputer.transform(feature_array)

    # Scale features
    features_scaled = scaler.transform(feature_array)

    task_mode = artifacts.get("task_mode", "classification")
    if task_mode == "classification":
        cli_raw = model.predict_proba(features_scaled)[0, 1]  # Probability of high load
    else:
        cli_raw = float(model.predict(features_scaled)[0])
        cli_raw = float(np.clip(cli_raw, 0.0, 1.0))

    confidence = abs(cli_raw - 0.5) * 2.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    confidence = float(np.clip(confidence * np.clip(valid_frame_ratio, 0.0, 1.0), 0.0, 1.0))

    cli = float(cli_raw)

    return cli, confidence


def extract_features_from_window(
    frame_data: List[Dict],
    config: Dict,
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Extract window-level features from list of per-frame features.

    This is a convenience function that wraps compute_window_features.

    Args:
        frame_data: List of per-frame feature dictionaries
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Dictionary of window-level features

    Example:
        >>> frame_data = [
        ...     {"ear_mean": 0.25, "pupil_mean": 0.3, "brightness": 120, "valid": True},
        ...     {"ear_mean": 0.26, "pupil_mean": 0.31, "brightness": 121, "valid": True},
        ...     # ... more frames
        ... ]
        >>> features = extract_features_from_window(frame_data, config, fps=30.0)
        >>> cli, conf = predict_window(features, artifacts)
    """
    features = compute_window_features(frame_data, config, fps)
    return features


def predict_from_frame_data(
    frame_data: List[Dict],
    artifacts: Dict,
    config: Dict,
    fps: float = 30.0,
) -> Tuple[float, float]:
    """
    End-to-end prediction from per-frame features.

    Combines feature extraction and prediction in one call.

    Args:
        frame_data: List of per-frame feature dictionaries
        artifacts: Model artifacts from load_model()
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Tuple of (cli, confidence)
    """
    # Extract window features
    window_features = extract_features_from_window(frame_data, config, fps)

    # Predict
    cli, confidence = predict_window(window_features, artifacts)

    return cli, confidence
