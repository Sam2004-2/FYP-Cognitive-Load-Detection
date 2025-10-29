"""
Cognitive Load Estimation (CLE) - Production-ready MVP for inferring cognitive load from video.

This package provides tools to extract ocular features from video (offline or real-time),
train machine learning models, and predict continuous Cognitive Load Index (CLI) values.

Public API:
    - load_model: Load trained model artifacts
    - predict_window: Predict CLI from features
    - extract_features_from_window: Extract features from frame data
    - predict_from_frame_data: End-to-end prediction from per-frame features
"""

from src.cle.api import (
    extract_features_from_window,
    load_model,
    predict_from_frame_data,
    predict_window,
)

__version__ = "0.1.0"
__all__ = [
    "load_model",
    "predict_window",
    "extract_features_from_window",
    "predict_from_frame_data",
]

