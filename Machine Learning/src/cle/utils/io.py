"""
I/O utilities for CLE.

Functions for reading videos, loading/saving CSVs, and model serialization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def open_video(video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    """
    Open video file and extract metadata.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (VideoCapture object, metadata dict)

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be opened
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Extract metadata
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        if cap.get(cv2.CAP_PROP_FPS) > 0
        else 0,
    }

    logger.info(
        f"Opened video: {video_path.name} "
        f"({metadata['width']}x{metadata['height']}, "
        f"{metadata['fps']:.1f} fps, "
        f"{metadata['frame_count']} frames)"
    )

    return cap, metadata


def read_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Read all frames from video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None for all)

    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap, metadata = open_video(video_path)
    frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_idx += 1

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    logger.info(f"Read {len(frames)} frames from {Path(video_path).name}")
    return frames


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load video manifest CSV.

    Expected columns: video_file, label, role, user_id, notes

    Args:
        manifest_path: Path to manifest CSV file

    Returns:
        DataFrame with manifest data

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If required columns are missing
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Validate required columns
    required_cols = ["video_file", "label", "role", "user_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in manifest: {missing_cols}")

    logger.info(f"Loaded manifest with {len(df)} entries from {manifest_path.name}")
    return df


def save_features_csv(
    df: pd.DataFrame, output_path: str, include_index: bool = False
) -> None:
    """
    Save features DataFrame to CSV.

    Args:
        df: Features DataFrame
        output_path: Path to output CSV file
        include_index: Whether to include index in CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=include_index)
    logger.info(f"Saved {len(df)} feature rows to {output_path}")


def load_features_csv(features_path: str) -> pd.DataFrame:
    """
    Load features CSV.

    Automatically adapts AVCAffe schema to expected training format:
    - 'participant_id' -> 'user_id'
    - 'cognitive_load' -> 'load_0_1'

    Args:
        features_path: Path to features CSV file

    Returns:
        Features DataFrame

    Raises:
        FileNotFoundError: If features file doesn't exist
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} feature rows from {features_path.name}")
    
    # Adapt AVCAffe schema if detected
    rename_map = {}
    if "participant_id" in df.columns and "user_id" not in df.columns:
        rename_map["participant_id"] = "user_id"
    if "cognitive_load" in df.columns and "load_0_1" not in df.columns:
        rename_map["cognitive_load"] = "load_0_1"
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Adapted AVCAffe schema: {rename_map}")
    
    return df


def save_model_artifact(obj: Any, path: str) -> None:
    """
    Save model artifact using joblib.

    Args:
        obj: Object to save (model, scaler, etc.)
        path: Path to save artifact
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved model artifact to {path}")


def load_model_artifact(path: str) -> Any:
    """
    Load model artifact using joblib.

    Args:
        path: Path to artifact file

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If artifact file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    obj = joblib.load(path)
    logger.info(f"Loaded model artifact from {path.name}")
    return obj


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Path to JSON file
        indent: Indentation level for pretty printing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with JSON data

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {path.name}")
    return data

