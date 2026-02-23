"""
I/O utilities for CLE.

Functions for reading videos, loading/saving CSVs, and model serialization.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def install_numpy_bitgenerator_compatibility_patch() -> bool:
    """
    Patch numpy's private bit-generator constructor for legacy pickles.

    Some older model artifacts store bit-generator state with class objects
    instead of string names (for example ``<class '...MT19937'>``). Newer NumPy
    versions expect the generator name as a string and raise:
    "not a known BitGenerator module".
    """
    numpy_pickle = getattr(np.random, "_pickle", None)
    if numpy_pickle is None:
        return False

    ctor_name = "__bit_generator_ctor"
    original_ctor = getattr(numpy_pickle, ctor_name, None)
    if original_ctor is None:
        return False

    if getattr(original_ctor, "_cle_compat_wrapped", False):
        return True

    def compat_ctor(bit_generator_name="MT19937"):  # type: ignore[no-untyped-def]
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        elif isinstance(bit_generator_name, str) and bit_generator_name.startswith("<class "):
            match = re.search(r"\.([A-Za-z0-9_]+)'>$", bit_generator_name)
            if match:
                bit_generator_name = match.group(1)
        return original_ctor(bit_generator_name)

    setattr(compat_ctor, "_cle_compat_wrapped", True)
    setattr(numpy_pickle, ctor_name, compat_ctor)
    logger.warning("Applied NumPy bit-generator compatibility patch for legacy model artifacts")
    return True


def install_numpy_private_core_aliases() -> bool:
    """
    Alias legacy NumPy private module paths used by older pickles.

    Some artifacts reference ``numpy._core.numeric`` while NumPy 1.x exposes
    modules under ``numpy.core``. Adding import aliases allows unpickling.
    """
    changed = False

    try:
        import numpy.core as np_core  # type: ignore[attr-defined]
        import numpy.core.numeric as np_core_numeric  # type: ignore[attr-defined]
    except Exception:
        return False

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np_core
        changed = True

    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = np_core_numeric
        changed = True

    if changed:
        logger.warning("Applied NumPy private-core alias compatibility patch for legacy model artifacts")

    return changed


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

    patched_bitgenerator = False
    patched_private_core = False

    while True:
        try:
            obj = joblib.load(path)
            break
        except ValueError as err:
            if "not a known BitGenerator module" not in str(err):
                raise
            if patched_bitgenerator:
                raise
            if not install_numpy_bitgenerator_compatibility_patch():
                raise
            patched_bitgenerator = True
            logger.warning(
                "Retrying model artifact load after applying NumPy bit-generator compatibility patch"
            )
        except ModuleNotFoundError as err:
            missing_name = err.name or str(err)
            if "numpy._core" not in missing_name:
                raise
            if patched_private_core:
                raise
            if not install_numpy_private_core_aliases():
                raise
            patched_private_core = True
            logger.warning(
                "Retrying model artifact load after applying NumPy private-core alias compatibility patch"
            )

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
