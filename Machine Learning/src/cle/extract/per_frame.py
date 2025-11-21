"""
Per-frame feature computation.

Computes ocular features from landmarks for each video frame.
Focus on EAR (Eye Aspect Ratio) based features - pupil tracking removed.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Distance between points
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_coords: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_coords: Array of 6 eye landmark coordinates [(x, y, z), ...]
                   Points order: [outer, top1, top2, inner, bottom2, bottom1]

    Returns:
        EAR value (typically 0.2-0.3 for open eye, <0.2 for closed)
    """
    if eye_coords is None or len(eye_coords) != 6:
        return 0.0

    # Extract 2D coordinates
    points = eye_coords[:, :2]  # Use only x, y

    # Vertical distances
    v1 = euclidean_distance(points[1], points[5])  # top1 to bottom1
    v2 = euclidean_distance(points[2], points[4])  # top2 to bottom2

    # Horizontal distance
    h = euclidean_distance(points[0], points[3])  # outer to inner

    # Avoid division by zero
    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


# Pupil/iris functions removed - no longer tracking pupil diameter
# Focus shifted to EAR-based features which are more robust and calibration-free


def compute_brightness(frame: np.ndarray, face_roi: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    Compute mean brightness of frame or face ROI.

    Args:
        frame: Input frame (BGR or grayscale)
        face_roi: Optional face bounding box (x, y, w, h)

    Returns:
        Mean brightness value (0-255)
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Use ROI if provided
    if face_roi is not None:
        x, y, w, h = face_roi
        h_frame, w_frame = gray.shape
        # Clip ROI to frame bounds
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        gray = gray[y : y + h, x : x + w]

    return float(np.mean(gray))


def extract_face_roi(landmarks_px: np.ndarray, margin: float = 0.2) -> Tuple[int, int, int, int]:
    """
    Extract face bounding box from landmarks.

    Args:
        landmarks_px: Landmarks in pixel coordinates
        margin: Margin to add around face (fraction of face size)

    Returns:
        Bounding box (x, y, w, h)
    """
    if landmarks_px is None or len(landmarks_px) == 0:
        return (0, 0, 1, 1)

    # Get bounding box
    x_min = int(np.min(landmarks_px[:, 0]))
    x_max = int(np.max(landmarks_px[:, 0]))
    y_min = int(np.min(landmarks_px[:, 1]))
    y_max = int(np.max(landmarks_px[:, 1]))

    # Add margin
    w = x_max - x_min
    h = y_max - y_min
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x = max(0, x_min - margin_x)
    y = max(0, y_min - margin_y)
    w = w + 2 * margin_x
    h = h + 2 * margin_y

    return (x, y, w, h)


def extract_frame_features(frame: np.ndarray, landmark_result: Dict) -> Dict:
    """
    Extract all per-frame features from frame and landmarks.

    Note: Pupil tracking removed - focus on EAR-based features only.

    Args:
        frame: Input frame (BGR)
        landmark_result: Result from FaceMeshExtractor.process_frame()

    Returns:
        Dictionary with per-frame features:
            - ear_left: Left eye aspect ratio
            - ear_right: Right eye aspect ratio
            - ear_mean: Mean EAR (blink proxy)
            - brightness: Face region brightness
            - quality: Detection quality
            - valid: Whether frame has valid features
    """
    # Default invalid result
    invalid_result = {
        "ear_left": 0.0,
        "ear_right": 0.0,
        "ear_mean": 0.0,
        "brightness": 0.0,
        "quality": 0.0,
        "valid": False,
    }

    if not landmark_result["detected"]:
        return invalid_result

    # Extract EAR for both eyes
    ear_left = eye_aspect_ratio(landmark_result["left_eye"]["coords"])
    ear_right = eye_aspect_ratio(landmark_result["right_eye"]["coords"])
    ear_mean = (ear_left + ear_right) / 2.0

    # Pupil extraction removed - no longer tracking pupil diameter

    # Extract brightness from face ROI
    face_roi = extract_face_roi(landmark_result["landmarks_px"])
    brightness = compute_brightness(frame, face_roi)

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": ear_mean,
        "brightness": brightness,
        "quality": landmark_result["quality"],
        "valid": True,
    }

