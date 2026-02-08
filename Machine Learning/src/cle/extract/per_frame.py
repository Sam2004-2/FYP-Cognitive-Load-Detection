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

# MediaPipe FaceMesh landmark indices (must match UI/src/config/featureConfig.ts)
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14

# Head pose estimation landmarks
NOSE_TIP_IDX = 4
FOREHEAD_IDX = 10
CHIN_IDX = 152
LEFT_EAR_TRAGION_IDX = 234
RIGHT_EAR_TRAGION_IDX = 454


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


def eye_outer_center(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Compute midpoint between the outer eye corners (indices 33 and 263).

    Args:
        landmarks: Array of (x, y, z) normalized coordinates

    Returns:
        (x, y) midpoint in normalized coordinates
    """
    if landmarks is None or len(landmarks) <= max(LEFT_EYE_OUTER_IDX, RIGHT_EYE_OUTER_IDX):
        return (0.0, 0.0)

    left = landmarks[LEFT_EYE_OUTER_IDX]
    right = landmarks[RIGHT_EYE_OUTER_IDX]
    return (float((left[0] + right[0]) / 2.0), float((left[1] + right[1]) / 2.0))


def mouth_aspect_ratio(landmarks: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Mouth Aspect Ratio (MAR).

    MAR = dist2D(upper_lip, lower_lip) / max(dist2D(left_corner, right_corner), eps)

    Args:
        landmarks: Array of (x, y, z) normalized coordinates
        eps: Small constant to avoid division by zero

    Returns:
        MAR value
    """
    if landmarks is None or len(landmarks) <= max(MOUTH_RIGHT_IDX, MOUTH_LOWER_IDX):
        return 0.0

    left = landmarks[MOUTH_LEFT_IDX]
    right = landmarks[MOUTH_RIGHT_IDX]
    upper = landmarks[MOUTH_UPPER_IDX]
    lower = landmarks[MOUTH_LOWER_IDX]

    width = euclidean_distance((float(left[0]), float(left[1])), (float(right[0]), float(right[1])))
    height = euclidean_distance((float(upper[0]), float(upper[1])), (float(lower[0]), float(lower[1])))

    if width < eps:
        return 0.0
    return float(height / width)


def head_roll(landmarks: np.ndarray) -> float:
    """
    Compute a head roll proxy (radians) from the outer eye corner line.

    roll = atan2(y_right - y_left, x_right - x_left)
    """
    if landmarks is None or len(landmarks) <= max(LEFT_EYE_OUTER_IDX, RIGHT_EYE_OUTER_IDX):
        return 0.0

    left = landmarks[LEFT_EYE_OUTER_IDX]
    right = landmarks[RIGHT_EYE_OUTER_IDX]
    dy = float(right[1] - left[1])
    dx = float(right[0] - left[0])
    return float(np.arctan2(dy, dx))


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


def head_pitch(landmarks: np.ndarray) -> float:
    """
    Compute a head pitch proxy (radians) from forehead-to-chin depth angle.

    Uses the z-coordinate difference between forehead (landmark 10) and
    chin (landmark 152) relative to their vertical separation.
    Positive values indicate head tilted forward (looking down).
    """
    if landmarks is None or len(landmarks) <= max(FOREHEAD_IDX, CHIN_IDX):
        return 0.0

    forehead = landmarks[FOREHEAD_IDX]
    chin = landmarks[CHIN_IDX]
    dy = float(chin[1] - forehead[1])  # vertical (y increases downward in normalised coords)
    dz = float(chin[2] - forehead[2])  # depth

    if abs(dy) < 1e-6:
        return 0.0
    return float(np.arctan2(dz, dy))


def head_yaw(landmarks: np.ndarray) -> float:
    """
    Compute a head yaw proxy (radians) from nose-to-face-center offset.

    Uses the horizontal offset of the nose tip (landmark 4) from the
    midpoint between the ear tragion landmarks (234 / 454), normalised
    by face width.  Positive values indicate head turned to the right.
    """
    if landmarks is None or len(landmarks) <= max(NOSE_TIP_IDX, LEFT_EAR_TRAGION_IDX, RIGHT_EAR_TRAGION_IDX):
        return 0.0

    nose = landmarks[NOSE_TIP_IDX]
    left = landmarks[LEFT_EAR_TRAGION_IDX]
    right = landmarks[RIGHT_EAR_TRAGION_IDX]

    mid_x = (float(left[0]) + float(right[0])) / 2.0
    face_width = abs(float(right[0]) - float(left[0]))

    if face_width < 1e-6:
        return 0.0
    return float(np.arctan2(float(nose[0]) - mid_x, face_width / 2.0))


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
        "eye_center_x": 0.0,
        "eye_center_y": 0.0,
        "mouth_mar": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "valid": False,
    }

    if not landmark_result.get("detected", False):
        return invalid_result

    # Extract EAR for both eyes
    ear_left = eye_aspect_ratio(landmark_result["left_eye"]["coords"])
    ear_right = eye_aspect_ratio(landmark_result["right_eye"]["coords"])
    ear_mean = (ear_left + ear_right) / 2.0

    # Pupil extraction removed - no longer tracking pupil diameter

    # Extract brightness from face ROI
    face_roi = extract_face_roi(landmark_result["landmarks_px"])
    brightness = compute_brightness(frame, face_roi)

    # Geometry features
    landmarks = landmark_result.get("landmarks")
    eye_center_x, eye_center_y = eye_outer_center(landmarks)
    mouth_mar = mouth_aspect_ratio(landmarks)
    roll = head_roll(landmarks)
    pitch = head_pitch(landmarks)
    yaw = head_yaw(landmarks)

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": ear_mean,
        "brightness": brightness,
        "quality": landmark_result["quality"],
        "eye_center_x": eye_center_x,
        "eye_center_y": eye_center_y,
        "mouth_mar": mouth_mar,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "valid": True,
    }
