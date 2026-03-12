"""
Landmark extraction using MediaPipe Face Mesh.

Extracts 468 facial landmarks with iris landmarks for eye tracking.
"""

from typing import Dict, List, Optional, Tuple

import os

# Headless/CI environments (and some macOS setups) can fail to create an OpenGL context.
# Force MediaPipe to use CPU-only graph execution by default.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp
import numpy as np

from src.cle.logging_setup import get_logger

logger = get_logger(__name__)

# MediaPipe Face Mesh landmark indices from https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
# Eyes (for EAR computation)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 6 points for left eye these come from google mediapipe documentation
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 6 points for right eye these come from google mediapipe documentation

# Iris landmarks (4 points per iris)
LEFT_IRIS_INDICES = [469, 470, 471, 472]
RIGHT_IRIS_INDICES = [474, 475, 476, 477]

# Eye corners (inner and outer canthus)
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


class FaceMeshExtractor:
    """
    MediaPipe Face Mesh extractor for ocular feature extraction.

    Extracts facial landmarks with refined iris landmarks for pupil tracking.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ):
        """
        Initialize Face Mesh extractor.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            refine_landmarks: Enable iris landmark refinement
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.info(
            f"Initialized FaceMeshExtractor "
            f"(refine_landmarks={refine_landmarks}, "
            f"min_conf={min_detection_confidence})"
        )

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame and extract landmarks.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Dictionary containing:
                - landmarks: List of (x, y, z) normalized coordinates (468+ points)
                - landmarks_px: List of (x, y) pixel coordinates
                - quality: Detection confidence score (0-1)
                - detected: Whether face was detected
                - left_eye: Left eye landmark indices and coordinates
                - right_eye: Right eye landmark indices and coordinates
                - left_iris: Left iris landmark coordinates
                - right_iris: Right iris landmark coordinates
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Process frame
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "landmarks": None,
                "landmarks_px": None,
                "quality": 0.0,
                "detected": False,
                "left_eye": None,
                "right_eye": None,
                "left_iris": None,
                "right_iris": None,
            }

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Extract all landmarks (normalized coordinates)
        landmarks = []
        landmarks_px = []
        for lm in face_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
            landmarks_px.append((int(lm.x * w), int(lm.y * h)))

        # Extract specific eye and iris landmarks
        left_eye_coords = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye_coords = [landmarks[i] for i in RIGHT_EYE_INDICES]

        result = {
            "landmarks": np.array(landmarks),
            "landmarks_px": np.array(landmarks_px),
            "quality": 1.0,  # MediaPipe doesn't provide per-frame confidence
            "detected": True,
            "left_eye": {
                "indices": LEFT_EYE_INDICES,
                "coords": np.array(left_eye_coords),
            },
            "right_eye": {
                "indices": RIGHT_EYE_INDICES,
                "coords": np.array(right_eye_coords),
            },
        }

        # Add iris landmarks if available
        if self.refine_landmarks and len(landmarks) > 468:
            left_iris_coords = [landmarks[i] for i in LEFT_IRIS_INDICES]
            right_iris_coords = [landmarks[i] for i in RIGHT_IRIS_INDICES]

            result["left_iris"] = {
                "indices": LEFT_IRIS_INDICES,
                "coords": np.array(left_iris_coords),
            }
            result["right_iris"] = {
                "indices": RIGHT_IRIS_INDICES,
                "coords": np.array(right_iris_coords),
            }
        else:
            result["left_iris"] = None
            result["right_iris"] = None

        return result

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()
        logger.debug("Closed FaceMeshExtractor")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
