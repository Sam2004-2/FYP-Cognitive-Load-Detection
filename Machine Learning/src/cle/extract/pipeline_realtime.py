"""
Real-time video processing pipeline.

Processes webcam feed and outputs continuous CLI predictions.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.cle.api import load_model
from src.cle.config import load_config
from src.cle.extract.features import compute_window_features
from src.cle.extract.landmarks import FaceMeshExtractor
from src.cle.extract.per_frame import extract_frame_features
from src.cle.extract.windowing import WindowBuffer, validate_window_quality
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.utils.timers import FPSCounter

logger = get_logger(__name__)


class RealtimeCLIEstimator:
    """
    Real-time Cognitive Load Index estimator.

    Processes video stream and outputs CLI predictions at regular intervals.
    """

    def __init__(
        self,
        artifacts: dict,
        config: dict,
        fps: float = 30.0,
        smoothing_alpha: float = 0.4,
    ):
        """
        Initialize real-time estimator.

        Args:
            artifacts: Model artifacts from load_model()
            config: Configuration dictionary
            fps: Expected frames per second
            smoothing_alpha: EWMA smoothing parameter (0-1)
        """
        self.artifacts = artifacts
        self.config = config
        self.fps = fps
        self.smoothing_alpha = smoothing_alpha

        # Initialize window buffer
        window_length_s = config.get("windows.length_s", 20.0)
        self.buffer = WindowBuffer(window_length_s, fps)

        # Initialize face mesh extractor
        self.extractor = FaceMeshExtractor(
            min_detection_confidence=config.get("quality.min_face_conf", 0.5),
            refine_landmarks=True,
        )

        # State for predictions
        self.cli_smoothed = 0.5  # Start at neutral
        self.last_prediction_time = 0.0
        self.step_s = config.get("windows.step_s", 5.0)

        # FPS counter
        self.fps_counter = FPSCounter(alpha=0.1)

        logger.info(
            f"Initialized RealtimeCLIEstimator "
            f"(window={window_length_s}s, step={self.step_s}s, alpha={smoothing_alpha})"
        )

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame and extract features.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Dictionary with frame processing results
        """
        # Extract landmarks
        landmark_result = self.extractor.process_frame(frame)

        # Extract per-frame features
        frame_features = extract_frame_features(frame, landmark_result)

        # Update FPS
        current_fps = self.fps_counter.update()

        return {
            "features": frame_features,
            "detected": landmark_result["detected"],
            "fps": current_fps,
        }

    def add_frame(self, frame_features: dict) -> None:
        """
        Add frame features to buffer.

        Args:
            frame_features: Per-frame feature dictionary
        """
        self.buffer.add_frame(frame_features)

    def should_predict(self, current_time: float) -> bool:
        """
        Check if it's time to make a new prediction.

        Args:
            current_time: Current time in seconds

        Returns:
            True if prediction should be made
        """
        if not self.buffer.is_ready():
            return False

        time_since_last = current_time - self.last_prediction_time
        return time_since_last >= self.step_s

    def predict(self) -> dict:
        """
        Make CLI prediction from current buffer.

        Returns:
            Dictionary with prediction results:
                - cli: Raw CLI prediction
                - cli_smoothed: Smoothed CLI prediction
                - confidence: Prediction confidence
                - valid: Whether prediction is valid
        """
        if not self.buffer.is_ready():
            return {
                "cli": self.cli_smoothed,
                "cli_smoothed": self.cli_smoothed,
                "confidence": 0.0,
                "valid": False,
            }

        # Get window data
        window_data = self.buffer.get_window()

        # Validate window quality
        is_valid, bad_ratio = validate_window_quality(
            window_data,
            max_bad_ratio=self.config.get("quality.max_bad_frame_ratio", 0.2),
        )

        if not is_valid:
            logger.warning(f"Low quality window (bad_ratio={bad_ratio:.2f}), using previous CLI")
            return {
                "cli": self.cli_smoothed,
                "cli_smoothed": self.cli_smoothed,
                "confidence": 0.5 * (1 - bad_ratio),
                "valid": False,
            }

        # Compute window features
        window_features = compute_window_features(window_data, self.config, self.fps)

        # Predict CLI
        model = self.artifacts["model"]
        scaler = self.artifacts["scaler"]
        feature_names = self.artifacts["feature_spec"]["features"]

        # Extract features in correct order
        feature_array = np.array([window_features.get(name, 0.0) for name in feature_names])
        feature_array = np.nan_to_num(feature_array, nan=0.0).reshape(1, -1)

        # Scale and predict
        features_scaled = scaler.transform(feature_array)
        cli_raw = float(model.predict_proba(features_scaled)[0, 1])

        # Apply EWMA smoothing
        self.cli_smoothed = (
            self.smoothing_alpha * cli_raw + (1 - self.smoothing_alpha) * self.cli_smoothed
        )

        # Confidence based on prediction certainty and window quality
        confidence_pred = abs(cli_raw - 0.5) * 2.0
        confidence_quality = 1.0 - bad_ratio
        confidence = float(np.clip(confidence_pred * confidence_quality, 0.0, 1.0))

        self.last_prediction_time = time.time()

        return {
            "cli": cli_raw,
            "cli_smoothed": self.cli_smoothed,
            "confidence": confidence,
            "valid": True,
        }

    def close(self) -> None:
        """Release resources."""
        self.extractor.close()


def main():
    """Main entry point for real-time pipeline."""
    parser = argparse.ArgumentParser(description="Real-time cognitive load estimation")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Video source (0 for webcam, or video file path)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Expected FPS for processing",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video feed with overlay",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="pipeline_realtime.log")
    logger.info("=" * 80)
    logger.info("Starting real-time cognitive load estimation")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration (hash: {config.hash()[:8]})")

    # Load model artifacts
    try:
        artifacts = load_model(args.models)
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        sys.exit(1)

    # Open video source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        sys.exit(1)

    logger.info(f"Opened video source: {args.source}")

    # Get actual FPS if available
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        fps = video_fps
        logger.info(f"Using video FPS: {fps:.1f}")
    else:
        fps = args.fps
        logger.info(f"Using default FPS: {fps:.1f}")

    # Initialize estimator
    smoothing_alpha = config.get("realtime.smoothing_alpha", 0.4)
    estimator = RealtimeCLIEstimator(artifacts, config.to_dict(), fps, smoothing_alpha)

    logger.info("Starting real-time processing... (Press 'q' to quit)")
    logger.info("-" * 80)

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            # Process frame
            result = estimator.process_frame(frame)
            estimator.add_frame(result["features"])

            current_time = time.time() - start_time

            # Check if it's time to predict
            if estimator.should_predict(current_time):
                prediction = estimator.predict()

                if prediction["valid"]:
                    logger.info(
                        f"[{current_time:7.1f}s] CLI: {prediction['cli_smoothed']:.3f} "
                        f"(raw: {prediction['cli']:.3f}, "
                        f"conf: {prediction['confidence']:.3f}, "
                        f"fps: {result['fps']:.1f})"
                    )
                else:
                    logger.warning(
                        f"[{current_time:7.1f}s] Low quality window, "
                        f"CLI: {prediction['cli_smoothed']:.3f} "
                        f"(conf: {prediction['confidence']:.3f})"
                    )

            # Display video with overlay (if requested)
            if args.display:
                display_frame = frame.copy()

                # Add CLI overlay
                if estimator.buffer.is_ready():
                    cli_text = f"CLI: {estimator.cli_smoothed:.3f}"
                    fps_text = f"FPS: {result['fps']:.1f}"

                    cv2.putText(
                        display_frame,
                        cli_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        fps_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Draw face detection indicator
                    if result["detected"]:
                        cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
                    else:
                        cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)

                cv2.imshow("Cognitive Load Estimation", display_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    finally:
        # Cleanup
        cap.release()
        estimator.close()
        if args.display:
            cv2.destroyAllWindows()

        logger.info("=" * 80)
        logger.info("Real-time processing stopped")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

