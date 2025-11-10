"""
Automated webcam feature validation script.

Runs real-time pipeline for a specified duration and generates a validation report.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.api import load_model
from src.cle.config import load_config
from src.cle.extract.features import compute_window_features
from src.cle.extract.landmarks import FaceMeshExtractor
from src.cle.extract.per_frame import extract_frame_features
from src.cle.extract.windowing import WindowBuffer, validate_window_quality
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def validate_webcam_features(
    duration_s: float,
    models_dir: str,
    config_path: str = None,
    output_dir: str = "reports",
    source: int = 0,
) -> dict:
    """
    Run webcam validation test and generate report.

    Args:
        duration_s: Duration to run test in seconds
        models_dir: Directory containing model artifacts
        config_path: Path to config YAML
        output_dir: Directory to save reports
        source: Video source (0 for webcam)

    Returns:
        Dictionary with validation results
    """
    logger.info("=" * 80)
    logger.info("Starting webcam feature validation")
    logger.info(f"Duration: {duration_s}s")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(config_path)

    # Load model artifacts
    try:
        artifacts = load_model(models_dir)
        logger.info(f"Loaded model artifacts from {models_dir}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return {"status": "FAIL", "error": "Model loading failed"}

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return {"status": "FAIL", "error": "Camera not accessible"}

    logger.info(f"Opened video source: {source}")

    # Get FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = video_fps if video_fps > 0 else 30.0
    logger.info(f"FPS: {fps:.1f}")

    # Initialize components
    extractor = FaceMeshExtractor(
        min_detection_confidence=config.get("quality.min_face_conf", 0.5),
        refine_landmarks=True,
    )

    window_length_s = config.get("windows.length_s", 20.0)
    buffer = WindowBuffer(window_length_s, fps)

    # Storage for validation data
    frame_features_list = []
    window_features_list = []
    cli_predictions = []
    fps_samples = []

    start_time = time.time()
    frame_count = 0
    face_detected_count = 0

    logger.info(f"Recording for {duration_s} seconds...")
    logger.info("-" * 80)

    try:
        while time.time() - start_time < duration_s:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame_start_time = time.time()

            # Extract landmarks
            landmark_result = extractor.process_frame(frame)

            # Extract per-frame features
            features = extract_frame_features(frame, landmark_result)

            # Track detection
            if landmark_result["detected"]:
                face_detected_count += 1

            # Store frame features
            frame_features_list.append({
                "timestamp": time.time() - start_time,
                "ear_mean": features["ear_mean"],
                "pupil_mean": features["pupil_mean"],
                "brightness": features["brightness"],
                "quality": features["quality"],
                "valid": features["valid"],
            })

            # Add to buffer
            buffer.add_frame(features)

            # Make prediction if buffer ready
            if buffer.is_ready():
                window_data = buffer.get_window()

                # Validate quality
                is_valid, bad_ratio = validate_window_quality(
                    window_data,
                    max_bad_ratio=config.get("quality.max_bad_frame_ratio", 0.2),
                )

                if is_valid:
                    # Compute window features
                    window_features = compute_window_features(
                        window_data, config.to_dict(), fps
                    )

                    # Predict CLI
                    model = artifacts["model"]
                    scaler = artifacts["scaler"]
                    feature_names = artifacts["feature_spec"]["features"]

                    feature_array = np.array(
                        [window_features.get(name, 0.0) for name in feature_names]
                    )
                    feature_array = np.nan_to_num(feature_array, nan=0.0).reshape(1, -1)

                    features_scaled = scaler.transform(feature_array)
                    cli = float(model.predict_proba(features_scaled)[0, 1])

                    # Store prediction
                    cli_predictions.append({
                        "timestamp": time.time() - start_time,
                        "cli": cli,
                        "confidence": abs(cli - 0.5) * 2.0,
                    })

                    # Store window features
                    window_features["timestamp"] = time.time() - start_time
                    window_features_list.append(window_features)

            # Track FPS
            frame_time = time.time() - frame_start_time
            fps_samples.append(1.0 / frame_time if frame_time > 0 else 0)

            frame_count += 1

            # Progress update every 5 seconds
            elapsed = time.time() - start_time
            if frame_count % (int(fps) * 5) == 0:
                logger.info(f"[{elapsed:6.1f}s] Processed {frame_count} frames")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        extractor.close()

    # Generate validation report
    logger.info("=" * 80)
    logger.info("Generating validation report...")
    logger.info("=" * 80)

    elapsed_time = time.time() - start_time

    # Convert to DataFrames
    df_frames = pd.DataFrame(frame_features_list)
    df_windows = pd.DataFrame(window_features_list) if window_features_list else None
    df_cli = pd.DataFrame(cli_predictions) if cli_predictions else None

    # Compute statistics
    report = {
        "status": "PASS",  # Will be updated if issues found
        "timestamp": datetime.now().isoformat(),
        "duration_s": elapsed_time,
        "frame_count": frame_count,
        "average_fps": np.mean(fps_samples) if fps_samples else 0,
        "face_detection": {
            "frames_detected": face_detected_count,
            "detection_rate": face_detected_count / frame_count if frame_count > 0 else 0,
        },
        "per_frame_features": {},
        "window_features": {},
        "cli_predictions": {},
        "quality_metrics": {},
        "warnings": [],
    }

    # Per-frame feature statistics
    if len(df_frames) > 0:
        valid_frames = df_frames[df_frames["valid"] == True]

        if len(valid_frames) > 0:
            report["per_frame_features"] = {
                "ear": {
                    "mean": float(valid_frames["ear_mean"].mean()),
                    "min": float(valid_frames["ear_mean"].min()),
                    "max": float(valid_frames["ear_mean"].max()),
                    "std": float(valid_frames["ear_mean"].std()),
                },
                "pupil": {
                    "mean": float(valid_frames["pupil_mean"].mean()),
                    "min": float(valid_frames["pupil_mean"].min()),
                    "max": float(valid_frames["pupil_mean"].max()),
                    "std": float(valid_frames["pupil_mean"].std()),
                },
                "brightness": {
                    "mean": float(valid_frames["brightness"].mean()),
                    "min": float(valid_frames["brightness"].min()),
                    "max": float(valid_frames["brightness"].max()),
                    "std": float(valid_frames["brightness"].std()),
                },
                "quality": {
                    "mean": float(valid_frames["quality"].mean()),
                    "min": float(valid_frames["quality"].min()),
                    "max": float(valid_frames["quality"].max()),
                },
            }

            # Validate ranges
            ear_mean = report["per_frame_features"]["ear"]["mean"]
            if not (0.20 <= ear_mean <= 0.35):
                report["warnings"].append(f"EAR mean ({ear_mean:.3f}) outside expected range [0.20, 0.35]")

            pupil_mean = report["per_frame_features"]["pupil"]["mean"]
            if not (0.15 <= pupil_mean <= 0.60):
                report["warnings"].append(f"Pupil mean ({pupil_mean:.3f}) outside expected range [0.15, 0.60]")

            brightness_mean = report["per_frame_features"]["brightness"]["mean"]
            if not (60 <= brightness_mean <= 200):
                report["warnings"].append(
                    f"Brightness mean ({brightness_mean:.1f}) outside expected range [60, 200]"
                )

    # Window feature statistics
    if df_windows is not None and len(df_windows) > 0:
        report["window_features"] = {
            "count": len(df_windows),
            "tepr_delta_mean": {
                "mean": float(df_windows["tepr_delta_mean"].mean()),
                "min": float(df_windows["tepr_delta_mean"].min()),
                "max": float(df_windows["tepr_delta_mean"].max()),
            },
            "blink_rate": {
                "mean": float(df_windows["blink_rate"].mean()),
                "min": float(df_windows["blink_rate"].min()),
                "max": float(df_windows["blink_rate"].max()),
            },
            "perclos": {
                "mean": float(df_windows["perclos"].mean()),
                "min": float(df_windows["perclos"].min()),
                "max": float(df_windows["perclos"].max()),
            },
        }

        # Validate blink rate
        blink_rate_mean = report["window_features"]["blink_rate"]["mean"]
        if not (5 <= blink_rate_mean <= 35):
            report["warnings"].append(
                f"Blink rate mean ({blink_rate_mean:.1f}) outside expected range [5, 35] blinks/min"
            )

    # CLI prediction statistics
    if df_cli is not None and len(df_cli) > 0:
        report["cli_predictions"] = {
            "count": len(df_cli),
            "mean": float(df_cli["cli"].mean()),
            "min": float(df_cli["cli"].min()),
            "max": float(df_cli["cli"].max()),
            "std": float(df_cli["cli"].std()),
            "confidence_mean": float(df_cli["confidence"].mean()),
        }

        # Check for invalid values
        if df_cli["cli"].isna().any():
            report["warnings"].append("CLI predictions contain NaN values")

        if (df_cli["cli"] < 0).any() or (df_cli["cli"] > 1).any():
            report["warnings"].append("CLI predictions outside valid range [0, 1]")

    # Quality metrics
    report["quality_metrics"] = {
        "valid_frame_ratio": face_detected_count / frame_count if frame_count > 0 else 0,
        "average_quality": report["per_frame_features"].get("quality", {}).get("mean", 0),
        "average_fps": report["average_fps"],
    }

    # Check quality thresholds
    if report["quality_metrics"]["valid_frame_ratio"] < 0.80:
        report["warnings"].append(
            f"Valid frame ratio ({report['quality_metrics']['valid_frame_ratio']:.2f}) below 0.80"
        )
        report["status"] = "PARTIAL"

    if report["quality_metrics"]["average_quality"] < 0.85:
        report["warnings"].append(
            f"Average quality ({report['quality_metrics']['average_quality']:.2f}) below 0.85"
        )
        report["status"] = "PARTIAL"

    if report["average_fps"] < 15:
        report["warnings"].append(f"Average FPS ({report['average_fps']:.1f}) below 15")
        report["status"] = "PARTIAL"

    # Set final status
    if len(report["warnings"]) == 0:
        report["status"] = "PASS"
    elif len(report["warnings"]) >= 3:
        report["status"] = "FAIL"

    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    import json

    report_file = output_path / f"webcam_validation_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report to: {report_file}")

    # Save CSV data
    if len(df_frames) > 0:
        csv_file = output_path / f"webcam_validation_frames_{timestamp}.csv"
        df_frames.to_csv(csv_file, index=False)
        logger.info(f"Saved frame data to: {csv_file}")

    if df_windows is not None and len(df_windows) > 0:
        csv_file = output_path / f"webcam_validation_windows_{timestamp}.csv"
        df_windows.to_csv(csv_file, index=False)
        logger.info(f"Saved window data to: {csv_file}")

    if df_cli is not None and len(df_cli) > 0:
        csv_file = output_path / f"webcam_validation_cli_{timestamp}.csv"
        df_cli.to_csv(csv_file, index=False)
        logger.info(f"Saved CLI data to: {csv_file}")

    # Print summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {report['status']}")
    logger.info(f"Duration: {elapsed_time:.1f}s")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Average FPS: {report['average_fps']:.1f}")
    logger.info(f"Face detection rate: {report['face_detection']['detection_rate']:.2%}")

    if report["per_frame_features"]:
        logger.info("\nPer-Frame Features:")
        logger.info(f"  EAR: {report['per_frame_features']['ear']['mean']:.3f} "
                    f"(range: {report['per_frame_features']['ear']['min']:.3f}-"
                    f"{report['per_frame_features']['ear']['max']:.3f})")
        logger.info(f"  Pupil: {report['per_frame_features']['pupil']['mean']:.3f} "
                    f"(range: {report['per_frame_features']['pupil']['min']:.3f}-"
                    f"{report['per_frame_features']['pupil']['max']:.3f})")
        logger.info(f"  Brightness: {report['per_frame_features']['brightness']['mean']:.1f} "
                    f"(range: {report['per_frame_features']['brightness']['min']:.1f}-"
                    f"{report['per_frame_features']['brightness']['max']:.1f})")

    if report["window_features"]:
        logger.info("\nWindow Features:")
        logger.info(f"  Windows computed: {report['window_features']['count']}")
        logger.info(f"  Blink rate: {report['window_features']['blink_rate']['mean']:.1f} blinks/min")
        logger.info(f"  PERCLOS: {report['window_features']['perclos']['mean']:.3f}")

    if report["cli_predictions"]:
        logger.info("\nCLI Predictions:")
        logger.info(f"  Count: {report['cli_predictions']['count']}")
        logger.info(f"  Mean CLI: {report['cli_predictions']['mean']:.3f}")
        logger.info(f"  CLI range: {report['cli_predictions']['min']:.3f}-"
                    f"{report['cli_predictions']['max']:.3f}")

    if report["warnings"]:
        logger.info("\nWarnings:")
        for warning in report["warnings"]:
            logger.info(f"  ⚠️  {warning}")

    logger.info("=" * 80)

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate webcam feature extraction"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration to run test in seconds (default: 60)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models/",
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Video source (default: 0 for webcam)",
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
    setup_logging(level=args.log_level, log_dir="logs", log_file="webcam_validation.log")

    # Run validation
    report = validate_webcam_features(
        duration_s=args.duration,
        models_dir=args.models,
        config_path=args.config,
        output_dir=args.output,
        source=args.source,
    )

    # Exit with appropriate code
    if report["status"] == "PASS":
        sys.exit(0)
    elif report["status"] == "PARTIAL":
        sys.exit(1)
    else:  # FAIL
        sys.exit(2)


if __name__ == "__main__":
    main()
