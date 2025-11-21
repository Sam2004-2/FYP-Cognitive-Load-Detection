"""
Automated webcam feature validation script.

Runs real-time pipeline for a specified duration and generates a validation report.
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.api import load_model
from src.cle.config import load_config
from src.cle.extract.features import compute_window_features, detect_blinks
from src.cle.extract.landmarks import FaceMeshExtractor
from src.cle.extract.per_frame import extract_frame_features
from src.cle.extract.windowing import WindowBuffer, validate_window_quality
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class PilotRecordingStatus:
    active: bool = False
    label: str = ""
    role: str = ""
    filename: str = ""
    elapsed: float = 0.0


class PilotRecorder:
    """
    Helper for pilot study recordings.
    Manages folder structure, manifest, and OpenCV VideoWriter lifecycle.
    """

    def __init__(
        self,
        root_dir: str,
        participant_id: str,
        fps: float,
    ):
        timestamp_id = datetime.now().strftime("participant_%Y%m%d_%H%M%S")
        self.participant_id = participant_id or timestamp_id
        self.root_dir = Path(root_dir)
        self.participant_dir = self.root_dir / self.participant_id
        self.participant_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.participant_dir / "meta.csv"
        self.video_writer: cv2.VideoWriter | None = None
        self.recording_label: str | None = None
        self.recording_role: str | None = None
        self.recording_file: Path | None = None
        self.record_start_time: float | None = None
        self.fps = fps

        if not self.manifest_path.exists():
            with self.manifest_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["video_file", "label", "role", "user_id", "notes"],
                )
                writer.writeheader()

    def is_recording(self) -> bool:
        return self.video_writer is not None

    def start(self, label: str, role: str, frame_shape) -> bool:
        if self.is_recording():
            logger.warning("Recording already in progress. Stop before starting a new one.")
            return False

        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.mp4"
        output_path = self.participant_dir / filename

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            logger.error("Unable to start recording. VideoWriter failed to open.")
            return False

        self.video_writer = writer
        self.recording_label = label
        self.recording_role = role
        self.recording_file = output_path
        self.record_start_time = time.time()

        logger.info(f"Started {label} recording for participant {self.participant_id}")
        return True

    def write_frame(self, frame):
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def stop(self):
        if not self.is_recording():
            logger.warning("Stop requested but no active recording.")
            return None

        self.video_writer.release()
        self.video_writer = None
        elapsed = time.time() - self.record_start_time if self.record_start_time else 0

        manifest_row = {
            "video_file": self.recording_file.name if self.recording_file else "",
            "label": self.recording_label,
            "role": self.recording_role,
            "user_id": self.participant_id,
            "notes": f"Recorded via GUI on {datetime.now().isoformat()}",
        }

        with self.manifest_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video_file", "label", "role", "user_id", "notes"],
            )
            writer.writerow(manifest_row)

        logger.info(
            f"Saved recording {self.recording_file.name} "
            f"({self.recording_label}, {elapsed:.1f}s)"
        )

        status = PilotRecordingStatus(
            active=False,
            label=self.recording_label or "",
            role=self.recording_role or "",
            filename=str(self.recording_file) if self.recording_file else "",
            elapsed=elapsed,
        )

        self.recording_label = None
        self.recording_role = None
        self.recording_file = None
        self.record_start_time = None

        return status

    def status(self) -> PilotRecordingStatus:
        if not self.is_recording():
            return PilotRecordingStatus(active=False)

        elapsed = time.time() - self.record_start_time if self.record_start_time else 0
        return PilotRecordingStatus(
            active=True,
            label=self.recording_label or "",
            role=self.recording_role or "",
            filename=str(self.recording_file) if self.recording_file else "",
            elapsed=elapsed,
        )

    def shutdown(self):
        if self.is_recording():
            self.stop()


def render_metrics_dashboard(
    frame_features: dict,
    window_features: dict,
    cumulative_stats: dict,
    buffer_ready: bool,
    window_quality: str,
    pilot_status: PilotRecordingStatus | None = None,
    width: int = 640,
    height: int = 720,
) -> np.ndarray:
    """
    Render comprehensive metrics dashboard.

    Args:
        frame_features: Current per-frame features
        window_features: Current window features (if available)
        cumulative_stats: Cumulative session statistics
        buffer_ready: Whether window buffer is ready
        window_quality: "Good" or "Low"
        width: Dashboard width
        height: Dashboard height

    Returns:
        Dashboard image (BGR)
    """
    # Create dark canvas
    dashboard = np.zeros((height, width, 3), dtype=np.uint8)
    dashboard[:, :] = (30, 30, 30)  # Dark gray background

    # Define colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_GRAY = (128, 128, 128)
    COLOR_HEADER = (200, 200, 200)

    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_HEADER = 0.6
    FONT_SCALE_TEXT = 0.5
    FONT_SCALE_SMALL = 0.4
    THICKNESS = 1
    LINE_HEIGHT = 25
    SMALL_LINE_HEIGHT = 20

    y_pos = 30

    # Helper function to add text
    def add_text(text, color=COLOR_WHITE, x=10, scale=FONT_SCALE_TEXT):
        nonlocal y_pos
        cv2.putText(dashboard, text, (x, y_pos), FONT, scale, color, THICKNESS)
        y_pos += LINE_HEIGHT

    def add_small_text(text, color=COLOR_WHITE, x=10):
        nonlocal y_pos
        cv2.putText(dashboard, text, (x, y_pos), FONT, FONT_SCALE_SMALL, color, THICKNESS)
        y_pos += SMALL_LINE_HEIGHT

    def add_separator():
        nonlocal y_pos
        cv2.line(dashboard, (10, y_pos), (width - 10, y_pos), COLOR_GRAY, 1)
        y_pos += 15

    # ========== SECTION 1: Per-Frame Metrics ==========
    add_text("PER-FRAME METRICS", COLOR_HEADER, scale=FONT_SCALE_HEADER)
    add_separator()

    if frame_features and frame_features.get("valid", False):
        ear_mean = frame_features.get("ear_mean", 0.0)
        ear_left = frame_features.get("ear_left", 0.0)
        ear_right = frame_features.get("ear_right", 0.0)
        
        # Color code EAR (red if blinking)
        ear_color = COLOR_RED if ear_mean < 0.21 else COLOR_GREEN
        
        add_small_text(f"EAR Mean:  {ear_mean:.3f}", ear_color)
        add_small_text(f"EAR Left:  {ear_left:.3f}")
        add_small_text(f"EAR Right: {ear_right:.3f}")
        
        pupil_mean = frame_features.get("pupil_mean", 0.0)
        pupil_left = frame_features.get("pupil_left", 0.0)
        pupil_right = frame_features.get("pupil_right", 0.0)
        
        add_small_text(f"Pupil Mean:  {pupil_mean:.3f}")
        add_small_text(f"Pupil Left:  {pupil_left:.3f}")
        add_small_text(f"Pupil Right: {pupil_right:.3f}")
        
        brightness = frame_features.get("brightness", 0.0)
        quality = frame_features.get("quality", 0.0)
        
        add_small_text(f"Brightness: {brightness:.1f}")
        
        quality_color = COLOR_GREEN if quality > 0.85 else (COLOR_YELLOW if quality > 0.7 else COLOR_RED)
        add_small_text(f"Quality:    {quality:.3f}", quality_color)
    else:
        add_small_text("No valid frame data", COLOR_RED)

    y_pos += 10

    # ========== SECTION 2: Window Metrics ==========
    add_text("WINDOW METRICS", COLOR_HEADER, scale=FONT_SCALE_HEADER)
    add_separator()

    if buffer_ready and window_features:
        blink_rate = window_features.get("blink_rate", 0.0)
        perclos = window_features.get("perclos", 0.0)
        tepr_delta_mean = window_features.get("tepr_delta_mean", 0.0)
        tepr_delta_peak = window_features.get("tepr_delta_peak", 0.0)
        tepr_baseline = window_features.get("tepr_baseline", 0.0)
        
        add_small_text(f"Blink Rate:     {blink_rate:.1f} /min")
        add_small_text(f"PERCLOS:        {perclos:.3f}")
        add_small_text(f"TEPR Delta Mn:  {tepr_delta_mean:.4f}")
        add_small_text(f"TEPR Delta Pk:  {tepr_delta_peak:.4f}")
        add_small_text(f"TEPR Baseline:  {tepr_baseline:.3f}")
        
        # Window quality indicator
        valid_ratio = cumulative_stats.get("valid_frame_ratio", 0.0)
        add_small_text(f"Valid Frames:   {valid_ratio:.1%}")
    else:
        add_small_text("Buffer not ready...", COLOR_YELLOW)

    y_pos += 10

    # ========== SECTION 3: Cumulative Stats ==========
    add_text("CUMULATIVE STATS", COLOR_HEADER, scale=FONT_SCALE_HEADER)
    add_separator()

    total_blinks = cumulative_stats.get("total_blinks", 0)
    duration = cumulative_stats.get("duration", 0.0)
    avg_fps = cumulative_stats.get("avg_fps", 0.0)
    detection_rate = cumulative_stats.get("detection_rate", 0.0)

    add_small_text(f"Total Blinks:   {total_blinks}")
    add_small_text(f"Duration:       {duration:.1f}s")
    add_small_text(f"Average FPS:    {avg_fps:.1f}")
    add_small_text(f"Detection Rate: {detection_rate:.1%}")

    y_pos += 10

    # ========== SECTION 4: Status ==========
    add_text("STATUS", COLOR_HEADER, scale=FONT_SCALE_HEADER)
    add_separator()

    face_detected = frame_features.get("valid", False) if frame_features else False
    face_icon = "✓" if face_detected else "✗"
    face_color = COLOR_GREEN if face_detected else COLOR_RED
    add_small_text(f"Face Detected:  {face_icon}", face_color)

    buffer_icon = "✓" if buffer_ready else "✗"
    buffer_color = COLOR_GREEN if buffer_ready else COLOR_YELLOW
    add_small_text(f"Buffer Ready:   {buffer_icon}", buffer_color)

    quality_color = COLOR_GREEN if window_quality == "Good" else COLOR_YELLOW
    add_small_text(f"Window Quality: {window_quality}", quality_color)

    y_pos += 20

    # ========== SECTION 5: Pilot Recording ==========
    if pilot_status:
        add_text("PILOT RECORDING", COLOR_HEADER, scale=FONT_SCALE_HEADER)
        add_separator()
        if pilot_status.active:
            add_small_text(f"Recording: {pilot_status.label}", COLOR_RED)
            add_small_text(f"Role: {pilot_status.role}", COLOR_RED)
            add_small_text(f"Elapsed: {pilot_status.elapsed:5.1f}s", COLOR_RED)
        else:
            add_small_text("Idle", COLOR_GRAY)
        y_pos += 10

    # ========== SECTION 6: Controls ==========
    add_text("CONTROLS", COLOR_HEADER, scale=FONT_SCALE_HEADER)
    add_separator()
    add_small_text("'r' - Reset counters", COLOR_GRAY)
    add_small_text("'s' - Save screenshot", COLOR_GRAY)
    if pilot_status:
        add_small_text("'1' - Start Calibration", COLOR_GRAY)
        add_small_text("'2' - Start Low", COLOR_GRAY)
        add_small_text("'3' - Start High", COLOR_GRAY)
        add_small_text("'0'/'x' - Stop recording", COLOR_GRAY)
    add_small_text("'q' - Quit", COLOR_GRAY)

    return dashboard


def validate_webcam_features(
    duration_s: float,
    models_dir: str,
    config_path: str = None,
    output_dir: str = "reports",
    source: int = 0,
    live_display: bool = False,
    pilot_mode: bool = False,
    pilot_output_dir: str = "data/raw",
    pilot_participant_id: str | None = None,
    recording_fps: float = 30.0,
) -> dict:
    """
    Run webcam validation test and generate report.

    Args:
        duration_s: Duration to run test in seconds
        models_dir: Directory containing model artifacts
        config_path: Path to config YAML
        output_dir: Directory to save reports
        source: Video source (0 for webcam)
        live_display: Whether to show live split-screen display
        pilot_mode: Enable pilot recording controls
        pilot_output_dir: Root directory for pilot recordings
        pilot_participant_id: Optional participant identifier
        recording_fps: FPS for saved pilot recordings

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
    
    # Live display tracking variables
    total_blinks = 0
    ear_history = []  # For blink detection
    last_blink_count = 0
    blink_flash_frames = 0  # For visual blink indicator
    current_window_features = None
    current_window_quality = "Unknown"
    reset_counters_flag = False

    if pilot_mode and not live_display:
        logger.warning("Pilot mode requested without live display; enabling live display.")
        live_display = True

    run_indefinitely = pilot_mode or duration_s <= 0
    if run_indefinitely:
        logger.info("Recording until user quits (pilot/indefinite mode).")
    else:
        logger.info(f"Recording for {duration_s} seconds...")
    if live_display:
        logger.info("Live display mode enabled")
        if pilot_mode:
            logger.info("Pilot recording controls enabled")
    logger.info("-" * 80)

    pilot_recorder = None
    if pilot_mode:
        pilot_recorder = PilotRecorder(
            root_dir=pilot_output_dir,
            participant_id=pilot_participant_id,
            fps=recording_fps,
        )

    try:
        while True:
            elapsed = time.time() - start_time
            if not run_indefinitely and elapsed >= duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame_start_time = time.time()

            if pilot_recorder and pilot_recorder.is_recording():
                pilot_recorder.write_frame(frame)

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
                    
                    # Update for live display
                    current_window_features = window_features
                    current_window_quality = "Good"
                else:
                    current_window_quality = "Low"

            # Track blinks for live display
            if live_display and features["valid"]:
                ear_history.append(features["ear_mean"])
                # Keep only last 10 seconds of EAR history
                max_ear_samples = int(10 * fps)
                if len(ear_history) > max_ear_samples:
                    ear_history = ear_history[-max_ear_samples:]
                
                # Detect blinks from EAR history
                if len(ear_history) >= int(fps * 0.5):  # Need at least 0.5s of data
                    ear_array = np.array(ear_history)
                    blinks = detect_blinks(
                        ear_array,
                        fps,
                        ear_threshold=config.get("blinks.ear_threshold", 0.21),
                        min_blink_ms=config.get("blinks.min_duration_ms", 120),
                        max_blink_ms=config.get("blinks.max_duration_ms", 400),
                    )
                    total_blinks = len(blinks)
                    
                    # Check if new blink detected
                    if total_blinks > last_blink_count:
                        blink_flash_frames = int(fps * 0.2)  # Flash for 0.2 seconds
                        last_blink_count = total_blinks

            # Track FPS
            frame_time = time.time() - frame_start_time
            fps_samples.append(1.0 / frame_time if frame_time > 0 else 0)

            frame_count += 1
            
            # Live display rendering
            if live_display:
                elapsed = time.time() - start_time
                
                # Prepare cumulative stats
                cumulative_stats = {
                    "total_blinks": total_blinks,
                    "duration": elapsed,
                    "avg_fps": np.mean(fps_samples[-30:]) if len(fps_samples) >= 30 else np.mean(fps_samples) if fps_samples else 0.0,
                    "detection_rate": face_detected_count / frame_count if frame_count > 0 else 0.0,
                    "valid_frame_ratio": face_detected_count / frame_count if frame_count > 0 else 0.0,
                }
                
                pilot_status = pilot_recorder.status() if pilot_recorder else None

                # Render dashboard
                dashboard = render_metrics_dashboard(
                    frame_features=features,
                    window_features=current_window_features,
                    cumulative_stats=cumulative_stats,
                    buffer_ready=buffer.is_ready(),
                    window_quality=current_window_quality,
                    pilot_status=pilot_status,
                )
                
                # Prepare webcam display
                display_frame = frame.copy()
                
                # Add blink flash effect
                if blink_flash_frames > 0:
                    # Draw thick border
                    cv2.rectangle(
                        display_frame,
                        (0, 0),
                        (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                        (0, 0, 255),
                        10,
                    )
                    blink_flash_frames -= 1
                
                # Add face detection indicator
                if features["valid"]:
                    cv2.circle(display_frame, (30, 30), 15, (0, 255, 0), -1)
                else:
                    cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)

                if pilot_status and pilot_status.active:
                    cv2.putText(
                        display_frame,
                        f"REC {pilot_status.label.upper()} {pilot_status.elapsed:04.1f}s",
                        (60, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                
                # Resize webcam frame to match dashboard height
                target_height = dashboard.shape[0]
                aspect_ratio = display_frame.shape[1] / display_frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                display_frame_resized = cv2.resize(display_frame, (target_width, target_height))
                
                # Combine horizontally
                combined = np.hstack([display_frame_resized, dashboard])
                
                # Display
                cv2.imshow("Metric Validation", combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('r'):
                    # Reset counters
                    logger.info("Resetting counters...")
                    total_blinks = 0
                    last_blink_count = 0
                    ear_history = []
                    start_time = time.time()
                    frame_count = 0
                    face_detected_count = 0
                    fps_samples = []
                    frame_features_list = []
                    window_features_list = []
                    cli_predictions = []
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = Path(output_dir) / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(screenshot_path), combined)
                    logger.info(f"Screenshot saved to {screenshot_path}")
                elif key in (ord('1'), ord('c')):
                    if pilot_recorder:
                        pilot_recorder.start("calibration", "calibration", frame.shape)
                elif key in (ord('2'), ord('l')):
                    if pilot_recorder:
                        pilot_recorder.start("low", "train", frame.shape)
                elif key in (ord('3'), ord('h')):
                    if pilot_recorder:
                        pilot_recorder.start("high", "train", frame.shape)
                elif key in (ord('0'), ord('x'), ord(' ')):
                    if pilot_recorder:
                        pilot_recorder.stop()

            # Progress update every 5 seconds
            if frame_count % (int(fps) * 5) == 0:
                logger.info(f"[{elapsed:6.1f}s] Processed {frame_count} frames")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        extractor.close()
        if live_display:
            cv2.destroyAllWindows()
        if pilot_recorder:
            pilot_recorder.shutdown()

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
    parser.add_argument(
        "--live-display",
        action="store_true",
        help="Show live split-screen display with metrics dashboard",
    )
    parser.add_argument(
        "--pilot-mode",
        action="store_true",
        help="Enable pilot recording controls within the GUI",
    )
    parser.add_argument(
        "--pilot-output",
        type=str,
        default="data/raw",
        help="Root directory to store pilot recordings (default: data/raw)",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default=None,
        help="Participant identifier (default: timestamp-based)",
    )
    parser.add_argument(
        "--recording-fps",
        type=float,
        default=30.0,
        help="FPS to use when saving pilot recordings",
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
        live_display=args.live_display,
        pilot_mode=args.pilot_mode,
        pilot_output_dir=args.pilot_output,
        pilot_participant_id=args.participant_id,
        recording_fps=args.recording_fps,
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
