"""
Extract features from video files using MediaPipe FaceLandmarker.

Processes video files and extracts per-frame and window-level features
compatible with the cognitive load estimation model.

Usage:
    python scripts/extract_video_features.py --input_dir <videos_dir> --output <output.csv>
    
Example:
    python scripts/extract_video_features.py \
        --input_dir "E:/FYP/Dataset/AVCAffe/codes/downloader/data/videos/per_participant_per_task" \
        --output "data/processed/avcaffe_features.csv"
"""

import argparse
import csv
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from src.cle.config import Config
from src.cle.extract.features import compute_window_features
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# MediaPipe landmark indices for eyes (same as frontend)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Model URL for face landmarker
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Processing optimization settings
DEFAULT_RESIZE_WIDTH = 640  # Resize frames for faster processing
DEFAULT_NUM_WORKERS = 4  # Number of parallel workers


def download_model(model_path: Path) -> None:
    """Download the face landmarker model if not present."""
    if model_path.exists():
        return
    
    logger.info(f"Downloading face landmarker model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, str(model_path))
    logger.info("Model downloaded successfully")


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_coords: List[Tuple[float, float, float]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Reference: Dewi et al. (2022)
    
    Args:
        eye_coords: 6 eye landmark coordinates [outer, top1, top2, inner, bottom1, bottom2]
    
    Returns:
        EAR value (typically 0.25-0.35 for open eye, <0.21 for blink)
    """
    if len(eye_coords) != 6:
        return 0.0
    
    # Extract 2D coordinates
    points = [(p[0], p[1]) for p in eye_coords]
    
    # Vertical distances
    v1 = euclidean_distance(points[1], points[5])  # top1 to bottom1
    v2 = euclidean_distance(points[2], points[4])  # top2 to bottom2
    
    # Horizontal distance
    h = euclidean_distance(points[0], points[3])  # outer to inner
    
    if h < 1e-6:
        return 0.0
    
    return (v1 + v2) / (2.0 * h)


def compute_brightness(frame: np.ndarray, landmarks: Optional[List] = None) -> float:
    """
    Compute mean brightness of frame (optionally in face ROI).
    
    Args:
        frame: BGR image frame
        landmarks: Optional MediaPipe landmarks to extract face ROI
    
    Returns:
        Mean brightness value (0-255)
    """
    if landmarks is not None:
        # Extract face bounding box from landmarks
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        
        x_min = max(0, int(min(xs) - 0.1 * (max(xs) - min(xs))))
        x_max = min(w, int(max(xs) + 0.1 * (max(xs) - min(xs))))
        y_min = max(0, int(min(ys) - 0.1 * (max(ys) - min(ys))))
        y_max = min(h, int(max(ys) + 0.1 * (max(ys) - min(ys))))
        
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            roi = frame
    else:
        roi = frame
    
    # Convert to grayscale and compute mean
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def extract_frame_features(
    frame: np.ndarray,
    face_landmarker: vision.FaceLandmarker,
    min_detection_confidence: float = 0.5
) -> Dict:
    """
    Extract per-frame features from a video frame using MediaPipe.
    
    Args:
        frame: BGR image frame
        face_landmarker: MediaPipe FaceLandmarker instance
        min_detection_confidence: Minimum face detection confidence
    
    Returns:
        Dictionary with per-frame features
    """
    invalid_result = {
        "ear_left": 0.0,
        "ear_right": 0.0,
        "ear_mean": 0.0,
        "brightness": 0.0,
        "quality": 0.0,
        "valid": False,
    }
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect face landmarks
    detection_result = face_landmarker.detect(mp_image)
    
    if not detection_result.face_landmarks:
        return invalid_result
    
    # Use first detected face
    landmarks = detection_result.face_landmarks[0]
    
    # Extract eye landmarks (landmarks are normalized [0,1])
    left_eye_coords = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in LEFT_EYE_INDICES]
    right_eye_coords = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in RIGHT_EYE_INDICES]
    
    # Calculate EAR for both eyes
    ear_left = eye_aspect_ratio(left_eye_coords)
    ear_right = eye_aspect_ratio(right_eye_coords)
    ear_mean = (ear_left + ear_right) / 2.0
    
    # Compute brightness using face bounding box
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min = max(0, int(min(xs) - 0.1 * (max(xs) - min(xs))))
    x_max = min(w, int(max(xs) + 0.1 * (max(xs) - min(xs))))
    y_min = max(0, int(min(ys) - 0.1 * (max(ys) - min(ys))))
    y_max = min(h, int(max(ys) + 0.1 * (max(ys) - min(ys))))
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    
    # Quality score (detection confidence - use presence score if available)
    # FaceLandmarker doesn't provide per-landmark visibility, use a default high value
    quality = 0.9
    
    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": ear_mean,
        "brightness": brightness,
        "quality": quality,
        "valid": True,
    }


def process_video(
    video_path: Path,
    config: Dict,
    face_landmarker: vision.FaceLandmarker,
    resize_width: int = DEFAULT_RESIZE_WIDTH
) -> List[Dict]:
    """
    Process a video file and extract window-level features.
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
        face_landmarker: MediaPipe FaceLandmarker instance
        resize_width: Width to resize frames for faster processing (0 = no resize)
    
    Returns:
        List of window feature dictionaries
    """
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        fps = config.get("fps_fallback", 30.0)
        logger.warning(f"Could not detect FPS, using fallback: {fps}")
    
    # Calculate resize dimensions
    if resize_width > 0 and orig_width > resize_width:
        scale = resize_width / orig_width
        resize_height = int(orig_height * scale)
        logger.info(f"Video: {total_frames} frames @ {fps:.1f} FPS, resizing {orig_width}x{orig_height} -> {resize_width}x{resize_height}")
    else:
        resize_width = 0
        resize_height = 0
        logger.info(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    
    # Windowing parameters
    window_length_s = config.get("windows", {}).get("length_s", 10.0)
    window_step_s = config.get("windows", {}).get("step_s", 2.5)
    window_length_frames = int(window_length_s * fps)
    window_step_frames = int(window_step_s * fps)
    
    # Collect all frame features
    all_frame_features = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if needed
        if resize_width > 0:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        
        features = extract_frame_features(frame, face_landmarker)
        features["frame_idx"] = frame_idx
        features["timestamp"] = frame_idx / fps
        all_frame_features.append(features)
        
        frame_idx += 1
        
        # Progress logging every 1000 frames
        if frame_idx % 1000 == 0:
            logger.info(f"  Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.0f}%)")
    
    cap.release()
    
    logger.info(f"Extracted {len(all_frame_features)} frame features")
    
    # Compute window-level features
    window_features_list = []
    
    for window_start in range(0, len(all_frame_features) - window_length_frames + 1, window_step_frames):
        window_end = window_start + window_length_frames
        window_data = all_frame_features[window_start:window_end]
        
        # Use existing compute_window_features from features.py
        window_features = compute_window_features(window_data, config, fps)
        
        # Add metadata
        window_features["window_start_frame"] = window_start
        window_features["window_end_frame"] = window_end
        window_features["window_start_s"] = window_start / fps
        window_features["window_end_s"] = window_end / fps
        
        window_features_list.append(window_features)
    
    logger.info(f"Computed {len(window_features_list)} windows")
    
    return window_features_list


def process_video_worker(args: Tuple) -> Tuple[str, str, List[Dict]]:
    """
    Worker function for parallel video processing.
    Each worker creates its own FaceLandmarker instance.
    
    Args:
        args: Tuple of (video_path, participant_id, task, config_dict, model_path, resize_width, use_gpu)
    
    Returns:
        Tuple of (participant_id, task, window_features_list)
    """
    video_path, participant_id, task, config_dict, model_path, resize_width, use_gpu = args
    
    # Create FaceLandmarker for this worker
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    base_options = python.BaseOptions(model_asset_path=str(model_path), delegate=delegate)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    try:
        window_features = process_video(Path(video_path), config_dict, face_landmarker, resize_width)
        return (participant_id, task, window_features)
    finally:
        face_landmarker.close()


def discover_videos(input_dir: Path, extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")) -> List[Dict]:
    """
    Discover video files in directory structure.
    
    Expects structure: input_dir/<participant_id>/<task_file>
    
    Args:
        input_dir: Root directory to search
        extensions: Tuple of valid video extensions
    
    Returns:
        List of dicts with video_path, participant_id, task
    """
    videos = []
    
    for participant_dir in sorted(input_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        
        participant_id = participant_dir.name
        
        for video_file in sorted(participant_dir.iterdir()):
            if video_file.suffix.lower() in extensions:
                task = video_file.stem  # e.g., "task_1" from "task_1.mp4"
                videos.append({
                    "video_path": video_file,
                    "participant_id": participant_id,
                    "task": task,
                })
    
    return videos


def main():
    """Main entry point for video feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from video files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing participant video folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/video_features.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--max_participants",
        type=int,
        default=0,
        help="Maximum number of participants to process (0 = all)"
    )
    parser.add_argument(
        "--start_participant",
        type=int,
        default=0,
        help="Skip first N participants (for parallel processing across terminals)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have features in output file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})"
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=DEFAULT_RESIZE_WIDTH,
        help=f"Resize frames to this width (0 = no resize, default: {DEFAULT_RESIZE_WIDTH})"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (requires NVIDIA GPU with CUDA)"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Load configuration
    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()
    
    logger.info(f"Loaded config from {args.config}")
    
    # Discover videos
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    videos = discover_videos(input_dir)
    logger.info(f"Found {len(videos)} videos")
    
    # Get all unique participants and apply start/max filters
    all_participants = sorted(set(v["participant_id"] for v in videos))
    
    if args.start_participant > 0:
        all_participants = all_participants[args.start_participant:]
        logger.info(f"Skipping first {args.start_participant} participants")
    
    if args.max_participants > 0:
        all_participants = all_participants[:args.max_participants]
    
    if args.start_participant > 0 or args.max_participants > 0:
        videos = [v for v in videos if v["participant_id"] in all_participants]
        logger.info(f"Processing {len(videos)} videos from {len(all_participants)} participants")
    
    # Check for existing output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_keys = set()
    if args.skip_existing and output_path.exists():
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['participant_id']}_{row['task']}"
                existing_keys.add(key)
        logger.info(f"Found {len(existing_keys)} existing entries, will skip")
    
    # Download MediaPipe model
    model_path = Path("models/face_landmarker.task").resolve()
    download_model(model_path)
    
    # Filter videos to process
    videos_to_process = []
    for video_info in videos:
        key = f"{video_info['participant_id']}_{video_info['task']}"
        if key not in existing_keys:
            videos_to_process.append(video_info)
        else:
            logger.info(f"Skipping {key} (already processed)")
    
    if args.gpu:
        logger.info("GPU acceleration enabled")
    
    logger.info(f"Will process {len(videos_to_process)} videos with {args.workers} workers")
    
    # Process videos in parallel using threads
    all_results = []
    processed_count = 0
    total_videos = len(videos_to_process)
    
    if args.workers > 1 and len(videos_to_process) > 1:
        # Parallel processing with ThreadPoolExecutor
        # Each thread gets its own FaceLandmarker instance
        worker_args = [
            (
                str(v["video_path"]),
                v["participant_id"],
                v["task"],
                config_dict,
                str(model_path),
                args.resize_width,
                args.gpu
            )
            for v in videos_to_process
        ]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_video_worker, arg): arg for arg in worker_args}
            
            for future in as_completed(futures):
                try:
                    participant_id, task, window_features = future.result()
                    
                    video_path = next(
                        v["video_path"] for v in videos_to_process 
                        if v["participant_id"] == participant_id and v["task"] == task
                    )
                    
                    for i, features in enumerate(window_features):
                        result = {
                            "participant_id": participant_id,
                            "task": task,
                            "window_idx": i,
                            "video_file": str(video_path),
                            **features
                        }
                        all_results.append(result)
                    
                    processed_count += 1
                    logger.info(f"Completed {processed_count}/{total_videos}: {participant_id}/{task} ({len(window_features)} windows)")
                    
                except Exception as e:
                    arg = futures[future]
                    logger.error(f"Error processing {arg[0]}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential processing (single worker or single video)
        delegate = python.BaseOptions.Delegate.GPU if args.gpu else python.BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(model_asset_path=str(model_path), delegate=delegate)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        for video_info in videos_to_process:
            video_path = video_info["video_path"]
            participant_id = video_info["participant_id"]
            task = video_info["task"]
            
            try:
                window_features = process_video(video_path, config_dict, face_landmarker, args.resize_width)
                
                for i, features in enumerate(window_features):
                    result = {
                        "participant_id": participant_id,
                        "task": task,
                        "window_idx": i,
                        "video_file": str(video_path),
                        **features
                    }
                    all_results.append(result)
                
                processed_count += 1
                logger.info(f"Completed {processed_count}/{total_videos}: {participant_id}/{task} ({len(window_features)} windows)")
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        face_landmarker.close()
    
    # Write results to CSV
    if all_results:
        # Get all column names
        all_columns = set()
        for result in all_results:
            all_columns.update(result.keys())
        
        # Organize columns
        meta_columns = ["participant_id", "task", "window_idx", "video_file", 
                       "window_start_s", "window_end_s", "window_start_frame", "window_end_frame"]
        feature_columns = sorted([c for c in all_columns if c not in meta_columns])
        columns = meta_columns + feature_columns
        
        # Append or write
        file_exists = output_path.exists() and args.skip_existing
        mode = "a" if file_exists else "w"
        
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"Saved {len(all_results)} window features to {output_path}")
    else:
        logger.warning("No results to save")
    
    logger.info("Feature extraction complete!")
    return 0


if __name__ == "__main__":
    exit(main())
