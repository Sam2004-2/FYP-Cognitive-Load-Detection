#!/usr/bin/env python3
"""
Extract StressID video features from a manifest CSV using MediaPipe FaceLandmarker (CPU).

This is an alternative to src/cle/extract/pipeline_offline.py for environments where
mp.solutions.face_mesh requires an OpenGL context (headless/CI/macOS issues).

Input manifest columns (minimum):
  - video_file, label, role, user_id
Optional:
  - task

Output CSV schema matches the rest of the project:
  user_id,task,video,label,role,t_start_s,t_end_s,<9 base window features>
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

# Ensure src imports work when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cle.config import load_config
from src.cle.data.load_data import extract_task_from_video_path
from src.cle.extract.features import compute_window_features, get_feature_names
from src.cle.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_coords: List[Tuple[float, float, float]]) -> float:
    if len(eye_coords) != 6:
        return 0.0
    points = [(p[0], p[1]) for p in eye_coords]
    v1 = euclidean_distance(points[1], points[5])
    v2 = euclidean_distance(points[2], points[4])
    h = euclidean_distance(points[0], points[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_brightness(frame: np.ndarray, landmarks) -> float:
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
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def extract_frame_features(frame: np.ndarray, face_landmarker: vision.FaceLandmarker) -> Dict:
    invalid = {
        "ear_left": 0.0,
        "ear_right": 0.0,
        "ear_mean": 0.0,
        "brightness": 0.0,
        "quality": 0.0,
        "valid": False,
    }

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = face_landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        return invalid

    landmarks = detection_result.face_landmarks[0]
    left_eye = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in LEFT_EYE_INDICES]
    right_eye = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in RIGHT_EYE_INDICES]

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    ear_mean = (ear_left + ear_right) / 2.0
    brightness = compute_brightness(frame, landmarks)

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": ear_mean,
        "brightness": brightness,
        "quality": 1.0,
        "valid": True,
    }


def process_video(
    video_path: Path,
    face_landmarker: vision.FaceLandmarker,
    fps_fallback: float,
    resize_width: int = 0,
) -> Tuple[List[Dict], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps if fps and fps > 0 else fps_fallback)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_height = 0
    if resize_width and orig_width and orig_width > resize_width:
        scale = resize_width / orig_width
        resize_height = int(orig_height * scale)

    frames: List[Dict] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width and resize_height:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

        feats = extract_frame_features(frame, face_landmarker)
        feats["frame_idx"] = frame_idx
        feats["timestamp_s"] = frame_idx / fps
        frames.append(feats)
        frame_idx += 1

    cap.release()
    return frames, fps


def extract_windows(
    frame_features: List[Dict],
    fps: float,
    config_dict: Dict,
    user_id: str,
    task: str,
    video: str,
    label: str,
    role: str,
) -> List[Dict]:
    length_s = float(config_dict.get("windows", {}).get("length_s", 10.0))
    step_s = float(config_dict.get("windows", {}).get("step_s", 2.5))
    max_bad_ratio = float(config_dict.get("quality", {}).get("max_bad_frame_ratio", 0.05))

    length_frames = int(length_s * fps)
    step_frames = int(step_s * fps)

    out = []
    start = 0
    while start + length_frames <= len(frame_features):
        end = start + length_frames
        window = frame_features[start:end]
        bad_ratio = sum(1 for f in window if not f.get("valid", False)) / len(window) if window else 1.0
        if bad_ratio <= max_bad_ratio:
            feats = compute_window_features(window, config_dict, fps)
            feats.update(
                {
                    "user_id": user_id,
                    "task": task,
                    "video": video,
                    "label": label,
                    "role": role,
                    "t_start_s": start / fps,
                    "t_end_s": end / fps,
                }
            )
            out.append(feats)
        start += step_frames

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract video features from manifest (FaceLandmarker, CPU)")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max-videos", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--resize-width", type=int, default=640, help="Resize width for speed (0 = no resize)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    config_dict = config.to_dict()

    manifest = pd.read_csv(args.manifest)
    if args.max_videos and args.max_videos > 0:
        manifest = manifest.head(args.max_videos).copy()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (Path(__file__).parent.parent / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"FaceLandmarker model not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=float(config.get("quality.min_face_conf", 0.5)),
        min_face_presence_confidence=float(config.get("quality.min_face_conf", 0.5)),
        min_tracking_confidence=float(config.get("quality.min_face_conf", 0.5)),
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    try:
        all_rows: List[Dict] = []

        for idx, row in manifest.iterrows():
            video_file = row["video_file"]
            video_path = Path(video_file)
            if not video_path.is_absolute():
                video_path = (Path(args.manifest).parent / video_path).resolve()
            if not video_path.exists():
                logger.warning(f"Missing video, skipping: {video_path}")
                continue

            user_id = str(row["user_id"])
            task = str(row["task"]) if "task" in manifest.columns and pd.notna(row.get("task")) else (
                extract_task_from_video_path(str(video_file)) or ""
            )
            label = str(row["label"])
            role = str(row["role"])

            logger.info(f"[{idx+1}/{len(manifest)}] Processing {user_id}/{task}: {video_path.name}")
            frames, fps = process_video(video_path, face_landmarker, fps_fallback=float(config.get("fps_fallback", 30.0)), resize_width=args.resize_width)
            windows = extract_windows(
                frame_features=frames,
                fps=fps,
                config_dict=config_dict,
                user_id=user_id,
                task=task,
                video=str(video_file),
                label=label,
                role=role,
            )
            all_rows.extend(windows)

        if not all_rows:
            raise RuntimeError("No features extracted from any videos.")

        df = pd.DataFrame(all_rows)
        feature_names = get_feature_names(config_dict)
        metadata_cols = ["user_id", "task", "video", "label", "role", "t_start_s", "t_end_s"]
        ordered = metadata_cols + feature_names
        for col in ordered:
            if col not in df.columns:
                df[col] = "" if col in metadata_cols else 0.0
        df = df[ordered]

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} windows to {out_path}")
        return 0

    finally:
        face_landmarker.close()


if __name__ == "__main__":
    raise SystemExit(main())

