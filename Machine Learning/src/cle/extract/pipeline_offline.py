"""
Offline video processing pipeline.

Extracts features from video files in batch mode.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.cle.config import load_config
from src.cle.extract.features import compute_window_features, get_feature_names
from src.cle.extract.landmarks import FaceMeshExtractor
from src.cle.extract.per_frame import extract_frame_features
from src.cle.extract.windowing import (
    interpolate_gaps,
    sliding_window_indices,
    validate_window_quality,
)
from src.cle.logging_setup import get_logger, setup_logging
from src.cle.utils.io import load_manifest, open_video, save_features_csv
from src.cle.utils.timers import timer

logger = get_logger(__name__)


def process_video(
    video_path: str,
    config,
    extractor: FaceMeshExtractor,
    video_metadata: Dict,
) -> List[Dict]:
    """
    Process a single video and extract per-frame features.

    Args:
        video_path: Path to video file
        config: Configuration dictionary
        extractor: FaceMeshExtractor instance
        video_metadata: Video metadata from manifest

    Returns:
        List of per-frame feature dictionaries
    """
    cap, vid_meta = open_video(video_path)
    fps = vid_meta["fps"] if vid_meta["fps"] > 0 else config.get("fps_fallback", 30.0)

    frame_features = []
    frame_idx = 0

    with tqdm(total=vid_meta["frame_count"], desc=f"Processing {Path(video_path).name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks
            landmark_result = extractor.process_frame(frame)

            # Extract per-frame features
            features = extract_frame_features(frame, landmark_result)
            features["frame_idx"] = frame_idx
            features["timestamp_s"] = frame_idx / fps

            frame_features.append(features)
            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Interpolate only single-frame dropouts (avoid creating fake blinks)
    frame_features = interpolate_gaps(frame_features, max_gap=1)

    logger.info(
        f"Processed {len(frame_features)} frames from {Path(video_path).name} "
        f"({len([f for f in frame_features if f['valid']])} valid)"
    )

    return frame_features


def extract_windows(
    frame_features: List[Dict],
    fps: float,
    config,
    video_metadata: Dict,
) -> List[Dict]:
    """
    Extract window-level features from per-frame features.

    Args:
        frame_features: List of per-frame feature dictionaries
        fps: Frames per second
        config: Configuration dictionary
        video_metadata: Video metadata from manifest

    Returns:
        List of window feature dictionaries
    """
    # Generate sliding windows
    windows = sliding_window_indices(
        n_frames=len(frame_features),
        fps=fps,
        length_s=config.get("windows.length_s", 10.0),
        step_s=config.get("windows.step_s", 2.5),
    )

    window_features_list = []

    for start_idx, end_idx, start_time_s, end_time_s in windows:
        # Extract window data
        window_data = frame_features[start_idx:end_idx]

        # Validate window quality
        is_valid, bad_ratio = validate_window_quality(
            window_data,
            max_bad_ratio=config.get("quality.max_bad_frame_ratio", 0.05),
        )

        if not is_valid:
            logger.debug(
                f"Skipping window [{start_time_s:.1f}s - {end_time_s:.1f}s] "
                f"(bad_ratio={bad_ratio:.2f})"
            )
            continue

        # Compute window features
        window_features = compute_window_features(window_data, config.to_dict(), fps)

        # Add metadata
        window_features.update({
            "t_start_s": start_time_s,
            "t_end_s": end_time_s,
            "user_id": video_metadata["user_id"],
            "task": video_metadata.get("task"),
            "video": video_metadata["video_file"],
            "label": video_metadata["label"],
            "role": video_metadata["role"],
        })

        window_features_list.append(window_features)

    logger.info(
        f"Extracted {len(window_features_list)} valid windows "
        f"from {len(windows)} total windows"
    )

    return window_features_list


def main():
    """Main entry point for offline pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract features from videos in batch mode"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for features CSV",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum number of videos to process (0 = all).",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="logs", log_file="pipeline_offline.log")
    logger.info("=" * 80)
    logger.info("Starting offline feature extraction pipeline")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration (hash: {config.hash()[:8]})")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(str(manifest_path))
    logger.info(f"Loaded manifest with {len(manifest)} videos")
    if args.max_videos and args.max_videos > 0:
        manifest = manifest.head(args.max_videos).copy()
        logger.info(f"Limiting to first {len(manifest)} videos due to --max-videos")

    # Initialize face mesh extractor
    with FaceMeshExtractor(
        min_detection_confidence=config.get("quality.min_face_conf", 0.5),
        refine_landmarks=True,
    ) as extractor:

        all_window_features = []

        # Process each video in manifest
        for idx, row in manifest.iterrows():
            video_file = row["video_file"]

            # Handle relative paths
            if not Path(video_file).is_absolute():
                video_path = manifest_path.parent / video_file
            else:
                video_path = Path(video_file)

            if not video_path.exists():
                logger.warning(f"Video file not found, skipping: {video_path}")
                continue

            logger.info(f"Processing video {idx + 1}/{len(manifest)}: {video_path.name}")

            # Video metadata
            video_metadata = {
                "video_file": video_file,
                "label": row["label"],
                "role": row["role"],
                "user_id": row["user_id"],
                "task": (
                    row["task"]
                    if "task" in manifest.columns and pd.notna(row.get("task"))
                    else Path(video_file).stem.split("_", 1)[-1] if "_" in Path(video_file).stem else "unknown"
                ),
            }

            try:
                with timer(f"Video {video_path.name}"):
                    # Process video to get per-frame features
                    frame_features = process_video(
                        str(video_path),
                        config,
                        extractor,
                        video_metadata,
                    )

                    # Get FPS
                    cap, vid_meta = open_video(str(video_path))
                    fps = vid_meta["fps"] if vid_meta["fps"] > 0 else config.get("fps_fallback", 30.0)
                    cap.release()

                    # Extract window features
                    window_features = extract_windows(
                        frame_features,
                        fps,
                        config,
                        video_metadata,
                    )

                    all_window_features.extend(window_features)

            except Exception as e:
                logger.error(f"Error processing video {video_path.name}: {e}", exc_info=True)
                continue

    # Convert to DataFrame
    if not all_window_features:
        logger.error("No features extracted from any video")
        sys.exit(1)

    df = pd.DataFrame(all_window_features)

    # Reorder columns: metadata first, then model features, then monitoring features
    feature_names = get_feature_names(config.to_dict())
    metadata_cols = ["user_id", "task", "video", "label", "role", "t_start_s", "t_end_s"]
    ordered_cols = metadata_cols + feature_names

    # Ensure all model columns exist
    for col in ordered_cols:
        if col not in df.columns:
            logger.warning(f"Missing column {col}, adding with zeros")
            df[col] = "" if col in metadata_cols else 0.0

    # Also keep monitoring/quality columns that are computed but not model inputs
    monitoring_cols = ["mean_brightness", "std_brightness", "mean_quality", "valid_frame_ratio"]
    for col in monitoring_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    df = df[ordered_cols]

    # Save features
    save_features_csv(df, args.out, include_index=False)

    # Summary statistics
    logger.info("=" * 80)
    logger.info("Feature extraction complete!")
    logger.info(f"Total windows extracted: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info(f"Features saved to: {args.out}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
