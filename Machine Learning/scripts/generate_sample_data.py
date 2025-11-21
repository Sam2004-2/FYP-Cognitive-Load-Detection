"""
Generate synthetic sample videos for testing.

Creates simple videos with simulated face-like shapes and varying pupil sizes
to test the full pipeline without requiring real video data.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def draw_face(
    frame: np.ndarray,
    center: tuple,
    pupil_size: float,
    eye_open_ratio: float = 1.0,
) -> np.ndarray:
    """
    Draw a simple face with eyes and pupils.

    Args:
        frame: Output frame
        center: Face center (x, y)
        pupil_size: Pupil diameter (normalized)
        eye_open_ratio: Eye openness (0=closed, 1=fully open)

    Returns:
        Frame with face drawn
    """
    cx, cy = center

    # Draw face circle
    cv2.circle(frame, (cx, cy), 80, (200, 180, 150), -1)  # Skin tone

    # Draw eyes
    left_eye_center = (cx - 30, cy - 20)
    right_eye_center = (cx + 30, cy - 20)
    eye_radius = 15

    # Eye openness affects vertical size
    eye_height = int(eye_radius * eye_open_ratio)

    # Draw eye whites
    cv2.ellipse(frame, left_eye_center, (eye_radius, eye_height), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, right_eye_center, (eye_radius, eye_height), 0, 0, 360, (255, 255, 255), -1)

    # Draw pupils
    pupil_radius = int(5 + pupil_size * 5)  # 5-10 pixels
    if eye_open_ratio > 0.3:  # Only draw pupils if eyes are reasonably open
        cv2.circle(frame, left_eye_center, pupil_radius, (50, 50, 50), -1)
        cv2.circle(frame, right_eye_center, pupil_radius, (50, 50, 50), -1)

    # Draw nose
    nose_pts = np.array([
        [cx, cy],
        [cx - 8, cy + 20],
        [cx + 8, cy + 20]
    ], np.int32)
    cv2.polylines(frame, [nose_pts], False, (150, 130, 100), 2)

    # Draw mouth
    cv2.ellipse(frame, (cx, cy + 40), (20, 8), 0, 0, 180, (100, 80, 80), 2)

    return frame


def generate_video(
    output_path: str,
    duration_s: float,
    fps: float,
    pupil_baseline: float,
    pupil_variation: float,
    blink_rate: float,
) -> None:
    """
    Generate synthetic video.

    Args:
        output_path: Path to output video file
        duration_s: Video duration in seconds
        fps: Frames per second
        pupil_baseline: Baseline pupil size (0-1)
        pupil_variation: Pupil size variation amplitude (0-1)
        blink_rate: Blinks per minute
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Video parameters
    width, height = 640, 480
    n_frames = int(duration_s * fps)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    # Generate pupil size time series
    t = np.linspace(0, duration_s, n_frames)

    # Baseline + slow variations (task load) + fast variations (natural fluctuation)
    pupil_size = (
        pupil_baseline
        + pupil_variation * np.sin(2 * np.pi * 0.1 * t)  # Slow variation
        + 0.05 * np.random.randn(n_frames)  # Noise
    )
    pupil_size = np.clip(pupil_size, 0.0, 1.0)

    # Generate blinks
    blink_frames = []
    if blink_rate > 0:
        blink_interval_frames = int((60.0 / blink_rate) * fps)
        # Add randomness to intervals
        current_frame = np.random.randint(0, blink_interval_frames)
        while current_frame < n_frames:
            blink_frames.append(current_frame)
            # Next blink with +/- 20% variability
            variation = int(blink_interval_frames * 0.2)
            next_interval = blink_interval_frames + np.random.randint(-variation, variation)
            current_frame += next_interval
    
    print(f"Generating {output_path.name}: {n_frames} frames @ {fps} fps")
    print(f"  Pupil: baseline={pupil_baseline:.2f}, variation={pupil_variation:.2f}")
    print(f"  Blinks: {len(blink_frames)} blinks (~{blink_rate:.1f}/min)")

    # Generate frames
    for frame_idx in range(n_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 180  # Gray background

        # Determine eye openness (blink state)
        eye_open_ratio = 1.0
        for blink_frame in blink_frames:
            # Blink lasts ~4-6 frames (120-200ms at 30fps)
            blink_duration = 4
            dist = frame_idx - blink_frame
            if 0 <= dist < blink_duration:
                # Simple closing/opening animation
                if dist < blink_duration / 2:
                    eye_open_ratio = 0.1  # Closing
                else:
                    eye_open_ratio = 0.5  # Opening

        # Draw face
        face_center = (width // 2, height // 2)
        draw_face(frame, face_center, pupil_size[frame_idx], eye_open_ratio)

        # Write frame
        writer.write(frame)

    writer.release()
    print(f"  Saved to {output_path}")


def generate_participant(
    subject_id: str,
    output_dir: Path,
    fps: float = 30.0
):
    """Generate data for a single participant with randomized traits."""
    subject_dir = output_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomize physiological traits
    # Base pupil size (0.3 - 0.6)
    base_pupil = np.random.uniform(0.3, 0.6)
    # Base blink rate (12 - 22 per min)
    base_blink = np.random.uniform(12, 22)
    
    print(f"\nGenerating {subject_id}:")
    print(f"  Traits: Base Pupil={base_pupil:.2f}, Base Blink={base_blink:.1f}/min")

    # 1. Calibration (Baseline)
    # Normal state
    generate_video(
        output_path=subject_dir / "calib_60s.mp4",
        duration_s=60.0,
        fps=fps,
        pupil_baseline=base_pupil,
        pupil_variation=0.05,
        blink_rate=base_blink,
    )

    # 2. Low Load
    # Slightly dilated pupil (+5%), slightly higher blink rate (+10%)
    # (Low load often increases blink rate due to engagement without stress)
    generate_video(
        output_path=subject_dir / "task_low_1.mp4",
        duration_s=60.0,
        fps=fps,
        pupil_baseline=base_pupil * 1.05, 
        pupil_variation=0.08,
        blink_rate=base_blink * 1.1,
    )

    # 3. High Load
    # Dilated pupil (+15-25%), SUPPRESSED blink rate (-50%)
    # (High load consistently suppresses blinks)
    generate_video(
        output_path=subject_dir / "task_high_1.mp4",
        duration_s=60.0,
        fps=fps,
        pupil_baseline=base_pupil * 1.20, 
        pupil_variation=0.03,  # Less variation (focused)
        blink_rate=base_blink * 0.5,
    )

    # Generate manifest
    manifest_data = {
        "video_file": ["calib_60s.mp4", "task_low_1.mp4", "task_high_1.mp4"],
        "label": ["none", "low", "high"],
        "role": ["calibration", "train", "train"],
        "user_id": [subject_id] * 3,
        "notes": ["neutral baseline", "low cognitive load", "high cognitive load"],
    }
    
    pd.DataFrame(manifest_data).to_csv(subject_dir / "meta.csv", index=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic sample videos")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Root output directory",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=1,
        help="Number of subjects to generate",
    )

    args = parser.parse_args()
    root_dir = Path(args.output_dir)

    print("=" * 80)
    print(f"Generating synthetic data for {args.num_subjects} participants")
    print("=" * 80)

    for i in range(args.num_subjects):
        subject_id = f"synthetic_{i+1:03d}"
        generate_participant(subject_id, root_dir, args.fps)

    print("\n" + "=" * 80)
    print("Generation complete!")
    print(f"Output directory: {root_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
