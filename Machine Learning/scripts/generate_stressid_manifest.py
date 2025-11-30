"""
Generate manifest CSV for StressID dataset videos.

Creates a manifest file mapping videos to cognitive load labels:
- HIGH load: Math, Counting2
- LOW load: Counting1, Speaking

Usage:
    python scripts/generate_stressid_manifest.py --videos_dir ../Videos --participants 10 --output data/raw/stressid_manifest.csv
"""

import argparse
from pathlib import Path


# Task to label mapping based on cognitive load research
TASK_LABELS = {
    "Math": "high",       # Mental arithmetic - high cognitive load
    "Counting2": "high",  # Complex counting - high cognitive load
    "Counting1": "low",   # Simple counting - low cognitive load
    "Speaking": "low",    # Free speaking - low cognitive load
}


def generate_manifest(videos_dir: Path, num_participants: int = 0) -> list:
    """
    Generate manifest entries for StressID videos.
    
    Args:
        videos_dir: Path to Videos folder
        num_participants: Number of participants to include (0 = all)
    
    Returns:
        List of manifest entries (dicts)
    """
    # Get participant directories
    participant_dirs = sorted([
        d for d in videos_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    
    if num_participants > 0:
        participant_dirs = participant_dirs[:num_participants]
    
    print(f"Found {len(participant_dirs)} participants")
    
    entries = []
    missing_videos = []
    
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        
        for task, label in TASK_LABELS.items():
            video_file = participant_dir / f"{participant_id}_{task}.mp4"
            
            if video_file.exists():
                # Use absolute path for reliable resolution
                absolute_path = str(video_file.resolve())
                
                entries.append({
                    "video_file": absolute_path,
                    "label": label,
                    "role": "train",  # All data used for training
                    "user_id": participant_id,
                    "task": task,
                    "participant_id": participant_id,
                })
            else:
                missing_videos.append(f"{participant_id}/{task}")
    
    if missing_videos:
        print(f"Warning: {len(missing_videos)} videos not found")
        for v in missing_videos[:5]:
            print(f"  - {v}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos) - 5} more")
    
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate StressID manifest")
    parser.add_argument("--videos_dir", type=str, default="../Videos",
                       help="Path to Videos folder")
    parser.add_argument("--output", type=str, default="data/raw/stressid_manifest.csv",
                       help="Output manifest CSV path")
    parser.add_argument("--participants", type=int, default=10,
                       help="Number of participants (0 = all)")
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"Error: Videos directory not found: {videos_dir}")
        return 1
    
    # Generate manifest entries
    entries = generate_manifest(videos_dir, args.participants)
    
    if not entries:
        print("Error: No videos found!")
        return 1
    
    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Header (must include role and user_id for pipeline_offline.py)
        f.write("video_file,label,role,user_id,task,notes\n")
        
        # Data rows
        for entry in entries:
            f.write(f"{entry['video_file']},{entry['label']},{entry['role']},{entry['user_id']},{entry['task']},StressID\n")
    
    print(f"\nGenerated manifest: {output_path}")
    print(f"Total entries: {len(entries)}")
    
    # Summary
    high_count = sum(1 for e in entries if e['label'] == 'high')
    low_count = sum(1 for e in entries if e['label'] == 'low')
    print(f"  HIGH load: {high_count} videos (Math, Counting2)")
    print(f"  LOW load: {low_count} videos (Counting1, Speaking)")
    
    return 0


if __name__ == "__main__":
    exit(main())

