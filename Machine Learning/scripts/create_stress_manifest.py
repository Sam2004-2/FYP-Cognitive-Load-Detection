"""
Create manifest for stress-based classification training.
Labels based on average stress ratings from self-assessments.

LOW (stress < 3/10): Relax, Breathing, Video1
HIGH (stress > 5/10): Counting2, Stroop, Speaking
"""

import os
from pathlib import Path
import pandas as pd

# Define task labels based on stress ratings
TASK_LABELS = {
    # LOW stress tasks (< 3/10)
    "Relax": "low",      # 2.20/10
    "Breathing": "low",  # 2.29/10
    "Video1": "low",     # 2.92/10
    
    # HIGH stress tasks (> 5/10)
    "Counting2": "high", # 6.05/10
    "Stroop": "high",    # 5.72/10
    "Speaking": "high",  # 5.54/10
}

def create_manifest(videos_dir: str, output_path: str):
    """Create manifest CSV for pipeline_offline.py"""
    videos_dir = Path(videos_dir)
    entries = []
    
    # Iterate through all subject folders
    for subject_dir in sorted(videos_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Look for videos matching our tasks
        for task, label in TASK_LABELS.items():
            video_file = subject_dir / f"{subject_id}_{task}.mp4"
            
            if video_file.exists():
                entries.append({
                    "video_file": str(video_file),
                    "label": label,
                    "role": "train",  # All for training
                    "user_id": subject_id,
                    "task": task,
                    "notes": f"stress_task_{task}"
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(entries)
    df.to_csv(output_path, index=False)
    
    print(f"Created manifest with {len(df)} entries")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nTask distribution:")
    print(df['task'].value_counts())
    print(f"\nSaved to: {output_path}")
    
    return df

if __name__ == "__main__":
    videos_dir = "/Users/sam/Desktop/FYP/Videos"
    output_path = "data/raw/stress_manifest.csv"
    
    create_manifest(videos_dir, output_path)