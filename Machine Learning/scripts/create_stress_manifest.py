#!/usr/bin/env python3
"""
Create manifest for stress-based classification training.

Thin wrapper around src.cle.data.manifest.create_stress_manifest().
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cle.data.manifest import create_stress_manifest

if __name__ == "__main__":
    videos_dir = "/Users/sam/Desktop/FYP/Videos"
    output_path = "data/raw/stress_manifest.csv"

    df = create_stress_manifest(videos_dir, output_path)
    print(f"Created manifest with {len(df)} entries")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nTask distribution:\n{df['task'].value_counts()}")
