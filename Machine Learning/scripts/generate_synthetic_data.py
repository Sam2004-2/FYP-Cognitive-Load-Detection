"""
Generate synthetic training data for testing the pipeline.

Creates realistic feature values based on cognitive load labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# Define feature ranges for low and high cognitive load
# Based on literature and expected values

def generate_features(n_samples, label, user_id):
    """Generate synthetic features for a given label."""
    
    if label == "low":
        # Low cognitive load - fewer blinks, stable EAR
        blink_rate = np.random.uniform(10, 18, n_samples)  # blinks per minute
        blink_count = np.random.uniform(3, 6, n_samples)  # in 20s window
        mean_blink_duration = np.random.uniform(150, 250, n_samples)  # ms
        ear_std = np.random.uniform(0.02, 0.05, n_samples)  # low variability
        perclos = np.random.uniform(0.05, 0.15, n_samples)  # low closure
    else:  # high
        # High cognitive load - more blinks, more variability
        blink_rate = np.random.uniform(18, 30, n_samples)  # blinks per minute
        blink_count = np.random.uniform(6, 10, n_samples)  # in 20s window
        mean_blink_duration = np.random.uniform(180, 280, n_samples)  # ms
        ear_std = np.random.uniform(0.05, 0.12, n_samples)  # high variability
        perclos = np.random.uniform(0.15, 0.30, n_samples)  # more closure
    
    # Brightness and quality features (less affected by cognitive load)
    mean_brightness = np.random.uniform(100, 150, n_samples)
    std_brightness = np.random.uniform(10, 30, n_samples)
    mean_quality = np.random.uniform(0.85, 0.98, n_samples)
    valid_frame_ratio = np.random.uniform(0.92, 1.0, n_samples)
    
    # Add some noise to make it realistic
    blink_rate += np.random.normal(0, 0.5, n_samples)
    ear_std += np.random.normal(0, 0.005, n_samples)
    
    return pd.DataFrame({
        'blink_rate': blink_rate,
        'blink_count': blink_count,
        'mean_blink_duration': mean_blink_duration,
        'ear_std': ear_std,
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'perclos': perclos,
        'mean_quality': mean_quality,
        'valid_frame_ratio': valid_frame_ratio,
    })


def main():
    """Generate synthetic dataset."""
    
    # Configuration
    n_users = 5  # Number of synthetic users
    windows_per_condition = 20  # Windows per user per condition
    
    all_data = []
    
    for user_id in range(1, n_users + 1):
        # Generate low load data (train role for first 3 users)
        role = "train" if user_id <= 3 else "test"
        
        # Low cognitive load
        low_features = generate_features(windows_per_condition, "low", user_id)
        low_features['user_id'] = f"user_{user_id:02d}"
        low_features['video'] = f"user_{user_id:02d}_low.mp4"
        low_features['label'] = "low"
        low_features['role'] = role
        low_features['t_start_s'] = np.arange(0, windows_per_condition * 5, 5)  # 5s steps
        low_features['t_end_s'] = low_features['t_start_s'] + 20  # 20s windows
        
        # High cognitive load
        high_features = generate_features(windows_per_condition, "high", user_id)
        high_features['user_id'] = f"user_{user_id:02d}"
        high_features['video'] = f"user_{user_id:02d}_high.mp4"
        high_features['label'] = "high"
        high_features['role'] = role
        high_features['t_start_s'] = np.arange(0, windows_per_condition * 5, 5)
        high_features['t_end_s'] = high_features['t_start_s'] + 20
        
        all_data.append(low_features)
        all_data.append(high_features)
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: metadata first, then features
    metadata_cols = ['user_id', 'video', 'label', 'role', 't_start_s', 't_end_s']
    feature_cols = [
        'blink_rate', 'blink_count', 'mean_blink_duration', 'ear_std',
        'mean_brightness', 'std_brightness', 'perclos',
        'mean_quality', 'valid_frame_ratio'
    ]
    df = df[metadata_cols + feature_cols]
    
    # Save to CSV
    output_path = Path("data/processed/features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"Generated synthetic dataset:")
    print(f"  Total samples: {len(df)}")
    print(f"  Users: {n_users}")
    print(f"  Train samples: {len(df[df['role'] == 'train'])}")
    print(f"  Test samples: {len(df[df['role'] == 'test'])}")
    print(f"  Low load samples: {len(df[df['label'] == 'low'])}")
    print(f"  High load samples: {len(df[df['label'] == 'high'])}")
    print(f"\nLabel distribution by role:")
    print(df.groupby(['role', 'label']).size())
    print(f"\nSaved to: {output_path}")
    

if __name__ == "__main__":
    main()



