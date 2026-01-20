# AVCAffe Dataset Integration

Comprehensive documentation for AVCAffe dataset integration with continuous cognitive load regression.

## Overview

The AVCAffe (Automated Visual Caffe) dataset contains video recordings of participants performing cognitive tasks with varying mental demand levels. We extract eye-tracking features from these videos and train regression models to predict continuous cognitive load scores.

### Dataset Statistics

- **Participants**: 106
- **Tasks per participant**: 9
- **Total videos**: ~950
- **Feature windows extracted**: 132,862 (10-second windows, 2.5s step)
- **Label coverage**: 100%
- **Label type**: Continuous [0,1] (normalized NASA-TLX mental demand scores)

## Ground Truth Labels

### Source
- **File**: `E:\FYP\Dataset\AVCAffe\codes\downloader\data\ground_truths\mental_demand.txt`
- **Format**: `participant_task, score` (e.g., `"aiim001_task_1, 1.0"`)
- **Scale**: NASA-TLX mental demand subscale (0-21)
- **Normalization**: Scores divided by 21.0 to get [0,1] range

### Label Distribution
- **Min**: 0.000 (no mental demand)
- **Max**: 1.000 (maximum mental demand)
- **Mean**: 0.473 ± 0.279
- **Median**: 0.524
- **Quartiles**: [0.190, 0.524, 0.714]

### Important Note: Label Granularity
Labels are collected **per task** (~9 per participant), but features are extracted at **window level** (~140 windows per task). This means:
- All windows from the same task receive the same cognitive load label
- High label repetition is expected (same label for ~140 windows)
- Cross-validation uses **GroupKFold by participant** to prevent data leakage

## Feature Extraction

### Method
Features extracted using **MediaPipe FaceLandmarker** for face detection and eye landmark tracking.

### Extraction Pipeline
```bash
# Extraction script: Machine Learning/scripts/extract_video_features.py
# Parallel processing: Machine Learning/scripts/run_parallel_extraction.ps1
# Model: Machine Learning/models/face_landmarker.task (3.6 MB)
```

### Window Configuration
- **Window length**: 10 seconds (250 frames at 25 fps)
- **Window step**: 2.5 seconds (sliding window with overlap)
- **Result**: ~140 windows per 10-minute task video

### Features Extracted (9 total)

| Feature | Description | Type |
|---------|-------------|------|
| `blink_count` | Number of blinks detected in window | Integer |
| `blink_rate` | Blinks per minute | Float |
| `ear_std` | Standard deviation of Eye Aspect Ratio | Float |
| `mean_blink_duration` | Average blink duration (frames) | Float |
| `mean_brightness` | Mean brightness of face ROI | Float |
| `mean_quality` | Mean face detection confidence | Float [0,1] |
| `perclos` | Percentage of Eye Closure | Float [0,1] |
| `std_brightness` | Std dev of brightness | Float |
| `valid_frame_ratio` | Ratio of frames with valid detection | Float [0,1] |

### Feature Quality
- **Mean quality**: 0.90 (excellent detection)
- **Valid frame ratio**: 0.992 (99.2% frames have valid face detection)
- **Missing values**: 99 windows (0.07%) have NaN features - handled by imputation

## Data Files

### Raw Extracted Features (Gitignored)
Located in `Machine Learning/data/processed/`:
- `part1.csv`: 35,173 rows (aiim001-aiim029)
- `part2.csv`: 65,984 rows (aiim030-aiim057)
- `part3.csv`: 63,659 rows (aiim058-aiim083)
- `part4.csv`: 31,705 rows (aiim084-aiim108)

**Note**: Part files have overlaps (63,659 duplicates removed during combination)

### Combined Labeled Dataset
**File**: `Machine Learning/data/processed/avcaffe_labeled_features.csv`
- **Size**: 32.8 MB
- **Rows**: 132,862 (unique windows after deduplication)
- **Columns**: 17 (metadata + 9 features + cognitive_load label)

**Schema**:
```
participant_id, task, window_idx, video_file,
window_start_s, window_end_s, window_start_frame, window_end_frame,
blink_count, blink_rate, ear_std, mean_blink_duration,
mean_brightness, mean_quality, perclos, std_brightness,
valid_frame_ratio, cognitive_load
```

### Backup Location
**Critical**: CSV files are gitignored. Backup maintained at:
```
E:\FYP\Dataset\AVCAffe\extracted_features_backup\20260120_163901\
```

## Data Preparation Scripts

### 1. Label Loading
**File**: [src/cle/data/load_avcaffe_labels.py](../src/cle/data/load_avcaffe_labels.py)

Parses `mental_demand.txt` and normalizes scores to [0,1]:
```python
from src.cle.data.load_avcaffe_labels import load_mental_demand_labels

labels_df = load_mental_demand_labels()
# Returns DataFrame with columns: participant_id, task, raw_score, cognitive_load
```

### 2. Feature Combination & Labeling
**File**: [scripts/combine_and_label_avcaffe.py](../scripts/combine_and_label_avcaffe.py)

Combines part files and joins with labels:
```bash
python scripts/combine_and_label_avcaffe.py \
    --input data/processed/part1.csv \
            data/processed/part2.csv \
            data/processed/part3.csv \
            data/processed/part4.csv \
    --labels "E:/FYP/Dataset/AVCAffe/codes/downloader/data/ground_truths/mental_demand.txt" \
    --output data/processed/avcaffe_labeled_features.csv \
    --validate
```

### 3. Dataset Validation
**File**: [scripts/validate_avcaffe_dataset.py](../scripts/validate_avcaffe_dataset.py)

Generates comprehensive validation report:
```bash
python scripts/validate_avcaffe_dataset.py \
    --input data/processed/avcaffe_labeled_features.csv \
    --output-dir reports
```

### 4. Data Adapter for Training
**File**: [src/cle/data/load_avcaffe_data.py](../src/cle/data/load_avcaffe_data.py)

Adapts AVCAffe schema to match training pipeline expectations:
- Renames `participant_id` → `user_id`
- Renames `cognitive_load` → `load_0_1`
- Applies quality filters

## Training Regression Models

### Configuration
**File**: [configs/avcaffe_regression.yaml](../configs/avcaffe_regression.yaml)

Key settings:
- **Model**: GradientBoostingRegressor (200 trees, depth 4, lr 0.05)
- **Cross-validation**: 5-fold GroupKFold (grouped by participant)
- **Features**: All 9 eye-tracking features
- **Quality filters**: valid_frame_ratio ≥ 0.80, mean_quality ≥ 0.85
- **Metrics**: MAE, RMSE, R², Spearman correlation

### Training Command

**Note**: The main branch's `train_regression.py` expects StressID-style schema. Use the data adapter or modify column names:

```bash
cd "Machine Learning"

# Option 1: Using adapted data (recommended)
python src/cle/train/train_regression.py \
    --in data/processed/avcaffe_labeled_features.csv \
    --out models/avcaffe_baseline \
    --config configs/avcaffe_regression.yaml
```

### Expected Performance

Based on similar datasets:
- **R²**: 0.2 - 0.4 (moderate predictive power)
- **Spearman r**: 0.4 - 0.6 (moderate correlation)
- **MAE**: 0.10 - 0.15 (10-15% error on normalized scale)

**Note**: Performance may be limited by:
- Subjective self-reported labels (label noise)
- High inter-individual variability
- Label granularity (task-level labels, window-level predictions)
- Limited feature set (only eye metrics, no physiological signals)

### Predicted Model Artifacts

After training:
```
models/avcaffe_baseline/
├── model.bin              # Trained GradientBoostingRegressor (gitignored)
├── scaler.bin             # StandardScaler for features (gitignored)
├── feature_spec.json      # Feature metadata (can commit)
└── metrics.json           # Evaluation results (can commit)
```

## Validation Report

Latest validation: [reports/avcaffe_validation_20260120_165007.json](../reports/avcaffe_validation_20260120_165007.json)

Key findings:
- ✅ 132,862 unique windows preserved
- ✅ 100% label coverage (all windows labeled)
- ✅ No duplicates after combination
- ✅ All time and frame ranges valid
- ⚠️ 99 windows (0.07%) have NaN features (handled by imputation)

## Usage Examples

### Load and Explore Data
```python
import pandas as pd

# Load labeled features
df = pd.read_csv("data/processed/avcaffe_labeled_features.csv")

print(f"Total windows: {len(df):,}")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Label distribution:\n{df['cognitive_load'].describe()}")
```

### Train Custom Model
```python
from src.cle.data.load_avcaffe_data import prepare_avcaffe_for_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

# Load and prepare data
feature_names = ["blink_rate", "perclos", "ear_std", "mean_blink_duration"]
quality_filters = {"valid_frame_ratio": 0.80, "mean_quality": 0.85}

df = prepare_avcaffe_for_regression(
    features_path="data/processed/avcaffe_labeled_features.csv",
    feature_names=feature_names,
    quality_filters=quality_filters
)

# Prepare features and labels
X = df[feature_names].values
y = df["load_0_1"].values
groups = df["user_id"].factorize()[0]  # For GroupKFold

# Train with cross-validation
model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
scaler = StandardScaler()

cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv, groups=groups,
                          scoring="r2")

print(f"Cross-validated R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Known Limitations

1. **Label Granularity Mismatch**
   - Labels are task-level (9 per participant)
   - Features are window-level (~140 per task)
   - Same label repeated for all windows in a task
   - Potential for overfitting if not using proper CV

2. **Missing Data**
   - 4 participant-task pairs missing from expected 954 (950 present)
   - 99 windows (0.07%) have NaN features (likely due to failed face detection)
   - Handled automatically by SimpleImputer during training

3. **Label Noise**
   - Self-reported NASA-TLX scores (subjective)
   - Collected post-task (memory bias)
   - Individual differences in perception

4. **Limited Features**
   - Only eye-tracking metrics (9 features)
   - No physiological signals (HR, EDA, etc.)
   - No behavioral metrics (typing speed, mouse movement, etc.)

## Citation

If using AVCAffe dataset, cite:
```bibtex
@inproceedings{avcaffe2020,
  title={AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work},
  author={...},
  booktitle={...},
  year={2020}
}
```

## Contact & Support

For questions or issues:
- Check validation report: `reports/avcaffe_validation_*.json`
- Review extraction logs: Check console output from `extract_video_features.py`
- Data issues: Verify backup at `E:\FYP\Dataset\AVCAffe\extracted_features_backup\`

## Version History

- **2026-01-20**: Initial integration
  - Extracted 132,862 feature windows from 106 participants
  - Implemented continuous regression pipeline
  - Validated 100% label coverage
  - Created comprehensive documentation
