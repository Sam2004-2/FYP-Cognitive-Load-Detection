# Data Guide

Documentation for datasets, data formats, and preprocessing.

## Supported Datasets

### AVCAffe Dataset

**Overview:**
- 106 participants
- 9 tasks per participant (cognitive tasks with varying difficulty)
- ~950 videos
- NASA-TLX self-assessment labels

**Labels:**
- Continuous cognitive load: 0-1 (normalized mental demand score)
- Raw scores: 0-21 (NASA-TLX mental demand subscale)
- Normalization: `cognitive_load = raw_score / 21.0`

**Statistics:**
```
Total windows:    132,862
Label range:      0.0 - 1.0
Mean:             0.473 ± 0.279
Median:           0.524
Quartiles:        [0.190, 0.524, 0.714]
```

**Directory Structure:**
```
Dataset/
└── AVCAffe/
    ├── codes/downloader/data/
    │   ├── ground_truths/
    │   │   ├── mental_demand.txt    # Primary label file
    │   │   ├── arousal.txt
    │   │   ├── dominance.txt
    │   │   └── ...
    │   └── info/
    │       ├── train.txt
    │       ├── val.txt
    │       └── public_face_ids.txt
    └── videos/                      # Downloaded videos
```

**Documentation:** See [Machine Learning/docs/AVCAFFE_DATASET.md](../Machine%20Learning/docs/AVCAFFE_DATASET.md)

---

### StressID Dataset

**Overview:**
- Physiological stress detection dataset
- Video recordings with stress/no-stress labels
- Binary classification task

---

### Collected Data

Data collected via the Data Collection mode in the UI.

**Storage Location:** `Machine Learning/data/collected/`

**Files Generated:**
- `collected_{participant_id}_{timestamp}.csv` - Training data
- `collected_{participant_id}_{timestamp}.json` - Metadata sidecar

---

## Data Formats

### Feature CSV Format

Main format for training data.

**Schema:**
```csv
user_id,timestamp,window_index,label,difficulty,task_type,blink_rate,blink_count,mean_blink_duration,ear_std,mean_brightness,std_brightness,perclos,mean_quality,valid_frame_ratio,role
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Participant identifier |
| `timestamp` | int | Unix timestamp (milliseconds) |
| `window_index` | int | Window sequence number |
| `label` | int | 0=low, 1=high (classification) |
| `difficulty` | string | easy, medium, hard |
| `task_type` | string | Task description |
| `blink_rate` | float | Blinks per minute |
| `blink_count` | float | Total blinks in window |
| `mean_blink_duration` | float | Mean blink duration (ms) |
| `ear_std` | float | EAR standard deviation |
| `mean_brightness` | float | Mean face brightness |
| `std_brightness` | float | Brightness std deviation |
| `perclos` | float | Percentage eye closure |
| `mean_quality` | float | Detection quality |
| `valid_frame_ratio` | float | Valid frame ratio |
| `role` | string | train or test |

**Example:**
```csv
user_id,timestamp,window_index,label,difficulty,task_type,blink_rate,blink_count,mean_blink_duration,ear_std,mean_brightness,std_brightness,perclos,mean_quality,valid_frame_ratio,role
P001,1706123456789,0,0,easy,baseline,15.2,3,165.0,0.038,145.0,10.5,0.05,0.95,0.99,train
P001,1706123459289,1,0,easy,baseline,16.1,3,170.0,0.042,144.2,11.0,0.06,0.94,0.98,train
```

---

### AVCAffe Labeled Features Format

Combined features with cognitive load labels.

**File:** `Machine Learning/data/processed/avcaffe_labeled_features.csv`

**Schema:**
```csv
participant_id,task,window_idx,video_file,window_start_s,window_end_s,window_start_frame,window_end_frame,blink_count,blink_rate,ear_std,mean_blink_duration,mean_brightness,mean_quality,perclos,std_brightness,valid_frame_ratio,cognitive_load
```

**Additional Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | string | e.g., "aiim001" |
| `task` | string | e.g., "task_1" |
| `video_file` | string | Source video path |
| `window_start_s` | float | Window start time |
| `window_end_s` | float | Window end time |
| `cognitive_load` | float | Normalized 0-1 |

---

### Video Manifest Format

For batch video processing.

**Schema:**
```csv
video_file,label,role,user_id,notes
```

**Example:**
```csv
video_file,label,role,user_id,notes
videos/aiim001_task_1.mp4,low,train,aiim001,baseline
videos/aiim001_task_5.mp4,high,train,aiim001,difficult task
videos/aiim002_task_1.mp4,low,test,aiim002,baseline
```

---

### Label Files

**NASA-TLX Mental Demand (AVCAffe):**

**File:** `ground_truths/mental_demand.txt`

**Format:**
```
aiim001_task_1, 5.0
aiim001_task_2, 12.0
aiim001_task_3, 8.0
...
```

Each line: `{participant_id}_{task}, {score}`

Score range: 0-21

---

### Model Artifacts

**Directory Structure:**
```
models/my_model/
├── model.bin           # Serialized model (joblib)
├── scaler.bin          # StandardScaler (joblib)
├── feature_spec.json   # Feature names and count
├── calibration.json    # Training metadata
└── cv_results.json     # Cross-validation results (optional)
```

**feature_spec.json:**
```json
{
  "features": [
    "blink_rate",
    "blink_count",
    "mean_blink_duration",
    "ear_std",
    "mean_brightness",
    "std_brightness",
    "perclos",
    "mean_quality",
    "valid_frame_ratio"
  ],
  "n_features": 9
}
```

**calibration.json:**
```json
{
  "method": "platt",
  "trained_at": "2026-01-20T15:30:00",
  "training_samples": 1500,
  "config_hash": "a1b2c3d4",
  "metrics": {
    "accuracy": 0.78,
    "auc": 0.82,
    "f1": 0.75
  }
}
```

---

## Data Preprocessing

### Feature Extraction from Videos

```bash
# Extract features from a single video
python scripts/extract_video_features.py \
    --video path/to/video.mp4 \
    --out features.csv

# Parallel extraction for multiple videos
pwsh scripts/run_parallel_extraction.ps1
```

### Combining Feature Files

```bash
python scripts/combine_and_label_avcaffe.py \
    --input data/processed/part1.csv data/processed/part2.csv \
    --labels "path/to/ground_truths/mental_demand.txt" \
    --output data/processed/avcaffe_labeled_features.csv \
    --validate
```

### Dataset Validation

```bash
python scripts/validate_avcaffe_dataset.py \
    --input data/processed/avcaffe_labeled_features.csv \
    --output-dir reports
```

**Validation Checks:**
- Column schema validation
- Missing value detection
- Label distribution analysis
- Feature range validation
- Participant coverage

---

## Data Quality

### Quality Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `min_face_conf` | 0.5 | Minimum detection confidence |
| `max_bad_frame_ratio` | 0.05 | Max 5% invalid frames |
| `valid_frame_ratio` | >= 0.80 | Minimum for window |
| `mean_quality` | >= 0.85 | Minimum mean quality |

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer

# Median imputation (preserves distribution)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
```

### Quality Filtering

```python
# Filter low-quality windows
df_filtered = df[
    (df['valid_frame_ratio'] >= 0.80) &
    (df['mean_quality'] >= 0.85)
]
```

---

## Label Distribution

### AVCAffe Dataset

```
Label Distribution (continuous):
  Min:    0.000
  Max:    1.000
  Mean:   0.473
  Std:    0.279
  Median: 0.524

Binned Distribution:
  0.0-0.2: 15.2%
  0.2-0.4: 18.7%
  0.4-0.6: 25.1%
  0.6-0.8: 28.3%
  0.8-1.0: 12.7%
```

### Binary Classification

For binary classification, labels are thresholded:

```python
# Common threshold: 0.5
df['label_binary'] = (df['cognitive_load'] >= 0.5).astype(int)

# Alternative: median split
median = df['cognitive_load'].median()
df['label_binary'] = (df['cognitive_load'] >= median).astype(int)
```

---

## Data Pipeline Scripts

### extract_video_features.py

Extract features from AVCAffe videos.

```bash
python scripts/extract_video_features.py \
    --video videos/aiim001_task_1.mp4 \
    --out features_aiim001_task1.csv \
    --config configs/default.yaml
```

### combine_and_label_avcaffe.py

Combine multiple feature files and add labels.

```bash
python scripts/combine_and_label_avcaffe.py \
    --input part1.csv part2.csv part3.csv part4.csv \
    --labels mental_demand.txt \
    --output combined_features.csv \
    --validate
```

### validate_avcaffe_dataset.py

Validate dataset integrity.

```bash
python scripts/validate_avcaffe_dataset.py \
    --input avcaffe_labeled_features.csv \
    --output-dir reports
```

**Output:**
- `avcaffe_validation_{timestamp}.json` - Validation results
- Console summary of issues found

---

## Data Loading Utilities

### load_avcaffe_data.py

```python
from src.cle.data.load_avcaffe_data import load_avcaffe_data

df = load_avcaffe_data(
    features_path="data/processed/avcaffe_labeled_features.csv",
    config=config
)

# Applies:
# - Schema renaming (participant_id -> user_id)
# - Quality filtering
# - NaN handling
```

### load_avcaffe_labels.py

```python
from src.cle.data.load_avcaffe_labels import load_mental_demand_labels

labels = load_mental_demand_labels(
    "path/to/ground_truths/mental_demand.txt"
)

# Returns: Dict[str, float]
# e.g., {"aiim001_task_1": 0.238, "aiim001_task_2": 0.571, ...}
```

---

## Data Storage

### Gitignored Files

Large data files are gitignored. Locations:

```
Machine Learning/data/raw/           # Raw videos
Machine Learning/data/processed/     # Processed CSVs
Machine Learning/models/             # Model artifacts (except configs)
Machine Learning/reports/            # Validation reports
Machine Learning/logs/               # Server logs
```

### Backups

Feature backups maintained at:
```
E:\FYP\Dataset\AVCAffe\extracted_features_backup\
└── YYYYMMDD_HHMMSS/
    ├── part1.csv
    ├── part2.csv
    ├── part3.csv
    └── part4.csv
```

---

## Best Practices

### Data Collection

1. **Consistent protocol**: Use same baseline/task procedures
2. **Balanced classes**: Aim for equal low/high samples
3. **Multiple participants**: At least 10 for generalization
4. **Quality control**: Remove low-quality windows
5. **Metadata**: Record session notes, conditions

### Data Storage

1. **Version control**: Track config files, not large data
2. **Backups**: Maintain copies of extracted features
3. **Documentation**: Record preprocessing steps
4. **Validation**: Run validation script regularly

### Privacy

1. **No video storage**: Only store extracted features
2. **Anonymize IDs**: Use participant codes, not names
3. **Secure storage**: Protect collected data files
