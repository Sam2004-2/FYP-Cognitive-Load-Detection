# Configuration Reference

Complete reference for all configuration options.

## Configuration Files

### Backend (YAML)

Located in `Machine Learning/configs/`:

| File | Description |
|------|-------------|
| `config.yaml` | **Single unified configuration** for binary classification |

> **Note:** The project has been simplified to use a single configuration file. Previous config files (`default.yaml`, `regression.yaml`, `avcaffe_*.yaml`) have been archived.

### Frontend (TypeScript)

Located in `UI/src/config/featureConfig.ts`

---

## Backend Configuration (YAML)

### Complete Configuration Reference

```yaml
# =============================================================================
# Cognitive Load Detection - Simplified Configuration
# Binary classification (HIGH/LOW) with trend detection
# =============================================================================

# =============================================================================
# Data Paths
# =============================================================================
data:
  features_path: "data/avcaffe_features_final.csv"
  labels_path: "data/avcaffe_labels.csv"

# =============================================================================
# Feature Configuration
# =============================================================================
# 9 base features - no derived features for simplicity
features:
  - blink_rate         # Blinks per minute
  - blink_count        # Total blinks in window
  - mean_blink_duration # Average blink length (ms)
  - ear_std            # Eye Aspect Ratio std dev
  - perclos            # % eye closure
  - mean_brightness    # Face region brightness
  - std_brightness     # Brightness variability
  - mean_quality       # Detection quality
  - valid_frame_ratio  # Valid frames ratio

features_enabled:
  blinks: true         # Blink rate and statistics
  brightness: true     # Ambient brightness control
  perclos: true        # Percentage of eye closure

# -----------------------------------------------------------------------------
# Blink Detection Parameters
# -----------------------------------------------------------------------------
blink:
  ear_thresh: 0.21     # Eye Aspect Ratio threshold for blink detection
                       # Below this = eye considered closed
                       # Typical: 0.21 (most people), 0.18-0.25 (individual variation)

  min_blink_ms: 120    # Minimum blink duration in milliseconds
                       # Filters out noise/measurement errors

  max_blink_ms: 400    # Maximum blink duration in milliseconds
                       # Filters out intentional eye closures

# =============================================================================
# Model Configuration
# =============================================================================
model:
  type: xgboost        # GradientBoostingClassifier
  params:
    n_estimators: 100  # Number of trees
    max_depth: 6       # Maximum tree depth
    learning_rate: 0.1 # Learning rate

# =============================================================================
# Training Configuration
# =============================================================================
training:
  cv_folds: 5          # Number of cross-validation folds
  cv_method: group_kfold  # Subject-wise cross-validation (prevents data leakage)
  group_column: user_id   # Column to group by
  seed: 42             # Random seed for reproducibility

# =============================================================================
# Classification Thresholds
# =============================================================================
threshold:
  binary: 0.5          # >= 0.5 is HIGH, < 0.5 is LOW

# =============================================================================
# Trend Detection
# =============================================================================
trend:
  window: 5            # Compare last 5 vs previous 5 predictions
  threshold: 0.1       # 10% change required to detect trend
  max_history: 100     # Maximum predictions to keep in buffer

# =============================================================================
# Window Configuration (for feature extraction)
# =============================================================================
window:
  length_s: 10.0       # Window length in seconds
                       # 10s provides stable measurements

  step_s: 2.5          # Step size in seconds (75% overlap)
                       # Prediction every 2.5 seconds

  min_valid_ratio: 0.5 # Minimum valid frame ratio to accept window
```

---

## Frontend Configuration (TypeScript)

### featureConfig.ts

```typescript
// Feature extraction configuration
export const FEATURE_CONFIG = {
  // Window parameters
  windows: {
    length_s: 10.0,    // Window length in seconds
    step_s: 2.5,       // Prediction interval
  },

  // Blink detection
  blink: {
    ear_thresh: 0.21,  // EAR threshold
    min_blink_ms: 120, // Minimum blink duration
    max_blink_ms: 400, // Maximum blink duration
  },

  // Quality thresholds
  quality: {
    min_face_conf: 0.5,
    max_bad_frame_ratio: 0.05,
  },

  // Real-time processing
  realtime: {
    smoothing_alpha: 0.4,    // EWMA smoothing
    conf_threshold: 0.6,     // Minimum confidence
  },

  // Video settings
  video: {
    fps: 30.0,               // Expected FPS
  },
};

// MediaPipe landmark indices
export const LANDMARK_INDICES = {
  // Eye landmarks for EAR calculation (6 points each)
  LEFT_EYE: [33, 160, 158, 133, 153, 144],
  RIGHT_EYE: [362, 385, 387, 263, 373, 380],

  // Iris landmarks (4 points each)
  LEFT_IRIS: [469, 470, 471, 472],
  RIGHT_IRIS: [474, 475, 476, 477],
};
```

---

## Configuration Usage

### Loading Configuration (Python)

```python
from src.cle.config import load_config

# Load default config (configs/config.yaml)
config = load_config(None)

# Load specific config
config = load_config("configs/config.yaml")

# Access values with dot notation
ear_thresh = config.get("blink.ear_thresh", 0.21)
window_length = config.get("window.length_s", 10.0)
model_type = config.get("model.type", "xgboost")
```

### Training with Configuration

```bash
cd "Machine Learning"

# Train binary classifier with default config
python -m src.cle.train.train_binary \
    --input data/avcaffe_features_final.csv \
    --output models/binary_classifier \
    --cv-folds 5
```

---

## Parameter Tuning Guide

### Window Parameters

| Scenario | length_s | step_s | Notes |
|----------|----------|--------|-------|
| **Production** | 10.0 | 2.5 | Balanced stability/responsiveness |
| **Responsive** | 5.0 | 1.0 | Faster updates, more noise |
| **Stable** | 15.0 | 5.0 | Very smooth, delayed response |
| **Training** | 10.0 | 10.0 | Non-overlapping windows |

### Blink Detection

| Scenario | ear_thresh | Notes |
|----------|------------|-------|
| **Default** | 0.21 | Works for most users |
| **Glasses** | 0.18 | Lower threshold for glasses |
| **Large eyes** | 0.25 | Higher threshold |

### Trend Detection

| Scenario | window | threshold | Notes |
|----------|--------|-----------|-------|
| **Responsive** | 3 | 0.15 | Faster trend detection |
| **Balanced** | 5 | 0.1 | Default |
| **Stable** | 10 | 0.05 | Very stable trend signals |

### Smoothing

| Scenario | smoothing_alpha | Notes |
|----------|-----------------|-------|
| **Responsive** | 0.6 | More weight on new predictions |
| **Balanced** | 0.4 | Default |
| **Smooth** | 0.2 | Stable but delayed |

---

## Archived Configurations

> **Note:** The following configurations have been archived in `archive/configs/` and are no longer actively used. The project now uses a single unified `config.yaml`.

### Archived Files

| File | Description |
|------|-------------|
| `default.yaml` | Old base template |
| `regression.yaml` | StressID regression |
| `avcaffe_regression.yaml` | AVCAffe continuous regression |
| `avcaffe_classification.yaml` | AVCAffe classification |
| `avcaffe_classification_improved.yaml` | Enhanced with feature engineering |

These configurations can still be referenced for thesis documentation or if you need to experiment with alternative approaches.

---

## Creating Custom Configurations

1. **Start from config.yaml:**
   ```bash
   cp configs/config.yaml configs/my_config.yaml
   ```

2. **Modify parameters:**
   ```yaml
   # my_config.yaml
   window:
     length_s: 8.0
     step_s: 2.0

   model:
     type: xgboost
     params:
       n_estimators: 200
       max_depth: 8

   trend:
     window: 7
     threshold: 0.08
   ```

3. **The training script uses hardcoded model parameters** but you can modify `train_binary.py` directly if needed.

---

## Configuration Validation

Configurations are validated on load. Invalid configurations raise `ValueError` with descriptive message.

---

## Troubleshooting Configuration

### "Config file not found"

```python
# Use absolute path or ensure CWD is correct
config = load_config("/full/path/to/config.yaml")
```

### "Invalid parameter combination"

```yaml
# Ensure step_s <= length_s
window:
  length_s: 10.0
  step_s: 2.5  # Not 15.0
```
