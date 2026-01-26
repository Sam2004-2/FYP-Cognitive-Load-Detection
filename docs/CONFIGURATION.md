# Configuration Reference

Complete reference for all configuration options.

## Configuration Files

### Backend (YAML)

Located in `Machine Learning/configs/`:

| File | Description |
|------|-------------|
| `default.yaml` | Base configuration template |
| `regression.yaml` | StressID regression config |
| `avcaffe_regression.yaml` | AVCAffe regression config |
| `avcaffe_classification.yaml` | AVCAffe classification config |
| `avcaffe_classification_improved.yaml` | Enhanced classification |

### Frontend (TypeScript)

Located in `UI/src/config/featureConfig.ts`

---

## Backend Configuration (YAML)

### Complete Configuration Reference

```yaml
# =============================================================================
# Cognitive Load Estimation - Configuration Reference
# =============================================================================

# Random seed for reproducibility
seed: 42

# Fallback FPS if video metadata is unavailable
fps_fallback: 30.0

# -----------------------------------------------------------------------------
# Windowing Parameters
# -----------------------------------------------------------------------------
windows:
  length_s: 10.0    # Window length in seconds
                    # Longer = more stable, shorter = more responsive
                    # Typical: 10s (production), 5s (responsive)

  step_s: 2.5       # Step size in seconds (overlap = length - step)
                    # 2.5s with 10s window = 75% overlap
                    # Smaller step = more predictions but higher CPU

# -----------------------------------------------------------------------------
# Quality Control Thresholds
# -----------------------------------------------------------------------------
quality:
  min_face_conf: 0.5        # Minimum face detection confidence (0-1)
                            # Higher = more reliable but may reject valid frames
                            # Typical: 0.5 (balanced), 0.7 (strict)

  max_bad_frame_ratio: 0.05 # Maximum ratio of invalid frames per window
                            # 0.05 = reject windows with >5% bad frames
                            # Lower = stricter quality, may drop more windows

# -----------------------------------------------------------------------------
# Blink Detection Parameters
# -----------------------------------------------------------------------------
blink:
  ear_thresh: 0.21    # Eye Aspect Ratio threshold for blink detection
                      # Below this = eye considered closed
                      # Typical: 0.21 (most people), 0.18-0.25 (individual variation)

  min_blink_ms: 120   # Minimum blink duration in milliseconds
                      # Filters out noise/measurement errors
                      # Typical: 100-150ms

  max_blink_ms: 400   # Maximum blink duration in milliseconds
                      # Filters out intentional eye closures
                      # Typical: 350-500ms

# -----------------------------------------------------------------------------
# TEPR Parameters (Disabled by default)
# -----------------------------------------------------------------------------
tepr:
  baseline_s: 10.0          # Baseline window duration in seconds
  min_baseline_samples: 150 # Minimum samples for valid baseline

# -----------------------------------------------------------------------------
# Feature Flags
# -----------------------------------------------------------------------------
features_enabled:
  tepr: false         # Task-Evoked Pupillary Response
                      # Disabled - requires calibration and special lighting

  blinks: true        # Blink rate and statistics
                      # Primary cognitive load indicator

  perclos: true       # Percentage of eye closure
                      # Key fatigue/drowsiness metric

  brightness: true    # Ambient brightness control
                      # Helps normalize for lighting conditions

  fix_sac: false      # Fixations and saccades
                      # Disabled - requires calibration

  gaze_entropy: false # Gaze entropy
                      # Disabled - requires calibration

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
model:
  type: "logreg"      # Model type: "logreg", "gbt", "rf", "xgb"
                      # logreg: Fast, interpretable
                      # gbt: Good accuracy, slower
                      # rf: Robust, handles outliers
                      # xgb: Best accuracy, slowest

  calibration: "platt" # Calibration method: "platt" or "isotonic"
                       # platt: Sigmoid fit, works well for most
                       # isotonic: Non-parametric, needs more data

  # Logistic Regression parameters
  logreg_params:
    max_iter: 1000    # Maximum iterations
    C: 0.1            # Regularization strength (smaller = stronger)
    penalty: "l2"     # Regularization type

  # Gradient Boosted Trees parameters
  gbt_params:
    n_estimators: 100   # Number of trees
    max_depth: 3        # Maximum tree depth
    learning_rate: 0.1  # Learning rate

  # Random Forest parameters
  rf_params:
    n_estimators: 100   # Number of trees
    max_depth: 10       # Maximum tree depth
    min_samples_leaf: 5 # Minimum samples per leaf

  # XGBoost parameters (regression)
  xgb_params:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.05
    min_samples_leaf: 5

# -----------------------------------------------------------------------------
# Evaluation Configuration
# -----------------------------------------------------------------------------
eval:
  metric: ["auc", "f1"]  # Metrics to compute
                         # Options: auc, f1, accuracy, precision, recall

  split_by_role: true    # Use 'role' column for train/test split
                         # If true, respects existing train/test assignments

  test_size: 0.2         # Test split ratio (if split_by_role is false)

  stratify: true         # Stratify splits by label
                         # Maintains class distribution in splits

# -----------------------------------------------------------------------------
# Cross-Validation Configuration
# -----------------------------------------------------------------------------
cv:
  method: group_kfold    # CV method: "group_kfold", "stratified_kfold"
                         # group_kfold: Groups by participant (recommended)
                         # stratified_kfold: Stratifies by label

  n_splits: 5            # Number of CV folds

# -----------------------------------------------------------------------------
# Real-time Processing
# -----------------------------------------------------------------------------
realtime:
  smoothing_alpha: 0.4   # EWMA smoothing parameter (0-1)
                         # Higher = more responsive, lower = smoother
                         # 0.4 = balanced responsiveness and stability

  conf_threshold: 0.6    # Minimum confidence to display result
                         # Below this, prediction may be suppressed

  display_fps: true      # Show FPS in real-time output

# -----------------------------------------------------------------------------
# Paths (relative to project root)
# -----------------------------------------------------------------------------
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  models_dir: "models"
  reports_dir: "reports"

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging:
  level: "INFO"        # Logging level: DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  save_logs: true      # Save logs to file
  log_dir: "logs"
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

# Load default config
config = load_config(None)

# Load specific config
config = load_config("configs/avcaffe_regression.yaml")

# Access values with dot notation
ear_thresh = config.get("blink.ear_thresh", 0.21)
window_length = config.get("windows.length_s", 10.0)
model_type = config.get("model.type", "logreg")
```

### Overriding via CLI

```bash
# Override model type
python -m src.cle.train.train \
    --features data.csv \
    --model-type rf

# Override config file
python -m src.cle.train.train \
    --features data.csv \
    --config configs/custom.yaml
```

### Environment-Based Configuration

Create environment-specific configs:

```yaml
# configs/production.yaml
windows:
  length_s: 10.0
  step_s: 5.0  # Longer interval for stability

realtime:
  smoothing_alpha: 0.3  # More smoothing

logging:
  level: "WARNING"
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

### Smoothing

| Scenario | smoothing_alpha | Notes |
|----------|-----------------|-------|
| **Responsive** | 0.6 | More weight on new predictions |
| **Balanced** | 0.4 | Default |
| **Smooth** | 0.2 | Stable but delayed |

### Model Selection

| Scenario | Model | Notes |
|----------|-------|-------|
| **Quick/Simple** | logreg | Fast, interpretable |
| **Balanced** | gbt | Good accuracy |
| **Best accuracy** | rf/xgb | Slower, robust |
| **Regression** | xgb | Continuous output |

---

## Available Configurations

### default.yaml

Base template with sensible defaults for classification.

### regression.yaml

StressID regression configuration:
- Binary stress labels
- GroupKFold CV
- Standard features

### avcaffe_regression.yaml

AVCAffe continuous regression:
- Cognitive load 0-1
- XGBoost regressor
- GroupKFold by participant

```yaml
task_mode: regression
regression:
  model_type: xgb
  xgb_params:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.05
```

### avcaffe_classification.yaml

AVCAffe binary classification:
- Low/High labels
- Threshold at 0.5

### avcaffe_classification_improved.yaml

Enhanced classification with:
- Feature engineering
- Advanced hyperparameters
- Ensemble methods

---

## Creating Custom Configurations

1. **Start from default:**
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   ```

2. **Modify parameters:**
   ```yaml
   # my_config.yaml
   windows:
     length_s: 8.0
     step_s: 2.0

   model:
     type: "rf"
     rf_params:
       n_estimators: 200
   ```

3. **Use in training:**
   ```bash
   python -m src.cle.train.train \
       --features data.csv \
       --config configs/my_config.yaml
   ```

---

## Configuration Validation

Configurations are validated on load:

```python
# In src/cle/config.py
def validate_config(config):
    """Validate configuration values."""
    assert config.get("windows.length_s") > 0
    assert 0 < config.get("windows.step_s") <= config.get("windows.length_s")
    assert 0 < config.get("blink.ear_thresh") < 1
    assert config.get("model.type") in ["logreg", "gbt", "rf", "xgb"]
```

Invalid configurations raise `ValueError` with descriptive message.

---

## Troubleshooting Configuration

### "Config file not found"

```python
# Use absolute path or ensure CWD is correct
config = load_config("/full/path/to/config.yaml")
```

### "Unknown model type"

```yaml
# Ensure model.type is one of: logreg, gbt, rf, xgb
model:
  type: "rf"  # Not "random_forest"
```

### "Invalid parameter combination"

```yaml
# Ensure step_s <= length_s
windows:
  length_s: 10.0
  step_s: 2.5  # Not 15.0
```
