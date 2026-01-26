# Machine Learning Pipeline

Comprehensive documentation of the ML pipeline for cognitive load estimation.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                                   │
│  Video Files → MediaPipe FaceMesh → Per-Frame Features → CSV Storage    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                                │
│  Load CSV → Sliding Windows (10s) → Window Features → Quality Filter    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING                                    │
│  Data Prep → GroupKFold CV → Train Model → Calibrate → Save Artifacts   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE                                         │
│  Real-time Features → Scale → Predict → Calibrated Probability → CLI    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Feature Extraction

### Per-Frame Features

Each video frame produces these features:

| Feature | Description | Extraction Method |
|---------|-------------|-------------------|
| `ear_left` | Left eye aspect ratio | EAR formula on left eye landmarks |
| `ear_right` | Right eye aspect ratio | EAR formula on right eye landmarks |
| `ear_mean` | Mean of both eyes | (ear_left + ear_right) / 2 |
| `brightness` | Face region brightness | Mean grayscale in face ROI |
| `quality` | Detection confidence | MediaPipe detection score |
| `valid` | Frame validity flag | True if face detected |

### Eye Aspect Ratio (EAR)

The EAR measures eye openness and is the core signal for blink detection:

```
        ||p2-p6|| + ||p3-p5||
EAR = ─────────────────────────
           2 × ||p1-p4||
```

**Landmark indices (MediaPipe):**
```
Left eye:  [33, 160, 158, 133, 153, 144]
Right eye: [362, 385, 387, 263, 373, 380]

Eye diagram:
    p2 ─── p3
   /         \
  p1          p4
   \         /
    p6 ─── p5
```

**Typical values:**
- Open eye: 0.25 - 0.35
- Closed eye: < 0.21
- Blink threshold: 0.21 (configurable)

### Blink Detection

State machine algorithm for detecting blinks:

```
State: OPEN
  │
  │ EAR < threshold
  ▼
State: CLOSED
  │ Record start frame
  │
  │ EAR >= threshold
  ▼
State: OPEN
  │ Record end frame
  │ Validate duration
  │
  ├── 120ms <= duration <= 400ms → Valid blink
  └── Otherwise → Reject (noise or intentional closure)
```

**Parameters:**
- `ear_thresh`: 0.21 - Below this, eye considered closed
- `min_blink_ms`: 120 - Minimum blink duration (filter noise)
- `max_blink_ms`: 400 - Maximum blink duration (filter prolonged closures)

## Windowing

### Sliding Window Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `length_s` | 10.0 | Window duration in seconds |
| `step_s` | 2.5 | Step between windows (75% overlap) |
| FPS | 30 | Frames per second |
| Frames/window | ~300 | 10s × 30fps |

### Window Buffer (Ring Buffer)

```typescript
class WindowBuffer {
  private buffer: FrameFeatures[];
  private capacity: number;  // 300 frames for 10s at 30fps
  private writeIndex: number;

  addFrame(frame: FrameFeatures): void {
    this.buffer[this.writeIndex % this.capacity] = frame;
    this.writeIndex++;
  }

  isReady(): boolean {
    return this.length >= this.capacity;
  }
}
```

### Quality Validation

Windows are validated before processing:

```python
def validate_window_quality(window_data, max_bad_ratio=0.05):
    valid_frames = sum(1 for f in window_data if f['valid'])
    bad_ratio = 1 - (valid_frames / len(window_data))
    return bad_ratio <= max_bad_ratio
```

**Quality thresholds:**
- `min_face_conf`: 0.5 - Minimum face detection confidence
- `max_bad_frame_ratio`: 0.05 - Maximum 5% invalid frames per window

## Window-Level Features

### Core Features (9 total)

| Feature | Type | Description | Cognitive Load Indicator |
|---------|------|-------------|--------------------------|
| `blink_rate` | Float | Blinks per minute | Increases with load |
| `blink_count` | Int | Total blinks in window | Activity measure |
| `mean_blink_duration` | Float (ms) | Average blink length | Fatigue indicator |
| `ear_std` | Float | EAR standard deviation | Variability with load |
| `perclos` | Float (0-1) | % eye closure time | Key fatigue metric |
| `mean_brightness` | Float (0-255) | Mean face brightness | Environmental control |
| `std_brightness` | Float | Brightness variability | Stability indicator |
| `mean_quality` | Float (0-1) | Detection confidence | Data quality |
| `valid_frame_ratio` | Float (0-1) | Valid frame proportion | Reliability metric |

### Feature Computation

**Blink Rate:**
```python
window_duration_min = len(ear_series) / fps / 60.0
blink_rate = len(blinks) / window_duration_min
```

**PERCLOS (Percentage of Eye Closure):**
```python
closed_frames = sum(1 for ear in ear_series if ear < threshold)
perclos = closed_frames / len(ear_series)
```

**EAR Variability:**
```python
valid_ear = [ear for ear in ear_series if ear > 0]
ear_std = np.std(valid_ear)
```

### Derived Features (Feature Engineering)

For improved models, additional derived features:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log_blink_rate` | log(blink_rate + 1) | Handle skewed distribution |
| `blink_rate_sq` | blink_rate² | Capture non-linear effects |
| `perclos_sq` | perclos² | Capture non-linear effects |
| `blink_perclos_interaction` | blink_rate × perclos | Combined fatigue indicator |
| `fatigue_index` | 0.4×norm(blink_rate) + 0.4×norm(perclos) + 0.2×norm(ear_std) | Composite metric |

## Model Training

### Supported Models

**Classification:**

| Model | Description | Parameters |
|-------|-------------|------------|
| LogReg | Logistic Regression | C=0.1, max_iter=1000, L2 penalty |
| GBT | Gradient Boosted Trees | n_estimators=100, max_depth=3 |
| RF | Random Forest | n_estimators=100, max_depth=10 |

**Regression:**

| Model | Description | Parameters |
|-------|-------------|------------|
| XGBoost | Gradient Boosting | n_estimators=200, max_depth=4, lr=0.05 |

### Cross-Validation Strategy

**GroupKFold** prevents data leakage by keeping all data from one participant in the same fold:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=participant_ids):
    # All windows from a participant are in either train OR test
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**Why GroupKFold?**
- Windows from same participant are correlated
- Standard K-fold would leak participant-specific patterns
- GroupKFold ensures model generalizes to new users

### Training Pipeline

```python
# 1. Load and prepare data
df = pd.read_csv("features.csv")
X = df[feature_columns].values
y = df["label"].values
groups = df["participant_id"].values

# 2. Handle NaN values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train model with cross-validation
model = GradientBoostingClassifier(**params)
scores = cross_val_score(model, X_scaled, y, cv=gkf, groups=groups)

# 5. Fit final model on all data
model.fit(X_scaled, y)

# 6. Calibrate probabilities
calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5)
calibrated.fit(X_scaled, y)

# 7. Save artifacts
joblib.dump(calibrated, "model.bin")
joblib.dump(scaler, "scaler.bin")
```

### Probability Calibration

Raw classifier scores are often poorly calibrated. Calibration ensures:
- Predicted probability 0.7 means ~70% chance of high load
- Enables meaningful confidence thresholds

**Methods:**
- **Platt Scaling (sigmoid)**: Fits sigmoid to classifier scores
- **Isotonic Regression**: Non-parametric monotonic mapping

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",  # or "isotonic"
    cv=5
)
```

### Model Artifacts

After training, these files are saved:

| File | Contents |
|------|----------|
| `model.bin` | Calibrated classifier (joblib format) |
| `scaler.bin` | Fitted StandardScaler |
| `feature_spec.json` | Feature names and count |
| `calibration.json` | Training metadata and metrics |

## Evaluation Metrics

### Classification Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability |
| **F1 Score** | Harmonic mean of precision/recall | Balanced measure |
| **Accuracy** | Correct predictions / total | Overall performance |
| **Precision** | TP / (TP + FP) | When false positives costly |
| **Recall** | TP / (TP + FN) | When false negatives costly |

### Regression Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average prediction error |
| **RMSE** | Root Mean Squared Error | Penalizes large errors |
| **R²** | Coefficient of determination | Variance explained |
| **Spearman r** | Rank correlation | Monotonic relationship |

### Current Performance

AVCAffe baseline model (XGBoost regression):

```
Window-level (132,862 samples):
  MAE:  0.244
  RMSE: 0.285
  R²:   -0.046
  Spearman r: 0.029

Session-level (950 sessions):
  MAE:  0.243
  Spearman r: 0.030
```

**Limitations:**
- Task-level labels applied to window-level predictions
- Individual differences in baseline EAR
- Environmental variations in lighting

## Inference Pipeline

### Real-time Prediction Flow

```
1. Receive window features from frontend
      │
      ▼
2. Validate feature dimensions (9 features)
      │
      ▼
3. Handle NaN values (replace with 0)
      │
      ▼
4. Scale features using saved scaler
      │
      ▼
5. Get probability prediction
      │
      ▼
6. Calculate confidence
      │
      ▼
7. Return CLI and confidence
```

### Confidence Calculation

```python
cli_proba = model.predict_proba(features)[0, 1]  # P(high load)

# Confidence = distance from decision boundary (0.5)
confidence = abs(cli_proba - 0.5) * 2.0

# Maps: 0.5 → 0.0 confidence, 0.0 or 1.0 → 1.0 confidence
```

### EWMA Smoothing

Predictions are smoothed to reduce jitter:

```python
# Exponential Weighted Moving Average
alpha = 0.4  # Smoothing factor
smoothed_load = alpha * new_prediction + (1 - alpha) * previous_load

# alpha = 1.0: No smoothing (use raw predictions)
# alpha = 0.0: No updates (ignore new predictions)
# alpha = 0.4: Balance between responsiveness and stability
```

## Training Commands

### Classification

```bash
python -m src.cle.train.train \
    --features data/processed/training_data.csv \
    --out models/my_classifier \
    --config configs/default.yaml \
    --model-type rf  # logreg, gbt, or rf
```

### Regression

```bash
python -m src.cle.train.train_regression \
    --in data/processed/avcaffe_labeled_features.csv \
    --out models/avcaffe_baseline \
    --config configs/avcaffe_regression.yaml
```

### Evaluation

```bash
# Classification evaluation
python -m src.cle.train.eval \
    --model models/my_classifier \
    --test data/processed/test_data.csv

# Regression evaluation
python -m src.cle.train.eval_regression \
    --model models/avcaffe_baseline \
    --test data/processed/test_data.csv
```

## Feature Engineering Module

Located at `src/cle/extract/feature_engineering.py`:

```python
# Transform features
log_blink_rate = np.log1p(blink_rate)
blink_rate_sq = blink_rate ** 2

# Interaction terms
blink_perclos_interaction = blink_rate * perclos

# Stability metrics
ear_stability = 1 / (ear_std + 0.01)

# Composite index
fatigue_index = (
    0.4 * normalize(blink_rate) +
    0.4 * normalize(perclos) +
    0.2 * normalize(ear_std)
)

# User-relative features (requires baseline)
blink_rate_deviation = blink_rate - user_baseline_blink_rate
```

## Best Practices

### Data Collection

1. **Balanced classes**: Aim for equal low/high samples
2. **Multiple participants**: At least 10 for generalization
3. **Varied conditions**: Different lighting, times of day
4. **Quality control**: Remove low-quality windows
5. **Consistent protocol**: Same baseline/task procedures

### Model Selection

1. **Start simple**: LogReg baseline first
2. **Validate properly**: Always use GroupKFold
3. **Check calibration**: Plot reliability diagrams
4. **Monitor overfitting**: Compare train/test metrics
5. **Consider ensemble**: Combine multiple models

### Feature Selection

1. **Remove correlated**: blink_count ≈ blink_rate
2. **Keep controls**: brightness for environmental variation
3. **Engineer carefully**: Domain-relevant interactions
4. **Scale properly**: StandardScaler for most models
5. **Handle missing**: Median imputation preserves distribution
