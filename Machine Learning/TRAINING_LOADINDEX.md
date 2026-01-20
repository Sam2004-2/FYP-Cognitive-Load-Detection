# Load Index Regression Training

This document explains how to train regression models to predict continuous cognitive load (`load_0_1`) from video-derived ocular features using StressID self-assessment data.

## Overview

Instead of binary classification (low/high), this pipeline predicts a continuous load score in the range [0, 1], derived from participant self-assessments of stress and relaxation.

## Label Construction

The `load_0_1` target is computed from StressID self-assessment questionnaires:

```
load_raw = (stress + (10 - relax)) / 2
load_0_1 = (load_raw - min) / (max - min)
```

Where:
- `stress`: Self-reported stress level (0-10 scale)
- `relax`: Self-reported relaxation level (0-10 scale, inverted)
- Result: Higher cognitive load when stress is high AND relaxation is low

The pre-computed values are in `data/raw/assessments/self_assessments_loadindex.csv`.

## Data Join Process

Features and labels are joined on `(subject, task)`:

1. **Extract task from video path**: The video column contains paths like `/path/to/2ea4_Math.mp4`. We extract the task name (`Math`) using regex.

2. **Filter to target tasks**: Only these 4 tasks are used for training:
   - `Math`
   - `Counting1`
   - `Counting2`
   - `Speaking`

3. **Inner join**: Features are joined with load labels on `(user_id=subject, task)`. Windows without matching labels are dropped.

### Merge Validation

The pipeline prints detailed merge statistics:
```
Features before task filter: 356 rows
Features after task filter: 356 rows
Unique (subject, task) pairs in features: 40
Matching pairs: 40
Rows after merge: 356
```

Always verify these counts match expectations before training.

## Subject-Wise Splitting

**Critical**: To prevent data leakage, all splits are performed by subject, not by individual windows.

### GroupKFold (Default)
- Splits subjects into K groups (default: 5 folds)
- Each fold tests on ~2 subjects (with 10 subjects total)
- No subject appears in both train and test within a fold

### Leave-One-Subject-Out (LOSO)
- Each fold holds out exactly one subject
- Most rigorous but higher variance with small N
- Use `--loso` flag to enable

### Validation
The pipeline automatically validates no leakage:
```python
assert len(train_subjects & test_subjects) == 0
```

## Evaluation Metrics

### Session-Level (Primary)
Window predictions are aggregated per (subject, task) using mean:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - average prediction error |
| **Spearman r** | Rank correlation - captures monotonic relationship |
| **RMSE** | Root Mean Squared Error |
| **R2** | Coefficient of determination |

### Window-Level (Secondary)
Raw per-window metrics are also reported for debugging but are NOT the primary evaluation since windows within a session share the same label.

## How to Run

### Training with Cross-Validation

```bash
cd "Machine Learning"

# Basic training (Ridge regression, 5-fold GroupKFold)
python -m src.cle.train.train_regression \
    --in data/processed/stressid_features.csv \
    --load-index data/raw/assessments/self_assessments_loadindex.csv \
    --out models/regression/

# With different model type
python -m src.cle.train.train_regression \
    --in data/processed/stressid_features.csv \
    --load-index data/raw/assessments/self_assessments_loadindex.csv \
    --out models/regression/ \
    --model-type rf  # or "xgb"

# With Leave-One-Subject-Out CV
python -m src.cle.train.train_regression \
    --in data/processed/stressid_features.csv \
    --load-index data/raw/assessments/self_assessments_loadindex.csv \
    --out models/regression/ \
    --loso

# Custom config
python -m src.cle.train.train_regression \
    --in data/processed/stressid_features.csv \
    --load-index data/raw/assessments/self_assessments_loadindex.csv \
    --out models/regression/ \
    --config configs/regression.yaml
```

### Evaluation

```bash
python -m src.cle.train.eval_regression \
    --in data/processed/stressid_features.csv \
    --load-index data/raw/assessments/self_assessments_loadindex.csv \
    --models models/regression/ \
    --report reports/regression_eval.json
```

### Output Files

After training, the output directory contains:

| File | Description |
|------|-------------|
| `model_regression.bin` | Trained model (Ridge/RF/XGB) |
| `scaler.bin` | Fitted StandardScaler |
| `imputer.bin` | Fitted median imputer |
| `feature_spec.json` | Feature names and metadata |
| `cv_results.json` | Cross-validation results |
| `window_predictions.csv` | All window-level predictions |
| `session_predictions.csv` | Session-level aggregated predictions |

## Model Types

### Ridge Regression (default)
- Linear model with L2 regularization
- Fast, interpretable, good baseline
- Config: `regression.ridge_params.alpha`

### Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Config: `regression.rf_params.*`

### Gradient Boosting (XGBoost-style)
- Sequential boosted trees
- Often best performance
- Config: `regression.xgb_params.*`

## Expected Results

With 10 subjects and 40 sessions:

| Metric | Expected Range |
|--------|---------------|
| Session MAE | 0.10 - 0.20 |
| Spearman r | 0.40 - 0.70 |

Note: Results vary significantly across folds due to small sample size.

## Troubleshooting

### No matching (subject, task) pairs
- Check that subject IDs in features match load index (case-sensitive)
- Verify task names extracted from video paths are correct
- The pipeline prints unmatched keys for debugging

### High variance across folds
- Expected with only 10 subjects
- Consider using LOSO for more robust estimate
- Check for outlier subjects

### NaN in predictions
- Features with NaN are imputed with median
- If all values are NaN for a feature, check data quality

### Overfitting (train MAE << test MAE)
- Increase Ridge alpha (e.g., 10.0)
- Reduce RF max_depth
- Use fewer features

## Configuration Reference

See `configs/regression.yaml` for all options:

```yaml
task_mode: "regression"

regression:
  model_type: "ridge"  # ridge, rf, xgb
  ridge_params:
    alpha: 1.0
  rf_params:
    n_estimators: 100
    max_depth: 10
  xgb_params:
    n_estimators: 100
    max_depth: 4

cv:
  method: "group_kfold"  # or "loso"
  n_splits: 5

session_agg:
  method: "mean"  # or "median"

tasks:
  - "Math"
  - "Counting1"
  - "Counting2"
  - "Speaking"
```

## Comparison with Classification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Target | Binary (low/high) | Continuous [0,1] |
| Loss | Cross-entropy | MSE |
| Metrics | AUC, F1 | MAE, Spearman |
| Granularity | Coarse | Fine-grained |
| Calibration | Platt/Isotonic | N/A |

The regression approach provides richer predictions that better capture the continuous nature of cognitive load.
