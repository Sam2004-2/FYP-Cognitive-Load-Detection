# FYP Refactor - Simplified Binary Classification

## What Changed

The project has been simplified from a multi-class, multi-approach system to a focused **binary classifier with trend detection**:

### Before
- 5 training scripts (train.py, train_classification.py, train_classification_improved.py, train_regression.py, eval scripts)
- 33 features (9 base + 24 derived)
- 3-class ordinal classification + regression + ensembles
- 5 config files
- 4 model folders

### After
- **1 training script** ([train_binary.py](Machine Learning/src/cle/train/train_binary.py))
- **9 base features** (no derived features)
- **Binary classification** (HIGH vs LOW)
- **Simple trend detection** (INCREASING/DECREASING/STABLE)
- **1 config file** ([config.yaml](Machine Learning/configs/config.yaml))
- **1 model folder** (binary_classifier)

---

## Architecture

```
Machine Learning/
├── src/cle/
│   ├── train/
│   │   └── train_binary.py       # Single binary training script
│   ├── predict/
│   │   ├── __init__.py
│   │   └── trend.py              # Trend detection (no ML)
│   ├── api.py                    # Updated for binary + trend
│   └── server.py                 # Updated API endpoints
├── configs/
│   └── config.yaml               # Single unified config
├── models/
│   └── binary_classifier/        # Model artifacts go here
└── archive/                      # Old files preserved here
    ├── train/
    ├── configs/
    └── models/
```

---

## Binary Classification

**Threshold:** `load_0_1 >= 0.5` → **HIGH**, `load_0_1 < 0.5` → **LOW**

### 9 Base Features (No Derived)
1. `blink_rate` - Blinks per minute
2. `blink_count` - Total blinks in window
3. `mean_blink_duration` - Average blink length (ms)
4. `ear_std` - Eye Aspect Ratio std dev
5. `perclos` - % eye closure
6. `mean_brightness` - Face region brightness
7. `std_brightness` - Brightness variability
8. `mean_quality` - Detection quality
9. `valid_frame_ratio` - Valid frames ratio

### Model
- **Algorithm:** XGBoost (GradientBoostingClassifier)
- **Cross-validation:** 5-fold GroupKFold (subject-wise)
- **Prevents data leakage:** Same subject never in train + test

---

## Trend Detection

**Algorithm:** Moving average comparison (no ML needed)

```python
# Compare last 5 predictions vs previous 5
recent_avg = mean(predictions[-5:])
earlier_avg = mean(predictions[-10:-5])

if recent_avg > earlier_avg + 0.1:
    trend = "INCREASING"
elif recent_avg < earlier_avg - 0.1:
    trend = "DECREASING"
else:
    trend = "STABLE"
```

**Parameters:**
- Window: 5 predictions
- Threshold: 0.1 (10% change)

---

## How to Train the Model

### 1. Ensure you have the feature data

```bash
cd "Machine Learning"
ls data/avcaffe_features_final.csv
```

### 2. Run training

```bash
python -m src.cle.train.train_binary \
    --input data/avcaffe_features_final.csv \
    --output models/binary_classifier \
    --cv-folds 5
```

### 3. Check results

The script will output:
- Session-level accuracy
- Confusion matrix
- Model artifacts saved to `models/binary_classifier/`

**Files created:**
- `model.bin` - Trained XGBoost model
- `scaler.bin` - Feature scaler
- `imputer.bin` - Missing value imputer
- `feature_spec.json` - Feature metadata
- `metrics.json` - Training metrics

---

## API Changes

### New Prediction Response

**Before:**
```json
{
  "cli": 0.73,
  "confidence": 0.82
}
```

**After:**
```json
{
  "level": "HIGH",
  "confidence": 0.82,
  "trend": "INCREASING",
  "raw_score": 0.73
}
```

### New Endpoints

- `POST /predict` - Returns binary classification + trend
- `POST /reset-trend` - Reset trend detector state
- `GET /health` - Health check
- `GET /model-info` - Model information

---

## Running the Server

```bash
cd "Machine Learning"

# Make sure model is trained first
python -m src.cle.train.train_binary --input data/avcaffe_features_final.csv --output models/binary_classifier

# Start server
python -m src.cle.server --port 8000
```

The server will:
1. Look for `models/binary_classifier/`
2. If not found, fall back to `models/stress_classifier_rf/`
3. Initialize trend detector automatically

---

## Why This Simplification?

### For Your Thesis

1. **Clearer scope:** "Binary classifier with trend detection" is easy to explain
2. **Easier to demonstrate:** HIGH/LOW + trend is intuitive
3. **Reduced complexity:** 9 features vs 33, 1 script vs 5
4. **Still academically sound:** GroupKFold CV, proper evaluation

### Technical Benefits

1. **Faster training:** Single model vs multiple approaches
2. **Easier debugging:** One script to maintain
3. **Clearer API:** Simple response format
4. **No feature engineering overhead:** +6% accuracy wasn't worth 24 extra features

### What You Can Still Discuss

You can mention in your thesis that you explored:
- 3-class ordinal classification
- Continuous regression
- Feature engineering (24 derived features)
- Ensemble methods

But ultimately chose binary classification for simplicity and clarity.

---

## Verification Checklist

- [ ] `train_binary.py` runs without errors
- [ ] Binary model achieves >65% session-level accuracy
- [ ] API server starts successfully
- [ ] `/predict` endpoint returns `{level, confidence, trend, raw_score}`
- [ ] Trend detection works (test with multiple predictions)
- [ ] UI displays binary classification + trend

---

## Files Archived

All old files are in `Machine Learning/archive/` for reference:
- `archive/train/` - Old training scripts
- `archive/configs/` - Old config files
- `archive/models/` - Old model artifacts
- `archive/extract/feature_engineering.py` - Derived features code

You can still reference these for your thesis methodology chapter.

---

## Next Steps

1. **Train the model:**
   ```bash
   python -m src.cle.train.train_binary --input data/avcaffe_features_final.csv --output models/binary_classifier
   ```

2. **Test the API:**
   ```bash
   python -m src.cle.server
   # In another terminal:
   curl http://localhost:8000/health
   ```

3. **Update UI** (optional):
   - Change display to show HIGH/LOW instead of continuous value
   - Add trend indicator (↑ ↓ →)

4. **Document in thesis:**
   - Binary classification approach
   - Trend detection algorithm
   - Results and evaluation

---

## Questions?

If you encounter issues:
1. Check logs in `logs/`
2. Verify data file exists at `data/avcaffe_features_final.csv`
3. Ensure all dependencies are installed
4. Check that Python environment is activated
