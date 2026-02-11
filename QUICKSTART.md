# Thesis Quickstart (Pruned Scope)

This guide covers the only supported thesis workflow:
1. Realtime CLI inference backend
2. Study protocol UI flow
3. Canonical physio-aligned training pipeline

## 1) Start Backend

```bash
cd "Machine Learning"
python3 -m pip install -e ".[dev]"
python3 -m src.cle.server --host 0.0.0.0 --port 8000 \
  --models-dir models/video_physio_regression_z01_geom
```

Expected health check:

```bash
curl http://localhost:8000/health
```

## 2) Start UI (Study Workflow)

```bash
cd UI
npm install
npm start
```

Then run:
1. `http://localhost:3000/`
2. Click `Start Study Setup`
3. Complete the flow:
`/study/setup -> /study/session -> /study/summary -> /study/delayed`

## 3) Canonical Training Pipeline

```bash
cd "Machine Learning"

# A. Extract physiological features
python3 scripts/extract_all_physio_features.py \
  --physio-dir ../Physiological \
  --output data/processed/physio_features.csv

# B. Generate physiological stress labels
python3 scripts/generate_physio_labels.py

# C. Train the video student model
python3 scripts/train_video_physio_regression.py \
  --video-features data/processed/stress_features_10s_geom.csv \
  --physio-labels data/processed/physio_stress_labels.csv \
  --out models/video_physio_regression_z01_geom \
  --report reports/video_physio_regression_z01_geom_eval.json \
  --merge-mode overlap \
  --target z01
```

## 4) Validation Commands

```bash
cd "Machine Learning" && python3 -m pytest tests -q
cd UI && npm test -- --watchAll=false --passWithNoTests
cd UI && npm run build
```
