# Cognitive Load Estimation (StressID) — ML Backend

This directory contains the machine-learning backend, feature extraction code, and training scripts for the **StressID** real-time cognitive load system.

## Run the API (real-time)

```bash
cd "Machine Learning"
python3 -m pip install -e ".[dev]"

# Uses the physio-aligned student by default if you pass --models-dir
python3 -m src.cle.server --host 0.0.0.0 --port 8000 \
  --models-dir models/video_physio_regression_z01_geom
```

## Re-extract StressID video window features

```bash
cd "Machine Learning" && python3 -m src.cle.extract.pipeline_offline \
  --manifest data/raw/stress_manifest.csv \
  --out data/processed/stress_features_10s_geom.csv \
  --config configs/default.yaml
```

## Canonical physio-aligned training chain (recommended)

This workflow is the supported thesis training path:
1) extract physiological features, 2) generate physio stress labels, 3) train the video student.

```bash
cd "Machine Learning" && python3 scripts/extract_all_physio_features.py \
  --physio-dir ../Physiological \
  --output data/processed/physio_features.csv

cd "Machine Learning" && python3 scripts/generate_physio_labels.py

cd "Machine Learning" && python3 scripts/train_video_physio_regression.py \
  --video-features data/processed/stress_features_10s_geom.csv \
  --physio-labels data/processed/physio_stress_labels.csv \
  --out models/video_physio_regression_z01_geom \
  --report reports/video_physio_regression_z01_geom_eval.json \
  --merge-mode overlap \
  --target z01
```

## Layout

```text
Machine Learning/
  configs/       # Feature + windowing config
  data/raw/      # Manifests / self assessments
  src/cle/       # API, extraction, training library
  scripts/       # Extraction + training entrypoints
  models/        # Tracked demo models (lean allowlist)
  tests/         # Pytest suite
```
