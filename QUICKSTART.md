# Quickstart

This is the shortest supported local workflow for the current repository.

## Prerequisites

- Python `3.10+`
- Node.js and npm

## 1. Start the Backend

```sh
cd machine_learning
python -m pip install -e ".[dev]"
python -m src.cle.server --host 0.0.0.0 --port 8000 --models-dir models/video_physio_regression_z01_geom
```

Health check:

```sh
curl http://localhost:8000/health
```

## 2. Start the UI

Point the UI at the backend with `REACT_APP_API_URL=http://localhost:8000`.

Create `UI/.env.local` from `UI/.env.example` if needed.

Then run:

```sh
cd UI
npm install
npm start
```

Open `http://localhost:3000/`.

Main routes:

- `/`
- `/study/setup`
- `/study/session`
- `/study/summary`
- `/study/delayed`
- `/admin`

## 3. Validate the Repo

```sh
cd machine_learning
python -m pytest tests -q
```

```sh
cd UI
npm test -- --watchAll=false --passWithNoTests
npm run build
```

## 4. Optional Training Workflow

The training scripts depend on external datasets that are not committed to this repo. Adjust paths to match your local data layout.

```sh
cd machine_learning

python scripts/extract_all_physio_features.py --physio-dir <path-to-physio-dir> --output data/processed/physio_features.csv
python scripts/generate_physio_labels.py
python scripts/train_video_physio_regression.py --video-features data/processed/stress_features_10s_geom.csv --physio-labels data/processed/physio_stress_labels.csv --out models/video_physio_regression_z01_geom --report reports/video_physio_regression_z01_geom_eval.json --merge-mode overlap --target z01
```

For API, config, and study notes, see [docs/README.md](./docs/README.md).
