# Cognitive Load Estimation System

A real-time cognitive load detection system that uses machine learning to estimate mental workload from facial features captured via webcam.

## Overview

This system provides non-invasive, real-time cognitive load monitoring using eye-tracking metrics extracted from facial landmarks. It combines a Python/FastAPI backend for ML inference with a React/TypeScript frontend for real-time visualization.

### Key Features

- **Real-time Detection**: Continuous cognitive load monitoring at 2.5-second intervals
- **Non-invasive**: Uses standard webcam - no specialized hardware required
- **Calibration-free**: Eye Aspect Ratio (EAR) based features work without calibration
- **Multiple Modes**: Active monitoring and data collection for model training
- **Extensible**: Support for multiple datasets and model architectures

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   Webcam    │───▶│  MediaPipe   │───▶│   Feature   │───▶│   Window   │  │
│  │    Feed     │    │  FaceMesh    │    │  Extraction │    │   Buffer   │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘  │
│                                                                   │         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────▼──────┐  │
│  │  Cognitive  │◀───│    Load      │◀───│     Window Features (JSON)     │  │
│  │ Load Gauge  │    │    Chart     │    │ Every 2.5 seconds (10s window) │  │
│  └─────────────┘    └──────────────┘    └─────────────────────────┬──────┘  │
└───────────────────────────────────────────────────────────────────┼─────────┘
                                                                    │ HTTP POST
                                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   /predict  │───▶│   Feature    │───▶│    Scale    │───▶│  ML Model  │  │
│  │   Endpoint  │    │  Validation  │    │  (Standard) │    │ (Calibrated)│  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘  │
│                                                                   │         │
│                    ┌──────────────────────────────────────────────▼──────┐  │
│                    │  Cognitive Load Index (0-1) + Confidence Score     │  │
│                    └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.10+, FastAPI, scikit-learn, XGBoost |
| **Frontend** | React 19, TypeScript, Tailwind CSS, Recharts |
| **Face Detection** | MediaPipe FaceMesh (468 landmarks) |
| **ML Models** | Logistic Regression, Gradient Boosting, Random Forest |
| **Data** | AVCAffe Dataset (106 participants), StressID |

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

### TL;DR

**Terminal 1 - Backend:**
```bash
cd "Machine Learning"
pip install -e .
python -m src.cle.server --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd UI
npm install
npm start
```

Open http://localhost:3000 in your browser.

## Project Structure

```
FYP-Cognitive-Load-Detection/
├── Machine Learning/           # Python ML backend
│   ├── configs/               # YAML configuration files
│   ├── data/                  # Data files (gitignored)
│   ├── docs/                  # ML-specific documentation
│   ├── models/                # Trained model artifacts
│   ├── scripts/               # Data processing scripts
│   ├── src/cle/               # Core library
│   │   ├── api.py            # Public API
│   │   ├── server.py         # FastAPI server
│   │   ├── config.py         # Configuration management
│   │   ├── data/             # Data loading utilities
│   │   ├── extract/          # Feature extraction pipeline
│   │   ├── train/            # Model training scripts
│   │   └── utils/            # Common utilities
│   └── tests/                 # pytest test suite
│
├── UI/                        # React frontend
│   ├── public/               # Static assets
│   └── src/
│       ├── pages/            # Page components
│       ├── components/       # Reusable UI components
│       ├── services/         # API client, MediaPipe
│       ├── config/           # Feature configuration
│       └── types/            # TypeScript types
│
├── docs/                      # Project documentation
│   ├── INSTALLATION.md       # Setup guide
│   ├── USER_GUIDE.md         # Usage instructions
│   ├── DEVELOPER_GUIDE.md    # Development guide
│   ├── ML_PIPELINE.md        # ML pipeline documentation
│   ├── API_REFERENCE.md      # API documentation
│   ├── DATA_GUIDE.md         # Dataset documentation
│   ├── CONFIGURATION.md      # Config reference
│   └── TROUBLESHOOTING.md    # Common issues
│
├── QUICKSTART.md             # Quick setup guide
└── README.md                 # This file
```

## Features Extracted

The system extracts 9 core features from 10-second sliding windows:

| Feature | Description | Cognitive Load Indicator |
|---------|-------------|--------------------------|
| `blink_rate` | Blinks per minute | Increases with load |
| `blink_count` | Total blinks in window | Activity level |
| `mean_blink_duration` | Average blink length (ms) | Fatigue indicator |
| `ear_std` | Eye Aspect Ratio variability | Increases with load |
| `perclos` | Percentage of eye closure | Key fatigue metric |
| `mean_brightness` | Face region brightness | Environmental control |
| `std_brightness` | Brightness variability | Stability indicator |
| `mean_quality` | Detection confidence | Data quality |
| `valid_frame_ratio` | Valid frames ratio | Reliability metric |

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5 minutes |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Detailed setup instructions |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | How to use the application |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Codebase structure and extending |
| [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) | Feature extraction and training |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | REST API and Python API |
| [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md) | Datasets and preprocessing |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [Machine Learning/docs/AVCAFFE_DATASET.md](Machine%20Learning/docs/AVCAFFE_DATASET.md) | AVCAffe dataset specifics |

## Application Modes

### Active Session Mode
Real-time cognitive load monitoring with:
- Live webcam feed with face detection overlay
- Cognitive Load Index gauge (0-1 scale)
- Time-series load chart
- Live feature values panel

### Data Collection Mode
Collect labeled training data with:
- Structured baseline/task protocol
- Low/High cognitive load labeling
- NASA-TLX self-assessment integration
- Export to CSV/JSON or save to server

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/predict` | POST | Cognitive load prediction |
| `/model-info` | GET | Model information |
| `/save-training-data` | POST | Save collected training data |

## Model Performance

Current AVCAffe baseline model (XGBoost regression):

| Metric | Window-level | Session-level |
|--------|--------------|---------------|
| MAE | 0.244 | 0.243 |
| RMSE | 0.285 | 0.283 |
| Spearman r | 0.029 | 0.030 |

Note: Performance limited by task-level labels applied to window-level predictions.

## Development

### Running Tests

```bash
# Backend tests
cd "Machine Learning"
pytest tests/ -v

# Frontend tests
cd UI
npm test
```

### Code Quality

```bash
# Backend linting
cd "Machine Learning"
black src/ tests/
mypy src/

# Frontend linting
cd UI
npm run lint
```

## Requirements

### Backend
- Python 3.10+
- See `Machine Learning/pyproject.toml` for dependencies

### Frontend
- Node.js 18+
- See `UI/package.json` for dependencies

### Hardware
- Webcam (720p minimum recommended)
- Modern browser (Chrome/Edge recommended)

## License

See LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for facial landmark detection
- [AVCAffe Dataset](https://github.com/avcaffe/avcaffe) for training data
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
