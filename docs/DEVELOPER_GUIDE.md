# Developer Guide

Technical guide for understanding and extending the Cognitive Load Estimation system.

## Repository Structure

```
FYP-Cognitive-Load-Detection/
├── Machine Learning/           # Python backend
│   ├── configs/               # YAML configuration files
│   ├── data/                  # Data files (gitignored)
│   ├── docs/                  # ML-specific docs
│   ├── models/                # Trained model artifacts
│   ├── scripts/               # Utility scripts
│   ├── src/cle/               # Core library
│   │   ├── api.py            # Public API
│   │   ├── server.py         # FastAPI server
│   │   ├── config.py         # Configuration
│   │   ├── data/             # Data loaders
│   │   ├── extract/          # Feature extraction
│   │   ├── train/            # Training pipelines
│   │   └── utils/            # Utilities
│   └── tests/                 # pytest tests
│
├── UI/                        # React frontend
│   └── src/
│       ├── pages/            # Page components
│       ├── components/       # Reusable components
│       ├── services/         # Business logic
│       ├── config/           # Configuration
│       └── types/            # TypeScript types
│
└── docs/                      # Project documentation
```

## Backend Architecture

### Module Overview

```
src/cle/
├── api.py              # Public API: load_model(), predict_window()
├── server.py           # FastAPI server with REST endpoints
├── config.py           # Configuration loading and validation
├── logging_setup.py    # Logging configuration
├── viz.py              # Visualization utilities
│
├── data/               # Data loading
│   ├── load_data.py           # Generic data loading
│   ├── load_avcaffe_data.py   # AVCAffe dataset adapter
│   └── load_avcaffe_labels.py # Label parsing
│
├── extract/            # Feature extraction pipeline
│   ├── landmarks.py           # MediaPipe face mesh
│   ├── per_frame.py           # Per-frame feature extraction
│   ├── features.py            # Window-level aggregation
│   ├── feature_engineering.py # Derived features
│   ├── windowing.py           # Sliding window logic
│   └── pipeline_offline.py    # Batch processing
│
├── train/              # Model training
│   ├── train.py               # Classification training
│   ├── train_regression.py    # Regression training
│   ├── calibrate.py           # Probability calibration
│   ├── eval.py                # Classification evaluation
│   └── eval_regression.py     # Regression evaluation
│
└── utils/              # Utilities
    ├── io.py                  # File I/O, video handling
    └── timers.py              # Performance timing
```

### Key Classes and Functions

**api.py - Public API**
```python
# Load trained model artifacts
artifacts = load_model("models/stress_classifier_rf")

# Predict cognitive load from features
cli, confidence = predict_window(features_dict, artifacts)

# Extract features from frame data
features = extract_features_from_window(frame_data, config, fps=30.0)

# End-to-end prediction
cli, confidence = predict_from_frame_data(frame_data, artifacts, config)
```

**server.py - REST API**
```python
# Endpoints
GET  /health         # Health check
GET  /model-info     # Model information
POST /predict        # Cognitive load prediction
POST /save-training-data  # Save collected data
```

**extract/features.py - Feature Computation**
```python
# Main aggregation function
features = compute_window_features(window_data, config, fps)

# Individual feature computations
blinks = detect_blinks(ear_series, fps, threshold=0.21)
blink_features = compute_blink_features(ear_series, fps, config)
perclos = compute_perclos(ear_series, threshold=0.21)
```

### Data Flow

```
Video Frame
    │
    ▼
MediaPipe FaceMesh (468 landmarks)
    │
    ▼
Per-Frame Features (extract/per_frame.py)
    ├── ear_left, ear_right, ear_mean
    ├── brightness
    └── quality
    │
    ▼
Sliding Window Buffer (10s, 2.5s step)
    │
    ▼
Window Features (extract/features.py)
    ├── blink_rate, blink_count, mean_blink_duration
    ├── ear_std, perclos
    ├── mean_brightness, std_brightness
    └── mean_quality, valid_frame_ratio
    │
    ▼
StandardScaler Transform
    │
    ▼
ML Model Prediction
    │
    ▼
Cognitive Load Index (0-1)
```

## Frontend Architecture

### Component Hierarchy

```
App
├── SessionSetup (landing page)
├── ActiveSession
│   ├── CognitiveLoadGauge
│   ├── WebcamFeed
│   ├── LiveFeaturePanel
│   └── TaskPanel
│       ├── MemoryTask
│       └── MathTask
├── DataCollection
│   ├── WebcamFeed
│   ├── LiveFeaturePanel
│   └── Task components
├── Summary
│   └── LoadChart
└── Settings
```

### Services Layer

**services/mediapipeManager.ts**
- Initializes MediaPipe FaceMesh
- Manages face detection lifecycle
- Handles model loading

**services/featureExtraction.ts**
- Computes Eye Aspect Ratio (EAR)
- Extracts per-frame features
- Aggregates window-level features
- Detects blinks via state machine

**services/windowBuffer.ts**
- Ring buffer for frame storage
- Sliding window management
- Quality validation

**services/apiClient.ts**
- Backend HTTP client
- Health checks
- Prediction requests
- Training data upload

### State Management

The application uses React's built-in state management with hooks:

**Pattern: Ref Sync for Callbacks**
```typescript
// Problem: useCallback captures stale state values
// Solution: Mirror state to refs, read refs in callbacks

const currentLoadRef = useRef(currentLoad);
useEffect(() => { currentLoadRef.current = currentLoad; }, [currentLoad]);

const makePrediction = useCallback(async () => {
  // Use ref to get current value
  const smoothedLoad = alpha * result.cli + (1 - alpha) * currentLoadRef.current;
}, []); // Empty deps - function is stable
```

### Feature Extraction (TypeScript)

The frontend mirrors the Python feature extraction:

```typescript
// Eye Aspect Ratio calculation
function eyeAspectRatio(eyeCoords: Point[]): number {
  const v1 = euclideanDistance(points[1], points[5]);
  const v2 = euclideanDistance(points[2], points[4]);
  const h = euclideanDistance(points[0], points[3]);
  return (v1 + v2) / (2.0 * h);
}

// Blink detection state machine
function detectBlinks(earSeries: number[], fps: number): Blink[] {
  // OPEN -> CLOSED (EAR drops) -> OPEN (EAR rises)
  // Duration filtering: 120-400ms
}

// Window feature aggregation
function computeWindowFeatures(frameData: FrameFeatures[], fps: number): WindowFeatures
```

## Configuration System

### YAML Configuration Files

Located in `Machine Learning/configs/`:

```yaml
# default.yaml - Base configuration
seed: 42
fps_fallback: 30.0

windows:
  length_s: 10.0    # Window duration
  step_s: 2.5       # Step size (75% overlap)

quality:
  min_face_conf: 0.5
  max_bad_frame_ratio: 0.05

blink:
  ear_thresh: 0.21
  min_blink_ms: 120
  max_blink_ms: 400

features_enabled:
  tepr: false       # Disabled (requires calibration)
  blinks: true
  perclos: true
  brightness: true

model:
  type: "logreg"
  calibration: "platt"
```

### Frontend Configuration

Located in `UI/src/config/featureConfig.ts`:

```typescript
export const FEATURE_CONFIG = {
  windows: { length_s: 10.0, step_s: 2.5 },
  blink: { ear_thresh: 0.21, min_blink_ms: 120, max_blink_ms: 400 },
  quality: { min_face_conf: 0.5, max_bad_frame_ratio: 0.05 },
  realtime: { smoothing_alpha: 0.4, conf_threshold: 0.6 },
  video: { fps: 30.0 },
};

export const LANDMARK_INDICES = {
  LEFT_EYE: [33, 160, 158, 133, 153, 144],
  RIGHT_EYE: [362, 385, 387, 263, 373, 380],
};
```

## Adding New Features

### Adding a New Window Feature

1. **Backend (Python)**

   Update `src/cle/extract/features.py`:
   ```python
   def compute_new_feature(data: np.ndarray) -> float:
       """Compute your new feature."""
       return float(np.mean(data))

   def compute_window_features(window_data, config, fps):
       # ... existing code ...
       features["new_feature"] = compute_new_feature(some_series)
       return features
   ```

2. **Update feature names**:
   ```python
   def get_feature_names(config):
       feature_names = [...]
       feature_names.append("new_feature")
       return feature_names
   ```

3. **Frontend (TypeScript)**

   Update `services/featureExtraction.ts`:
   ```typescript
   export interface WindowFeatures {
     // ... existing fields ...
     new_feature: number;
   }

   export function computeWindowFeatures(...): WindowFeatures {
     // ... add new feature computation ...
   }
   ```

4. **Retrain model** with updated features

### Adding a New Model Type

1. **Update training script** (`src/cle/train/train.py`):
   ```python
   def create_model(model_type: str, config: Dict):
       if model_type == "new_model":
           return NewModelClass(**config.get("new_model_params", {}))
   ```

2. **Update configuration** (`configs/default.yaml`):
   ```yaml
   model:
     type: "new_model"
     new_model_params:
       param1: value1
   ```

3. **Ensure compatibility** with calibration and prediction pipelines

### Adding a New Page (Frontend)

1. **Create component** in `UI/src/pages/NewPage.tsx`

2. **Add route** in `UI/src/App.tsx`:
   ```tsx
   import NewPage from './pages/NewPage';

   <Route path="/new-page" element={<NewPage />} />
   ```

3. **Add navigation** link where needed

## Testing

### Backend Tests

```bash
cd "Machine Learning"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src/cle --cov-report=html
```

**Test files:**
- `test_features.py` - Feature computation tests
- `test_per_frame.py` - Per-frame extraction tests
- `test_windowing.py` - Window buffer tests
- `test_config.py` - Configuration tests
- `test_integration.py` - End-to-end tests

### Frontend Tests

```bash
cd UI

# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

### Writing Tests

**Python test example:**
```python
def test_detect_blinks():
    # Create EAR series with known blink
    ear_series = np.array([0.3, 0.3, 0.15, 0.15, 0.15, 0.3, 0.3])
    fps = 30.0

    blinks = detect_blinks(ear_series, fps, ear_threshold=0.21)

    assert len(blinks) == 1
    assert blinks[0] == (2, 5)  # (start, end)
```

## Code Style

### Python

- **Formatter**: black (line length: 100)
- **Import sorting**: isort
- **Type checking**: mypy

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type check
mypy src/
```

### TypeScript

- **Linter**: ESLint (react-app config)
- **Formatter**: Prettier (via ESLint)

```bash
# Lint
npm run lint

# Fix issues
npm run lint -- --fix
```

### Pre-commit Hooks

```bash
cd "Machine Learning"
pre-commit install

# Run manually
pre-commit run --all-files
```

## CI/CD Pipeline

GitHub Actions workflows in `.github/workflows/`:

### Jobs

1. **test-ml-pipeline**
   - Python setup
   - Dependency installation
   - pytest execution

2. **lint-ml-pipeline**
   - black formatting check
   - isort import check
   - mypy type check

3. **test-ui**
   - Node.js setup
   - npm install
   - npm test

### Triggering

- Push to main/develop
- Pull requests
- Manual dispatch

## Debugging

### Backend Debugging

**Enable debug logging:**
```bash
python -m src.cle.server --log-level DEBUG
```

**Add debug prints:**
```python
from src.cle.logging_setup import get_logger
logger = get_logger(__name__)
logger.debug(f"Feature values: {features}")
```

### Frontend Debugging

**Browser DevTools:**
- Console for logs
- Network tab for API calls
- React DevTools for component state

**Add console logs:**
```typescript
console.log('Window features:', windowFeatures);
console.log('Prediction result:', result);
```

### Common Issues

**Feature mismatch:**
- Ensure frontend/backend compute same features
- Check feature order matches model expectation
- Verify scaling is applied correctly

**MediaPipe not loading:**
- Check browser console for WASM errors
- Ensure HTTPS or localhost
- Try clearing browser cache

**Model prediction errors:**
- Check input feature dimensions
- Verify no NaN values
- Ensure scaler is loaded

## Performance Optimization

### Backend

- Use NumPy vectorized operations
- Avoid Python loops where possible
- Profile with cProfile

### Frontend

- Minimize re-renders with useMemo/useCallback
- Use refs for values accessed in callbacks
- Optimize MediaPipe frame processing

### Reducing Latency

1. Increase prediction interval (step_s)
2. Reduce window length
3. Use lighter ML model
4. Enable GPU acceleration for MediaPipe

## Deployment

### Production Backend

```bash
# Using uvicorn with workers
uvicorn src.cle.server:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn
gunicorn src.cle.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Production Frontend

```bash
# Build optimized bundle
npm run build

# Serve with any static file server
npx serve -s build
```

### Docker (optional)

```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY Machine\ Learning/ .
RUN pip install -e .
CMD ["python", "-m", "src.cle.server", "--host", "0.0.0.0"]
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run linting and tests
5. Submit pull request

### PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted (black, isort)
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No secrets in code
