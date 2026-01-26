# API Reference

Complete reference for the Cognitive Load Estimation API.

## REST API Endpoints

### Base URL

```
http://localhost:8000
```

### Health Check

Check server status and model availability.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 9
}
```

**Status Codes:**
- `200 OK` - Server is running
- `503 Service Unavailable` - Model not loaded

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "healthy" or "degraded" |
| `model_loaded` | boolean | Whether model artifacts are loaded |
| `feature_count` | integer | Number of features model expects |

---

### Root Endpoint

Get API information.

**Endpoint:** `GET /`

**Response:**
```json
{
  "service": "Cognitive Load Estimation API",
  "version": "1.0.0",
  "endpoints": {
    "POST /predict": "Make cognitive load prediction",
    "GET /health": "Health check",
    "GET /model-info": "Model information"
  }
}
```

---

### Model Information

Get details about the loaded model.

**Endpoint:** `GET /model-info`

**Response:**
```json
{
  "features": [
    "blink_rate",
    "blink_count",
    "mean_blink_duration",
    "ear_std",
    "mean_brightness",
    "std_brightness",
    "perclos",
    "mean_quality",
    "valid_frame_ratio"
  ],
  "n_features": 9,
  "calibration": {
    "method": "platt",
    "trained_at": "2026-01-20T15:30:00",
    "training_samples": 1500
  }
}
```

**Status Codes:**
- `200 OK` - Success
- `503 Service Unavailable` - Model not loaded

---

### Predict Cognitive Load

Make a cognitive load prediction from window features.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "features": {
    "blink_rate": 18.5,
    "blink_count": 3.0,
    "mean_blink_duration": 180.5,
    "ear_std": 0.045,
    "mean_brightness": 142.3,
    "std_brightness": 12.8,
    "perclos": 0.08,
    "mean_quality": 0.92,
    "valid_frame_ratio": 0.98
  }
}
```

**Request Schema:**
```typescript
interface PredictionRequest {
  features: {
    blink_rate: number;      // Blinks per minute (0-60)
    blink_count: number;     // Total blinks in window (0-20)
    mean_blink_duration: number;  // Mean duration in ms (100-500)
    ear_std: number;         // EAR standard deviation (0-0.1)
    mean_brightness: number; // Mean brightness (0-255)
    std_brightness: number;  // Brightness std dev (0-50)
    perclos: number;         // Percentage eye closure (0-1)
    mean_quality: number;    // Detection quality (0-1)
    valid_frame_ratio: number;  // Valid frame ratio (0-1)
  }
}
```

**Response:**
```json
{
  "cli": 0.65,
  "confidence": 0.78,
  "success": true,
  "message": null
}
```

**Response Schema:**
```typescript
interface PredictionResponse {
  cli: number;          // Cognitive Load Index (0-1)
  confidence: number;   // Prediction confidence (0-1)
  success: boolean;     // Whether prediction succeeded
  message: string | null;  // Optional message
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid request body
- `500 Internal Server Error` - Prediction failed
- `503 Service Unavailable` - Model not loaded

**Example cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "blink_rate": 18.5,
      "blink_count": 3.0,
      "mean_blink_duration": 180.5,
      "ear_std": 0.045,
      "mean_brightness": 142.3,
      "std_brightness": 12.8,
      "perclos": 0.08,
      "mean_quality": 0.92,
      "valid_frame_ratio": 0.98
    }
  }'
```

---

### Save Training Data

Save collected training data from Data Collection mode.

**Endpoint:** `POST /save-training-data`

**Request Body:**
```json
{
  "participant_id": "P001",
  "session_notes": "Morning session, good lighting",
  "samples": [
    {
      "timestamp": 1706123456789,
      "window_index": 0,
      "label": "low",
      "difficulty": "easy",
      "task_type": "baseline",
      "features": {
        "blink_rate": 15.2,
        "blink_count": 2.0,
        "mean_blink_duration": 165.0,
        "ear_std": 0.038,
        "mean_brightness": 145.0,
        "std_brightness": 10.5,
        "perclos": 0.05,
        "mean_quality": 0.95,
        "valid_frame_ratio": 0.99
      },
      "valid_frame_ratio": 0.99
    }
  ]
}
```

**Request Schema:**
```typescript
interface TrainingDataRequest {
  participant_id: string;
  session_notes?: string;
  samples: Array<{
    timestamp: number;
    window_index: number;
    label: "low" | "high";
    difficulty: "easy" | "medium" | "hard";
    task_type: string;
    features: WindowFeatures;
    valid_frame_ratio: number;
  }>;
}
```

**Response:**
```json
{
  "success": true,
  "filename": "collected_P001_20260120_153000.csv",
  "samples_saved": 48,
  "message": "Data saved to data/collected/collected_P001_20260120_153000.csv"
}
```

**Status Codes:**
- `200 OK` - Data saved successfully
- `400 Bad Request` - No samples provided
- `500 Internal Server Error` - Save failed

---

## Python API

### Module: `src.cle.api`

High-level functions for model loading and prediction.

#### load_model

```python
def load_model(models_dir: str) -> Dict:
    """
    Load trained model artifacts.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with:
            - model: Trained (calibrated) model
            - scaler: Fitted scaler
            - feature_spec: Feature specification
            - calibration: Calibration metadata

    Raises:
        FileNotFoundError: If required artifacts are missing
    """
```

**Example:**
```python
from src.cle.api import load_model

artifacts = load_model("models/stress_classifier_rf")
print(f"Features: {artifacts['feature_spec']['features']}")
```

---

#### predict_window

```python
def predict_window(
    features: Union[Dict, np.ndarray, List],
    artifacts: Dict,
) -> Tuple[float, float]:
    """
    Predict Cognitive Load Index (CLI) for a window of features.

    Args:
        features: Window features as:
            - Dict: feature_name -> value mapping
            - np.ndarray: array of feature values (must match feature_spec order)
            - List: list of feature values (must match feature_spec order)
        artifacts: Model artifacts from load_model()

    Returns:
        Tuple of (cli, confidence):
            - cli: Cognitive Load Index in [0, 1]
            - confidence: Prediction confidence in [0, 1]

    Raises:
        ValueError: If features don't match expected format
    """
```

**Example:**
```python
from src.cle.api import load_model, predict_window

artifacts = load_model("models/stress_classifier_rf")

features = {
    "blink_rate": 18.5,
    "blink_count": 3.0,
    "mean_blink_duration": 180.5,
    "ear_std": 0.045,
    "mean_brightness": 142.3,
    "std_brightness": 12.8,
    "perclos": 0.08,
    "mean_quality": 0.92,
    "valid_frame_ratio": 0.98
}

cli, confidence = predict_window(features, artifacts)
print(f"CLI: {cli:.3f}, Confidence: {confidence:.3f}")
```

---

#### extract_features_from_window

```python
def extract_features_from_window(
    frame_data: List[Dict],
    config: Dict,
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Extract window-level features from list of per-frame features.

    Args:
        frame_data: List of per-frame feature dictionaries
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Dictionary of window-level features
    """
```

**Example:**
```python
from src.cle.api import extract_features_from_window
from src.cle.config import load_config

config = load_config(None)

# frame_data from video processing
frame_data = [
    {"ear_mean": 0.25, "brightness": 120, "quality": 0.9, "valid": True},
    {"ear_mean": 0.26, "brightness": 121, "quality": 0.9, "valid": True},
    # ... more frames
]

features = extract_features_from_window(frame_data, config, fps=30.0)
```

---

#### predict_from_frame_data

```python
def predict_from_frame_data(
    frame_data: List[Dict],
    artifacts: Dict,
    config: Dict,
    fps: float = 30.0,
) -> Tuple[float, float]:
    """
    End-to-end prediction from per-frame features.

    Combines feature extraction and prediction in one call.

    Args:
        frame_data: List of per-frame feature dictionaries
        artifacts: Model artifacts from load_model()
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Tuple of (cli, confidence)
    """
```

---

### Module: `src.cle.config`

Configuration management.

#### load_config

```python
def load_config(config_path: Optional[str]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file, or None for default

    Returns:
        Config object with get() method for accessing values
    """
```

**Example:**
```python
from src.cle.config import load_config

# Load default config
config = load_config(None)

# Load specific config
config = load_config("configs/avcaffe_regression.yaml")

# Access values
ear_thresh = config.get("blink.ear_thresh", 0.21)
window_length = config.get("windows.length_s", 10.0)
```

---

### Module: `src.cle.extract.features`

Feature extraction functions.

#### compute_window_features

```python
def compute_window_features(
    window_data: List[Dict],
    config: Dict,
    fps: float
) -> Dict[str, float]:
    """
    Compute all window-level features.

    Args:
        window_data: List of per-frame feature dictionaries
        config: Configuration dictionary
        fps: Frames per second

    Returns:
        Dictionary with all window features
    """
```

---

#### detect_blinks

```python
def detect_blinks(
    ear_series: np.ndarray,
    fps: float,
    ear_threshold: float = 0.21,
    min_blink_ms: int = 120,
    max_blink_ms: int = 400,
) -> List[Tuple[int, int]]:
    """
    Detect blinks from Eye Aspect Ratio (EAR) time series.

    Args:
        ear_series: Array of EAR values
        fps: Frames per second
        ear_threshold: EAR threshold for blink detection
        min_blink_ms: Minimum blink duration in milliseconds
        max_blink_ms: Maximum blink duration in milliseconds

    Returns:
        List of (start_idx, end_idx) tuples for each blink
    """
```

---

#### get_feature_names

```python
def get_feature_names(config: Dict) -> List[str]:
    """
    Get ordered list of feature names based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Ordered list of feature names
    """
```

---

## TypeScript API (Frontend)

### apiClient.ts

#### testConnection

```typescript
async function testConnection(): Promise<boolean>
```

Test backend connectivity.

---

#### predictCognitiveLoad

```typescript
async function predictCognitiveLoad(
  features: WindowFeatures
): Promise<PredictionResult>
```

Send features to backend for prediction.

---

#### saveTrainingData

```typescript
async function saveTrainingData(
  participantId: string,
  samples: TrainingSample[],
  sessionNotes?: string
): Promise<SaveResult>
```

Save collected training data to server.

---

### featureExtraction.ts

#### eyeAspectRatio

```typescript
function eyeAspectRatio(
  eyeCoords: { x: number; y: number; z: number }[]
): number
```

Calculate Eye Aspect Ratio from 6 eye landmarks.

---

#### extractFrameFeatures

```typescript
function extractFrameFeatures(
  imageData: ImageData,
  landmarkResult: LandmarkResult
): FrameFeatures
```

Extract per-frame features from video frame.

---

#### computeWindowFeatures

```typescript
function computeWindowFeatures(
  frameData: FrameFeatures[],
  fps: number
): WindowFeatures
```

Aggregate frame features into window features.

---

#### detectBlinks

```typescript
function detectBlinks(
  earSeries: number[],
  fps: number,
  earThreshold?: number,
  minBlinkMs?: number,
  maxBlinkMs?: number
): Blink[]
```

Detect blinks from EAR time series.

---

## Data Types

### WindowFeatures

```typescript
interface WindowFeatures {
  blink_rate: number;
  blink_count: number;
  mean_blink_duration: number;
  ear_std: number;
  mean_brightness: number;
  std_brightness: number;
  perclos: number;
  mean_quality: number;
  valid_frame_ratio: number;
}
```

### FrameFeatures

```typescript
interface FrameFeatures {
  ear_left: number;
  ear_right: number;
  ear_mean: number;
  brightness: number;
  quality: number;
  valid: boolean;
}
```

### PredictionResult

```typescript
interface PredictionResult {
  cli: number;
  confidence: number;
  success: boolean;
  message?: string;
}
```

---

## Error Handling

### HTTP Error Codes

| Code | Description | Action |
|------|-------------|--------|
| 400 | Bad Request | Check request body format |
| 500 | Internal Server Error | Check server logs |
| 503 | Service Unavailable | Model not loaded, check model files |

### Common Errors

**Feature dimension mismatch:**
```json
{
  "detail": "Feature dimension mismatch: got 8, expected 9"
}
```
Solution: Ensure all 9 features are provided.

**Model not loaded:**
```json
{
  "detail": "Model not loaded"
}
```
Solution: Check model directory exists and contains required files.

---

## Rate Limits

No rate limits are enforced by default. For production, consider adding rate limiting middleware.

## CORS

CORS is enabled for all origins by default. For production, restrict to specific origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
