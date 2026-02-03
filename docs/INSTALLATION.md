# Installation Guide

Complete setup instructions for the Cognitive Load Estimation system.

## Prerequisites

### Required Software

| Software | Minimum Version | Recommended | Check Command |
|----------|----------------|-------------|---------------|
| Python | 3.10 | 3.11 | `python --version` |
| Node.js | 18.0 | 20.x LTS | `node --version` |
| npm | 8.0 | 10.x | `npm --version` |
| Git | 2.0 | Latest | `git --version` |

### Hardware Requirements

- **Webcam**: 720p minimum (1080p recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: Modern multi-core processor
- **Browser**: Chrome 90+ or Edge 90+ (recommended for best MediaPipe performance)

## Backend Setup

### 1. Navigate to Backend Directory

```bash
cd "Machine Learning"
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

**Option A: Install as editable package (recommended)**
```bash
pip install -e .
```

**Option B: Install with development tools**
```bash
pip install -e ".[dev]"
```

### 4. Install Additional Runtime Dependencies

The FastAPI server requires additional packages:

```bash
pip install fastapi uvicorn
```

### 5. Verify MediaPipe Model

Ensure the face landmark model exists:

```bash
# Windows
dir models\face_landmarker.task

# Linux/macOS
ls models/face_landmarker.task
```

If missing, download from [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models).

### 6. Verify ML Model

Check that a trained model exists:

```bash
# Windows
dir models\binary_classifier\

# Linux/macOS
ls models/binary_classifier/
```

You should see:
- `model.bin`
- `scaler.bin`
- `imputer.bin`
- `feature_spec.json`
- `metrics.json`

If the model doesn't exist, train it first:

```bash
python -m src.cle.train.train_binary \
    --input data/avcaffe_features_final.csv \
    --output models/binary_classifier \
    --cv-folds 5
```

### 7. Test Backend Installation

```bash
# Start the server
python -m src.cle.server --host 127.0.0.1 --port 8000
```

Expected output:
```
INFO: Starting Cognitive Load Estimation API
INFO: Loaded configuration (hash: xxxxxxxx)
INFO: Loaded model artifacts with 9 features
INFO: Initialized trend detector (window=5, threshold=0.1)
INFO: API ready to accept requests
INFO: Uvicorn running on http://127.0.0.1:8000
```

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","model_loaded":true,"feature_count":9}
```

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd UI
```

### 2. Install Dependencies

```bash
npm install
```

This installs:
- React 19 and React DOM
- React Router
- MediaPipe libraries
- Tailwind CSS
- Recharts for visualization
- TypeScript and type definitions

### 3. Verify Installation

```bash
npm list --depth=0
```

### 4. Start Development Server

```bash
npm start
```

The browser should open automatically to http://localhost:3000.

### 5. Verify Frontend-Backend Connection

1. Ensure backend is running (Terminal 1)
2. Open http://localhost:3000 in browser
3. Look for "Backend Connected" indicator (green dot)
4. Check browser console (F12) for any errors

## Complete Installation Verification

### Checklist

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] Backend virtual environment created and activated
- [ ] Backend dependencies installed (`pip install -e .`)
- [ ] FastAPI/uvicorn installed (`pip install fastapi uvicorn`)
- [ ] MediaPipe model exists (`models/face_landmarker.task`)
- [ ] ML model trained (`models/binary_classifier/`)
- [ ] Backend starts without errors
- [ ] Health endpoint returns healthy status
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Frontend starts without errors
- [ ] Browser shows application
- [ ] Backend connection indicator is green

### Quick Test

1. **Terminal 1 - Backend:**
   ```bash
   cd "Machine Learning"
   .venv\Scripts\activate  # Windows
   python -m src.cle.server --host 0.0.0.0 --port 8000
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   cd UI
   npm start
   ```

3. **Browser:**
   - Open http://localhost:3000
   - Click "Start Session"
   - Allow camera access
   - Wait for face detection
   - Verify predictions appear after ~10 seconds

## Development Setup (Optional)

### Backend Development Tools

```bash
cd "Machine Learning"
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Code Formatting

```bash
# Format Python code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Backend tests
cd "Machine Learning"
pytest tests/ -v

# Frontend tests
cd UI
npm test
```

## Production Build (Optional)

### Frontend Production Build

```bash
cd UI
npm run build
```

This creates an optimized build in `UI/build/` that can be served by any static file server.

### Backend Production

For production deployment, use:

```bash
cd "Machine Learning"
uvicorn src.cle.server:app --host 0.0.0.0 --port 8000 --workers 4
```

Or with gunicorn:
```bash
pip install gunicorn
gunicorn src.cle.server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Troubleshooting Installation

### Python Version Issues

```bash
# Check Python version
python --version

# If wrong version, use specific version
python3.11 -m venv .venv
```

### pip Installation Failures

```bash
# Upgrade pip first
pip install --upgrade pip

# If SSL errors, try:
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -e .
```

### Node.js Permission Errors (Linux/macOS)

```bash
# Fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### MediaPipe Issues

If MediaPipe fails to load:

1. Ensure compatible Python version (3.10-3.11)
2. Try reinstalling: `pip uninstall mediapipe && pip install mediapipe`
3. On macOS M1/M2: `pip install mediapipe-silicon`

### Port Already in Use

```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000

# Linux/macOS
lsof -i :8000

# Kill the process or use different port
python -m src.cle.server --port 8001
```

## Next Steps

After successful installation:

1. Read the [User Guide](USER_GUIDE.md) to learn how to use the application
2. Read the [Configuration Guide](CONFIGURATION.md) to customize settings
3. Read the [ML Pipeline Guide](ML_PIPELINE.md) to understand the ML system
