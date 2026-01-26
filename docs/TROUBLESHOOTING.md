# Troubleshooting Guide

Common issues and solutions for the Cognitive Load Estimation system.

## Quick Diagnostics

### Health Check Commands

```bash
# Check backend health
curl http://localhost:8000/health

# Check Python environment
python --version
pip list | grep -E "mediapipe|fastapi|scikit"

# Check Node.js environment
node --version
npm --version

# Check model files
ls "Machine Learning/models/stress_classifier_rf/"
```

---

## Backend Issues

### Server Won't Start

**Symptom:** Error when running `python -m src.cle.server`

**Possible Causes:**

1. **Missing dependencies**
   ```bash
   pip install -e .
   pip install fastapi uvicorn
   ```

2. **Port already in use**
   ```bash
   # Find process using port 8000
   # Windows
   netstat -ano | findstr :8000

   # Linux/macOS
   lsof -i :8000

   # Use different port
   python -m src.cle.server --port 8001
   ```

3. **Model files missing**
   ```
   ERROR: Models directory not found: models/stress_classifier_rf
   ```

   Solution: Ensure model directory exists with:
   - `model.bin`
   - `scaler.bin`
   - `feature_spec.json`
   - `calibration.json`

4. **Python version incompatible**
   ```bash
   python --version  # Should be 3.10+
   ```

---

### Model Loading Errors

**Symptom:** `FileNotFoundError: Missing model artifact`

**Solution:**
```bash
# Verify model files exist
ls "Machine Learning/models/stress_classifier_rf/"

# Expected files:
# model.bin
# scaler.bin
# feature_spec.json
# calibration.json
```

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
cd "Machine Learning"
pip install -e .
```

---

### MediaPipe Initialization Failures

**Symptom:** `RuntimeError: Failed to load MediaPipe model`

**Solutions:**

1. **Check model file exists:**
   ```bash
   ls "Machine Learning/models/face_landmarker.task"
   ```

2. **Download if missing:**
   - Download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)

3. **Check Python version compatibility:**
   - MediaPipe works best with Python 3.10-3.11
   - May have issues with Python 3.12+

4. **macOS Apple Silicon:**
   ```bash
   pip uninstall mediapipe
   pip install mediapipe-silicon
   ```

---

### Prediction Errors

**Symptom:** `500 Internal Server Error` on `/predict`

**Check server logs:**
```bash
# View recent logs
tail -f "Machine Learning/logs/server.log"
```

**Common causes:**

1. **Feature dimension mismatch**
   - Ensure exactly 9 features are sent
   - Check feature names match model's `feature_spec.json`

2. **NaN values in features**
   - Frontend should filter out invalid windows
   - Backend replaces NaN with 0 (may affect accuracy)

3. **Scaler not fitted**
   - Retrain model if scaler.bin is corrupted

---

## Frontend Issues

### Webcam Not Detected

**Symptom:** Camera feed shows black or "Camera access denied"

**Solutions:**

1. **Check browser permissions:**
   - Click lock icon in address bar
   - Ensure camera is set to "Allow"

2. **Check system permissions:**
   - Windows: Settings → Privacy → Camera
   - macOS: System Preferences → Security & Privacy → Camera

3. **Check if another app is using camera:**
   - Close other video apps (Zoom, Teams, etc.)

4. **Try different browser:**
   - Chrome and Edge work best
   - Safari may have issues with MediaPipe

5. **Check if HTTPS or localhost:**
   - Camera only works on HTTPS or localhost
   - Not on plain HTTP with external IP

---

### MediaPipe Not Loading (Frontend)

**Symptom:** "Loading face detection..." never completes

**Solutions:**

1. **Check browser console (F12):**
   - Look for WASM loading errors
   - Check network tab for failed requests

2. **Clear browser cache:**
   ```
   Chrome: Ctrl+Shift+Delete → Cached images and files
   ```

3. **Try incognito mode:**
   - Rules out extension conflicts

4. **Check content security policy:**
   - MediaPipe needs to load WASM files
   - Check for blocked resources in console

---

### Backend Connection Failed

**Symptom:** Red "Backend Error" indicator

**Solutions:**

1. **Verify backend is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check CORS:**
   - Backend allows all origins by default
   - If changed, ensure frontend origin is allowed

3. **Check network:**
   - If using different host, ensure firewall allows connection
   - Use `--host 0.0.0.0` for external access

4. **Check port:**
   - Frontend expects backend on port 8000
   - If using different port, update `apiClient.ts`

---

### No Predictions Appearing

**Symptom:** Buffer fills but gauge doesn't update

**Solutions:**

1. **Wait for buffer to fill:**
   - Takes 10 seconds for first prediction
   - Check "Buffer: XX%" in feature panel

2. **Check face detection:**
   - Green "Face detected" should appear
   - Try better lighting

3. **Check console for errors:**
   - F12 → Console tab
   - Look for API errors or exceptions

4. **Check backend logs:**
   ```bash
   tail -f "Machine Learning/logs/server.log"
   ```

---

## Data Issues

### Feature Extraction Failures

**Symptom:** `ValueError` during feature extraction

**Possible causes:**

1. **Corrupted video file:**
   ```bash
   # Test video with ffprobe
   ffprobe input.mp4
   ```

2. **No face detected:**
   - Check video has visible faces
   - Check lighting in video

3. **Wrong FPS:**
   - Config assumes 30 FPS
   - Update `fps_fallback` if different

---

### Label Mismatch Errors

**Symptom:** `KeyError: 'aiim001_task_1'` when loading labels

**Solutions:**

1. **Check label file format:**
   ```
   aiim001_task_1, 5.0
   aiim001_task_2, 12.0
   ```
   - Comma-separated
   - Underscore between participant and task

2. **Check participant IDs match:**
   - Video filenames must match label IDs

3. **Check for whitespace:**
   - Trim whitespace from IDs
   ```python
   label_id = label_id.strip()
   ```

---

### NaN Values in Features

**Symptom:** Features CSV has NaN values

**Causes:**
1. Face not detected (invalid frames)
2. Division by zero in calculations
3. Empty windows

**Solutions:**

1. **Filter before training:**
   ```python
   df = df.dropna(subset=feature_columns)
   ```

2. **Impute missing values:**
   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy="median")
   X = imputer.fit_transform(X)
   ```

3. **Improve video quality:**
   - Better lighting
   - Face centered in frame

---

## Performance Issues

### High Latency Predictions

**Symptom:** Predictions lag behind by several seconds

**Solutions:**

1. **Increase prediction interval:**
   ```yaml
   windows:
     step_s: 5.0  # Instead of 2.5
   ```

2. **Use lighter model:**
   ```yaml
   model:
     type: "logreg"  # Fastest
   ```

3. **Check CPU usage:**
   - Close other applications
   - Check for thermal throttling

4. **Enable GPU for MediaPipe (frontend):**
   - Uses WebGL by default
   - Check GPU usage in browser task manager

---

### Frame Drops (Low FPS)

**Symptom:** FPS counter shows < 20 FPS

**Solutions:**

1. **Close browser tabs:**
   - Each tab uses memory
   - Keep only the app open

2. **Reduce video resolution:**
   ```typescript
   // In WebcamFeed.tsx
   const constraints = {
     video: {
       width: { ideal: 640 },  // Instead of 1280
       height: { ideal: 480 }, // Instead of 720
     }
   };
   ```

3. **Use Chrome/Edge:**
   - Best MediaPipe performance
   - Firefox may be slower

4. **Check hardware acceleration:**
   - Chrome: Settings → System → Use hardware acceleration

---

### Memory Leaks

**Symptom:** Browser becomes slow over time

**Solutions:**

1. **Clear window buffer:**
   - Handled automatically with ring buffer
   - If persists, refresh page periodically

2. **Check for event listener leaks:**
   - Components should clean up on unmount
   - Check useEffect cleanup functions

3. **Reduce prediction history:**
   - Limit loadHistory array size
   ```typescript
   setLoadHistory(prev => prev.slice(-100));  // Keep last 100
   ```

---

## Common Error Messages

### "Feature dimension mismatch: got X, expected 9"

**Cause:** Wrong number of features sent to API

**Solution:** Ensure all 9 features are computed and sent:
- blink_rate
- blink_count
- mean_blink_duration
- ear_std
- mean_brightness
- std_brightness
- perclos
- mean_quality
- valid_frame_ratio

---

### "Low quality window (bad ratio: X.XX), skipping prediction"

**Cause:** Too many invalid frames in window

**Solutions:**
1. Improve lighting
2. Keep face centered
3. Reduce head movement
4. Adjust `max_bad_frame_ratio` threshold

---

### "ModuleNotFoundError: No module named 'src.cle'"

**Cause:** Package not installed

**Solution:**
```bash
cd "Machine Learning"
pip install -e .
```

---

### "CORS error: No 'Access-Control-Allow-Origin' header"

**Cause:** Backend not allowing frontend origin

**Solution:** Backend includes CORS middleware by default. If issues:
```python
# In server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Getting Help

### Collecting Diagnostic Information

When reporting issues, include:

1. **System info:**
   ```bash
   python --version
   node --version
   pip list
   ```

2. **Error messages:**
   - Full stack trace
   - Browser console errors (F12)

3. **Configuration:**
   - Which config file
   - Any custom settings

4. **Steps to reproduce:**
   - What you did
   - What you expected
   - What happened

### Log Locations

| Component | Location |
|-----------|----------|
| Backend logs | `Machine Learning/logs/server.log` |
| Browser console | F12 → Console tab |
| npm output | Terminal running `npm start` |

### Useful Commands

```bash
# Reset backend
cd "Machine Learning"
pip install -e . --force-reinstall

# Reset frontend
cd UI
rm -rf node_modules
npm install

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear npm cache
npm cache clean --force
```
