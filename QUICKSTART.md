# Quick Start Guide

Get the cognitive load estimation system running in 5 minutes.

## Step 1: Start Backend (Terminal 1)

```bash
cd "Machine Learning"

# Install dependencies (first time only)
pip install -r requirements.txt

# Start server
python3 -m src.cle.server --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO: Starting Cognitive Load Estimation API
INFO: Loaded model artifacts with 9 features
INFO: API ready to accept requests
```

## Step 2: Start Frontend (Terminal 2)

```bash
cd UI

# Install dependencies (first time only)
npm install

# Start React app
npm start
```

Browser will open at `http://localhost:3000`

## Step 3: Test the System

1. Click **"Start Session"**
2. Allow camera access when prompted
3. Wait for green "Face detected" indicator
4. Wait ~10 seconds for buffer to fill
5. Watch cognitive load predictions update every 2.5 seconds

## Verification Checklist

✅ Backend shows "API ready to accept requests"
✅ Frontend shows "Backend Connected" (green dot)
✅ Camera feed shows your face
✅ FPS counter shows ~25-30 FPS
✅ Buffer fills to 100%
✅ Cognitive load gauge updates regularly

## Troubleshooting

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify model files exist
ls "Machine Learning/models/"
```

**Frontend won't start:**
```bash
# Check Node version
node --version  # Should be 16+

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**No predictions:**
- Ensure backend is running (check Terminal 1)
- Check browser console for errors (F12)
- Verify "Backend Connected" shows green dot

## What's Happening Behind the Scenes

1. **MediaPipe** extracts 468 facial landmarks from your webcam
2. **Feature extraction** computes Eye Aspect Ratio and brightness
3. **Window buffer** collects 10 seconds of frames (~300 frames)
4. Every **2.5 seconds**, window features are computed
5. Features are **sent to backend** for prediction
6. **ML model** returns Cognitive Load Index (0-1)
7. **Smoothed prediction** displayed in real-time

## Data Collection Mode

Use the Data Collection mode to collect labeled training data for improving your model:

1. Click **"Data Collection"** link on the home page
2. Enter a **Participant ID** (e.g., P001)
3. Click **"Start Data Collection"**
4. Follow the protocol:
   - **Baseline** (1 min): Relax, no tasks → LOW label
   - **Rest** (30s): Short break → LOW label
   - **Easy/Medium tasks**: Engage with tasks → LOW label
   - **Hard tasks**: High difficulty → HIGH label
5. Click **"Finish Collection"** when done
6. Export data:
   - **CSV**: Download for training with `python -m src.cle.train.train`
   - **JSON**: Full data with metadata
   - **Server**: Save directly to `data/collected/` (if backend running)

### Training with Collected Data

```bash
cd "Machine Learning"

# Combine collected data (if multiple sessions)
cat data/collected/*.csv > data/processed/combined_training.csv

# Train new model
python -m src.cle.train.train --features data/processed/combined_training.csv
```

## Next Steps

- Adjust settings in `UI/src/config/featureConfig.ts`
- Monitor backend logs: `Machine Learning/logs/server.log`
- Enable debug overlay: `showOverlay={true}` in WebcamFeed

## Common Issues

**"Backend Error" in UI:**
```bash
# Test backend directly
curl http://localhost:8000/health
```

**Camera not working:**
- Check browser permissions
- Try different browser (Chrome recommended)
- Ensure no other app is using camera

**Low FPS (<20):**
- Close other browser tabs
- Reduce video resolution in config
- Check CPU usage

## Performance Tips

- **Best FPS**: Chrome/Edge browsers
- **Best lighting**: Well-lit room, no glare
- **Best distance**: 50-70cm from camera
- **Best position**: Face camera directly

## Done!

You now have a working real-time cognitive load estimation system! 🎉

The system continuously monitors your cognitive state and can trigger interventions when load is high.

