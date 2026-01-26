# User Guide

Complete guide to using the Cognitive Load Estimation system.

## Getting Started

### Prerequisites

Before starting:
1. Backend server is running (see [INSTALLATION.md](INSTALLATION.md))
2. Modern browser (Chrome or Edge recommended)
3. Working webcam with good lighting
4. Stable internet connection (for local backend)

### Starting the Application

1. **Start the backend server** (Terminal 1):
   ```bash
   cd "Machine Learning"
   python -m src.cle.server --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend** (Terminal 2):
   ```bash
   cd UI
   npm start
   ```

3. **Open browser** to http://localhost:3000

## Session Setup Page

The landing page is where you begin each session.

### Camera Permission

1. Click **"Enable Camera"** button
2. Allow camera access when browser prompts
3. Green checkmark appears when permission is granted

### Starting Options

- **"Start Learning Session"** - Begin real-time cognitive load monitoring
- **"Settings"** - Configure application parameters
- **"Data Collection"** - Collect labeled training data

### Privacy Note

All video processing happens locally in your browser. No video data is transmitted to any server. Only computed features (numerical values) are sent to the backend for prediction.

## Active Session Mode

Real-time cognitive load monitoring during learning or work activities.

### Interface Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Timer  │  Load Gauge  │  Backend Status  │  Pause  │  End  │
├─────────────────────────────────────┬────────────────────────┤
│                                     │  Camera Feed           │
│           Task Panel                │  [Live Video]          │
│                                     │                        │
│      [Memory/Math Tasks]            ├────────────────────────┤
│                                     │  Live Feature Panel    │
│                                     │  • Buffer: XX%         │
│                                     │  • Blink Rate: XX      │
│                                     │  • EAR: X.XX           │
└─────────────────────────────────────┴────────────────────────┘
```

### Header Bar

| Element | Description |
|---------|-------------|
| **Timer** | Session duration (MM:SS format) |
| **Cognitive Load Gauge** | Visual indicator of current load (0-1 scale) |
| **Backend Status** | Green dot = connected, Red = error |
| **Confidence** | Prediction confidence percentage |
| **Pause** | Temporarily pause monitoring |
| **End Session** | End session and view summary |

### Cognitive Load Gauge

The circular gauge shows your current cognitive load:

| Load Level | Color | Meaning |
|------------|-------|---------|
| 0.0 - 0.3 | Green | Low cognitive load |
| 0.3 - 0.7 | Yellow | Moderate cognitive load |
| 0.7 - 1.0 | Red | High cognitive load |

### Task Panel

The task panel provides cognitive tasks to practice while monitoring:

**Memory Tasks:**
- Sequence recall (remember and repeat number sequences)
- N-back task (identify repeated items)
- Difficulty levels: Easy (3 items), Medium (5 items), Hard (7 items)

**Math Tasks:**
- Arithmetic problems (addition, subtraction, multiplication)
- Difficulty scales with problem complexity
- Timed responses

### Live Feature Panel

Shows real-time extracted features:

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| **Buffer Fill** | Progress bar showing window buffer status | 0-100% |
| **Blink Rate** | Blinks detected per minute | 10-30 |
| **EAR** | Current Eye Aspect Ratio | 0.2-0.35 |
| **Brightness** | Face region brightness | 80-200 |
| **Quality** | Face detection confidence | 0.8-1.0 |

### Intervention System

When high cognitive load is detected consistently:

1. A modal dialog appears suggesting a break
2. Options:
   - **Snooze (10 min)** - Dismiss temporarily
   - **Dismiss** - Continue without break

### Ending a Session

1. Click **"End Session"** button
2. View session summary with:
   - Total duration
   - Load history chart
   - Number of interventions triggered

## Data Collection Mode

Collect labeled training data to improve the model.

### Purpose

Data Collection mode lets you record labeled training samples with known cognitive load states. This data can be used to train or fine-tune the ML model for better accuracy.

### Collection Protocol

The recommended protocol follows this sequence:

| Phase | Duration | Label | Activity |
|-------|----------|-------|----------|
| **Baseline** | 60 seconds | LOW | Relax, no tasks |
| **Rest** | 30 seconds | LOW | Short break |
| **Easy Tasks** | 2 minutes | LOW | Simple tasks |
| **Rest** | 30 seconds | LOW | Short break |
| **Hard Tasks** | 2 minutes | HIGH | Difficult tasks |

### Starting Data Collection

1. Navigate to **Data Collection** from home page
2. Enter a **Participant ID** (e.g., "P001")
3. Optionally add **Session Notes**
4. Click **"Start Data Collection"**

### During Collection

**Recording Indicator:**
- Red pulsing dot shows recording is active
- Current phase displayed (BASELINE, TASK, REST)
- Current label shown (LOW or HIGH)

**Phase Controls:**
- **Easy/Medium/Hard** buttons - Switch to task mode with specified difficulty
- **Rest** button - Start rest period

**Sample Counter:**
- Shows total samples collected
- Samples collected every 2.5 seconds (window step)

### Task Selection

During task phases, choose between:

| Task | Description | Cognitive Load |
|------|-------------|----------------|
| **Memory Task** | Sequence recall, n-back | Adjustable by difficulty |
| **Math Task** | Arithmetic problems | Adjustable by difficulty |

**Difficulty Mapping:**
- Easy/Medium tasks → LOW label
- Hard tasks → HIGH label

### Completing Collection

1. Click **"Finish Collection"** when done
2. View collection statistics:
   - Total samples
   - Low CL samples count
   - High CL samples count
   - Total duration

### Exporting Data

Three export options available:

**CSV Export:**
```
user_id,timestamp,window_index,label,difficulty,task_type,blink_rate,...,role
P001,1234567890,0,0,easy,baseline,15.2,...,train
```
- Best for training with Python scripts
- Compatible with `src.cle.train.train`

**JSON Export:**
```json
{
  "metadata": {
    "participantId": "P001",
    "sessionNotes": "...",
    "totalSamples": 48
  },
  "samples": [...]
}
```
- Includes full metadata
- Good for archival

**Server Save:**
- Saves directly to `Machine Learning/data/collected/`
- Creates CSV + JSON metadata file
- Requires backend connection

### Training with Collected Data

After exporting:

```bash
cd "Machine Learning"

# Combine multiple collection sessions
cat data/collected/*.csv > data/processed/my_training_data.csv

# Train new model
python -m src.cle.train.train \
    --features data/processed/my_training_data.csv \
    --out models/my_custom_model
```

## Summary Page

After ending a session, the Summary page shows:

### Session Statistics

- **Duration**: Total session time
- **Average Load**: Mean cognitive load during session
- **Peak Load**: Maximum load reached
- **Interventions**: Number of break suggestions triggered

### Load History Chart

Time-series visualization showing:
- Cognitive load over time
- Intervention points marked
- Pause periods highlighted

## Settings Page

Configure application parameters:

### Display Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Show Feature Panel | Toggle live feature display | On |
| Show Overlay | Face mesh overlay on video | Off |
| Debug Mode | Additional logging | Off |

### Processing Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Window Length | Feature window duration | 10 seconds |
| Prediction Interval | Time between predictions | 2.5 seconds |
| Smoothing | EWMA smoothing factor | 0.4 |

### Threshold Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Intervention Threshold | Load level to trigger intervention | 0.7 |
| Confidence Threshold | Minimum confidence to show prediction | 0.6 |

## Tips for Best Results

### Lighting

- Well-lit room (natural or artificial light)
- Avoid backlighting (don't sit in front of window)
- Consistent lighting throughout session
- Avoid harsh shadows on face

### Camera Position

- Camera at eye level
- Face centered in frame
- Distance: 50-70cm from camera
- Entire face visible (forehead to chin)

### Physical Setup

- Sit comfortably
- Minimize head movement
- Keep eyes open naturally
- Avoid wearing glasses if possible (can affect EAR detection)

### Environment

- Quiet environment for focus
- Close unnecessary applications
- Disable notifications during session
- Chrome/Edge browsers work best

## Troubleshooting

### "Backend Error" Status

1. Check backend is running: `curl http://localhost:8000/health`
2. Restart backend server
3. Check for port conflicts

### No Face Detection

1. Ensure good lighting
2. Center face in camera view
3. Move closer to camera
4. Check browser console for MediaPipe errors

### Buffer Not Filling

1. Face detection may be failing intermittently
2. Check FPS counter (should be 25-30)
3. Close other browser tabs
4. Try different browser

### Low Confidence Predictions

1. Improve lighting conditions
2. Reduce head movement
3. Ensure face is clearly visible
4. Check quality indicator in feature panel

### High Latency

1. Close unused browser tabs
2. Check CPU usage
3. Reduce video resolution in settings
4. Ensure backend server is responsive

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume session |
| `Escape` | Dismiss intervention dialog |
| `Enter` | Submit task answer |

## FAQ

**Q: Is my video being recorded or transmitted?**
A: No. All video processing happens locally in your browser. Only numerical features (9 values per window) are sent to the local backend.

**Q: How accurate is the cognitive load prediction?**
A: Current model achieves ~75% accuracy in controlled conditions. Accuracy varies based on individual differences and environmental factors.

**Q: Can I use my own trained model?**
A: Yes. Train a new model using Data Collection mode and place it in `models/` directory. Update the server to load your model.

**Q: Why does the gauge sometimes not move?**
A: The system requires 10 seconds of data before making predictions. If face detection fails, predictions are skipped to maintain accuracy.

**Q: Can I use this without the tasks?**
A: Yes. The tasks are optional and provided for controlled data collection. During normal use, you can work on any activity while the system monitors.
