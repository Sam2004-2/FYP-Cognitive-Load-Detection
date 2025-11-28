/**
 * Feature extraction utilities.
 * 
 * Converted from Pythonr to typscript 
 */

import { FEATURE_CONFIG, LANDMARK_INDICES } from '../config/featureConfig';
import { Blink, FrameFeatures, LandmarkResult, WindowFeatures } from '../types/features';

/**
 * Calculate Euclidean distance between two 2D points.
 */
function euclideanDistance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

/**
 * Calculate Eye Aspect Ratio (EAR) for blink detection ***
 * 
 * EAR measures eye openness: ratio of vertical to horizontal eye dimensions ***
 * EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
 * 
 * Reference: Dewi et al. (2022) for EAR calculation
 * 
 * @param eyeCoords - Array of 6 eye landmark coordinates (specific order required) ***
 * @returns EAR value (typically 0.25-0.35 for open eye, <0.21 for closed/blink) ***
 */
export function eyeAspectRatio(eyeCoords: { x: number; y: number; z: number }[]): number {
  if (!eyeCoords || eyeCoords.length !== 6) {
    return 0.0;
  }

  // Extract 2D coordinates (use only x, y)
  const points = eyeCoords.map((p) => ({ x: p.x, y: p.y }));

  // Vertical distances
  const v1 = euclideanDistance(points[1], points[5]); // top1 to bottom1
  const v2 = euclideanDistance(points[2], points[4]); // top2 to bottom2

  // Horizontal distance
  const h = euclideanDistance(points[0], points[3]); // outer to inner

  // Avoid division by zero to avoid errors*** To make sure 
  if (h < 1e-6) {
    return 0.0;
  }

  const ear = (v1 + v2) / (2.0 * h);
  return ear;
}

/**
 * Compute mean brightness from canvas image data.
 * 
 * @param imageData - Canvas ImageData object
 * @param roi - Optional region of interest { x, y, width, height }
 * @returns Mean brightness value (0-255)
 */
export function computeBrightness(
  imageData: ImageData,
  roi?: { x: number; y: number; width: number; height: number }
): number {
  const data = imageData.data;
  const imgWidth = imageData.width;
  const imgHeight = imageData.height;

  // Default to full image if no ROI specified
  const x = roi ? Math.max(0, Math.floor(roi.x)) : 0;
  const y = roi ? Math.max(0, Math.floor(roi.y)) : 0;
  const width = roi ? Math.min(roi.width, imgWidth - x) : imgWidth;
  const height = roi ? Math.min(roi.height, imgHeight - y) : imgHeight;

  let sum = 0;
  let count = 0;

  // Iterate over ROI and compute grayscale value
  for (let row = y; row < y + height; row++) {
    for (let col = x; col < x + width; col++) {
      const idx = (row * imgWidth + col) * 4;
      // Convert RGB to grayscale using standard weights
      const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      sum += gray;
      count++;
    }
  }

  return count > 0 ? sum / count : 0.0;
}

/**
 * Extract face bounding box from landmarks.
 * 
 * @param landmarks - All facial landmarks
 * @param margin - Margin to add around face (fraction of face size)
 * @returns Bounding box { x, y, width, height }
 */
export function extractFaceROI(
  landmarks: { x: number; y: number; z: number }[],
  margin: number = 0.2
): { x: number; y: number; width: number; height: number } {
  if (!landmarks || landmarks.length === 0) {
    return { x: 0, y: 0, width: 1, height: 1 };
  }

  // Get bounding box
  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  // Add margin
  const width = xMax - xMin;
  const height = yMax - yMin;
  const marginX = width * margin;
  const marginY = height * margin;

  return {
    x: Math.max(0, xMin - marginX),
    y: Math.max(0, yMin - marginY),
    width: width + 2 * marginX,
    height: height + 2 * marginY,
  };
}

/**
 * Extract per-frame features from video frame and landmarks.
 * 
 * @param imageData - Canvas ImageData from video frame
 * @param landmarkResult - MediaPipe landmark detection result
 * @returns Per-frame features
 */
export function extractFrameFeatures(
  imageData: ImageData,
  landmarkResult: LandmarkResult
): FrameFeatures {
  // Default invalid result
  const invalidResult: FrameFeatures = {
    ear_left: 0.0,
    ear_right: 0.0,
    ear_mean: 0.0,
    brightness: 0.0,
    quality: 0.0,
    valid: false,
  };

  if (!landmarkResult.detected || !landmarkResult.landmarks) {
    return invalidResult;
  }

  const landmarks = landmarkResult.landmarks;

  // Extract eye landmarks
  const leftEyeCoords = LANDMARK_INDICES.LEFT_EYE.map((idx) => landmarks[idx]);
  const rightEyeCoords = LANDMARK_INDICES.RIGHT_EYE.map((idx) => landmarks[idx]);

  // Calculate EAR for both eyes
  const earLeft = eyeAspectRatio(leftEyeCoords);
  const earRight = eyeAspectRatio(rightEyeCoords);
  const earMean = (earLeft + earRight) / 2.0;

  // Extract face ROI
  const faceROI = extractFaceROI(landmarks);

  // Scale ROI to pixel coordinates
  const roiPixels = {
    x: faceROI.x * imageData.width,
    y: faceROI.y * imageData.height,
    width: faceROI.width * imageData.width,
    height: faceROI.height * imageData.height,
  };

  // Compute brightness
  const brightness = computeBrightness(imageData, roiPixels);

  return {
    ear_left: earLeft,
    ear_right: earRight,
    ear_mean: earMean,
    brightness,
    quality: landmarkResult.quality,
    valid: true,
  };
}

/**
 * State machine blink detector - identifies blinks in EAR time series ***
 * 
 * State transitions: OPEN -> CLOSED (EAR drops) -> OPEN (EAR rises) ***
 * Duration filtering excludes noise (too short) and extended closures (too long) ***
 * 
 * @param earSeries - Array of EAR values from frame features ***
 * @param fps - Frames per second (needed to convert ms to frames) ***
 * @param earThreshold - Below this = eye closed (default 0.21 from config) ***
 * @param minBlinkMs - Reject if shorter (120ms) - probably noise ***
 * @param maxBlinkMs - Reject if longer (400ms) - probably intentional closure ***
 * @returns Array of valid blinks with frame indices ***
 */
export function detectBlinks(
  earSeries: number[],
  fps: number,
  earThreshold: number = FEATURE_CONFIG.blink.ear_thresh,
  minBlinkMs: number = FEATURE_CONFIG.blink.min_blink_ms,
  maxBlinkMs: number = FEATURE_CONFIG.blink.max_blink_ms
): Blink[] {
  if (earSeries.length === 0) {
    return [];
  }

  // Convering to frames to handle video data
  const minBlinkFrames = Math.floor((minBlinkMs / 1000.0) * fps);
  const maxBlinkFrames = Math.floor((maxBlinkMs / 1000.0) * fps);

  const blinks: Blink[] = [];
  // Bool checks if in a blink
  let inBlink = false;
  //Frame blink starts
  let blinkStart = 0;

  for (let i = 0; i < earSeries.length; i++) {
    const ear = earSeries[i];

    if (!inBlink) {
      // Check for blink start
      if (ear < earThreshold) {
        inBlink = true;
        blinkStart = i;
      }
    } else {
      // Check for blink end
      if (ear >= earThreshold) {
        const blinkDuration = i - blinkStart;

        // Make sure blink duration is within range to exclude false positives***
        if (blinkDuration >= minBlinkFrames && blinkDuration <= maxBlinkFrames) {
          blinks.push({ start: blinkStart, end: i });
        }

        inBlink = false;
      }
    }
  }

  return blinks;
}

/**
 * Compute blink-related features from EAR series.
 * 
 * @param earSeries - Array of EAR values
 * @param fps - Frames per second
 * @returns Blink features
 */
export function computeBlinkFeatures(
  earSeries: number[],
  fps: number
): {
  blink_rate: number;
  blink_count: number;
  mean_blink_duration: number;
  ear_std: number;
} {
  if (earSeries.length === 0) {
    return {
      blink_rate: NaN,
      blink_count: NaN,
      mean_blink_duration: NaN,
      ear_std: NaN,
    };
  }

  // Detect blinks
  const blinks = detectBlinks(earSeries, fps);

  // Blink rarte per min 
  const windowDurationMin = earSeries.length / fps / 60.0;
  const blinkRate = windowDurationMin > 0 ? blinks.length / windowDurationMin : 0.0;

  // Calculate mean blink 
  let meanBlinkDuration = 0.0;
  if (blinks.length > 0) {
    const durations = blinks.map((b) => ((b.end - b.start) / fps) * 1000);
    meanBlinkDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;
  }

  // Compute EAR variability (std dev)
  // Filter out zeros (invalid frames)
  const validEar = earSeries.filter((ear) => ear > 0);
  let earStd = 0.0;
  if (validEar.length > 0) {
    const mean = validEar.reduce((sum, ear) => sum + ear, 0) / validEar.length;
    const variance =
      validEar.reduce((sum, ear) => sum + (ear - mean) ** 2, 0) / validEar.length;
    earStd = Math.sqrt(variance);
  }

  return {
    blink_rate: blinkRate,
    blink_count: blinks.length,
    mean_blink_duration: meanBlinkDuration,
    ear_std: earStd,
  };
}

/**
 * Compute PERCLOS - Percentage of Eye Closure ***
 * 
 * Classic drowsiness/fatigue indicator from driver monitoring research ***
 * High PERCLOS (>15%) suggests fatigue; correlates with cognitive overload ***
 * 
 * @param earSeries - Array of EAR values from window ***
 * @param earThreshold - EAR threshold (same as blink detection) ***
 * @returns PERCLOS value (0-1): proportion of frames with eyes closed ***
 */
export function computePERCLOS(
  earSeries: number[],
  earThreshold: number = FEATURE_CONFIG.blink.ear_thresh
): number {
  if (earSeries.length === 0) {
    return 0.0;
  }

  const closedFrames = earSeries.filter((ear) => ear < earThreshold).length;
  return closedFrames / earSeries.length;
}

/**
 * MAIN AGGREGATION FUNCTION: Computes the 9 window features for ML model ***
 * 
 * Input: ~300 frames (10 seconds at 30fps) of per-frame features ***
 * Output: Single WindowFeatures object ready for API prediction ***
 * 
 * @param frameData - Array of per-frame features from WindowBuffer ***
 * @param fps - Frames per second (needed for time-based calculations) ***
 * @returns WindowFeatures with all 9 features in expected order ***
 */
export function computeWindowFeatures(
  frameData: FrameFeatures[],
  fps: number
): WindowFeatures {
  // Extract valid frames
  const validFrames = frameData.filter((f) => f.valid);

  if (validFrames.length === 0) {
    // Return NaN features if no valid frames
    return {
      blink_rate: NaN,
      blink_count: NaN,
      mean_blink_duration: NaN,
      ear_std: NaN,
      mean_brightness: NaN,
      std_brightness: NaN,
      perclos: NaN,
      mean_quality: NaN,
      valid_frame_ratio: NaN,
    };
  }

  // Extract time series
  const earSeries = validFrames.map((f) => f.ear_mean);
  const brightnessSeries = validFrames.map((f) => f.brightness);
  const qualitySeries = validFrames.map((f) => f.quality);

  // Compute blink features
  const blinkFeatures = computeBlinkFeatures(earSeries, fps);

  // Compute brightness statistics
  const meanBrightness =
    brightnessSeries.reduce((sum, b) => sum + b, 0) / brightnessSeries.length;
  const brightnessVariance =
    brightnessSeries.reduce((sum, b) => sum + (b - meanBrightness) ** 2, 0) /
    brightnessSeries.length;
  const stdBrightness = Math.sqrt(brightnessVariance);

  // Compute PERCLOS
  const perclos = computePERCLOS(earSeries);

  // Compute quality metrics
  const meanQuality = qualitySeries.reduce((sum, q) => sum + q, 0) / qualitySeries.length;
  const validFrameRatio = validFrames.length / frameData.length;

  return {
    blink_rate: blinkFeatures.blink_rate,
    blink_count: blinkFeatures.blink_count,
    mean_blink_duration: blinkFeatures.mean_blink_duration,
    ear_std: blinkFeatures.ear_std,
    mean_brightness: meanBrightness,
    std_brightness: stdBrightness,
    perclos,
    mean_quality: meanQuality,
    valid_frame_ratio: validFrameRatio,
  };
}

