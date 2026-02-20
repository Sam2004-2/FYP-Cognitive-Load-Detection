/**
 * TypeScript type definitions for facial feature extraction.
 * 
 * Mirrors the Python feature extraction pipeline types.
 */

/**
 * Per-frame features extracted from a single video frame.
 */
export interface FrameFeatures {
  /** Left eye aspect ratio */
  ear_left: number;
  /** Right eye aspect ratio */
  ear_right: number;
  /** Mean eye aspect ratio (average of left and right) */
  ear_mean: number;
  /** Mean brightness of face region (0-255) */
  brightness: number;
  /** Detection quality score (0-1) */
  quality: number;
  /** Eye center X (midpoint of outer eye corners, normalized 0-1) */
  eye_center_x: number;
  /** Eye center Y (midpoint of outer eye corners, normalized 0-1) */
  eye_center_y: number;
  /** Mouth aspect ratio (MAR): lip distance / mouth width */
  mouth_mar: number;
  /** Head roll proxy (radians) from outer eye corner line angle */
  roll: number;
  /** Head pitch proxy (radians) from forehead-chin depth angle */
  pitch: number;
  /** Head yaw proxy (radians) from nose-to-face-center offset */
  yaw: number;
  /** Whether this frame has valid features */
  valid: boolean;
}

/**
 * Window-level features aggregated over a time window.
 * These are sent to the backend for prediction.
 */
export interface WindowFeatures {
  /** Blinks per minute */
  blink_rate: number;
  /** Total number of blinks in window */
  blink_count: number;
  /** Mean blink duration in milliseconds */
  mean_blink_duration: number;
  /** Standard deviation of EAR values */
  ear_std: number;
  /** Percentage of eye closure (0-1) */
  perclos: number;
  /** Mean mouth openness (MAR) */
  mouth_open_mean: number;
  /** Standard deviation of mouth openness (MAR) */
  mouth_open_std: number;
  /** Standard deviation of head roll proxy (radians) */
  roll_std: number;
  /** Standard deviation of head pitch proxy (radians) */
  pitch_std: number;
  /** Standard deviation of head yaw proxy (radians) */
  yaw_std: number;
  /** Mean motion speed from eye center (normalized units/s) */
  motion_mean: number;
  /** Standard deviation of motion speed */
  motion_std: number;
  // --- Monitoring features (not used as model inputs) ---
  /** Mean brightness of face region */
  mean_brightness: number;
  /** Standard deviation of brightness */
  std_brightness: number;
  /** Mean detection quality */
  mean_quality: number;
  /** Ratio of valid frames in window (0-1) */
  valid_frame_ratio: number;
}

/**
 * MediaPipe FaceMesh landmark result.
 */
export interface LandmarkResult {
  /** All 468+ facial landmarks (x, y, z normalized coordinates) */
  landmarks: { x: number; y: number; z: number }[] | null;
  /** Whether a face was detected */
  detected: boolean;
  /** Detection confidence (0-1) */
  quality: number;
}

/**
 * Blink detection result (start and end frame indices).
 */
export interface Blink {
  /** Start frame index */
  start: number;
  /** End frame index */
  end: number;
}

/**
 * Cognitive load prediction result from backend.
 */
export interface PredictionResult {
  /** Cognitive Load Index (0-1) */
  cli: number;
  /** Whether prediction succeeded */
  success: boolean;
  /** Optional message */
  message?: string;
}

/**
 * Model information from backend.
 */
export interface ModelInfo {
  /** List of feature names in expected order */
  features: string[];
  /** Number of features */
  n_features: number;
  /** Calibration metadata */
  calibration: Record<string, any>;
}

/**
 * Health check response from backend.
 */
export interface HealthStatus {
  /** Service status */
  status: string;
  /** Whether model is loaded */
  model_loaded: boolean;
  /** Number of features */
  feature_count?: number;
}
