/**
 * Feature extraction configuration.
 * 
 * Ported from Machine Learning/configs/default.yaml
 */

// 'as const' makes all values readonly and preserves literal types ***
// This prevents accidental modification and enables better type inference ***
export const FEATURE_CONFIG = {
  /** Windowing parameters - must match Python backend config */
  windows: {
    /** Window length in seconds - time span for feature aggregation */
    length_s: 10.0,
    /** Step size (2.5s) means 75% overlap between consecutive windows ***
     *  Higher overlap = smoother predictions but more compute */
    step_s: 2.5,
  },

  /** Quality control thresholds */
  quality: {
    /** Minimum face detection confidence */
    min_face_conf: 0.5,
    /** Maximum ratio of bad frames per window */
    max_bad_frame_ratio: 0.05,
  },

  /** Blink detection parameters */
  blink: {
    /** Eye Aspect Ratio threshold for blink detection */
    ear_thresh: 0.21,
    /** Minimum blink duration in milliseconds */
    min_blink_ms: 120,
    /** Maximum blink duration in milliseconds */
    max_blink_ms: 400,
  },

  /** Video processing parameters */
  video: {
    /** Target frames per second */
    fps: 30.0,
    /** Video width for MediaPipe */
    width: 640,
    /** Video height for MediaPipe */
    height: 480,
  },

  /** Real-time processing parameters */
  realtime: {
    /** EWMA smoothing parameter (0=no smoothing, 1=no memory) */
    smoothing_alpha: 0.4,
    /** Minimum confidence to display result */
    conf_threshold: 0.6,
  },

  /** Backend API configuration */
  api: {
    /** Base URL for backend API */
    base_url: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  },
} as const;

/** MediaPipe FaceMesh landmark indices - ported from Python landmarks.py ***
 *  MediaPipe provides 468 facial landmarks; these are the key eye-related ones ***
 */
export const LANDMARK_INDICES = {
  /** 6 landmarks per eye arranged for EAR calculation: corners + top/bottom pairs ***
   *  Order: [outer_corner, top1, top2, inner_corner, bottom1, bottom2] ***
   */
  LEFT_EYE: [33, 160, 158, 133, 153, 144],
  RIGHT_EYE: [362, 385, 387, 263, 373, 380],
  /** Iris landmarks used with refineLandmarks: true option in MediaPipe ***  */
  LEFT_IRIS: [469, 470, 471, 472],
  RIGHT_IRIS: [474, 475, 476, 477],
  /** Eye corners - useful for gaze direction estimation ***  */
  LEFT_EYE_INNER: 133,
  LEFT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 362,
  RIGHT_EYE_OUTER: 263,
  /** Mouth landmarks for mouth aspect ratio (MAR) ***
   *  Used as a proxy for talking/effort and mouth openness variability */
  MOUTH_LEFT: 61,
  MOUTH_RIGHT: 291,
  MOUTH_UPPER: 13,
  MOUTH_LOWER: 14,
} as const;

/** Feature names in the EXACT order expected by the ML model ***
 *  CRITICAL: This order must match feature_spec.json in the backend ***
 *  Changing order will cause model to produce incorrect predictions ***
 */
export const FEATURE_NAMES = [
  'blink_rate',          // Blinks per minute ***
  'blink_count',         // Total blinks in window ***
  'mean_blink_duration', // Average blink length in ms ***
  'ear_std',             // Eye Aspect Ratio variability ***
  'mean_brightness',     // Average face region brightness ***
  'std_brightness',      // Brightness variability ***
  'perclos',             // Percentage of Eye Closure (fatigue indicator) ***
  'mean_quality',        // Average detection confidence ***
  'valid_frame_ratio',   // Fraction of usable frames ***
  'mouth_open_mean',     // Mean mouth openness (MAR) ***
  'mouth_open_std',      // Mouth openness variability (MAR std) ***
  'roll_std',            // Head roll variability (radians std) ***
  'motion_mean',         // Mean eye-center motion speed (norm units/s) ***
  'motion_std',          // Motion speed variability ***
] as const;
