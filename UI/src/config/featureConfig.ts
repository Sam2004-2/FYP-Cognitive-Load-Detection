/**
 * Feature extraction configuration.
 * 
 * Ported from Machine Learning/configs/default.yaml
 */

export const FEATURE_CONFIG = {
  /** Windowing parameters */
  windows: {
    /** Window length in seconds */
    length_s: 10.0,
    /** Step size in seconds (window overlap = length - step) */
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

/** MediaPipe FaceMesh landmark indices */
export const LANDMARK_INDICES = {
  /** Left eye landmarks (6 points) for EAR calculation */
  LEFT_EYE: [33, 160, 158, 133, 153, 144],
  /** Right eye landmarks (6 points) for EAR calculation */
  RIGHT_EYE: [362, 385, 387, 263, 373, 380],
  /** Left iris landmarks (4 points) */
  LEFT_IRIS: [469, 470, 471, 472],
  /** Right iris landmarks (4 points) */
  RIGHT_IRIS: [474, 475, 476, 477],
  /** Eye corners */
  LEFT_EYE_INNER: 133,
  LEFT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 362,
  RIGHT_EYE_OUTER: 263,
} as const;

/** Feature names in the expected order (must match backend model) */
export const FEATURE_NAMES = [
  'blink_rate',
  'blink_count',
  'mean_blink_duration',
  'ear_std',
  'mean_brightness',
  'std_brightness',
  'perclos',
  'mean_quality',
  'valid_frame_ratio',
] as const;

