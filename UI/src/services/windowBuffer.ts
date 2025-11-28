/**
 * Window buffer for temporal feature aggregation.
 * 
 * Ported from Machine Learning/src/cle/extract/windowing.py
 */

import { FEATURE_CONFIG } from '../config/featureConfig';
import { FrameFeatures } from '../types/features';

/**
 * Quality gate for predictions - rejects windows with too many invalid frames ***
 * 
 * Invalid frames occur when face detection fails (lighting, occlusion, etc.) ***
 * Default threshold (5%) prevents unreliable predictions from poor data ***
 * 
 * @param frameFeatures - Array of per-frame features from buffer ***
 * @param maxBadRatio - Maximum allowed invalid frame ratio (default 0.05) ***
 * @returns Tuple [isValid, badRatio] - TypeScript tuple type ***
 */
export function validateWindowQuality(
  frameFeatures: FrameFeatures[],
  maxBadRatio: number = FEATURE_CONFIG.quality.max_bad_frame_ratio
): [boolean, number] {
  if (frameFeatures.length === 0) {
    return [false, 1.0];
  }

  const badCount = frameFeatures.filter((f) => !f.valid).length;
  const badRatio = badCount / frameFeatures.length;

  const isValid = badRatio <= maxBadRatio;

  return [isValid, badRatio];
}

/**
 * RING BUFFER DATA STRUCTURE for sliding window feature aggregation ***
 * 
 * Stores most recent N frames (e.g., 300 for 10s at 30fps) ***
 * When full, new frames push out oldest (FIFO behaviour) ***
 * Enables constant-memory real-time processing ***
 */
export class WindowBuffer {
  private windowLengthS: number;  // Window duration in seconds ***
  private fps: number;            // Frames per second ***
  private maxFrames: number;      // Capacity: windowLengthS * fps ***
  private buffer: FrameFeatures[]; // The actual frame storage ***
  private frameCount: number;      // Total frames ever added (for timing) ***

  /**
   * Initialize window buffer.
   * 
   * @param windowLengthS - Window length in seconds
   * @param fps - Frames per second
   */
  constructor(
    windowLengthS: number = FEATURE_CONFIG.windows.length_s,
    fps: number = FEATURE_CONFIG.video.fps
  ) {
    this.windowLengthS = windowLengthS;
    this.fps = fps;
    this.maxFrames = Math.floor(windowLengthS * fps);
    this.buffer = [];
    this.frameCount = 0;

    console.log(
      `Initialized WindowBuffer (length=${windowLengthS}s, fps=${fps}, maxFrames=${this.maxFrames})`
    );
  }

  /**
   * Add frame to buffer - O(1) average, O(n) when shift occurs ***
   * 
   * @param frameFeatures - Per-frame feature object
   */
  addFrame(frameFeatures: FrameFeatures): void {
    this.buffer.push(frameFeatures);

    // Ring buffer behaviour: remove oldest when at capacity ***
    if (this.buffer.length > this.maxFrames) {
      this.buffer.shift();  // Note: shift() is O(n) but acceptable at ~300 frames ***
    }

    this.frameCount++;  // Running total for window timing calculations ***
  }

  /**
   * Check if buffer has enough frames for a window.
   * 
   * @returns True if buffer is full
   */
  isReady(): boolean {
    return this.buffer.length >= this.maxFrames;
  }

  /**
   * Get current window of frame features ***
   * 
   * Returns COPY to prevent external mutation of internal buffer ***
   * Spread operator [...] creates shallow copy (sufficient for immutable frames) ***
   */
  getWindow(): FrameFeatures[] {
    return [...this.buffer];
  }

  /**
   * Get start and end times for current window.
   * 
   * @returns Tuple of [startTimeS, endTimeS]
   */
  getWindowTimes(): [number, number] {
    if (this.buffer.length === 0) {
      return [0.0, 0.0];
    }

    // Approximate times based on frame count
    const endTimeS = this.frameCount / this.fps;
    const startTimeS = Math.max(0.0, endTimeS - this.windowLengthS);

    return [startTimeS, endTimeS];
  }

  /**
   * Reset buffer.
   */
  reset(): void {
    this.buffer = [];
    this.frameCount = 0;
  }

  /**
   * Get current buffer size.
   */
  get length(): number {
    return this.buffer.length;
  }

  /**
   * Get maximum buffer capacity.
   */
  get capacity(): number {
    return this.maxFrames;
  }

  /**
   * Get fill percentage (0-1).
   */
  get fillRatio(): number {
    return this.buffer.length / this.maxFrames;
  }
}

