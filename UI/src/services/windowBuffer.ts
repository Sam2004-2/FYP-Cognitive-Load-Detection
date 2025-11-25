/**
 * Window buffer for temporal feature aggregation.
 * 
 * Ported from Machine Learning/src/cle/extract/windowing.py
 */

import { FEATURE_CONFIG } from '../config/featureConfig';
import { FrameFeatures } from '../types/features';

/**
 * Validate window quality based on fraction of bad frames.
 * 
 * @param frameFeatures - Array of per-frame features
 * @param maxBadRatio - Maximum allowed ratio of bad frames
 * @returns Tuple of [isValid, badRatio]
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
 * Ring buffer for real-time windowing.
 * 
 * Maintains a fixed-size buffer of recent frame features for real-time processing.
 */
export class WindowBuffer {
  private windowLengthS: number;
  private fps: number;
  private maxFrames: number;
  private buffer: FrameFeatures[];
  private frameCount: number;

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
   * Add frame features to buffer.
   * 
   * @param frameFeatures - Per-frame feature object
   */
  addFrame(frameFeatures: FrameFeatures): void {
    this.buffer.push(frameFeatures);

    // Remove oldest frame if buffer is full
    if (this.buffer.length > this.maxFrames) {
      this.buffer.shift();
    }

    this.frameCount++;
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
   * Get current window of frame features.
   * 
   * @returns Array of frame features for current window
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

