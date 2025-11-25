/**
 * MediaPipe FaceMesh manager for landmark extraction.
 * 
 * Handles initialization and processing of video frames using MediaPipe FaceMesh.
 */

import { FaceMesh, Results } from '@mediapipe/face_mesh';
import { FEATURE_CONFIG } from '../config/featureConfig';
import { LandmarkResult } from '../types/features';

/**
 * MediaPipe FaceMesh manager class.
 * 
 * Wraps MediaPipe FaceMesh for easier integration with React components.
 */
export class MediaPipeManager {
  private faceMesh: FaceMesh | null = null;
  private isInitialized: boolean = false;
  private lastResults: Results | null = null;

  /**
   * Initialize MediaPipe FaceMesh.
   * 
   * @param onResults - Callback function called when landmarks are detected
   */
  async initialize(onResults?: (results: Results) => void): Promise<void> {
    if (this.isInitialized) {
      console.warn('MediaPipe already initialized');
      return;
    }

    console.log('Initializing MediaPipe FaceMesh...');

    this.faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    // Configure FaceMesh
    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true, // Enable iris landmarks
      minDetectionConfidence: FEATURE_CONFIG.quality.min_face_conf,
      minTrackingConfidence: FEATURE_CONFIG.quality.min_face_conf,
    });

    // Set results callback
    this.faceMesh.onResults((results: Results) => {
      this.lastResults = results;
      if (onResults) {
        onResults(results);
      }
    });

    this.isInitialized = true;
    console.log('MediaPipe FaceMesh initialized successfully');
  }

  /**
   * Process a video frame and extract landmarks.
   * 
   * @param video - HTMLVideoElement to process
   * @returns Promise that resolves when processing is complete
   */
  async processFrame(video: HTMLVideoElement): Promise<void> {
    if (!this.faceMesh || !this.isInitialized) {
      throw new Error('MediaPipe not initialized. Call initialize() first.');
    }

    await this.faceMesh.send({ image: video });
  }

  /**
   * Get landmark result from last processed frame.
   * 
   * @returns LandmarkResult object
   */
  getLastResult(): LandmarkResult {
    if (!this.lastResults) {
      return {
        landmarks: null,
        detected: false,
        quality: 0.0,
      };
    }

    // Check if face was detected
    if (
      !this.lastResults.multiFaceLandmarks ||
      this.lastResults.multiFaceLandmarks.length === 0
    ) {
      return {
        landmarks: null,
        detected: false,
        quality: 0.0,
      };
    }

    // Get first face landmarks (we only process one face)
    const faceLandmarks = this.lastResults.multiFaceLandmarks[0];

    // Convert to our format
    const landmarks = faceLandmarks.map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z || 0,
    }));

    return {
      landmarks,
      detected: true,
      quality: 1.0, // MediaPipe doesn't provide per-frame confidence
    };
  }

  /**
   * Check if MediaPipe is initialized.
   */
  get initialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Close and clean up resources.
   */
  close(): void {
    if (this.faceMesh) {
      this.faceMesh.close();
      this.faceMesh = null;
    }
    this.isInitialized = false;
    this.lastResults = null;
    console.log('MediaPipe FaceMesh closed');
  }
}

/**
 * Global singleton instance (optional - can also create instances per component).
 */
let globalInstance: MediaPipeManager | null = null;

/**
 * Get or create global MediaPipe manager instance.
 */
export function getMediaPipeManager(): MediaPipeManager {
  if (!globalInstance) {
    globalInstance = new MediaPipeManager();
  }
  return globalInstance;
}

/**
 * Clean up global MediaPipe manager instance.
 */
export function cleanupMediaPipeManager(): void {
  if (globalInstance) {
    globalInstance.close();
    globalInstance = null;
  }
}

