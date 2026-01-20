/**
 * MediaPipe FaceMesh manager for landmark extraction.
 * 
 * Handles initialization and processing of video frames using MediaPipe FaceMesh.
 */

import { FaceMesh, Results } from '@mediapipe/face_mesh';
import { FEATURE_CONFIG } from '../config/featureConfig';
import { LandmarkResult } from '../types/features';

/**
 * MediaPipe FaceMesh manager class ***
 * 
 * Encapsulates MediaPipe lifecycle: initialisation, processing, cleanup ***
 * Class-based for stateful tracking of the FaceMesh instance ***
 */
export class MediaPipeManager {
  private faceMesh: FaceMesh | null = null;    // The MediaPipe instance ***
  private isInitialized: boolean = false;       // Guard against double-init ***
  private lastResults: Results | null = null;   // Cached for getLastResult() ***

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

    // locateFile tells MediaPipe where to find WASM/model files ***
    // CDN hosting avoids bundling large binary files with the app ***
    this.faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    // Configure FaceMesh - these settings match Python backend expectations ***
    this.faceMesh.setOptions({
      maxNumFaces: 1,                                              // Single user ***
      refineLandmarks: true,                                       // Enable 478 landmarks including iris ***
      minDetectionConfidence: FEATURE_CONFIG.quality.min_face_conf, // 0.5 minimum ***
      minTrackingConfidence: FEATURE_CONFIG.quality.min_face_conf,  // 0.5 minimum ***
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
