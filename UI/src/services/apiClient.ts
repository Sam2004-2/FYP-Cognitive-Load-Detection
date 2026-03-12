/**
 * API client for cognitive load prediction backend.
 * 
 * Handles communication with FastAPI backend server.
 */

import { FEATURE_CONFIG } from '../config/featureConfig';
import { HealthStatus, PredictionResult } from '../types/features';

const API_BASE_URL = FEATURE_CONFIG.api.base_url;

/**
 * Custom error class for API errors ***
 * Extends built-in Error with additional metadata for better error handling ***
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,  // HTTP status code (e.g., 404, 500) ***
    public details?: any     // Response body for debugging ***
  ) {
    super(message);
    this.name = 'APIError';  // Allows checking error type with instanceof ***
  }
}

/**
 * Check API health status.
 * 
 * @returns Health status information
 */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new APIError(
        `Health check failed: ${response.statusText}`,
        response.status
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      `Failed to connect to backend: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Predict cognitive load from window features.
 * 
 * @param features - Window-level features
 * @param retries - Number of retries on failure (default: 2)
 * @returns Prediction result with CLI
 */
// Main prediction API call with retry logic and NaN sanitisation ***
export async function predictCognitiveLoad(
  features: Record<string, number>,
  retries: number = 2  // Default 2 retries = 3 total attempts ***
): Promise<PredictionResult> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // NaN/Infinity values break JSON serialisation - replace with 0 ***
      // This can happen with insufficient valid frames in a window ***
      const cleanedFeatures = Object.fromEntries(
        Object.entries(features).map(([key, value]) => [
          key,
          isNaN(value) || !isFinite(value) ? 0 : value,
        ])
      ) as Record<string, number>;

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: cleanedFeatures,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          `Prediction failed: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      const data = await response.json();
      return data;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');

      // Don't retry on certain errors
      if (error instanceof APIError && error.status && error.status >= 400 && error.status < 500) {
        throw error;
      }

      // EXPONENTIAL BACKOFF: 1s, 2s, 4s... capped at 5s ***
      // Prevents hammering a struggling server while giving it time to recover ***
      if (attempt < retries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 5000);
        console.warn(`Prediction failed (attempt ${attempt + 1}/${retries + 1}), retrying in ${delay}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  // All retries failed
  throw new APIError(
    `Prediction failed after ${retries + 1} attempts: ${lastError?.message || 'Unknown error'}`
  );
}

/**
 * Test connection to backend API.
 * 
 * @returns True if backend is reachable and model is loaded
 */
export async function testConnection(): Promise<boolean> {
  try {
    const health = await checkHealth();
    return health.model_loaded;
  } catch (error) {
    console.error('Backend connection test failed:', error);
    return false;
  }
}

