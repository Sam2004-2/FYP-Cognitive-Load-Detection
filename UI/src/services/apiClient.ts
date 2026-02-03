/**
 * API client for cognitive load prediction backend.
 * 
 * Handles communication with FastAPI backend server.
 */

import { FEATURE_CONFIG } from '../config/featureConfig';
import { HealthStatus, ModelInfo, PredictionResult, WindowFeatures } from '../types/features';

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
 * Get model information from backend.
 * 
 * @returns Model information including feature names
 */
export async function getModelInfo(): Promise<ModelInfo> {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new APIError(
        `Failed to get model info: ${response.statusText}`,
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
      `Failed to get model info: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Predict cognitive load from window features.
 * 
 * @param features - Window-level features
 * @param retries - Number of retries on failure (default: 2)
 * @returns Prediction result with CLI and confidence
 */
// Main prediction API call with retry logic and NaN sanitisation ***
export async function predictCognitiveLoad(
  features: WindowFeatures,
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
      ) as WindowFeatures;  // Type assertion tells TS the result is WindowFeatures ***

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

/**
 * Wait for backend to be ready.
 * 
 * @param timeoutMs - Maximum time to wait in milliseconds (default: 30000)
 * @param intervalMs - Check interval in milliseconds (default: 1000)
 * @returns True if backend is ready, false if timeout
 */
export async function waitForBackend(
  timeoutMs: number = 30000,
  intervalMs: number = 1000
): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    try {
      const isReady = await testConnection();
      if (isReady) {
        console.log('Backend is ready');
        return true;
      }
    } catch (error) {
      // Ignore errors while waiting
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  console.error('Backend failed to become ready within timeout');
  return false;
}

/**
 * Training sample for data collection.
 */
export interface TrainingSample {
  timestamp: number;
  window_index: number;
  label: 'low' | 'high';
  difficulty: string;
  task_type: string;
  features: WindowFeatures;
  valid_frame_ratio: number;
}

/**
 * Response from saving training data.
 */
export interface SaveTrainingDataResponse {
  success: boolean;
  filename: string;
  samples_saved: number;
  message?: string;
}

/**
 * Save collected training data to the backend.
 * 
 * @param participantId - Participant identifier
 * @param samples - Collected training samples
 * @param sessionNotes - Optional session notes
 * @returns Response with save status
 */
export async function saveTrainingData(
  participantId: string,
  samples: TrainingSample[],
  sessionNotes?: string
): Promise<SaveTrainingDataResponse> {
  try {
    // Clean up samples - ensure all feature values are valid numbers
    const cleanedSamples = samples.map(sample => ({
      ...sample,
      features: Object.fromEntries(
        Object.entries(sample.features).map(([key, value]) => [
          key,
          isNaN(value) || !isFinite(value) ? 0 : value,
        ])
      ) as WindowFeatures,
    }));

    const response = await fetch(`${API_BASE_URL}/save-training-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        participant_id: participantId,
        session_notes: sessionNotes,
        samples: cleanedSamples,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        `Failed to save training data: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      `Failed to save training data: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

// ============================================================================
// Pilot Study API Functions
// ============================================================================

import { StudySession, TaskPerformance } from '../types';

export interface StudySessionResponse {
  success: boolean;
  session_id: string;
  filename: string;
  message?: string;
}

/**
 * Save a complete pilot study session.
 */
export async function saveStudySession(session: StudySession): Promise<StudySessionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/study/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(session),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        `Failed to save study session: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(
      `Failed to save study session: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Get a study session by ID for delayed testing.
 */
export async function getStudySession(sessionId: string): Promise<StudySession> {
  try {
    const response = await fetch(`${API_BASE_URL}/study/session/${sessionId}`);

    if (!response.ok) {
      throw new APIError(
        `Session not found: ${sessionId}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(
      `Failed to get study session: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Save delayed test results.
 */
export async function saveDelayedTestResult(
  sessionId: string,
  performance: TaskPerformance
): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/study/delayed-result`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        test_date: new Date().toISOString(),
        performance,
      }),
    });

    if (!response.ok) {
      throw new APIError(
        `Failed to save delayed test: ${response.statusText}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(
      `Failed to save delayed test: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * List all pilot study sessions.
 */
export async function listStudySessions(): Promise<{
  sessions: Array<{
    filename: string;
    participant_id: string;
    session_number: number;
    condition: string;
    timestamp: string;
    has_delayed_test: boolean;
  }>;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/study/sessions`);
    if (!response.ok) {
      throw new APIError('Failed to list sessions', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(
      `Failed to list sessions: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

