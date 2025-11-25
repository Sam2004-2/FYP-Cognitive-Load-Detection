import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import CognitiveLoadGauge from '../components/CognitiveLoadGauge';
import WebcamFeed from '../components/WebcamFeed';
import LiveFeaturePanel from '../components/LiveFeaturePanel';
import TaskPanel from '../components/tasks/TaskPanel';
import { CognitiveLoadData } from '../types';
import { WindowBuffer, validateWindowQuality } from '../services/windowBuffer';
import { computeWindowFeatures } from '../services/featureExtraction';
import { predictCognitiveLoad, testConnection } from '../services/apiClient';
import { FrameFeatures, WindowFeatures } from '../types/features';
import { FEATURE_CONFIG } from '../config/featureConfig';

const ActiveSession: React.FC = () => {
  const navigate = useNavigate();
  const [sessionTime, setSessionTime] = useState<number>(0);
  const [currentLoad, setCurrentLoad] = useState<number>(0.5);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [showIntervention, setShowIntervention] = useState<boolean>(false);
  const [interventionCount, setInterventionCount] = useState<number>(0);
  const [loadHistory, setLoadHistory] = useState<CognitiveLoadData[]>([]);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const [lastPredictionTime, setLastPredictionTime] = useState<number>(0);
  const [confidence, setConfidence] = useState<number>(0);
  
  // Live feature display state
  const [currentFrameFeatures, setCurrentFrameFeatures] = useState<FrameFeatures | null>(null);
  const [currentWindowFeatures, setCurrentWindowFeatures] = useState<WindowFeatures | null>(null);
  const [totalBlinkCount, setTotalBlinkCount] = useState<number>(0);
  const [showFeaturePanel, setShowFeaturePanel] = useState<boolean>(true);

  // Window buffer for collecting frame features
  const windowBufferRef = useRef<WindowBuffer>(
    new WindowBuffer(
      FEATURE_CONFIG.windows.length_s,
      FEATURE_CONFIG.video.fps
    )
  );

  // Refs to hold current values for use in callbacks (avoids stale closure issues)
  const currentLoadRef = useRef(currentLoad);
  const sessionTimeRef = useRef(sessionTime);
  const showInterventionRef = useRef(showIntervention);
  const lastPredictionTimeRef = useRef(lastPredictionTime);
  const isPausedRef = useRef(isPaused);

  // Keep refs in sync with state
  useEffect(() => { currentLoadRef.current = currentLoad; }, [currentLoad]);
  useEffect(() => { sessionTimeRef.current = sessionTime; }, [sessionTime]);
  useEffect(() => { showInterventionRef.current = showIntervention; }, [showIntervention]);
  useEffect(() => { lastPredictionTimeRef.current = lastPredictionTime; }, [lastPredictionTime]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);

  // Check backend connection on mount
  useEffect(() => {
    const checkBackend = async () => {
      const isConnected = await testConnection();
      setBackendStatus(isConnected ? 'connected' : 'error');
      if (!isConnected) {
        console.error('Backend is not available. Please start the backend server.');
      }
    };
    checkBackend();
  }, []);

  // Session timer
  useEffect(() => {
    if (!isPaused) {
      const timer = setInterval(() => {
        setSessionTime((prev) => prev + 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [isPaused]);

  // Make prediction from current window (uses refs to avoid stale closures)
  const makePrediction = useCallback(async () => {
    try {
      // Get window data
      const windowData = windowBufferRef.current.getWindow();

      // Validate window quality
      const [isValid, badRatio] = validateWindowQuality(windowData);

      if (!isValid) {
        console.warn(`Low quality window (bad ratio: ${badRatio.toFixed(2)}), skipping prediction`);
        // Don't update lastPredictionTime - allow immediate retry when quality improves
        return;
      }

      // Compute window features
      const windowFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);

      // Send to backend for prediction
      const result = await predictCognitiveLoad(windowFeatures);

      if (result.success) {
        // Apply EWMA smoothing (use ref for current value)
        const alpha = FEATURE_CONFIG.realtime.smoothing_alpha;
        const smoothedLoad = alpha * result.cli + (1 - alpha) * currentLoadRef.current;

        setCurrentLoad(smoothedLoad);
        setConfidence(result.confidence);
        setLoadHistory((prev) => [...prev, { timestamp: sessionTimeRef.current, load: smoothedLoad }]);

        console.log(
          `Prediction: CLI=${smoothedLoad.toFixed(3)}, ` +
          `confidence=${result.confidence.toFixed(3)}, ` +
          `valid_ratio=${windowFeatures.valid_frame_ratio.toFixed(3)}`
        );

        // Trigger intervention if load is high and confidence is sufficient
        if (
          smoothedLoad > 0.7 &&
          result.confidence > FEATURE_CONFIG.realtime.conf_threshold &&
          !showInterventionRef.current
        ) {
          setShowIntervention(true);
          setInterventionCount((prev) => prev + 1);
        }
      }

      setLastPredictionTime(Date.now() / 1000);
    } catch (error) {
      console.error('Prediction error:', error);
      setBackendStatus('error');
    }
  }, []); // No dependencies needed - uses refs for mutable values

  // Track last window feature update time
  const lastWindowUpdateRef = useRef<number>(0);
  const prevEarRef = useRef<number>(0.3);

  // Handle frame features from webcam
  const handleFrameFeatures = useCallback((features: FrameFeatures) => {
    if (isPausedRef.current) return;

    // Update current frame features for display
    setCurrentFrameFeatures(features);

    // Simple blink detection: EAR drops below threshold then rises
    if (features.valid && features.ear_mean < FEATURE_CONFIG.blink.ear_thresh && prevEarRef.current >= FEATURE_CONFIG.blink.ear_thresh) {
      setTotalBlinkCount(prev => prev + 1);
    }
    prevEarRef.current = features.ear_mean;

    // Add frame to buffer
    windowBufferRef.current.addFrame(features);

    // Update window features for display every 500ms
    const now = Date.now() / 1000;
    if (windowBufferRef.current.length > 0 && now - lastWindowUpdateRef.current >= 0.5) {
      const windowData = windowBufferRef.current.getWindow();
      if (windowData.length > 0) {
        const windowFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
        setCurrentWindowFeatures(windowFeatures);
      }
      lastWindowUpdateRef.current = now;
    }

    // Check if it's time to make a prediction
    const stepS = FEATURE_CONFIG.windows.step_s;

    if (windowBufferRef.current.isReady() && now - lastPredictionTimeRef.current >= stepS) {
      makePrediction();
    }
  }, [makePrediction]); // Only depends on makePrediction which is stable

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const handleEndSession = () => {
    navigate('/summary', {
      state: {
        duration: sessionTime,
        loadHistory,
        interventionCount
      }
    });
  };

  const dismissIntervention = () => {
    setShowIntervention(false);
  };

  const snoozeIntervention = () => {
    setShowIntervention(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="text-2xl font-bold text-gray-800">
                {formatTime(sessionTime)}
              </div>
              <CognitiveLoadGauge load={currentLoad} />
              
              {/* Backend status indicator */}
              <div className="flex items-center space-x-2">
                {backendStatus === 'checking' && (
                  <span className="text-xs text-gray-500">Connecting...</span>
                )}
                {backendStatus === 'connected' && (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-xs text-gray-600">Backend Connected</span>
                  </div>
                )}
                {backendStatus === 'error' && (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-red-500 rounded-full" />
                    <span className="text-xs text-red-600">Backend Error</span>
                  </div>
                )}
              </div>

              {/* Confidence indicator */}
              {confidence > 0 && (
                <div className="text-xs text-gray-600">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </div>
              )}
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => setIsPaused(!isPaused)}
                className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg transition-colors duration-200"
              >
                {isPaused ? 'Resume' : 'Pause'}
              </button>
              <button
                onClick={handleEndSession}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors duration-200"
              >
                End Session
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Task Area */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-6 min-h-[500px]">
              <TaskPanel 
                onTaskComplete={(correct, taskType, difficulty) => {
                  console.log(`Task completed: ${taskType} (${difficulty}) - ${correct ? 'Correct' : 'Incorrect'}`);
                }}
              />
            </div>
          </div>

          {/* Webcam Feed & Features */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Camera Feed</h3>
              <WebcamFeed 
                isActive={!isPaused} 
                onFrameFeatures={handleFrameFeatures}
                showOverlay={false}
              />
              <div className="mt-3 flex items-center justify-between">
                <span className="text-xs text-gray-500">Feature extraction active</span>
                <button
                  onClick={() => setShowFeaturePanel(!showFeaturePanel)}
                  className="text-xs text-blue-600 hover:text-blue-800 underline"
                >
                  {showFeaturePanel ? 'Hide' : 'Show'} Features
                </button>
              </div>
            </div>

            {/* Live Feature Panel */}
            {showFeaturePanel && (
              <LiveFeaturePanel
                frameFeatures={currentFrameFeatures}
                windowFeatures={currentWindowFeatures}
                blinkCount={totalBlinkCount}
                bufferFill={windowBufferRef.current?.fillRatio || 0}
              />
            )}
          </div>
        </div>
      </div>

      {/* Intervention Modal */}
      {showIntervention && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full">
            <div className="flex items-center mb-4">
              <div className="bg-yellow-100 rounded-full p-3 mr-4">
                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">High Cognitive Load Detected</h3>
                <p className="text-sm text-gray-600">Consider taking a short break</p>
              </div>
            </div>
            <p className="text-gray-700 mb-6">
              Your cognitive load has been elevated for a while. Taking a 5-10 minute break can help
              improve focus and retention.
            </p>
            <div className="flex space-x-3">
              <button
                onClick={snoozeIntervention}
                className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-lg transition-colors duration-200"
              >
                Snooze (10 min)
              </button>
              <button
                onClick={dismissIntervention}
                className="flex-1 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors duration-200"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ActiveSession;
