import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CognitiveLoadGauge from '../components/CognitiveLoadGauge';
import WebcamFeed from '../components/WebcamFeed';
import { generateMockCognitiveLoad, resetMockData } from '../services/mockData';
import { CognitiveLoadData } from '../types';

const ActiveSession: React.FC = () => {
  const navigate = useNavigate();
  const [sessionTime, setSessionTime] = useState<number>(0);
  const [currentLoad, setCurrentLoad] = useState<number>(0.5);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [showIntervention, setShowIntervention] = useState<boolean>(false);
  const [interventionCount, setInterventionCount] = useState<number>(0);
  const [loadHistory, setLoadHistory] = useState<CognitiveLoadData[]>([]);

  useEffect(() => {
    resetMockData();
  }, []);

  useEffect(() => {
    if (!isPaused) {
      const timer = setInterval(() => {
        setSessionTime((prev) => prev + 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [isPaused]);

  useEffect(() => {
    if (!isPaused) {
      const loadInterval = setInterval(() => {
        const newLoad = generateMockCognitiveLoad();
        setCurrentLoad(newLoad);
        setLoadHistory((prev) => [...prev, { timestamp: sessionTime, load: newLoad }]);

        // Trigger intervention if load is high
        if (newLoad > 0.7 && !showIntervention) {
          setShowIntervention(true);
          setInterventionCount((prev) => prev + 1);
        }
      }, 2000);
      return () => clearInterval(loadInterval);
    }
  }, [isPaused, sessionTime, showIntervention]);

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
    // Could implement re-trigger after delay if needed
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
          {/* Learning Content Area */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-8 min-h-[500px]">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">Learning Content</h2>
              <div className="text-gray-600 space-y-4">
                <p>
                  This is your learning content area. In a full implementation, you would import or display
                  study materials, lectures, or documents here.
                </p>
                <p>
                  The system monitors your cognitive load in real-time while you study and provides
                  intelligent interventions when needed.
                </p>
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mt-6">
                  <p className="text-sm text-blue-800">
                    <strong>Tip:</strong> The cognitive load indicator in the header updates every 2 seconds
                    based on facial feature analysis.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Webcam Feed */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Camera Feed</h3>
              <WebcamFeed isActive={!isPaused} />
              <div className="mt-4 text-xs text-gray-500">
                <p>All processing happens locally. No data is transmitted.</p>
              </div>
            </div>
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

