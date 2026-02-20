import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Landing page component - handles camera permissions before session starts ***
const SessionSetup: React.FC = () => {
  const navigate = useNavigate();  // React Router hook for programmatic navigation ***
  const [cameraPermission, setCameraPermission] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  // Requests camera access using browser's getUserMedia API ***
  // Immediately stops stream after permission granted - just testing access ***
  const requestCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((track) => track.stop());  // Release camera immediately ***
      setCameraPermission(true);
      setError('');
    } catch (err) {
      console.error('Camera access denied:', err);
      setError('Camera access denied. Please enable camera permissions to continue.');
    }
  };

  const startSession = () => {
    navigate('/study/setup');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-lg shadow-xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">Cognitive Load Monitor</h1>
            <p className="text-gray-600">Study protocol workflow for your pilot crossover sessions</p>
          </div>

          <div className="space-y-6">
            {/* Camera Permission */}
            <div className="border rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Camera Access</h2>
              <p className="text-gray-600 mb-4">
                We need access to your camera to monitor facial features. All processing happens locally on your device.
              </p>
              {!cameraPermission ? (
                <button
                  onClick={requestCameraPermission}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors duration-200"
                >
                  Enable Camera
                </button>
              ) : (
                <div className="flex items-center text-green-600">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Camera access granted
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                {error}
              </div>
            )}

            {/* Start Session Button */}
            {cameraPermission && (
              <button
                onClick={startSession}
                className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-all duration-200 shadow-lg"
              >
                Start Study Setup
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionSetup;
