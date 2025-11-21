import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const SessionSetup: React.FC = () => {
  const navigate = useNavigate();
  const [cameraPermission, setCameraPermission] = useState<boolean>(false);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calibrationTime, setCalibrationTime] = useState<number>(30);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isCalibrating && calibrationTime > 0) {
      interval = setInterval(() => {
        setCalibrationTime((prev) => prev - 1);
      }, 1000);
    } else if (calibrationTime === 0) {
      // Calibration complete
      setIsCalibrating(false);
    }
    return () => clearInterval(interval);
  }, [isCalibrating, calibrationTime]);

  const requestCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((track) => track.stop());
      setCameraPermission(true);
    } catch (error) {
      alert('Camera access denied. Please enable camera permissions to continue.');
    }
  };

  const startCalibration = () => {
    setIsCalibrating(true);
    setCalibrationTime(30);
  };

  const startSession = () => {
    navigate('/session');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-lg shadow-xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">Cognitive Load Monitor</h1>
            <p className="text-gray-600">Real-time learning assistance through mental effort tracking</p>
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

            {/* Calibration */}
            {cameraPermission && (
              <div className="border rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-3">Baseline Calibration</h2>
                <p className="text-gray-600 mb-4">
                  Sit comfortably and relax for 30 seconds while we establish your baseline cognitive state.
                </p>
                {!isCalibrating && calibrationTime === 30 ? (
                  <button
                    onClick={startCalibration}
                    className="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-2 rounded-lg transition-colors duration-200"
                  >
                    Start Calibration
                  </button>
                ) : isCalibrating ? (
                  <div className="text-center">
                    <div className="text-4xl font-bold text-indigo-600 mb-2">{calibrationTime}s</div>
                    <p className="text-gray-600">Calibrating... Please remain still</p>
                  </div>
                ) : (
                  <div className="flex items-center text-green-600">
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Calibration complete
                  </div>
                )}
              </div>
            )}

            {/* Lighting Quality */}
            {cameraPermission && (
              <div className="border rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-3">Environment Check</h2>
                <div className="flex items-center">
                  <span className="text-gray-600 mr-3">Lighting Quality:</span>
                  <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                    Good
                  </span>
                </div>
              </div>
            )}

            {/* Start Session */}
            {cameraPermission && calibrationTime === 0 && (
              <button
                onClick={startSession}
                className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-all duration-200 shadow-lg"
              >
                Start Learning Session
              </button>
            )}
          </div>

          <div className="mt-6 text-center">
            <button
              onClick={() => navigate('/settings')}
              className="text-gray-600 hover:text-gray-800 text-sm flex items-center justify-center mx-auto"
            >
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionSetup;

