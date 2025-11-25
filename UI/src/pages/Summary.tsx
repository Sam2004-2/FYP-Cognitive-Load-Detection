import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import LoadChart from '../components/LoadChart';
import NasaTLXForm from '../components/NasaTLXForm';
import { SessionData, NASATLXScores } from '../types';

const Summary: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get session data from navigation state
  const sessionData: SessionData = location.state || {
    duration: 0,
    loadHistory: [],
    interventionCount: 0
  };

  const hasData = sessionData.loadHistory.length > 0;

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const calculateFocusTime = (): string => {
    const focusedPoints = sessionData.loadHistory.filter(point => point.load >= 0.4 && point.load <= 0.7);
    const focusedMinutes = Math.floor((focusedPoints.length * 2) / 60); // 2 seconds per data point
    return `${focusedMinutes}m`;
  };

  const handleSubmitTLX = (scores: NASATLXScores) => {
    console.log('NASA-TLX Scores:', scores);
    // In a real app, would save this data
    alert('Thank you for your feedback!');
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Session Summary</h1>
          <p className="text-gray-600">Review your learning session and provide feedback</p>
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Session Duration</p>
                <p className="text-3xl font-bold text-gray-800">{formatDuration(sessionData.duration)}</p>
              </div>
              <div className="bg-blue-100 rounded-full p-3">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Interventions</p>
                <p className="text-3xl font-bold text-gray-800">{sessionData.interventionCount}</p>
              </div>
              <div className="bg-yellow-100 rounded-full p-3">
                <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Estimated Focus Time</p>
                <p className="text-3xl font-bold text-gray-800">{calculateFocusTime()}</p>
              </div>
              <div className="bg-green-100 rounded-full p-3">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Cognitive Load Chart */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Cognitive Load Over Time</h2>
          {hasData ? (
            <>
              <LoadChart data={sessionData.loadHistory} />
              <div className="mt-4 flex justify-center space-x-6 text-sm">
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-load-low rounded mr-2"></div>
                  <span className="text-gray-600">Low (&lt;40%)</span>
                </div>
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-load-medium rounded mr-2"></div>
                  <span className="text-gray-600">Medium (40-70%)</span>
                </div>
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-load-high rounded mr-2"></div>
                  <span className="text-gray-600">High (&gt;70%)</span>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p>No session data available.</p>
              <p className="text-sm mt-2">Complete a session to see your cognitive load history.</p>
            </div>
          )}
        </div>

        {/* NASA-TLX Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <NasaTLXForm onSubmit={handleSubmitTLX} />
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            onClick={() => navigate('/')}
            className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-6 py-2 rounded-lg transition-colors duration-200"
          >
            Back to Home
          </button>
          <button
            onClick={() => navigate('/history')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors duration-200"
          >
            View History
          </button>
        </div>
      </div>
    </div>
  );
};

export default Summary;

