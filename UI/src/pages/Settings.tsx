import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Settings as SettingsType } from '../types';  // Aliased to avoid name collision ***

// Settings page for customising study parameters ***
const Settings: React.FC = () => {
  const navigate = useNavigate();
  // useState<SettingsType> provides type safety for the settings object ***
  const [settings, setSettings] = useState<SettingsType>({
    interventionFrequency: 'medium',  // Default intervention sensitivity ***
    breakInterval: 25                 // Standard Pomodoro interval ***
  });
  const [showSaved, setShowSaved] = useState<boolean>(false);

  const handleSave = () => {
    console.log('Settings saved:', settings);
    setShowSaved(true);
    setTimeout(() => setShowSaved(false), 3000);
  };

  const handleReset = () => {
    setSettings({
      interventionFrequency: 'medium',
      breakInterval: 25
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">Settings</h1>
            <p className="text-gray-600">Customize your learning experience</p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="text-gray-600 hover:text-gray-800 flex items-center"
          >
            <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Close
          </button>
        </div>

        {/* Success Message */}
        {showSaved && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg mb-6">
            Settings saved successfully!
          </div>
        )}

        {/* Study Preferences */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Study Preferences</h2>
          
          <div className="space-y-6">
            {/* Intervention Frequency */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Intervention Frequency
              </label>
              <p className="text-sm text-gray-600 mb-3">
                How often should the system suggest breaks or adjustments?
              </p>
              <select
                value={settings.interventionFrequency}
                onChange={(e) => setSettings({ ...settings, interventionFrequency: e.target.value as SettingsType['interventionFrequency'] })}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="off">Off - No interventions</option>
                <option value="low">Low - Only critical alerts</option>
                <option value="medium">Medium - Balanced recommendations</option>
                <option value="high">High - Proactive assistance</option>
              </select>
            </div>

            {/* Break Interval */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Break Reminder Interval
              </label>
              <p className="text-sm text-gray-600 mb-3">
                Reminder to take breaks at regular intervals
              </p>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="15"
                    checked={settings.breakInterval === 15}
                    onChange={(e) => setSettings({ ...settings, breakInterval: parseInt(e.target.value) as SettingsType['breakInterval'] })}
                    className="mr-3 w-4 h-4 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-gray-700">Every 15 minutes (Pomodoro short)</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="25"
                    checked={settings.breakInterval === 25}
                    onChange={(e) => setSettings({ ...settings, breakInterval: parseInt(e.target.value) as SettingsType['breakInterval'] })}
                    className="mr-3 w-4 h-4 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-gray-700">Every 25 minutes (Pomodoro standard)</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="45"
                    checked={settings.breakInterval === 45}
                    onChange={(e) => setSettings({ ...settings, breakInterval: parseInt(e.target.value) as SettingsType['breakInterval'] })}
                    className="mr-3 w-4 h-4 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-gray-700">Every 45 minutes (Extended focus)</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Privacy & Data */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Privacy & Data</h2>
          
          <div className="space-y-4 text-gray-700">
            <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
              <h3 className="font-semibold text-blue-900 mb-2">Camera Usage</h3>
              <p className="text-sm text-blue-800">
                Your webcam is used exclusively for real-time facial feature analysis. All processing
                happens locally on your device. No video or images are stored or transmitted.
              </p>
            </div>
            
            <div className="bg-green-50 border-l-4 border-green-500 p-4">
              <h3 className="font-semibold text-green-900 mb-2">Data Storage</h3>
              <p className="text-sm text-green-800">
                Session data (cognitive load metrics, NASA-TLX scores) is stored only in your browser's
                memory during the session. No data is sent to external servers.
              </p>
            </div>
            
            <div className="bg-purple-50 border-l-4 border-purple-500 p-4">
              <h3 className="font-semibold text-purple-900 mb-2">Research Purpose</h3>
              <p className="text-sm text-purple-800">
                This is a research prototype for cognitive load estimation in learning environments.
                The system aims to provide personalized learning assistance based on mental effort tracking.
              </p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between">
          <button
            onClick={handleReset}
            className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-6 py-2 rounded-lg transition-colors duration-200"
          >
            Reset to Defaults
          </button>
          <button
            onClick={handleSave}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors duration-200"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;

