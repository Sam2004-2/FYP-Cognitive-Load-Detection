import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import WebcamFeed from '../components/WebcamFeed';
import LiveFeaturePanel from '../components/LiveFeaturePanel';
import MemoryTask from '../components/tasks/MemoryTask';
import MathTask from '../components/tasks/MathTask';
import { WindowBuffer, validateWindowQuality } from '../services/windowBuffer';
import { computeWindowFeatures } from '../services/featureExtraction';
import { FrameFeatures, WindowFeatures } from '../types/features';
import { FEATURE_CONFIG } from '../config/featureConfig';
import { saveTrainingData, testConnection } from '../services/apiClient';

// Union type with null allows "no task selected" state ***
type TaskType = 'memory' | 'math' | null;

// Structure for each collected training sample - matches backend expected format ***
interface CollectedSample {
  timestamp: number;                          // Unix timestamp for ordering ***
  windowIndex: number;                        // Sample sequence number ***
  label: 'low' | 'high';                      // Ground truth cognitive load label ***
  difficulty: 'easy' | 'medium' | 'hard';     // Task difficulty that induced the load ***
  taskType: string;                           // What task generated this sample ***
  features: WindowFeatures;                   // 9 computed window features ***
  validFrameRatio: number;                    // Data quality metric ***
}

// State machine phases for data collection flow ***
type CollectionPhase = 'setup' | 'baseline' | 'task' | 'rest' | 'complete';

const DataCollection: React.FC = () => {
  const navigate = useNavigate();
  
  // Collection state
  const [phase, setPhase] = useState<CollectionPhase>('setup');
  const [currentLabel, setCurrentLabel] = useState<'low' | 'high'>('low');
  const [currentDifficulty, setCurrentDifficulty] = useState<'easy' | 'medium' | 'hard'>('easy');
  const [currentTaskType, setCurrentTaskType] = useState<string>('baseline');
  const [collectedSamples, setCollectedSamples] = useState<CollectedSample[]>([]);
  const [phaseTime, setPhaseTime] = useState<number>(0);
  const [totalTime, setTotalTime] = useState<number>(0);
  const [windowIndex, setWindowIndex] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [participantId, setParticipantId] = useState<string>('');
  const [sessionNotes, setSessionNotes] = useState<string>('');

  // Task state
  const [selectedTask, setSelectedTask] = useState<TaskType>(null);
  const [taskStats, setTaskStats] = useState({
    totalAttempts: 0,
    correctAnswers: 0,
    streak: 0,
  });
  
  // Live feature state
  const [currentFrameFeatures, setCurrentFrameFeatures] = useState<FrameFeatures | null>(null);
  const [currentWindowFeatures, setCurrentWindowFeatures] = useState<WindowFeatures | null>(null);
  const [totalBlinkCount, setTotalBlinkCount] = useState<number>(0);
  
  // Backend state
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveMessage, setSaveMessage] = useState<string>('');
  
  // Refs
  const windowBufferRef = useRef<WindowBuffer>(
    new WindowBuffer(FEATURE_CONFIG.windows.length_s, FEATURE_CONFIG.video.fps)
  );
  const lastWindowUpdateRef = useRef<number>(0);
  const lastSampleTimeRef = useRef<number>(0);
  const prevEarRef = useRef<number>(0.3);

  // Phase durations (in seconds)
  const phaseDurations = {
    baseline: 60,    // 1 minute baseline
    task: 120,       // 2 minutes per task block
    rest: 30,        // 30 second rest between blocks
  };

  // Check backend connection
  useEffect(() => {
    const checkBackend = async () => {
      const isConnected = await testConnection();
      setBackendStatus(isConnected ? 'connected' : 'error');
    };
    checkBackend();
  }, []);

  // Timer effect
  useEffect(() => {
    if (!isRecording || phase === 'setup' || phase === 'complete') return;

    const timer = setInterval(() => {
      setPhaseTime(prev => prev + 1);
      setTotalTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [isRecording, phase]);

  // Auto-advance phases
  useEffect(() => {
    if (phase === 'baseline' && phaseTime >= phaseDurations.baseline) {
      setPhase('rest');
      setPhaseTime(0);
    } else if (phase === 'rest' && phaseTime >= phaseDurations.rest) {
      setPhase('task');
      setPhaseTime(0);
    } else if (phase === 'task' && phaseTime >= phaseDurations.task) {
      // Could auto-advance to next block or complete
    }
  }, [phase, phaseTime]);

  // REF SYNC PATTERN: Same pattern as ActiveSession ***
  // Refs provide stable references to current state in async callbacks ***
  const currentLabelRef = useRef(currentLabel);
  const currentDifficultyRef = useRef(currentDifficulty);
  const currentTaskTypeRef = useRef(currentTaskType);
  const windowIndexRef = useRef(0);  // Counter for sample numbering ***

  // Sync effects - run after each state change to update refs ***
  useEffect(() => { currentLabelRef.current = currentLabel; }, [currentLabel]);
  useEffect(() => { currentDifficultyRef.current = currentDifficulty; }, [currentDifficulty]);
  useEffect(() => { currentTaskTypeRef.current = currentTaskType; }, [currentTaskType]);

  // Collects one labeled training sample from current window buffer ***
  // Returns boolean indicating success - allows caller to handle quality failures ***
  const collectSample = useCallback((): boolean => {
    const windowData = windowBufferRef.current.getWindow();
    const [isValid, badRatio] = validateWindowQuality(windowData);

    if (!isValid) {
      console.warn(`Skipping low quality window (bad ratio: ${badRatio.toFixed(2)})`);
      return false; // Signal that collection failed
    }

    const features = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
    const currentIndex = windowIndexRef.current;
    windowIndexRef.current += 1;
    
    const sample: CollectedSample = {
      timestamp: Date.now(),
      windowIndex: currentIndex,
      label: currentLabelRef.current,
      difficulty: currentDifficultyRef.current,
      taskType: currentTaskTypeRef.current,
      features,
      validFrameRatio: features.valid_frame_ratio,
    };

    setCollectedSamples(prev => [...prev, sample]);
    setWindowIndex(currentIndex + 1);

    console.log(`Collected sample #${currentIndex + 1}: ${currentLabelRef.current} (${currentDifficultyRef.current})`);
    return true; // Signal success
  }, []);

  // Handle frame features from webcam
  const handleFrameFeatures = useCallback((features: FrameFeatures) => {
    if (!isRecording) return;

    setCurrentFrameFeatures(features);

    // Blink detection
    if (features.valid && features.ear_mean < FEATURE_CONFIG.blink.ear_thresh && 
        prevEarRef.current >= FEATURE_CONFIG.blink.ear_thresh) {
      setTotalBlinkCount(prev => prev + 1);
    }
    prevEarRef.current = features.ear_mean;

    // Add to buffer
    windowBufferRef.current.addFrame(features);

    // Update window features display
    const now = Date.now() / 1000;
    if (windowBufferRef.current.length > 0 && now - lastWindowUpdateRef.current >= 0.5) {
      const windowData = windowBufferRef.current.getWindow();
      if (windowData.length > 0) {
        const windowFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
        setCurrentWindowFeatures(windowFeatures);
      }
      lastWindowUpdateRef.current = now;
    }

    // Collect sample every step_s seconds when buffer is ready
    const stepS = FEATURE_CONFIG.windows.step_s;
    if (windowBufferRef.current.isReady() && now - lastSampleTimeRef.current >= stepS) {
      const success = collectSample();
      if (success) {
        // Only update timer on successful collection - allows immediate retry if quality improves
        lastSampleTimeRef.current = now;
      }
    }
  }, [isRecording, collectSample]);

  const startCollection = () => {
    if (!participantId.trim()) {
      alert('Please enter a participant ID');
      return;
    }
    setPhase('baseline');
    setIsRecording(true);
    setCurrentLabel('low');
    setCurrentTaskType('baseline');
    setPhaseTime(0);
    lastSampleTimeRef.current = Date.now() / 1000;
  };

  const setTaskBlock = (difficulty: 'easy' | 'medium' | 'hard') => {
    setCurrentDifficulty(difficulty);
    setCurrentLabel(difficulty === 'hard' ? 'high' : 'low');
    setCurrentTaskType(`${difficulty}_task`);
    setPhase('task');
    setPhaseTime(0);
  };

  const handleTaskComplete = (correct: boolean) => {
    setTaskStats(prev => ({
      totalAttempts: prev.totalAttempts + 1,
      correctAnswers: prev.correctAnswers + (correct ? 1 : 0),
      streak: correct ? prev.streak + 1 : 0,
    }));
  };

  const startRestPeriod = () => {
    setPhase('rest');
    setCurrentLabel('low');
    setCurrentTaskType('rest');
    setPhaseTime(0);
  };

  const finishCollection = () => {
    setIsRecording(false);
    setPhase('complete');
  };

  const exportToCSV = () => {
    if (collectedSamples.length === 0) {
      alert('No samples collected');
      return;
    }

    // Build CSV header
    const featureNames = [
      'blink_rate', 'blink_count', 'mean_blink_duration', 'ear_std',
      'perclos', 'mouth_open_mean', 'mouth_open_std', 'roll_std',
      'pitch_std', 'yaw_std', 'motion_mean', 'motion_std'
    ];
    
    const headers = [
      'user_id', 'timestamp', 'window_index', 'label', 'difficulty', 'task_type',
      ...featureNames, 'role'
    ];

    // Build CSV rows
    const rows = collectedSamples.map(sample => {
      const featureValues = featureNames.map(name => 
        sample.features[name as keyof WindowFeatures] ?? ''
      );
      return [
        participantId,
        sample.timestamp,
        sample.windowIndex,
        sample.label === 'high' ? 1 : 0,
        sample.difficulty,
        sample.taskType,
        ...featureValues,
        'train'  // All collected samples are for training
      ].join(',');
    });

    const csv = [headers.join(','), ...rows].join('\n');

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_data_${participantId}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    if (collectedSamples.length === 0) {
      alert('No samples collected');
      return;
    }

    const exportData = {
      metadata: {
        participantId,
        sessionNotes,
        collectionDate: new Date().toISOString(),
        totalDuration: totalTime,
        totalSamples: collectedSamples.length,
        samplesPerLabel: {
          low: collectedSamples.filter(s => s.label === 'low').length,
          high: collectedSamples.filter(s => s.label === 'high').length,
        }
      },
      samples: collectedSamples,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_data_${participantId}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const saveToServer = async () => {
    if (collectedSamples.length === 0) {
      alert('No samples collected');
      return;
    }

    setIsSaving(true);
    setSaveMessage('');

    try {
      // Convert to API format
      const samples = collectedSamples.map(s => ({
        timestamp: s.timestamp,
        window_index: s.windowIndex,
        label: s.label,
        difficulty: s.difficulty,
        task_type: s.taskType,
        features: s.features,
        valid_frame_ratio: s.validFrameRatio,
      }));

      const result = await saveTrainingData(participantId, samples, sessionNotes);
      
      if (result.success) {
        setSaveMessage(`✓ Saved ${result.samples_saved} samples to ${result.filename}`);
      } else {
        setSaveMessage('✗ Failed to save data');
      }
    } catch (error) {
      console.error('Save error:', error);
      setSaveMessage(`✗ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsSaving(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const getLabelColour = (label: 'low' | 'high') => {
    return label === 'high' ? 'bg-red-500' : 'bg-green-500';
  };

  const getPhaseColour = (p: CollectionPhase) => {
    switch (p) {
      case 'baseline': return 'bg-blue-500';
      case 'task': return 'bg-purple-500';
      case 'rest': return 'bg-gray-500';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-xl font-bold text-orange-400">Data Collection Mode</h1>
              {isRecording && (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                    <span className="text-sm">Recording</span>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${getPhaseColour(phase)}`}>
                    {phase.toUpperCase()}
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${getLabelColour(currentLabel)}`}>
                    Label: {currentLabel.toUpperCase()}
                  </div>
                </>
              )}
            </div>
            <div className="flex items-center gap-4">
              <div className="text-2xl font-mono">{formatTime(totalTime)}</div>
              <div className="text-sm text-gray-400">
                Samples: {collectedSamples.length}
              </div>
              {/* Backend status */}
              <div className="flex items-center gap-1.5">
                <div className={`w-2 h-2 rounded-full ${
                  backendStatus === 'connected' ? 'bg-green-500' : 
                  backendStatus === 'checking' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
                }`} />
                <span className="text-xs text-gray-400">
                  {backendStatus === 'connected' ? 'Server' : backendStatus === 'checking' ? 'Connecting...' : 'Offline'}
                </span>
              </div>
              <button
                onClick={() => navigate('/')}
                className="text-gray-400 hover:text-white"
              >
                Exit
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Control Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Setup Phase */}
            {phase === 'setup' && (
              <div className="bg-gray-800 rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-4">Collection Setup</h2>
                <p className="text-gray-400 mb-6">
                  This mode collects labeled training data for your cognitive load model.
                  Each window of features will be saved with a label based on task difficulty.
                </p>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Participant ID *</label>
                    <input
                      type="text"
                      value={participantId}
                      onChange={(e) => setParticipantId(e.target.value)}
                      placeholder="e.g., P001"
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-orange-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Session Notes (optional)</label>
                    <textarea
                      value={sessionNotes}
                      onChange={(e) => setSessionNotes(e.target.value)}
                      placeholder="Any notes about this session..."
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 h-20 focus:outline-none focus:border-orange-500"
                    />
                  </div>
                </div>

                <div className="mt-6 p-4 bg-gray-700 rounded-lg">
                  <h3 className="font-medium mb-2">Collection Protocol</h3>
                  <ol className="text-sm text-gray-300 space-y-1 list-decimal list-inside">
                    <li>1-minute baseline (relaxed state) → Label: LOW</li>
                    <li>Rest period (30s)</li>
                    <li>Easy tasks (2 min) → Label: LOW</li>
                    <li>Rest period (30s)</li>
                    <li>Hard tasks (2 min) → Label: HIGH</li>
                    <li>Export data for training</li>
                  </ol>
                </div>

                <button
                  onClick={startCollection}
                  disabled={!participantId.trim()}
                  className="mt-6 w-full bg-orange-500 hover:bg-orange-600 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-semibold transition-colors"
                >
                  Start Data Collection
                </button>
              </div>
            )}

            {/* Active Collection Phases */}
            {(phase === 'baseline' || phase === 'task' || phase === 'rest') && (
              <div className="bg-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold">
                    {phase === 'baseline' && '📊 Baseline Recording'}
                    {phase === 'task' && `🧠 ${currentDifficulty.toUpperCase()} Task Block`}
                    {phase === 'rest' && '😌 Rest Period'}
                  </h2>
                  <div className="flex items-center gap-4">
                    <div className="text-3xl font-mono text-orange-400">
                      {formatTime(phaseTime)}
                    </div>
                    {phase === 'task' && taskStats.totalAttempts > 0 && (
                      <div className="text-sm text-gray-400">
                        {taskStats.correctAnswers}/{taskStats.totalAttempts} correct
                      </div>
                    )}
                  </div>
                </div>

                {/* Phase-specific content */}
                {phase === 'baseline' && (
                  <div className="bg-gray-700 rounded-lg p-8 mb-6 text-center">
                    <div className="text-6xl mb-4">🧘</div>
                    <p className="text-xl mb-2">Sit comfortably and relax</p>
                    <p className="text-gray-400">Look at the screen naturally. No tasks required.</p>
                    <p className="text-sm text-green-400 mt-4">Recording LOW cognitive load samples</p>
                  </div>
                )}

                {phase === 'rest' && (
                  <div className="bg-gray-700 rounded-lg p-8 mb-6 text-center">
                    <div className="text-6xl mb-4">☕</div>
                    <p className="text-xl mb-2">Take a short break</p>
                    <p className="text-gray-400">Relax before the next task block.</p>
                    <p className="text-sm text-green-400 mt-4">Recording LOW cognitive load samples</p>
                  </div>
                )}

                {phase === 'task' && (
                  <div className="mb-6">
                    {/* Task Selection or Active Task */}
                    {!selectedTask ? (
                      <div className="space-y-4">
                        <p className="text-gray-400 text-sm text-center mb-4">
                          Select a task type to begin. Complete tasks at <span className={`font-semibold ${
                            currentDifficulty === 'hard' ? 'text-red-400' : currentDifficulty === 'medium' ? 'text-yellow-400' : 'text-green-400'
                          }`}>{currentDifficulty.toUpperCase()}</span> difficulty.
                        </p>
                        <div className="grid grid-cols-2 gap-4">
                          <button
                            onClick={() => setSelectedTask('memory')}
                            className="bg-gray-700 hover:bg-indigo-600 p-6 rounded-xl transition-all group border border-gray-600 hover:border-indigo-500"
                          >
                            <div className="text-4xl mb-3">🧠</div>
                            <h3 className="text-lg font-semibold mb-1">Memory Task</h3>
                            <p className="text-sm text-gray-400 group-hover:text-gray-200">
                              Sequence recall & n-back
                            </p>
                          </button>
                          <button
                            onClick={() => setSelectedTask('math')}
                            className="bg-gray-700 hover:bg-emerald-600 p-6 rounded-xl transition-all group border border-gray-600 hover:border-emerald-500"
                          >
                            <div className="text-4xl mb-3">🔢</div>
                            <h3 className="text-lg font-semibold mb-1">Math Task</h3>
                            <p className="text-sm text-gray-400 group-hover:text-gray-200">
                              Arithmetic challenges
                            </p>
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="flex items-center justify-between mb-4">
                          <button
                            onClick={() => setSelectedTask(null)}
                            className="flex items-center text-gray-400 hover:text-white text-sm"
                          >
                            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                            Change Task
                          </button>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              selectedTask === 'memory' ? 'bg-indigo-600' : 'bg-emerald-600'
                            }`}>
                              {selectedTask === 'memory' ? 'Memory' : 'Math'}
                            </span>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              currentDifficulty === 'hard' ? 'bg-red-600' : currentDifficulty === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                            }`}>
                              {currentDifficulty.toUpperCase()}
                            </span>
                            {taskStats.streak > 0 && (
                              <span className="text-orange-400 text-sm">🔥 {taskStats.streak}</span>
                            )}
                          </div>
                        </div>
                        
                        {/* Task Components with dark theme wrapper */}
                        <div className="task-dark-wrapper bg-gray-700 rounded-xl p-4">
                          {selectedTask === 'memory' && (
                            <MemoryTask difficulty={currentDifficulty} onComplete={handleTaskComplete} />
                          )}
                          {selectedTask === 'math' && (
                            <MathTask difficulty={currentDifficulty} onComplete={handleTaskComplete} />
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Label indicator */}
                    <div className={`mt-4 text-center text-sm ${currentLabel === 'high' ? 'text-red-400' : 'text-green-400'}`}>
                      Recording {currentLabel.toUpperCase()} cognitive load samples
                    </div>
                  </div>
                )}

                {/* Manual Phase Controls */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  <button
                    onClick={() => { setTaskBlock('easy'); setSelectedTask(null); }}
                    className={`py-3 rounded-lg font-medium transition-colors ${
                      currentDifficulty === 'easy' && phase === 'task'
                        ? 'bg-green-600'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Easy
                  </button>
                  <button
                    onClick={() => { setTaskBlock('medium'); setSelectedTask(null); }}
                    className={`py-3 rounded-lg font-medium transition-colors ${
                      currentDifficulty === 'medium' && phase === 'task'
                        ? 'bg-yellow-600'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Medium
                  </button>
                  <button
                    onClick={() => { setTaskBlock('hard'); setSelectedTask(null); }}
                    className={`py-3 rounded-lg font-medium transition-colors ${
                      currentDifficulty === 'hard' && phase === 'task'
                        ? 'bg-red-600'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Hard
                  </button>
                  <button
                    onClick={startRestPeriod}
                    className={`py-3 rounded-lg font-medium transition-colors ${
                      phase === 'rest'
                        ? 'bg-gray-500'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Rest
                  </button>
                </div>

                <button
                  onClick={finishCollection}
                  className="w-full bg-orange-500 hover:bg-orange-600 py-3 rounded-lg font-semibold transition-colors"
                >
                  Finish Collection
                </button>
              </div>
            )}

            {/* Collection Complete */}
            {phase === 'complete' && (
              <div className="bg-gray-800 rounded-xl p-6">
                <div className="text-center mb-6">
                  <div className="text-6xl mb-4">🎉</div>
                  <h2 className="text-2xl font-semibold mb-2">Collection Complete!</h2>
                  <p className="text-gray-400">
                    Collected {collectedSamples.length} labeled samples
                  </p>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-orange-400">{collectedSamples.length}</div>
                    <div className="text-sm text-gray-400">Total Samples</div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {collectedSamples.filter(s => s.label === 'low').length}
                    </div>
                    <div className="text-sm text-gray-400">Low CL</div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-red-400">
                      {collectedSamples.filter(s => s.label === 'high').length}
                    </div>
                    <div className="text-sm text-gray-400">High CL</div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold">{formatTime(totalTime)}</div>
                    <div className="text-sm text-gray-400">Duration</div>
                  </div>
                </div>

                {/* Export buttons */}
                <div className="grid grid-cols-3 gap-3 mb-6">
                  <button
                    onClick={exportToCSV}
                    className="bg-blue-600 hover:bg-blue-700 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    CSV
                  </button>
                  <button
                    onClick={exportToJSON}
                    className="bg-gray-700 hover:bg-gray-600 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    JSON
                  </button>
                  <button
                    onClick={saveToServer}
                    disabled={backendStatus !== 'connected' || isSaving}
                    className={`py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                      backendStatus === 'connected' && !isSaving
                        ? 'bg-green-600 hover:bg-green-700'
                        : 'bg-gray-600 cursor-not-allowed'
                    }`}
                  >
                    {isSaving ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    )}
                    Server
                  </button>
                </div>

                {/* Save message */}
                {saveMessage && (
                  <div className={`mb-4 p-3 rounded-lg text-sm ${
                    saveMessage.startsWith('✓') 
                      ? 'bg-green-900/50 text-green-300' 
                      : 'bg-red-900/50 text-red-300'
                  }`}>
                    {saveMessage}
                  </div>
                )}

                <div className="bg-gray-700 rounded-lg p-4 text-sm text-gray-300">
                  <h3 className="font-medium text-white mb-2">Next Steps:</h3>
                  <ol className="list-decimal list-inside space-y-1">
                    <li>Export the CSV file</li>
                    <li>Copy to <code className="bg-gray-600 px-1 rounded">Machine Learning/data/processed/</code></li>
                    <li>Run: <code className="bg-gray-600 px-1 rounded">python -m src.cle.train.train --features your_file.csv</code></li>
                    <li>New model will be saved to <code className="bg-gray-600 px-1 rounded">models/</code></li>
                  </ol>
                </div>

                <button
                  onClick={() => navigate('/')}
                  className="mt-6 w-full bg-gray-700 hover:bg-gray-600 py-3 rounded-lg font-semibold transition-colors"
                >
                  Return Home
                </button>
              </div>
            )}

            {/* Sample Log */}
            {phase !== 'setup' && phase !== 'complete' && collectedSamples.length > 0 && (
              <div className="bg-gray-800 rounded-xl p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-3">Recent Samples</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {collectedSamples.slice(-5).reverse().map((sample, i) => (
                    <div key={sample.windowIndex} className="flex items-center justify-between text-sm bg-gray-700 rounded px-3 py-2">
                      <span className="text-gray-400">#{sample.windowIndex + 1}</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        sample.label === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                      }`}>
                        {sample.label}
                      </span>
                      <span className="text-gray-500">{sample.difficulty}</span>
                      <span className="text-gray-500">
                        BR: {sample.features.blink_rate?.toFixed(1) || '—'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Camera Feed */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Camera Feed</h3>
              <WebcamFeed 
                isActive={isRecording}
                onFrameFeatures={handleFrameFeatures}
                showOverlay={false}
              />
            </div>

            {/* Live Features */}
            {isRecording && (
              <LiveFeaturePanel
                frameFeatures={currentFrameFeatures}
                windowFeatures={currentWindowFeatures}
                blinkCount={totalBlinkCount}
                bufferFill={windowBufferRef.current?.fillRatio || 0}
              />
            )}

            {/* Collection Stats */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Collection Stats</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Samples</span>
                  <span className="font-mono">{collectedSamples.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-green-400">Low CL</span>
                  <span className="font-mono">{collectedSamples.filter(s => s.label === 'low').length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-red-400">High CL</span>
                  <span className="font-mono">{collectedSamples.filter(s => s.label === 'high').length}</span>
                </div>
                <div className="pt-2 border-t border-gray-700">
                  <div className="text-xs text-gray-500">
                    Samples collected every {FEATURE_CONFIG.windows.step_s}s
                  </div>
                  <div className="text-xs text-gray-500">
                    Window size: {FEATURE_CONFIG.windows.length_s}s
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataCollection;

