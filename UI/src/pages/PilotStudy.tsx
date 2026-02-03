import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import WebcamFeed from '../components/WebcamFeed';
import NasaTLXForm from '../components/NasaTLXForm';
import PairedAssociatesTask, {
  WORD_PAIRS_FORM_A,
  WORD_PAIRS_FORM_B,
} from '../components/tasks/PairedAssociatesTask';
import { predictCognitiveLoad, saveStudySession } from '../services/apiClient';
import { WindowBuffer } from '../services/windowBuffer';
import { computeWindowFeatures } from '../services/featureExtraction';
import { FEATURE_CONFIG } from '../config/featureConfig';
import {
  StudyPhase,
  StudyCondition,
  CLIDataPoint,
  InterventionLog,
  TaskPerformance,
  CalibrationData,
  StudyNASATLX,
  NASATLXScores,
  WordPair,
} from '../types';
import { FrameFeatures } from '../types/features';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  calibration: {
    duration_s: 60,
  },
  learning: {
    easy: { pairCount: 6, exposureMs: 4000 },
    hard: { pairCount: 10, exposureMs: 2000 },
  },
  intervention: {
    cli_threshold: 0.7,
    consecutive_windows: 2,
    cooldown_s: 120,
    max_per_session: 2,
    break_duration_s: 60,
  },
  prediction: {
    interval_ms: 2500,
  },
};

// ============================================================================
// Main Component
// ============================================================================

const PilotStudy: React.FC = () => {
  const navigate = useNavigate();

  // Study setup state
  const [participantId, setParticipantId] = useState('');
  const [sessionNumber, setSessionNumber] = useState<1 | 2>(1);
  const [condition, setCondition] = useState<StudyCondition>('adaptive');
  const [formVersion, setFormVersion] = useState<'A' | 'B'>('A');
  const [consentGiven, setConsentGiven] = useState(false);

  // Phase management
  const [phase, setPhase] = useState<StudyPhase>('consent');
  const [phaseStartTime, setPhaseStartTime] = useState<number>(0);

  // Webcam and features
  const [cameraActive, setCameraActive] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const windowBufferRef = useRef<WindowBuffer | null>(null);
  const sessionStartRef = useRef<number>(0);

  // CLI tracking
  const [currentCLI, setCurrentCLI] = useState(0);
  const [cliTimeseries, setCLITimeseries] = useState<CLIDataPoint[]>([]);
  const consecutiveHighRef = useRef(0);

  // Calibration
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const calibrationCLIRef = useRef<number[]>([]);
  const calibrationEARRef = useRef<number[]>([]);

  // Interventions
  const [interventions, setInterventions] = useState<InterventionLog[]>([]);
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionType, setInterventionType] = useState<'micro_break' | 'pacing_adjustment'>('micro_break');
  const [breakTimeRemaining, setBreakTimeRemaining] = useState(0);
  const lastInterventionTimeRef = useRef(0);
  const [currentExposureBonus, setCurrentExposureBonus] = useState(0);

  // Task performance
  const [easyPerformance, setEasyPerformance] = useState<TaskPerformance | null>(null);
  const [hardPerformance, setHardPerformance] = useState<TaskPerformance | null>(null);
  const [immediateTestPerformance, setImmediateTestPerformance] = useState<TaskPerformance | null>(null);

  // NASA-TLX
  const [nasaTLX, setNasaTLX] = useState<StudyNASATLX | null>(null);

  // Initialize window buffer
  useEffect(() => {
    windowBufferRef.current = new WindowBuffer();
    return () => {
      windowBufferRef.current = null;
    };
  }, []);

  // Determine condition based on participant ID and session number
  useEffect(() => {
    if (!participantId) return;
    
    // Extract number from participant ID (e.g., P001 -> 1)
    const idNum = parseInt(participantId.replace(/\D/g, '')) || 0;
    const isOdd = idNum % 2 === 1;
    
    // Odd IDs: Session 1 = Adaptive, Session 2 = Baseline
    // Even IDs: Session 1 = Baseline, Session 2 = Adaptive
    if (sessionNumber === 1) {
      setCondition(isOdd ? 'adaptive' : 'baseline');
      setFormVersion('A');
    } else {
      setCondition(isOdd ? 'baseline' : 'adaptive');
      setFormVersion('B');
    }
  }, [participantId, sessionNumber]);

  // Get word pairs for current session
  const getWordPairs = useCallback((difficulty: 'easy' | 'hard'): WordPair[] => {
    const allPairs = formVersion === 'A' ? WORD_PAIRS_FORM_A : WORD_PAIRS_FORM_B;
    const count = CONFIG.learning[difficulty].pairCount;
    
    if (difficulty === 'easy') {
      return allPairs.slice(0, count);
    } else {
      return allPairs.slice(6, 6 + count);
    }
  }, [formVersion]);

  // Handle frame from webcam
  const handleFrame = useCallback(async (features: FrameFeatures) => {
    if (!windowBufferRef.current) return;

    setFaceDetected(features.valid);
    windowBufferRef.current.addFrame(features);

    // During calibration, collect EAR values
    if (phase === 'calibration' && features.valid) {
      calibrationEARRef.current.push(features.ear_mean);
    }
  }, [phase]);

  // Prediction loop
  useEffect(() => {
    if (phase === 'consent' || phase === 'setup' || phase === 'complete') return;
    if (!windowBufferRef.current) return;

    const interval = setInterval(async () => {
      if (!windowBufferRef.current?.isReady()) return;

      try {
        const windowData = windowBufferRef.current.getWindow();
        const windowFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
        if (!windowFeatures) return;

        const result = await predictCognitiveLoad(windowFeatures);
        const cli = result.cli;
        setCurrentCLI(cli);

        // Record CLI datapoint
        const t = (Date.now() - sessionStartRef.current) / 1000;
        const dataPoint: CLIDataPoint = { t, cli, confidence: result.confidence };
        setCLITimeseries(prev => [...prev, dataPoint]);

        // During calibration, track CLI
        if (phase === 'calibration') {
          calibrationCLIRef.current.push(cli);
        }

        // Intervention logic (only in adaptive condition during learning)
        if (condition === 'adaptive' && (phase === 'learning_easy' || phase === 'learning_hard')) {
          checkForIntervention(cli, t);
        }
      } catch (error) {
        console.error('Prediction error:', error);
      }
    }, CONFIG.prediction.interval_ms);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, condition]);

  // Check if intervention should trigger
  const checkForIntervention = useCallback((cli: number, t: number) => {
    // Skip if already showing intervention or in cooldown
    if (showIntervention) return;
    if (t - lastInterventionTimeRef.current < CONFIG.intervention.cooldown_s) return;
    if (interventions.length >= CONFIG.intervention.max_per_session) return;

    if (cli > CONFIG.intervention.cli_threshold) {
      consecutiveHighRef.current++;
      
      if (consecutiveHighRef.current >= CONFIG.intervention.consecutive_windows) {
        // Trigger intervention
        const type = interventions.length === 0 ? 'micro_break' : 'pacing_adjustment';
        setInterventionType(type);
        setShowIntervention(true);
        
        if (type === 'micro_break') {
          setBreakTimeRemaining(CONFIG.intervention.break_duration_s);
        }
      }
    } else {
      consecutiveHighRef.current = 0;
    }
  }, [showIntervention, interventions.length]);

  // Break countdown
  useEffect(() => {
    if (!showIntervention || interventionType !== 'micro_break' || breakTimeRemaining <= 0) return;

    const timer = setTimeout(() => {
      setBreakTimeRemaining(prev => prev - 1);
    }, 1000);

    if (breakTimeRemaining === 1) {
      // Auto-dismiss after countdown
      handleInterventionResponse(true);
    }

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showIntervention, interventionType, breakTimeRemaining]);

  // Handle intervention response
  const handleInterventionResponse = useCallback((accepted: boolean) => {
    const t = (Date.now() - sessionStartRef.current) / 1000;
    
    const log: InterventionLog = {
      t,
      cli: currentCLI,
      type: interventionType,
      accepted,
    };
    
    setInterventions(prev => [...prev, log]);
    lastInterventionTimeRef.current = t;
    consecutiveHighRef.current = 0;
    setShowIntervention(false);

    if (accepted && interventionType === 'pacing_adjustment') {
      setCurrentExposureBonus(prev => prev + 1000); // +1s per pair
    }
  }, [currentCLI, interventionType]);

  // Calibration countdown state
  const [calibrationRemaining, setCalibrationRemaining] = useState(CONFIG.calibration.duration_s);

  // Phase timer for calibration
  useEffect(() => {
    if (phase !== 'calibration') return;

    const interval = setInterval(() => {
      const elapsed = (Date.now() - phaseStartTime) / 1000;
      const remaining = CONFIG.calibration.duration_s - elapsed;
      
      setCalibrationRemaining(Math.max(0, Math.ceil(remaining)));

      if (remaining <= 0) {
        clearInterval(interval);
        
        // Complete calibration
        const meanCLI = calibrationCLIRef.current.length > 0
          ? calibrationCLIRef.current.reduce((a, b) => a + b, 0) / calibrationCLIRef.current.length
          : 0.5;
        const meanEAR = calibrationEARRef.current.length > 0
          ? calibrationEARRef.current.reduce((a, b) => a + b, 0) / calibrationEARRef.current.length
          : 0.25;

        setCalibrationData({
          baseline_cli: meanCLI,
          baseline_ear: meanEAR,
          duration_s: CONFIG.calibration.duration_s,
        });

        setPhase('learning_easy');
        setPhaseStartTime(Date.now());
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [phase, phaseStartTime]);

  // Start study session
  const handleStartStudy = () => {
    if (!consentGiven || !participantId.trim()) return;
    
    sessionStartRef.current = Date.now();
    setCameraActive(true);
    setPhase('calibration');
    setPhaseStartTime(Date.now());
    calibrationCLIRef.current = [];
    calibrationEARRef.current = [];
  };

  // Task completion handlers
  const handleEasyComplete = (perf: TaskPerformance) => {
    setEasyPerformance(perf);
    setPhase('learning_hard');
    setPhaseStartTime(Date.now());
    setCurrentExposureBonus(0);
  };

  const handleHardComplete = (perf: TaskPerformance) => {
    setHardPerformance(perf);
    setPhase('immediate_test');
    setPhaseStartTime(Date.now());
  };

  const handleTestComplete = (perf: TaskPerformance) => {
    setImmediateTestPerformance(perf);
    setPhase('nasa_tlx');
  };

  // NASA-TLX submission
  const handleNASATLXSubmit = async (scores: NASATLXScores) => {
    const rawTLX = (
      scores.mentalDemand +
      scores.physicalDemand +
      scores.temporalDemand +
      (100 - scores.performance) + // Inverted
      scores.effort +
      scores.frustration
    ) / 6;

    const studyTLX: StudyNASATLX = {
      mental: scores.mentalDemand,
      physical: scores.physicalDemand,
      temporal: scores.temporalDemand,
      performance: scores.performance,
      effort: scores.effort,
      frustration: scores.frustration,
      raw_tlx: rawTLX,
    };

    setNasaTLX(studyTLX);

    // Save session
    try {
      const session = {
        participant_id: participantId,
        session_number: sessionNumber,
        condition,
        timestamp: new Date().toISOString(),
        form_version: formVersion,
        calibration: calibrationData!,
        cli_timeseries: cliTimeseries,
        interventions,
        task_performance: {
          easy: easyPerformance!,
          hard: hardPerformance!,
        },
        nasa_tlx: studyTLX,
        immediate_test: immediateTestPerformance!,
        delayed_test: null,
      };

      await saveStudySession(session);
      setPhase('complete');
    } catch (error) {
      console.error('Failed to save session:', error);
      alert('Error saving session. Please contact researcher.');
    }
  };

  // ============================================================================
  // Render
  // ============================================================================

  // Consent phase
  if (phase === 'consent') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-2xl w-full">
          <h1 className="text-2xl font-bold text-gray-800 mb-6">
            Pilot Study: Cognitive Load Detection
          </h1>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Participant ID
              </label>
              <input
                type="text"
                value={participantId}
                onChange={(e) => setParticipantId(e.target.value.toUpperCase())}
                placeholder="e.g., P001"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Number
              </label>
              <div className="flex gap-4">
                <button
                  onClick={() => setSessionNumber(1)}
                  className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                    sessionNumber === 1
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Session 1
                </button>
                <button
                  onClick={() => setSessionNumber(2)}
                  className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                    sessionNumber === 2
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Session 2
                </button>
              </div>
            </div>

            {participantId && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>Condition:</strong> {condition === 'adaptive' ? 'Adaptive (with interventions)' : 'Baseline (no interventions)'}
                </p>
                <p className="text-sm text-blue-800">
                  <strong>Word Form:</strong> {formVersion}
                </p>
              </div>
            )}

            <div className="border-t pt-4">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={consentGiven}
                  onChange={(e) => setConsentGiven(e.target.checked)}
                  className="mt-1 h-4 w-4 text-blue-600 rounded"
                />
                <span className="text-sm text-gray-600">
                  I understand that this study will use my webcam to extract facial features
                  (no video is stored). I consent to participate and understand I can withdraw
                  at any time.
                </span>
              </label>
            </div>

            <button
              onClick={handleStartStudy}
              disabled={!consentGiven || !participantId.trim()}
              className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 
                         text-white py-3 rounded-lg font-semibold transition-colors"
            >
              Start Study
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Calibration phase
  if (phase === 'calibration') {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-4xl w-full">
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Calibration
              </h2>
              <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden">
                <WebcamFeed isActive={cameraActive} onFrameFeatures={handleFrame} />
              </div>
              <div className={`mt-2 text-center ${faceDetected ? 'text-green-600' : 'text-red-600'}`}>
                {faceDetected ? '✓ Face detected' : '✗ Face not detected'}
              </div>
            </div>
            <div className="flex flex-col justify-center">
              <div className="text-center">
                <div className="text-6xl font-bold text-blue-500 mb-4">
                  {calibrationRemaining}
                </div>
                <p className="text-gray-600">seconds remaining</p>
              </div>
              <p className="mt-8 text-gray-600 text-center">
                Please look at the screen naturally. This establishes your baseline.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Learning phases
  if (phase === 'learning_easy' || phase === 'learning_hard') {
    const difficulty = phase === 'learning_easy' ? 'easy' : 'hard';
    const pairs = getWordPairs(difficulty);
    const baseExposure = CONFIG.learning[difficulty].exposureMs;
    const totalExposure = baseExposure + (condition === 'adaptive' ? currentExposureBonus : 0);

    return (
      <div className="min-h-screen bg-gray-100 p-4">
        {/* Intervention modal */}
        {showIntervention && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl p-8 max-w-md">
              {interventionType === 'micro_break' ? (
                <>
                  <h3 className="text-xl font-semibold mb-4">Take a Short Break</h3>
                  <p className="text-gray-600 mb-4">
                    Your cognitive load appears high. Take a moment to relax.
                  </p>
                  <div className="text-4xl font-bold text-blue-500 text-center mb-4">
                    {breakTimeRemaining}s
                  </div>
                  <button
                    onClick={() => handleInterventionResponse(false)}
                    className="w-full py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                  >
                    Skip Break
                  </button>
                </>
              ) : (
                <>
                  <h3 className="text-xl font-semibold mb-4">Adjust Pacing?</h3>
                  <p className="text-gray-600 mb-4">
                    Would you like more time to study each pair?
                  </p>
                  <div className="flex gap-4">
                    <button
                      onClick={() => handleInterventionResponse(true)}
                      className="flex-1 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                    >
                      Yes, slow down
                    </button>
                    <button
                      onClick={() => handleInterventionResponse(false)}
                      className="flex-1 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                    >
                      No, continue
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-xl shadow-lg p-4 mb-4 flex justify-between items-center">
            <div>
              <span className="text-sm text-gray-500">Condition: </span>
              <span className={`font-medium ${condition === 'adaptive' ? 'text-blue-600' : 'text-gray-600'}`}>
                {condition === 'adaptive' ? 'Adaptive' : 'Baseline'}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm">
                CLI: <span className="font-mono font-medium">{currentCLI.toFixed(2)}</span>
              </div>
              <div className={`w-3 h-3 rounded-full ${faceDetected ? 'bg-green-500' : 'bg-red-500'}`} />
            </div>
          </div>

          {/* Small webcam preview */}
          <div className="fixed bottom-4 right-4 w-48 rounded-lg overflow-hidden shadow-lg">
            <WebcamFeed isActive={cameraActive} onFrameFeatures={handleFrame} />
          </div>

          {/* Task */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              {difficulty === 'easy' ? 'Easy Block' : 'Hard Block'} - Study Phase
            </h2>
            <p className="text-gray-600 mb-6">
              Memorize each word pair. You will be tested on these later.
            </p>
            <PairedAssociatesTask
              pairs={pairs}
              exposureTimeMs={totalExposure}
              mode="study"
              onComplete={difficulty === 'easy' ? handleEasyComplete : handleHardComplete}
            />
          </div>
        </div>
      </div>
    );
  }

  // Immediate test phase
  if (phase === 'immediate_test') {
    const allPairs = [
      ...getWordPairs('easy'),
      ...getWordPairs('hard'),
    ];

    return (
      <div className="min-h-screen bg-gray-100 p-4">
        <div className="max-w-2xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Immediate Recall Test
            </h2>
            <p className="text-gray-600 mb-6">
              Type the word that was paired with each cue.
            </p>
            <PairedAssociatesTask
              pairs={allPairs}
              exposureTimeMs={0}
              mode="test"
              onComplete={handleTestComplete}
            />
          </div>
        </div>

        {/* Small webcam preview */}
        <div className="fixed bottom-4 right-4 w-48 rounded-lg overflow-hidden shadow-lg">
          <WebcamFeed isActive={cameraActive} onFrameFeatures={handleFrame} />
        </div>
      </div>
    );
  }

  // NASA-TLX phase
  if (phase === 'nasa_tlx') {
    return (
      <div className="min-h-screen bg-gray-100 p-4">
        <div className="max-w-2xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <NasaTLXForm onSubmit={handleNASATLXSubmit} />
          </div>
        </div>
      </div>
    );
  }

  // Complete phase
  if (phase === 'complete') {
    const sessionId = `${participantId}_s${sessionNumber}_${condition}`;
    
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-2xl w-full text-center">
          <div className="text-6xl mb-4">✓</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Session Complete!
          </h2>
          <p className="text-gray-600 mb-6">
            Thank you for participating. Your data has been saved.
          </p>

          <div className="bg-gray-50 rounded-lg p-4 mb-6 text-left">
            <h3 className="font-medium text-gray-800 mb-2">Session Summary</h3>
            <div className="text-sm text-gray-600 space-y-1">
              <p>Participant: {participantId}</p>
              <p>Condition: {condition}</p>
              <p>Interventions: {interventions.length}</p>
              <p>Immediate Test: {immediateTestPerformance?.correct}/{immediateTestPerformance?.total}</p>
              <p>NASA-TLX (Raw): {nasaTLX?.raw_tlx.toFixed(1)}</p>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4 mb-6">
            <p className="text-sm text-blue-800">
              <strong>Delayed Test Link:</strong><br />
              <code className="text-xs">{window.location.origin}/study/delayed/{sessionId}</code>
            </p>
            <p className="text-xs text-blue-600 mt-2">
              Send this link to the participant in 7 days.
            </p>
          </div>

          <button
            onClick={() => navigate('/')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default PilotStudy;
