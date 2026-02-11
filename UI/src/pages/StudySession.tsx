import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import WebcamFeed from '../components/WebcamFeed';
import LiveFeaturePanel from '../components/LiveFeaturePanel';
import NasaTLXForm from '../components/NasaTLXForm';
import CuedRecallTest from '../components/study/CuedRecallTest';
import PairedAssociateLearningBlock from '../components/study/PairedAssociateLearningBlock';
import RecognitionTest from '../components/study/RecognitionTest';
import StudyInterventionModal from '../components/study/StudyInterventionModal';
import { FEATURE_CONFIG } from '../config/featureConfig';
import { STUDY_CONFIG, STUDY_QUALITY_CONFIG, STUDY_RECORD_VERSION } from '../config/studyConfig';
import { predictCognitiveLoad, testConnection } from '../services/apiClient';
import { StudyAdaptiveController } from '../services/studyAdaptiveController';
import { engineerFeatures, computeBaseline } from '../services/featureEngineering';
import { computeWindowFeatures } from '../services/featureExtraction';
import { createSessionRecordId, saveSessionDraft } from '../services/studyStorage';
import { getStimulusItemsForBlock } from '../services/studyStimuli';
import { WindowBuffer, validateWindowQuality } from '../services/windowBuffer';
import { FrameFeatures, WindowFeatures } from '../types/features';
import {
  StudyBlockSummary,
  StudyCliSample,
  StudyInterventionEvent,
  StudyCondition,
  StudyForm,
  StudyPhaseTag,
  StudySessionNumber,
  StudySessionRecord,
  StudySetupState,
  StudyTrialResult,
} from '../types/study';

type SessionPhase =
  | 'baseline'
  | 'learn_easy'
  | 'test_easy_recognition'
  | 'test_easy_cued'
  | 'learn_hard'
  | 'test_hard_recognition'
  | 'test_hard_cued'
  | 'break'
  | 'nasa_tlx'
  | 'complete';

const FALLBACK_SETUP: StudySetupState = {
  participantId: '',
  assignment: {
    participantId: '',
    participantIdNormalized: '',
    hashValue: 0,
    hashParity: 'even',
    conditionOrder: ['adaptive', 'baseline'],
    formOrder: ['A', 'B'],
    sessionNumber: 1 as StudySessionNumber,
    condition: 'adaptive' as StudyCondition,
    form: 'A' as StudyForm,
    delayedDueAtIso: new Date().toISOString(),
  },
  plan: STUDY_CONFIG,
  session2StartedEarlyOverride: false,
};

function phaseToTag(phase: SessionPhase): StudyPhaseTag {
  switch (phase) {
    case 'baseline':
      return 'baseline_calibration';
    case 'learn_easy':
      return 'learning_easy';
    case 'test_easy_recognition':
      return 'test_easy_recognition';
    case 'test_easy_cued':
      return 'test_easy_cued_recall';
    case 'learn_hard':
      return 'learning_hard';
    case 'test_hard_recognition':
      return 'test_hard_recognition';
    case 'test_hard_cued':
      return 'test_hard_cued_recall';
    case 'break':
      return 'break';
    case 'nasa_tlx':
      return 'nasa_tlx';
    case 'complete':
      return 'complete';
  }
}

function isLearningPhase(phase: SessionPhase): boolean {
  return phase === 'learn_easy' || phase === 'learn_hard';
}

function isActiveTaskPhase(phase: SessionPhase): boolean {
  return (
    phase === 'learn_easy' ||
    phase === 'learn_hard' ||
    phase === 'test_easy_recognition' ||
    phase === 'test_easy_cued' ||
    phase === 'test_hard_recognition' ||
    phase === 'test_hard_cued'
  );
}

function buildBlockSummary(
  blockIndex: 1 | 2,
  difficulty: 'easy' | 'hard',
  allTrials: StudyTrialResult[],
  effectiveExposureSeconds: number,
  interventions: StudyInterventionEvent[]
): StudyBlockSummary {
  const blockTrials = allTrials.filter((trial) => trial.blockIndex === blockIndex);

  const recognition = blockTrials.filter((trial) => trial.kind === 'recognition');
  const cued = blockTrials.filter((trial) => trial.kind === 'cued_recall');

  const recognitionAccuracy =
    recognition.length > 0
      ? recognition.filter((trial) => trial.correct).length / recognition.length
      : 0;
  const cuedAccuracy =
    cued.length > 0 ? cued.filter((trial) => trial.correct).length / cued.length : 0;

  const recognitionMeanRtMs =
    recognition.length > 0
      ? recognition.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / recognition.length
      : 0;
  const cuedMeanRtMs =
    cued.length > 0 ? cued.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / cued.length : 0;

  const adaptationApplied = interventions.some(
    (event) =>
      event.outcome === 'applied' &&
      ((blockIndex === 1 && (event.phase === 'learning_easy' || event.phase === 'test_easy_recognition' || event.phase === 'test_easy_cued_recall')) ||
        (blockIndex === 2 && (event.phase === 'learning_hard' || event.phase === 'test_hard_recognition' || event.phase === 'test_hard_cued_recall')))
  );

  return {
    blockIndex,
    difficulty,
    learningItemCount: blockTrials.filter((trial) => trial.kind === 'learning').length,
    recognitionAccuracy,
    recognitionMeanRtMs,
    cuedRecallAccuracy: cuedAccuracy,
    cuedRecallMeanRtMs: cuedMeanRtMs,
    effectiveExposureSeconds,
    adaptationApplied,
  };
}

const StudySession: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const setup = location.state as StudySetupState | undefined;
  const missingSetup = !setup;
  const effectiveSetup = setup ?? FALLBACK_SETUP;
  const { participantId, assignment, plan, session2StartedEarlyOverride } = effectiveSetup;

  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');

  const [phase, setPhase] = useState<SessionPhase>('baseline');
  const [resumePhase, setResumePhase] = useState<SessionPhase | null>(null);
  const [phaseSeconds, setPhaseSeconds] = useState(0);
  const [totalSessionSeconds, setTotalSessionSeconds] = useState(0);
  const [activeTaskSeconds, setActiveTaskSeconds] = useState(0);
  const [breakSeconds, setBreakSeconds] = useState(0);

  const [currentLoad, setCurrentLoad] = useState(0.5);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [lowConfidenceMode, setLowConfidenceMode] = useState(false);

  const [currentFrameFeatures, setCurrentFrameFeatures] = useState<FrameFeatures | null>(null);
  const [currentWindowFeatures, setCurrentWindowFeatures] = useState<WindowFeatures | null>(null);
  const [totalBlinkCount, setTotalBlinkCount] = useState(0);

  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [cliSamples, setCliSamples] = useState<StudyCliSample[]>([]);
  const [interventions, setInterventions] = useState<StudyInterventionEvent[]>([]);
  const [blockSummaries, setBlockSummaries] = useState<StudyBlockSummary[]>([]);

  const [scheduledBreak, setScheduledBreak] = useState(false);
  const [pacingOffsetSeconds, setPacingOffsetSeconds] = useState(0);
  const [hardInterferenceReduced, setHardInterferenceReduced] = useState(false);

  const [modalOpen, setModalOpen] = useState(false);
  const [pendingIntervention, setPendingIntervention] = useState<StudyInterventionEvent | null>(null);

  const [nasaPreviewOpen, setNasaPreviewOpen] = useState(false);

  const windowBufferRef = useRef(
    new WindowBuffer(FEATURE_CONFIG.windows.length_s, FEATURE_CONFIG.video.fps)
  );
  const lastPredictionTimeRef = useRef(0);
  const lastWindowUpdateRef = useRef(0);
  const prevEarRef = useRef(0.3);
  const adaptiveControllerRef = useRef(new StudyAdaptiveController());

  const baselineRef = useRef<Record<string, number> | null>(null);
  const baselineSamplesRef = useRef<WindowFeatures[]>([]);
  const prevCenteredRef = useRef<Record<string, number> | null>(null);

  const sessionStartMsRef = useRef(Date.now());
  const startedAtIsoRef = useRef(new Date().toISOString());
  const recordIdRef = useRef<string>('');

  if (!recordIdRef.current && !missingSetup) {
    recordIdRef.current = createSessionRecordId(participantId, assignment.sessionNumber, assignment.condition);
  }

  const easyItems = useMemo(
    () =>
      getStimulusItemsForBlock(
        assignment.form,
        'easy',
        plan.easyItemCount,
        `${participantId}:${assignment.sessionNumber}`
      ),
    [assignment.form, assignment.sessionNumber, participantId, plan.easyItemCount]
  );

  const hardItems = useMemo(
    () =>
      getStimulusItemsForBlock(
        assignment.form,
        'hard',
        plan.hardItemCount,
        `${participantId}:${assignment.sessionNumber}`
      ),
    [assignment.form, assignment.sessionNumber, participantId, plan.hardItemCount]
  );

  const persistDraft = useCallback(
    (override?: Partial<StudySessionRecord>) => {
      if (missingSetup) return;
      const draft: StudySessionRecord = {
        recordVersion: STUDY_RECORD_VERSION,
        recordId: recordIdRef.current,
        participantId,
        sessionNumber: assignment.sessionNumber,
        assignment,
        plan,
        startedAtIso: startedAtIsoRef.current,
        completedAtIso: undefined,
        session2StartedEarlyOverride,
        condition: assignment.condition,
        form: assignment.form,
        totalSessionSeconds,
        activeTaskSeconds,
        breakSeconds,
        cliSamples,
        interventions,
        trials,
        blockSummaries,
        pendingDelayedTest: true,
        delayedDueAtIso: assignment.delayedDueAtIso,
        ...override,
      };
      saveSessionDraft(draft);
    },
    [
      activeTaskSeconds,
      assignment,
      blockSummaries,
      breakSeconds,
      cliSamples,
      interventions,
      missingSetup,
      participantId,
      plan,
      session2StartedEarlyOverride,
      totalSessionSeconds,
      trials,
    ]
  );

  useEffect(() => {
    const checkBackend = async () => {
      const connected = await testConnection();
      setBackendStatus(connected ? 'connected' : 'error');
    };
    checkBackend();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTotalSessionSeconds((prev) => prev + 1);
      setPhaseSeconds((prev) => prev + 1);
      if (phase === 'break') {
        setBreakSeconds((prev) => prev + 1);
      }
      if (isActiveTaskPhase(phase)) {
        setActiveTaskSeconds((prev) => prev + 1);
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [phase]);

  useEffect(() => {
    if (phase === 'baseline' && phaseSeconds >= plan.baselineSeconds) {
      setPhase('learn_easy');
      setPhaseSeconds(0);
    }
  }, [phase, phaseSeconds, plan.baselineSeconds]);

  useEffect(() => {
    if (phase === 'break' && phaseSeconds >= plan.microBreakSeconds) {
      if (resumePhase) {
        setPhase(resumePhase);
      } else {
        setPhase('learn_easy');
      }
      setResumePhase(null);
      setPhaseSeconds(0);
    }
  }, [phase, phaseSeconds, plan.microBreakSeconds, resumePhase]);

  useEffect(() => {
    persistDraft();
  }, [persistDraft]);

  const transitionTo = useCallback(
    (nextPhase: SessionPhase) => {
      if (scheduledBreak && nextPhase !== 'break') {
        setResumePhase(nextPhase);
        setPhase('break');
        setPhaseSeconds(0);
        setScheduledBreak(false);
        return;
      }
      setPhase(nextPhase);
      setPhaseSeconds(0);
    },
    [scheduledBreak]
  );

  const finishSession = useCallback(() => {
    const completedRecord: StudySessionRecord = {
      recordVersion: STUDY_RECORD_VERSION,
      recordId: recordIdRef.current,
      participantId,
      sessionNumber: assignment.sessionNumber,
      assignment,
      plan,
      startedAtIso: startedAtIsoRef.current,
      completedAtIso: new Date().toISOString(),
      session2StartedEarlyOverride,
      condition: assignment.condition,
      form: assignment.form,
      totalSessionSeconds,
      activeTaskSeconds,
      breakSeconds,
      cliSamples,
      interventions,
      trials,
      blockSummaries,
      pendingDelayedTest: true,
      delayedDueAtIso: assignment.delayedDueAtIso,
    };

    persistDraft(completedRecord);
    navigate('/study/summary', { state: { record: completedRecord } });
  }, [
    activeTaskSeconds,
    assignment,
    blockSummaries,
    breakSeconds,
    cliSamples,
    interventions,
    navigate,
    participantId,
    persistDraft,
    plan,
    session2StartedEarlyOverride,
    totalSessionSeconds,
    trials,
  ]);

  const handleAdaptiveDecision = useCallback(
    (event: StudyInterventionEvent | undefined, actionType?: StudyInterventionEvent['type']) => {
      if (!event || !actionType) return;

      if (actionType === 'micro_break_60s') {
        setPendingIntervention(event);
        setModalOpen(true);
        return;
      }

      setInterventions((prev) => [...prev, event]);
    },
    []
  );

  const handlePrediction = useCallback(
    async (windowFeatures: WindowFeatures) => {
      if (backendStatus !== 'connected') return;

      if (!baselineRef.current) {
        baselineSamplesRef.current.push(windowFeatures);
        if (baselineSamplesRef.current.length >= 4) {
          baselineRef.current = computeBaseline(baselineSamplesRef.current.slice(-4));
          prevCenteredRef.current = null;
          setIsCalibrated(true);
        }
        return;
      }

      const engineered = engineerFeatures(windowFeatures, baselineRef.current, prevCenteredRef.current);
      prevCenteredRef.current = engineered.nextPrevCentered;

      try {
        const result = await predictCognitiveLoad(engineered.featureMap);
        if (!result.success) return;

        const alpha = FEATURE_CONFIG.realtime.smoothing_alpha;
        const smoothed = alpha * result.cli + (1 - alpha) * currentLoad;

        setCurrentLoad(smoothed);

        const phaseTag = phaseToTag(phase);
        const qualityFlags = {
          lowConfidence: result.confidence < STUDY_QUALITY_CONFIG.confidenceMin,
          lowValidFrameRatio: windowFeatures.valid_frame_ratio < STUDY_QUALITY_CONFIG.validFrameRatioMin,
          unstableIllumination: windowFeatures.std_brightness > STUDY_QUALITY_CONFIG.illuminationStdMax,
        };

        const cliSample: StudyCliSample = {
          timestampMs: Date.now(),
          sessionTimeS: totalSessionSeconds,
          phase: phaseTag,
          rawCli: result.cli,
          smoothedCli: smoothed,
          confidence: result.confidence,
          validFrameRatio: windowFeatures.valid_frame_ratio,
          illuminationStd: windowFeatures.std_brightness,
          qualityFlags,
        };

        setCliSamples((prev) => [...prev, cliSample]);

        if (!isLearningPhase(phase)) return;

        const decision = adaptiveControllerRef.current.ingest(
          {
            timestampMs: cliSample.timestampMs,
            cli: result.cli,
            confidence: result.confidence,
            validFrameRatio: windowFeatures.valid_frame_ratio,
            illuminationStd: windowFeatures.std_brightness,
            sessionTimeS: totalSessionSeconds,
            phase: phaseTag,
          },
          assignment.condition
        );

        setLowConfidenceMode(decision.state.lowConfidenceMode);
        setPacingOffsetSeconds(decision.state.pacingOffsetSeconds);
        setHardInterferenceReduced(decision.state.difficultySteppedDown);

        handleAdaptiveDecision(decision.event, decision.actionType);
      } catch (error) {
        console.error('Prediction failed:', error);
      }
    },
    [assignment.condition, backendStatus, currentLoad, handleAdaptiveDecision, phase, totalSessionSeconds]
  );

  const handleFrameFeatures = useCallback(
    (frameFeatures: FrameFeatures) => {
      if (phase === 'complete') return;

      setCurrentFrameFeatures(frameFeatures);

      if (
        frameFeatures.valid &&
        frameFeatures.ear_mean < FEATURE_CONFIG.blink.ear_thresh &&
        prevEarRef.current >= FEATURE_CONFIG.blink.ear_thresh
      ) {
        setTotalBlinkCount((prev) => prev + 1);
      }
      prevEarRef.current = frameFeatures.ear_mean;

      windowBufferRef.current.addFrame(frameFeatures);

      const now = Date.now() / 1000;
      if (windowBufferRef.current.length > 0 && now - lastWindowUpdateRef.current >= 0.5) {
        const windowData = windowBufferRef.current.getWindow();
        if (windowData.length > 0) {
          const wFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
          setCurrentWindowFeatures(wFeatures);
        }
        lastWindowUpdateRef.current = now;
      }

      const stepS = FEATURE_CONFIG.windows.step_s;
      if (windowBufferRef.current.isReady() && now - lastPredictionTimeRef.current >= stepS) {
        lastPredictionTimeRef.current = now;
        const windowData = windowBufferRef.current.getWindow();
        const [isValid] = validateWindowQuality(windowData);
        if (!isValid) return;
        const wFeatures = computeWindowFeatures(windowData, FEATURE_CONFIG.video.fps);
        void handlePrediction(wFeatures);
      }
    },
    [handlePrediction, phase]
  );

  const completeLearningBlock = useCallback(
    (learningTrials: StudyTrialResult[], _elapsed: number, nextPhase: SessionPhase) => {
      setTrials((prev) => [...prev, ...learningTrials]);
      transitionTo(nextPhase);
    },
    [transitionTo]
  );

  const completeRecognitionBlock = useCallback(
    (recognitionTrials: StudyTrialResult[], nextPhase: SessionPhase) => {
      setTrials((prev) => [...prev, ...recognitionTrials]);
      transitionTo(nextPhase);
    },
    [transitionTo]
  );

  const completeCuedBlock = useCallback(
    (cuedTrials: StudyTrialResult[], blockIndex: 1 | 2, difficulty: 'easy' | 'hard', nextPhase: SessionPhase) => {
      const merged = [...trials, ...cuedTrials];
      setTrials(merged);

      const summary = buildBlockSummary(
        blockIndex,
        difficulty,
        merged,
        difficulty === 'easy' ? plan.easyExposureSeconds + pacingOffsetSeconds : plan.hardExposureSeconds + pacingOffsetSeconds,
        interventions
      );
      setBlockSummaries((prev) => [...prev.filter((b) => b.blockIndex !== blockIndex), summary]);

      transitionTo(nextPhase);
    },
    [interventions, pacingOffsetSeconds, plan.easyExposureSeconds, plan.hardExposureSeconds, transitionTo, trials]
  );

  const onModalAccept = () => {
    if (!pendingIntervention) {
      setModalOpen(false);
      return;
    }

    const acceptedEvent: StudyInterventionEvent = {
      ...pendingIntervention,
      outcome: 'applied',
    };
    setInterventions((prev) => [...prev, acceptedEvent]);
    setModalOpen(false);
    setPendingIntervention(null);
    setScheduledBreak(true);
  };

  const onModalDismiss = () => {
    if (pendingIntervention) {
      const dismissed: StudyInterventionEvent = {
        ...pendingIntervention,
        outcome: 'dismissed',
      };
      setInterventions((prev) => [...prev, dismissed]);
    }
    setModalOpen(false);
    setPendingIntervention(null);
  };

  if (missingSetup) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-lg w-full text-center space-y-3">
          <h2 className="text-xl font-semibold text-gray-800">Missing study setup</h2>
          <p className="text-gray-600">Start from Study Setup to initialize participant/session assignment.</p>
          <button
            onClick={() => navigate('/study/setup')}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
          >
            Go to Study Setup
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-800">Study Session {assignment.sessionNumber}</h1>
            <p className="text-sm text-gray-600">
              Participant {participantId} • {assignment.condition} • Form {assignment.form}
            </p>
          </div>
          <div className="text-right text-sm text-gray-600">
            <div>Phase: {phase}</div>
            <div>Elapsed: {totalSessionSeconds}s</div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {phase === 'baseline' && (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center space-y-3">
              <div className="text-5xl">🧘</div>
              <h2 className="text-2xl font-semibold text-gray-800">Baseline Calibration</h2>
              <p className="text-gray-600">Look at the screen naturally while baseline normalization is collected.</p>
              <p className="text-lg text-blue-700 font-medium">
                {Math.max(0, plan.baselineSeconds - phaseSeconds)}s remaining
              </p>
            </div>
          )}

          {phase === 'learn_easy' && (
            <PairedAssociateLearningBlock
              key={`learn_easy_${pacingOffsetSeconds}`}
              items={easyItems}
              exposureSeconds={plan.easyExposureSeconds + pacingOffsetSeconds}
              blockIndex={1}
              phase="learning_easy"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              onComplete={(blockTrials, elapsed) => completeLearningBlock(blockTrials, elapsed, 'test_easy_recognition')}
            />
          )}

          {phase === 'test_easy_recognition' && (
            <RecognitionTest
              key="test_easy_recognition"
              items={easyItems}
              blockIndex={1}
              phase="test_easy_recognition"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              participantSeed={`${participantId}:easy`}
              choiceCount={plan.recognitionChoices}
              useInterferenceDistractors={false}
              onComplete={(blockTrials) => completeRecognitionBlock(blockTrials, 'test_easy_cued')}
            />
          )}

          {phase === 'test_easy_cued' && (
            <CuedRecallTest
              key="test_easy_cued"
              items={easyItems}
              blockIndex={1}
              phase="test_easy_cued_recall"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              onComplete={(blockTrials) => completeCuedBlock(blockTrials, 1, 'easy', 'learn_hard')}
            />
          )}

          {phase === 'learn_hard' && (
            <PairedAssociateLearningBlock
              key={`learn_hard_${pacingOffsetSeconds}`}
              items={hardItems}
              exposureSeconds={plan.hardExposureSeconds + pacingOffsetSeconds}
              blockIndex={2}
              phase="learning_hard"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              onComplete={(blockTrials, elapsed) => completeLearningBlock(blockTrials, elapsed, 'test_hard_recognition')}
            />
          )}

          {phase === 'test_hard_recognition' && (
            <RecognitionTest
              key="test_hard_recognition"
              items={hardItems}
              blockIndex={2}
              phase="test_hard_recognition"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              participantSeed={`${participantId}:hard`}
              choiceCount={plan.recognitionChoices}
              useInterferenceDistractors={!hardInterferenceReduced}
              onComplete={(blockTrials) => completeRecognitionBlock(blockTrials, 'test_hard_cued')}
            />
          )}

          {phase === 'test_hard_cued' && (
            <CuedRecallTest
              key="test_hard_cued"
              items={hardItems}
              blockIndex={2}
              phase="test_hard_cued_recall"
              condition={assignment.condition}
              form={assignment.form}
              sessionStartMs={sessionStartMsRef.current}
              onComplete={(blockTrials) => completeCuedBlock(blockTrials, 2, 'hard', 'nasa_tlx')}
            />
          )}

          {phase === 'break' && (
            <div className="bg-white rounded-lg border border-yellow-200 p-8 text-center space-y-3">
              <div className="text-5xl">☕</div>
              <h2 className="text-2xl font-semibold text-gray-800">Micro-break</h2>
              <p className="text-gray-600">Adaptive break in progress. Resume occurs automatically.</p>
              <p className="text-lg text-yellow-700 font-medium">
                {Math.max(0, plan.microBreakSeconds - phaseSeconds)}s remaining
              </p>
            </div>
          )}

          {phase === 'nasa_tlx' && (
            <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
              <h2 className="text-xl font-semibold text-gray-800">Session complete</h2>
              <p className="text-gray-600">
                Proceed to summary to submit NASA-TLX and export study artifacts.
              </p>
              <button
                onClick={finishSession}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
              >
                Continue to Study Summary
              </button>
              <button
                onClick={() => setNasaPreviewOpen((prev) => !prev)}
                className="ml-3 text-sm text-blue-700 hover:text-blue-900 underline"
              >
                {nasaPreviewOpen ? 'Hide' : 'Preview'} NASA-TLX form
              </button>
              {nasaPreviewOpen && (
                <div className="pt-3 border-t border-gray-100">
                  <NasaTLXForm
                    title="NASA-TLX Preview"
                    submitLabel="Preview Only"
                    onSubmit={() => {
                      alert('NASA-TLX will be saved in the Study Summary step.');
                    }}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-medium text-gray-600 mb-2">Camera</h3>
            <WebcamFeed isActive={phase !== 'complete'} onFrameFeatures={handleFrameFeatures} showOverlay={false} />
          </div>

          <LiveFeaturePanel
            frameFeatures={currentFrameFeatures}
            windowFeatures={currentWindowFeatures}
            blinkCount={totalBlinkCount}
            bufferFill={windowBufferRef.current.fillRatio}
          />

          <div className="bg-white rounded-lg border border-gray-200 p-4 text-sm text-gray-700 space-y-2">
            <div className="flex justify-between">
              <span>Backend</span>
              <span>{backendStatus}</span>
            </div>
            <div className="flex justify-between">
              <span>CLI</span>
              <span>{currentLoad.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>Calibrated</span>
              <span>{isCalibrated ? 'yes' : 'no'}</span>
            </div>
            <div className="flex justify-between">
              <span>Low confidence mode</span>
              <span>{lowConfidenceMode ? 'on' : 'off'}</span>
            </div>
            <div className="flex justify-between">
              <span>Pacing offset</span>
              <span>+{pacingOffsetSeconds.toFixed(0)}s</span>
            </div>
            <div className="flex justify-between">
              <span>Hard interference reduced</span>
              <span>{hardInterferenceReduced ? 'yes' : 'no'}</span>
            </div>
            <div className="flex justify-between">
              <span>Interventions logged</span>
              <span>{interventions.length}</span>
            </div>
            <div className="flex justify-between">
              <span>CLI samples</span>
              <span>{cliSamples.length}</span>
            </div>
          </div>
        </div>
      </div>

      <StudyInterventionModal
        open={modalOpen}
        type={pendingIntervention?.type ?? null}
        details={pendingIntervention?.details}
        onAccept={onModalAccept}
        onDismiss={onModalDismiss}
      />
    </div>
  );
};

export default StudySession;
