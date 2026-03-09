import { StudySessionPlan } from '../types/study';

export const STUDY_CONFIG: StudySessionPlan = {
  baselineSeconds: 45,
  easyItemCount: 8,
  easyExposureSeconds: 4.5,
  hardItemCount: 10,
  hardExposureSeconds: 3.0,
  hardInterferenceEnabled: true,
  recognitionChoices: 4,
  microBreakSeconds: 45,
  maxMicroBreaksPerSession: 1,
  adaptationCooldownSeconds: 120,
  decisionWindowSeconds: 5,
  smoothingWindows: 3,
  adaptiveMode: 'relative',
  absoluteThreshold: 0.45,
  relativeZThreshold: 1.0,
  warmupWindows: 4,
  minStdEpsilon: 0.02,
  overloadThreshold: 0.7,
};

export const STUDY_QUALITY_CONFIG = {
  validFrameRatioMin: 0.95,
  illuminationStdMax: 28,
};

export const STUDY_STORAGE_KEYS = {
  sessionDraftPrefix: 'cle_study_session_draft_',
  sessionFinalPrefix: 'cle_study_session_final_',
  delayedPrefix: 'cle_study_delayed_',
};

export const STUDY_RECORD_VERSION = 2;
