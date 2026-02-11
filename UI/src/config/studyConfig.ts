import { StudySessionPlan } from '../types/study';

export const STUDY_CONFIG: StudySessionPlan = {
  baselineSeconds: 60,
  easyItemCount: 12,
  easyExposureSeconds: 5.5,
  hardItemCount: 18,
  hardExposureSeconds: 3.5,
  hardInterferenceEnabled: true,
  recognitionChoices: 4,
  microBreakSeconds: 60,
  maxMicroBreaksPerSession: 2,
  adaptationCooldownSeconds: 120,
  decisionWindowSeconds: 5,
  smoothingWindows: 3,
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

export const STUDY_RECORD_VERSION = 1;
