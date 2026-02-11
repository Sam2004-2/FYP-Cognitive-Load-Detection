import { NASATLXScores } from '../types';

export type StudyCondition = 'adaptive' | 'baseline';
export type StudySessionNumber = 1 | 2;
export type StudyForm = 'A' | 'B';

export type StudyPhaseTag =
  | 'baseline_calibration'
  | 'learning_easy'
  | 'test_easy_recognition'
  | 'test_easy_cued_recall'
  | 'learning_hard'
  | 'test_hard_recognition'
  | 'test_hard_cued_recall'
  | 'break'
  | 'nasa_tlx'
  | 'complete';

export type StudyInterventionType =
  | 'micro_break_60s'
  | 'pacing_change'
  | 'difficulty_step_down'
  | 'suppressed_trigger'
  | 'low_confidence_pause';

export type StudyInterventionOutcome =
  | 'applied'
  | 'dismissed'
  | 'auto'
  | 'suppressed'
  | 'paused';

export type StudyTrialKind = 'learning' | 'recognition' | 'cued_recall';
export type StudyDifficulty = 'easy' | 'hard';

export interface StudyAssignment {
  participantId: string;
  participantIdNormalized: string;
  hashValue: number;
  hashParity: 'even' | 'odd';
  conditionOrder: [StudyCondition, StudyCondition];
  formOrder: [StudyForm, StudyForm];
  sessionNumber: StudySessionNumber;
  condition: StudyCondition;
  form: StudyForm;
  delayedDueAtIso: string;
}

export interface StudySessionPlan {
  baselineSeconds: number;
  easyItemCount: number;
  easyExposureSeconds: number;
  hardItemCount: number;
  hardExposureSeconds: number;
  hardInterferenceEnabled: boolean;
  recognitionChoices: number;
  microBreakSeconds: number;
  maxMicroBreaksPerSession: number;
  adaptationCooldownSeconds: number;
  decisionWindowSeconds: number;
  smoothingWindows: number;
  overloadThreshold: number;
}

export interface StudyCliQualityFlags {
  lowConfidence: boolean;
  lowValidFrameRatio: boolean;
  unstableIllumination: boolean;
}

export interface StudyCliSample {
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  rawCli: number;
  smoothedCli: number;
  confidence: number;
  validFrameRatio: number;
  illuminationStd: number;
  qualityFlags: StudyCliQualityFlags;
}

export interface StudyInterventionEvent {
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  type: StudyInterventionType;
  outcome: StudyInterventionOutcome;
  cli: number;
  smoothedCli: number;
  confidence: number;
  validFrameRatio: number;
  details?: string;
}

export interface StudyStimulusItem {
  id: string;
  cue: string;
  target: string;
  difficulty: StudyDifficulty;
  interferenceGroup: string;
}

export interface StudyRecognitionChoice {
  value: string;
  isCorrect: boolean;
}

export interface StudyTrialResult {
  trialId: string;
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  kind: StudyTrialKind;
  difficulty: StudyDifficulty;
  blockIndex: 1 | 2;
  itemId: string;
  cue: string;
  target: string;
  recognitionChoices?: string[];
  selectedChoice?: string;
  responseText?: string;
  correct: boolean;
  reactionTimeMs: number;
  condition: StudyCondition;
  form: StudyForm;
}

export interface StudyBlockSummary {
  blockIndex: 1 | 2;
  difficulty: StudyDifficulty;
  learningItemCount: number;
  recognitionAccuracy: number;
  recognitionMeanRtMs: number;
  cuedRecallAccuracy: number;
  cuedRecallMeanRtMs: number;
  effectiveExposureSeconds: number;
  adaptationApplied: boolean;
}

export interface StudySessionRecord {
  recordVersion: number;
  recordId: string;
  participantId: string;
  sessionNumber: StudySessionNumber;
  assignment: StudyAssignment;
  plan: StudySessionPlan;
  startedAtIso: string;
  completedAtIso?: string;
  session2StartedEarlyOverride: boolean;
  condition: StudyCondition;
  form: StudyForm;
  totalSessionSeconds: number;
  activeTaskSeconds: number;
  breakSeconds: number;
  cliSamples: StudyCliSample[];
  interventions: StudyInterventionEvent[];
  trials: StudyTrialResult[];
  blockSummaries: StudyBlockSummary[];
  nasaTlx?: NASATLXScores;
  pendingDelayedTest: boolean;
  delayedDueAtIso: string;
}

export interface StudyDelayedTestRecord {
  recordVersion: number;
  recordId: string;
  linkedSessionRecordId: string;
  participantId: string;
  sessionNumber: StudySessionNumber;
  condition: StudyCondition;
  form: StudyForm;
  dueAtIso: string;
  completedAtIso?: string;
  trials: StudyTrialResult[];
  recognitionAccuracy: number;
  recognitionMeanRtMs: number;
  cuedRecallAccuracy: number;
  cuedRecallMeanRtMs: number;
}

export interface StudySetupState {
  participantId: string;
  assignment: StudyAssignment;
  plan: StudySessionPlan;
  session2StartedEarlyOverride: boolean;
}

export interface StudySummaryState {
  record: StudySessionRecord;
}
