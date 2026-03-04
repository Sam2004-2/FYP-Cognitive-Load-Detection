import { NASATLXScores } from '../types';

export type StudyCondition = 'adaptive' | 'baseline';
export type StudySessionNumber = 1 | 2;
export type StudyForm = 'A' | 'B';
export type StudyAdaptiveMode = 'absolute' | 'relative';

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
  adaptiveMode: StudyAdaptiveMode;
  absoluteThreshold: number;
  relativeZThreshold: number;
  warmupWindows: number;
  minStdEpsilon: number;
  overloadThreshold?: number;
}

export interface StudyCliQualityFlags {
  lowValidFrameRatio: boolean;
  unstableIllumination: boolean;
}

export interface StudyCliSample {
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  rawCli: number;
  smoothedCli: number;
  /** In absolute mode: smoothed CLI (0-1). In relative mode: z-score (unbounded). Interpret via decisionMode. */
  decisionCli?: number;
  decisionThreshold?: number;
  decisionMode?: StudyAdaptiveMode;
  validFrameRatio: number;
  illuminationStd: number;
  qualityFlags: StudyCliQualityFlags;
}

export interface StudyFeatureWindow {
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  windowIndex: number;
  isCalibration: boolean;
  features: Record<string, number>;
}

export interface StudyInterventionEvent {
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
  type: StudyInterventionType;
  outcome: StudyInterventionOutcome;
  cli: number;
  smoothedCli: number;
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

export interface StudyRuntimeDiagnostics {
  phaseIntegrityOk: boolean;
  phaseCounts: Record<string, number>;
  uniquePhases?: StudyPhaseTag[];
  learningPhaseSampleCount?: number;
  adaptiveTriggerCount: number;
  adaptiveSuppressionCount: number;
  lowConfidencePauseCount: number;
  notes?: string[];
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
  featureWindows: StudyFeatureWindow[];
  interventions: StudyInterventionEvent[];
  trials: StudyTrialResult[];
  blockSummaries: StudyBlockSummary[];
  runtimeDiagnostics?: StudyRuntimeDiagnostics;
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

export interface StudyParticipantIdentity {
  participantId: string;
  createdAtIso: string;
}

export interface StudySessionUploadResponse {
  success: boolean;
  recordId: string;
  storedAtIso: string;
}

export interface StudyDelayedUploadResponse {
  success: boolean;
  recordId: string;
  storedAtIso: string;
}

export interface PendingDelayedTask {
  linkedSessionRecordId: string;
  participantId: string;
  sessionNumber: StudySessionNumber;
  condition: StudyCondition;
  form: StudyForm;
  dueAtIso: string;
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
}

export interface AdminExportQuery {
  participantId?: string;
  fromIso?: string;
  toIso?: string;
  format?: 'zip' | 'json';
}

export interface StudyActivityEventInput {
  eventType: string;
  page: string;
  participantId?: string;
  visitorId?: string;
  sessionNumber?: number;
  condition?: StudyCondition;
  metadata?: Record<string, unknown>;
}

export interface AdminReportIndexRecord {
  participantId: string;
  kind: 'sessions' | 'delayed';
  recordId: string;
  eventTimeIso?: string | null;
  storedAtIso: string;
  path: string;
}

export interface AdminReportIndexResponse {
  generatedAtIso: string;
  count: number;
  records: AdminReportIndexRecord[];
}

export interface AdminMonitoringDailyUpload {
  date: string;
  sessionRecords: number;
  delayedRecords: number;
  totalRecords: number;
}

export interface AdminMonitoringRecentRecord {
  participantId: string;
  kind: 'sessions' | 'delayed';
  recordId: string;
  condition?: string;
  sessionNumber?: number;
  storedAtIso: string;
}

export interface AdminMonitoringActivityEvent {
  occurredAtIso: string;
  eventType: string;
  page: string;
  participantId?: string;
  visitorId?: string;
  sessionNumber?: number;
  condition?: string;
}

export interface AdminMonitoringActivitySummary {
  activeLast15m: number;
  activeLast60m: number;
  visitorsLast24h: number;
  pageViewsLast24h: number;
  pageViewCounts: Record<string, number>;
  recentEvents: AdminMonitoringActivityEvent[];
}

export interface AdminMonitoringSummary {
  generatedAtIso: string;
  totals: Record<string, number>;
  conditionCounts: Record<string, number>;
  interventionCounts: Record<string, number>;
  dailyUploads: AdminMonitoringDailyUpload[];
  recentRecords: AdminMonitoringRecentRecord[];
  activity: AdminMonitoringActivitySummary;
}
