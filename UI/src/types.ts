export interface CognitiveLoadData {
  timestamp: number;
  load: number; // 0-1 range
}

export interface SessionData {
  duration: number; // seconds
  loadHistory: CognitiveLoadData[];
  interventionCount: number;
}

export interface Settings {
  interventionFrequency: 'off' | 'low' | 'medium' | 'high';
  breakInterval: 15 | 25 | 45;
}

export interface NASATLXScores {
  mentalDemand: number;
  physicalDemand: number;
  temporalDemand: number;
  performance: number;
  effort: number;
  frustration: number;
}

export type LoadLevel = 'low' | 'medium' | 'high';

// ============================================================================
// Pilot Study Types
// ============================================================================

export type StudyCondition = 'adaptive' | 'baseline';
export type StudyPhase = 
  | 'consent'
  | 'setup'
  | 'calibration'
  | 'learning_easy'
  | 'learning_hard'
  | 'immediate_test'
  | 'nasa_tlx'
  | 'complete';

export interface CalibrationData {
  baseline_cli: number;
  baseline_ear: number;
  duration_s: number;
}

export interface CLIDataPoint {
  t: number;
  cli: number;
  confidence: number;
}

export interface InterventionLog {
  t: number;
  cli: number;
  type: 'micro_break' | 'pacing_adjustment';
  accepted: boolean;
}

export interface TaskPerformance {
  correct: number;
  total: number;
  rt_mean_ms?: number;
  responses?: Array<{
    cue: string;
    expected: string;
    given: string;
    correct: boolean;
    rt_ms: number;
  }>;
}

export interface StudyNASATLX {
  mental: number;
  physical: number;
  temporal: number;
  performance: number;
  effort: number;
  frustration: number;
  raw_tlx: number;
}

export interface StudySession {
  participant_id: string;
  session_number: 1 | 2;
  condition: StudyCondition;
  timestamp: string;
  form_version: 'A' | 'B';
  calibration: CalibrationData;
  cli_timeseries: CLIDataPoint[];
  interventions: InterventionLog[];
  task_performance: {
    easy: TaskPerformance;
    hard: TaskPerformance;
  };
  nasa_tlx: StudyNASATLX;
  immediate_test: TaskPerformance;
  delayed_test: TaskPerformance | null;
}

export interface WordPair {
  cue: string;
  target: string;
}

export interface PairedAssociatesConfig {
  pairs: WordPair[];
  exposureTime: number; // ms per pair
  difficulty: 'easy' | 'hard';
}

