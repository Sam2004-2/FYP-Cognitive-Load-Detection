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

