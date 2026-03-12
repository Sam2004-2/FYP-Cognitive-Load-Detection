import { STUDY_CONFIG, STUDY_QUALITY_CONFIG } from '../config/studyConfig';
import {
  StudyAdaptiveMode,
  StudyCondition,
  StudyInterventionEvent,
  StudyInterventionType,
  StudyPhaseTag,
} from '../types/study';

interface CliInputSample {
  timestampMs: number;
  rawCli: number;
  smoothedCli: number;
  validFrameRatio: number;
  illuminationStd: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
}

export interface AdaptiveControllerState {
  lowConfidenceMode: boolean;
  pacingOffsetSeconds: number;
  difficultySteppedDown: boolean;
  microBreakCount: number;
  onCooldown: boolean;
}

export interface AdaptiveDecisionMetrics {
  decisionMode: StudyAdaptiveMode;
  decisionCli?: number;
  decisionThreshold?: number;
  onCooldown: boolean;
}

export interface AdaptiveDecision {
  event?: StudyInterventionEvent;
  actionType?: StudyInterventionType;
  state: AdaptiveControllerState;
  metrics?: AdaptiveDecisionMetrics;
}

interface DecisionWindowAggregate {
  rawCli: number[];
  smoothedCli: number[];
  validFrameRatio: number[];
  illuminationStd: number[];
  timestampMs: number;
  sessionTimeS: number;
  phase: StudyPhaseTag;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const mu = mean(values);
  const variance = values.reduce((sum, v) => sum + (v - mu) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

export class StudyAdaptiveController {
  private decisionBucketStartMs: number | null = null;
  private bucket: DecisionWindowAggregate | null = null;
  private rollingDecisionMeans: number[] = [];
  private relativeDecisionHistory: number[] = [];
  private consecutiveOverload = 0;
  private consecutiveLowQuality = 0;
  private lowConfidenceMode = false;
  private microBreakCount = 0;
  private pacingOffsetSeconds = 0;
  private difficultySteppedDown = false;
  private lastInterventionMs: number | null = null;
  private lastDecisionTimestampMs: number | null = null;

  private readonly decisionWindowMs = STUDY_CONFIG.decisionWindowSeconds * 1000;
  private readonly relativeHistoryLimit = STUDY_CONFIG.warmupWindows + 1;

  reset(): void {
    this.decisionBucketStartMs = null;
    this.bucket = null;
    this.rollingDecisionMeans = [];
    this.relativeDecisionHistory = [];
    this.consecutiveOverload = 0;
    this.consecutiveLowQuality = 0;
    this.lowConfidenceMode = false;
    this.microBreakCount = 0;
    this.pacingOffsetSeconds = 0;
    this.difficultySteppedDown = false;
    this.lastInterventionMs = null;
    this.lastDecisionTimestampMs = null;
  }

  getState(): AdaptiveControllerState {
    return {
      lowConfidenceMode: this.lowConfidenceMode,
      pacingOffsetSeconds: this.pacingOffsetSeconds,
      difficultySteppedDown: this.difficultySteppedDown,
      microBreakCount: this.microBreakCount,
      onCooldown: this.isOnCooldown(this.lastDecisionTimestampMs ?? Date.now()),
    };
  }

  ingest(sample: CliInputSample, condition: StudyCondition): AdaptiveDecision {
    this.addToBucket(sample);

    if (this.decisionBucketStartMs === null || sample.timestampMs - this.decisionBucketStartMs < this.decisionWindowMs) {
      return { state: this.getState() };
    }

    const finalized = this.finalizeBucket();
    if (!finalized) {
      return { state: this.getState() };
    }
    this.lastDecisionTimestampMs = finalized.timestampMs;

    const decisionMean = finalized.smoothedCli;
    this.rollingDecisionMeans.push(decisionMean);
    if (this.rollingDecisionMeans.length > STUDY_CONFIG.smoothingWindows) {
      this.rollingDecisionMeans.shift();
    }
    const smoothedDecisionCli = mean(this.rollingDecisionMeans);

    const isLowQuality =
      finalized.validFrameRatio < STUDY_QUALITY_CONFIG.validFrameRatioMin ||
      finalized.illuminationStd > STUDY_QUALITY_CONFIG.illuminationStdMax;

    if (isLowQuality) {
      this.consecutiveLowQuality += 1;
    } else {
      this.consecutiveLowQuality = 0;
    }

    const enteringLowConfidence = !this.lowConfidenceMode && this.consecutiveLowQuality >= 2;
    this.lowConfidenceMode = this.consecutiveLowQuality >= 2;

    const metrics = this.computeDecisionMetrics(smoothedDecisionCli);
    const onCooldown = this.isOnCooldown(finalized.timestampMs);
    metrics.onCooldown = onCooldown;

    if (enteringLowConfidence) {
      this.consecutiveOverload = 0;
      return {
        event: this.makeEvent(
          'low_confidence_pause',
          'paused',
          finalized,
          'Adaptation paused due to low signal quality.'
        ),
        actionType: 'low_confidence_pause',
        state: this.getState(),
        metrics,
      };
    }

    if (this.lowConfidenceMode) {
      return { state: this.getState(), metrics };
    }

    const overloadDetected =
      metrics.decisionCli !== undefined &&
      metrics.decisionThreshold !== undefined &&
      metrics.decisionCli >= metrics.decisionThreshold;

    if (overloadDetected) {
      this.consecutiveOverload += 1;
    } else {
      this.consecutiveOverload = 0;
      return { state: this.getState(), metrics };
    }

    if (this.consecutiveOverload < 2) {
      return { state: this.getState(), metrics };
    }

    if (onCooldown) {
      return { state: this.getState(), metrics };
    }

    if (condition === 'baseline') {
      this.lastInterventionMs = finalized.timestampMs;
      return {
        event: this.makeEvent(
          'suppressed_trigger',
          'suppressed',
          finalized,
          'Overload trigger detected but suppressed in baseline condition.'
        ),
        actionType: 'suppressed_trigger',
        state: this.getState(),
        metrics,
      };
    }

    let actionType: StudyInterventionType;
    let details = '';

    if (this.microBreakCount < STUDY_CONFIG.maxMicroBreaksPerSession) {
      this.microBreakCount += 1;
      actionType = 'micro_break_60s';
      details = 'Prompting a 60 second micro-break due to sustained overload.';
    } else if (this.pacingOffsetSeconds < 2) {
      this.pacingOffsetSeconds = clamp(this.pacingOffsetSeconds + 1, 0, 2);
      actionType = 'pacing_change';
      details = `Increased item exposure by +${this.pacingOffsetSeconds.toFixed(0)}s cumulative.`;
    } else {
      this.difficultySteppedDown = true;
      actionType = 'difficulty_step_down';
      details = 'Stepped down hard-block interference profile for remaining items.';
    }

    this.lastInterventionMs = finalized.timestampMs;

    return {
      event: this.makeEvent(actionType, 'applied', finalized, details),
      actionType,
      state: this.getState(),
      metrics,
    };
  }

  private addToBucket(sample: CliInputSample): void {
    if (this.bucket === null || this.decisionBucketStartMs === null) {
      this.decisionBucketStartMs = sample.timestampMs;
      this.bucket = {
        rawCli: [],
        smoothedCli: [],
        validFrameRatio: [],
        illuminationStd: [],
        timestampMs: sample.timestampMs,
        sessionTimeS: sample.sessionTimeS,
        phase: sample.phase,
      };
    }

    this.bucket.rawCli.push(sample.rawCli);
    this.bucket.smoothedCli.push(sample.smoothedCli);
    this.bucket.validFrameRatio.push(sample.validFrameRatio);
    this.bucket.illuminationStd.push(sample.illuminationStd);
    this.bucket.timestampMs = sample.timestampMs;
    this.bucket.sessionTimeS = sample.sessionTimeS;
    this.bucket.phase = sample.phase;
  }

  private finalizeBucket():
    | {
        rawCli: number;
        smoothedCli: number;
        validFrameRatio: number;
        illuminationStd: number;
        timestampMs: number;
        sessionTimeS: number;
        phase: StudyPhaseTag;
      }
    | null {
    if (!this.bucket) return null;

    const out = {
      rawCli: mean(this.bucket.rawCli),
      smoothedCli: mean(this.bucket.smoothedCli),
      validFrameRatio: mean(this.bucket.validFrameRatio),
      illuminationStd: mean(this.bucket.illuminationStd),
      timestampMs: this.bucket.timestampMs,
      sessionTimeS: this.bucket.sessionTimeS,
      phase: this.bucket.phase,
    };

    this.bucket = null;
    this.decisionBucketStartMs = out.timestampMs;

    return out;
  }

  private makeEvent(
    type: StudyInterventionType,
    outcome: 'applied' | 'suppressed' | 'paused',
    finalized: {
      rawCli: number;
      smoothedCli: number;
      validFrameRatio: number;
      illuminationStd: number;
      timestampMs: number;
      sessionTimeS: number;
      phase: StudyPhaseTag;
    },
    details: string
  ): StudyInterventionEvent {
    return {
      timestampMs: finalized.timestampMs,
      sessionTimeS: finalized.sessionTimeS,
      phase: finalized.phase,
      type,
      outcome,
      cli: finalized.rawCli,
      smoothedCli: finalized.smoothedCli,
      validFrameRatio: finalized.validFrameRatio,
      details,
    };
  }

  private computeDecisionMetrics(smoothedDecisionCli: number): AdaptiveDecisionMetrics {
    const adaptiveMode = STUDY_CONFIG.adaptiveMode;
    if (adaptiveMode === 'absolute') {
      return {
        decisionMode: 'absolute',
        decisionCli: smoothedDecisionCli,
        decisionThreshold: this.getAbsoluteThreshold(),
        onCooldown: false,
      };
    }

    const historyWindow = this.relativeDecisionHistory.slice(-STUDY_CONFIG.warmupWindows);
    this.relativeDecisionHistory.push(smoothedDecisionCli);
    if (this.relativeDecisionHistory.length > this.relativeHistoryLimit) {
      this.relativeDecisionHistory.shift();
    }

    if (historyWindow.length < STUDY_CONFIG.warmupWindows) {
      return {
        decisionMode: 'relative',
        decisionThreshold: STUDY_CONFIG.relativeZThreshold,
        onCooldown: false,
      };
    }

    const rollingMean = mean(historyWindow);
    const rollingStd = stdDev(historyWindow);
    if (rollingStd < STUDY_CONFIG.minStdEpsilon) {
      return {
        decisionMode: 'absolute',
        decisionCli: smoothedDecisionCli,
        decisionThreshold: this.getAbsoluteThreshold(),
        onCooldown: false,
      };
    }

    return {
      decisionMode: 'relative',
      decisionCli: (smoothedDecisionCli - rollingMean) / rollingStd,
      decisionThreshold: STUDY_CONFIG.relativeZThreshold,
      onCooldown: false,
    };
  }

  private getAbsoluteThreshold(): number {
    return Number.isFinite(STUDY_CONFIG.absoluteThreshold)
      ? STUDY_CONFIG.absoluteThreshold
      : STUDY_CONFIG.overloadThreshold ?? 0.7;
  }

  private isOnCooldown(timestampMs: number): boolean {
    return (
      this.lastInterventionMs !== null &&
      timestampMs - this.lastInterventionMs < STUDY_CONFIG.adaptationCooldownSeconds * 1000
    );
  }
}
