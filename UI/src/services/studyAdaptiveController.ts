import { STUDY_CONFIG, STUDY_QUALITY_CONFIG } from '../config/studyConfig';
import { StudyCondition, StudyInterventionEvent, StudyInterventionType, StudyPhaseTag } from '../types/study';

interface CliInputSample {
  timestampMs: number;
  cli: number;
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
}

export interface AdaptiveDecision {
  event?: StudyInterventionEvent;
  actionType?: StudyInterventionType;
  state: AdaptiveControllerState;
}

interface DecisionWindowAggregate {
  cli: number[];
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

export class StudyAdaptiveController {
  private decisionBucketStartMs: number | null = null;
  private bucket: DecisionWindowAggregate | null = null;
  private rollingDecisionMeans: number[] = [];
  private consecutiveOverload = 0;
  private consecutiveLowQuality = 0;
  private lowConfidenceMode = false;
  private microBreakCount = 0;
  private pacingOffsetSeconds = 0;
  private difficultySteppedDown = false;
  private lastInterventionMs: number | null = null;

  private readonly decisionWindowMs = STUDY_CONFIG.decisionWindowSeconds * 1000;

  reset(): void {
    this.decisionBucketStartMs = null;
    this.bucket = null;
    this.rollingDecisionMeans = [];
    this.consecutiveOverload = 0;
    this.consecutiveLowQuality = 0;
    this.lowConfidenceMode = false;
    this.microBreakCount = 0;
    this.pacingOffsetSeconds = 0;
    this.difficultySteppedDown = false;
    this.lastInterventionMs = null;
  }

  getState(): AdaptiveControllerState {
    return {
      lowConfidenceMode: this.lowConfidenceMode,
      pacingOffsetSeconds: this.pacingOffsetSeconds,
      difficultySteppedDown: this.difficultySteppedDown,
      microBreakCount: this.microBreakCount,
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

    const decisionMean = finalized.cli;
    this.rollingDecisionMeans.push(decisionMean);
    if (this.rollingDecisionMeans.length > STUDY_CONFIG.smoothingWindows) {
      this.rollingDecisionMeans.shift();
    }
    const smoothedCli = mean(this.rollingDecisionMeans);

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

    if (enteringLowConfidence) {
      return {
        event: this.makeEvent('low_confidence_pause', 'paused', finalized, smoothedCli, 'Adaptation paused due to low signal quality.'),
        actionType: 'low_confidence_pause',
        state: this.getState(),
      };
    }

    if (this.lowConfidenceMode) {
      return { state: this.getState() };
    }

    if (smoothedCli > STUDY_CONFIG.overloadThreshold) {
      this.consecutiveOverload += 1;
    } else {
      this.consecutiveOverload = 0;
      return { state: this.getState() };
    }

    if (this.consecutiveOverload < 2) {
      return { state: this.getState() };
    }

    const onCooldown =
      this.lastInterventionMs !== null &&
      finalized.timestampMs - this.lastInterventionMs < STUDY_CONFIG.adaptationCooldownSeconds * 1000;

    if (onCooldown) {
      return { state: this.getState() };
    }

    if (condition === 'baseline') {
      this.lastInterventionMs = finalized.timestampMs;
      return {
        event: this.makeEvent(
          'suppressed_trigger',
          'suppressed',
          finalized,
          smoothedCli,
          'Overload trigger detected but suppressed in baseline condition.'
        ),
        actionType: 'suppressed_trigger',
        state: this.getState(),
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
      event: this.makeEvent(actionType, 'applied', finalized, smoothedCli, details),
      actionType,
      state: this.getState(),
    };
  }

  private addToBucket(sample: CliInputSample): void {
    if (this.bucket === null || this.decisionBucketStartMs === null) {
      this.decisionBucketStartMs = sample.timestampMs;
      this.bucket = {
        cli: [],
        validFrameRatio: [],
        illuminationStd: [],
        timestampMs: sample.timestampMs,
        sessionTimeS: sample.sessionTimeS,
        phase: sample.phase,
      };
    }

    this.bucket.cli.push(sample.cli);
    this.bucket.validFrameRatio.push(sample.validFrameRatio);
    this.bucket.illuminationStd.push(sample.illuminationStd);
    this.bucket.timestampMs = sample.timestampMs;
    this.bucket.sessionTimeS = sample.sessionTimeS;
    this.bucket.phase = sample.phase;
  }

  private finalizeBucket():
    | {
        cli: number;
        validFrameRatio: number;
        illuminationStd: number;
        timestampMs: number;
        sessionTimeS: number;
        phase: StudyPhaseTag;
      }
    | null {
    if (!this.bucket) return null;

    const out = {
      cli: mean(this.bucket.cli),
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
      cli: number;
      validFrameRatio: number;
      illuminationStd: number;
      timestampMs: number;
      sessionTimeS: number;
      phase: StudyPhaseTag;
    },
    smoothedCli: number,
    details: string
  ): StudyInterventionEvent {
    return {
      timestampMs: finalized.timestampMs,
      sessionTimeS: finalized.sessionTimeS,
      phase: finalized.phase,
      type,
      outcome,
      cli: finalized.cli,
      smoothedCli,
      validFrameRatio: finalized.validFrameRatio,
      details,
    };
  }
}
