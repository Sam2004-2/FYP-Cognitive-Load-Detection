import { STUDY_CONFIG } from '../../config/studyConfig';
import { StudyAdaptiveController } from '../studyAdaptiveController';

function sample(timestampMs: number, cli = 0.85, vfr = 0.99, illum = 10) {
  return {
    timestampMs,
    rawCli: cli,
    smoothedCli: cli,
    validFrameRatio: vfr,
    illuminationStd: illum,
    sessionTimeS: timestampMs / 1000,
    phase: 'learning_easy' as const,
  };
}

describe('StudyAdaptiveController', () => {
  const originalConfig = JSON.parse(JSON.stringify(STUDY_CONFIG));

  afterEach(() => {
    Object.assign(STUDY_CONFIG, originalConfig);
  });

  it('matches absolute threshold behavior when adaptiveMode is absolute', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'absolute',
      absoluteThreshold: 0.7,
    });

    const controller = new StudyAdaptiveController();
    let eventType: string | undefined;

    for (let i = 0; i < 8; i += 1) {
      const decision = controller.ingest(sample(i * 5000, 0.85), 'adaptive');
      if (decision.actionType) {
        eventType = decision.actionType;
      }
    }

    expect(eventType).toBe('pacing_change');
  });

  it('logs suppressed trigger events in baseline condition', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'absolute',
      absoluteThreshold: 0.7,
    });

    const controller = new StudyAdaptiveController();
    let suppressedSeen = false;

    for (let i = 0; i < 8; i += 1) {
      const decision = controller.ingest(sample(i * 5000, 0.85), 'baseline');
      if (decision.actionType === 'suppressed_trigger') {
        suppressedSeen = true;
      }
    }

    expect(suppressedSeen).toBe(true);
  });

  it('does not trigger in relative mode for tightly clustered CLI values', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'relative',
      absoluteThreshold: 0.7,
      relativeZThreshold: 2.0,
      warmupWindows: 4,
      minStdEpsilon: 0.001,
      decisionWindowSeconds: 1,
    });

    const controller = new StudyAdaptiveController();
    const sequence = [0.5, 0.501, 0.499, 0.5, 0.502, 0.501, 0.5, 0.503, 0.502, 0.501, 0.5, 0.502];
    let actionSeen = false;

    sequence.forEach((value, i) => {
      const decision = controller.ingest(sample(i * 5000, value), 'adaptive');
      if (decision.actionType) {
        actionSeen = true;
      }
    });

    expect(actionSeen).toBe(false);
  });

  it('triggers in relative mode after sustained increase beyond z-threshold', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'relative',
      relativeZThreshold: 1.0,
      warmupWindows: 4,
      minStdEpsilon: 0.001,
      decisionWindowSeconds: 1,
    });

    const controller = new StudyAdaptiveController();
    const sequence = [0.5, 0.5, 0.49, 0.5, 0.5, 0.49, 0.5, 0.5, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72];
    let actionType: string | undefined;

    sequence.forEach((value, i) => {
      const decision = controller.ingest(sample(i * 5000, value), 'adaptive');
      if (decision.actionType) {
        actionType = decision.actionType;
      }
    });

    expect(actionType).toBe('pacing_change');
  });

  it('falls back to absolute decision mode when relative variance is tiny', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'relative',
      absoluteThreshold: 0.58,
      relativeZThreshold: 1.0,
      warmupWindows: 4,
      minStdEpsilon: 0.05,
      decisionWindowSeconds: 1,
    });

    const controller = new StudyAdaptiveController();
    const sequence = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.59, 0.6, 0.61, 0.62];
    const modesSeen = new Set<string>();

    sequence.forEach((value, i) => {
      const decision = controller.ingest(sample(i * 5000, value), 'adaptive');
      if (decision.metrics?.decisionMode) {
        modesSeen.add(decision.metrics.decisionMode);
      }
    });

    expect(modesSeen.has('absolute')).toBe(true);
  });

  it('reset() clears relative history so warmup is required again', () => {
    Object.assign(STUDY_CONFIG, {
      adaptiveMode: 'relative',
      relativeZThreshold: 1.0,
      warmupWindows: 4,
      minStdEpsilon: 0.001,
      decisionWindowSeconds: 1,
    });

    const controller = new StudyAdaptiveController();
    // Ingest enough samples to pass warmup
    for (let i = 0; i < 8; i += 1) {
      controller.ingest(sample(i * 5000, 0.5), 'adaptive');
    }

    controller.reset();

    // After reset, the first few decision windows should lack decisionCli (warmup)
    const decision = controller.ingest(sample(50000, 0.9), 'adaptive');
    expect(decision.metrics?.decisionCli).toBeUndefined();
  });

  it('enters low confidence mode after repeated low-quality windows', () => {
    const controller = new StudyAdaptiveController();
    let pauseSeen = false;

    for (let i = 0; i < 6; i += 1) {
      const decision = controller.ingest(sample(i * 5000, 0.7, 0.5, 50), 'adaptive');
      if (decision.actionType === 'low_confidence_pause') {
        pauseSeen = true;
      }
    }

    expect(pauseSeen).toBe(true);
    expect(controller.getState().lowConfidenceMode).toBe(true);
  });
});
