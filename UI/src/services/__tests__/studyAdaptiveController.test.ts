import { StudyAdaptiveController } from '../studyAdaptiveController';

function sample(timestampMs: number, cli = 0.85, vfr = 0.99, illum = 10) {
  return {
    timestampMs,
    cli,
    validFrameRatio: vfr,
    illuminationStd: illum,
    sessionTimeS: timestampMs / 1000,
    phase: 'learning_easy' as const,
  };
}

describe('StudyAdaptiveController', () => {
  it('triggers micro-break after sustained overload in adaptive condition', () => {
    const controller = new StudyAdaptiveController();
    let eventType: string | undefined;

    // Feed enough samples to finalize multiple decision windows
    for (let i = 0; i < 8; i += 1) {
      const decision = controller.ingest(sample(i * 5000), 'adaptive');
      if (decision.actionType) {
        eventType = decision.actionType;
      }
    }

    expect(eventType).toBe('micro_break_60s');
  });

  it('logs suppressed triggers in baseline condition', () => {
    const controller = new StudyAdaptiveController();
    let suppressedSeen = false;

    for (let i = 0; i < 8; i += 1) {
      const decision = controller.ingest(sample(i * 5000), 'baseline');
      if (decision.actionType === 'suppressed_trigger') {
        suppressedSeen = true;
      }
    }

    expect(suppressedSeen).toBe(true);
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
