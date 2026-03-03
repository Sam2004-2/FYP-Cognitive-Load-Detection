import { computeSessionRuntimeDiagnostics } from '../studyRuntimeDiagnostics';
import { StudyCliSample, StudyInterventionEvent } from '../../types/study';

function sample(overrides: Partial<StudyCliSample>): StudyCliSample {
  return {
    timestampMs: 0,
    sessionTimeS: 0,
    phase: 'baseline_calibration',
    rawCli: 0.5,
    smoothedCli: 0.5,
    validFrameRatio: 0.99,
    illuminationStd: 10,
    qualityFlags: {
      lowValidFrameRatio: false,
      unstableIllumination: false,
    },
    ...overrides,
  };
}

function intervention(overrides: Partial<StudyInterventionEvent>): StudyInterventionEvent {
  return {
    timestampMs: 0,
    sessionTimeS: 0,
    phase: 'learning_easy',
    type: 'suppressed_trigger',
    outcome: 'suppressed',
    cli: 0.5,
    smoothedCli: 0.5,
    validFrameRatio: 0.99,
    ...overrides,
  };
}

describe('computeSessionRuntimeDiagnostics', () => {
  it('marks integrity true when multiple phases and learning samples are present', () => {
    const diagnostics = computeSessionRuntimeDiagnostics(
      [
        sample({ phase: 'baseline_calibration' }),
        sample({ phase: 'learning_easy' }),
        sample({ phase: 'test_easy_recognition' }),
      ],
      [intervention({ type: 'micro_break_60s', outcome: 'applied' })]
    );

    expect(diagnostics.phaseIntegrityOk).toBe(true);
    expect(diagnostics.phaseCounts.learning_easy).toBe(1);
    expect(diagnostics.learningPhaseSampleCount).toBe(1);
    expect(diagnostics.adaptiveTriggerCount).toBe(1);
    expect(diagnostics.adaptiveSuppressionCount).toBe(0);
    expect(diagnostics.lowConfidencePauseCount).toBe(0);
  });

  it('marks integrity false when all samples remain baseline calibration', () => {
    const diagnostics = computeSessionRuntimeDiagnostics(
      [sample({ phase: 'baseline_calibration' }), sample({ phase: 'baseline_calibration' })],
      [intervention({ type: 'suppressed_trigger', outcome: 'suppressed' })]
    );

    expect(diagnostics.phaseIntegrityOk).toBe(false);
    expect(diagnostics.uniquePhases).toEqual(['baseline_calibration']);
    expect(diagnostics.learningPhaseSampleCount).toBe(0);
    expect(diagnostics.notes?.join(' ')).toContain('baseline_calibration');
    expect(diagnostics.adaptiveSuppressionCount).toBe(1);
    expect(diagnostics.lowConfidencePauseCount).toBe(0);
  });

  it('counts paused and suppressed outcomes separately', () => {
    const diagnostics = computeSessionRuntimeDiagnostics(
      [sample({ phase: 'learning_easy' })],
      [
        intervention({ type: 'suppressed_trigger', outcome: 'suppressed' }),
        intervention({ type: 'low_confidence_pause', outcome: 'paused' }),
        intervention({ type: 'low_confidence_pause', outcome: 'paused' }),
      ]
    );

    expect(diagnostics.adaptiveSuppressionCount).toBe(1);
    expect(diagnostics.lowConfidencePauseCount).toBe(2);
  });
});
