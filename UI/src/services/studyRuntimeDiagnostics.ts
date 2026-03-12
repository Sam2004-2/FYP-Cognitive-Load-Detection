import { StudyCliSample, StudyInterventionEvent, StudyPhaseTag, StudyRuntimeDiagnostics } from '../types/study';

const LEARNING_PHASES: StudyPhaseTag[] = ['learning_easy', 'learning_hard'];

function isLearningPhaseTag(phase: StudyPhaseTag): boolean {
  return LEARNING_PHASES.includes(phase);
}

export function computeSessionRuntimeDiagnostics(
  cliSamples: StudyCliSample[],
  interventions: StudyInterventionEvent[]
): StudyRuntimeDiagnostics {
  const phaseCounts: Record<string, number> = {};
  for (const sample of cliSamples) {
    phaseCounts[sample.phase] = (phaseCounts[sample.phase] ?? 0) + 1;
  }

  const uniquePhases = Object.keys(phaseCounts) as StudyPhaseTag[];
  const learningPhaseSampleCount = cliSamples.filter((sample) => isLearningPhaseTag(sample.phase)).length;
  const onlyBaselinePhase =
    uniquePhases.length === 1 && uniquePhases[0] === 'baseline_calibration';

  const phaseIntegrityOk =
    uniquePhases.length > 1 && !onlyBaselinePhase && learningPhaseSampleCount > 0;

  const adaptiveTriggerCount = interventions.filter(
    (event) => event.outcome === 'applied' && event.type !== 'low_confidence_pause'
  ).length;
  const adaptiveSuppressionCount = interventions.filter(
    (event) => event.outcome === 'suppressed'
  ).length;
  const lowConfidencePauseCount = interventions.filter(
    (event) => event.outcome === 'paused'
  ).length;

  const notes: string[] = [];
  if (onlyBaselinePhase) {
    notes.push('All CLI samples were tagged as baseline_calibration.');
  }
  if (learningPhaseSampleCount === 0) {
    notes.push('No learning-phase samples were captured.');
  }

  return {
    phaseIntegrityOk,
    phaseCounts,
    uniquePhases,
    learningPhaseSampleCount,
    adaptiveTriggerCount,
    adaptiveSuppressionCount,
    lowConfidencePauseCount,
    notes: notes.length > 0 ? notes : undefined,
  };
}
