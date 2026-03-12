import { StudyDelayedTestRecord, StudySessionRecord } from '../types/study';

export interface StudyExportTables {
  session_summary: string;
  cli_windows: string;
  feature_windows: string;
  trials: string;
  arithmetic: string;
  interventions: string;
  tlx: string;
  delayed: string;
}

export interface StudyExportPackage {
  participantId: string;
  exportedAtIso: string;
  sessions: StudySessionRecord[];
  delayedTests: StudyDelayedTestRecord[];
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (/[",\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function toCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return '';
  const keys = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((k) => set.add(k));
    return set;
  }, new Set<string>()));

  const header = keys.join(',');
  const lines = rows.map((row) => keys.map((k) => csvEscape(row[k])).join(','));
  return [header, ...lines].join('\n');
}

export function buildStudyExportPackage(
  participantId: string,
  sessions: StudySessionRecord[],
  delayedTests: StudyDelayedTestRecord[]
): StudyExportPackage {
  return {
    participantId,
    exportedAtIso: new Date().toISOString(),
    sessions,
    delayedTests,
  };
}

export function buildStudyExportTables(
  sessions: StudySessionRecord[],
  delayedTests: StudyDelayedTestRecord[]
): StudyExportTables {
  const sessionSummaryRows = sessions.map((s) => ({
    record_id: s.recordId,
    participant_id: s.participantId,
    session_number: s.sessionNumber,
    condition: s.condition,
    form: s.form,
    started_at: s.startedAtIso,
    completed_at: s.completedAtIso ?? '',
    session2_early_override: s.session2StartedEarlyOverride,
    total_session_seconds: s.totalSessionSeconds,
    active_task_seconds: s.activeTaskSeconds,
    break_seconds: s.breakSeconds,
    intervention_count: s.interventions.filter((e) => e.outcome === 'applied').length,
    phase_integrity_ok: s.runtimeDiagnostics?.phaseIntegrityOk ?? '',
    learning_phase_sample_count: s.runtimeDiagnostics?.learningPhaseSampleCount ?? '',
    unique_phases: s.runtimeDiagnostics?.uniquePhases?.join('|') ?? '',
    adaptive_trigger_count: s.runtimeDiagnostics?.adaptiveTriggerCount ?? '',
    adaptive_suppression_count: s.runtimeDiagnostics?.adaptiveSuppressionCount ?? '',
    low_confidence_pause_count: s.runtimeDiagnostics?.lowConfidencePauseCount ?? '',
    arithmetic_scored_count: s.arithmeticChallenge?.totalScoredCount ?? '',
    arithmetic_accuracy: s.arithmeticChallenge?.overallAccuracy ?? '',
    arithmetic_mean_rt_ms: s.arithmeticChallenge?.overallMeanRtMs ?? '',
    arithmetic_timeout_count: s.arithmeticChallenge?.totalTimeoutCount ?? '',
    delayed_due_at: s.delayedDueAtIso,
    delayed_pending: s.pendingDelayedTest,
  }));

  const cliRows = sessions.flatMap((s) =>
    s.cliSamples.map((sample) => ({
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      timestamp_ms: sample.timestampMs,
      session_time_s: sample.sessionTimeS,
      phase: sample.phase,
      raw_cli: sample.rawCli,
      smoothed_cli: sample.smoothedCli,
      decision_cli: sample.decisionCli ?? '',
      decision_threshold: sample.decisionThreshold ?? '',
      decision_mode: sample.decisionMode ?? '',
      valid_frame_ratio: sample.validFrameRatio,
      illumination_std: sample.illuminationStd,
      low_vfr_flag: sample.qualityFlags.lowValidFrameRatio,
      unstable_illumination_flag: sample.qualityFlags.unstableIllumination,
    }))
  );

  const featureRows = sessions.flatMap((s) =>
    (s.featureWindows ?? []).map((fw) => ({
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      timestamp_ms: fw.timestampMs,
      session_time_s: fw.sessionTimeS,
      phase: fw.phase,
      window_index: fw.windowIndex,
      is_calibration: fw.isCalibration,
      ...fw.features,
    }))
  );

  const trialRows = sessions.flatMap((s) =>
    s.trials.map((trial) => ({
      record_version: s.recordVersion,
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      trial_id: trial.trialId,
      phase: trial.phase,
      kind: trial.kind,
      difficulty: trial.difficulty,
      block_index: trial.blockIndex,
      item_id: trial.itemId,
      cue: trial.cue,
      target: trial.target,
      recognition_choices: trial.recognitionChoices?.join('|') ?? '',
      selected_choice: trial.selectedChoice ?? '',
      response_text: trial.responseText ?? '',
      correct: trial.correct,
      scoring_version: trial.scoring?.version ?? '',
      scoring_method: trial.scoring?.method ?? '',
      scoring_match_type: trial.scoring?.matchType ?? '',
      normalized_response: trial.scoring?.normalizedResponse ?? '',
      normalized_target: trial.scoring?.normalizedTarget ?? '',
      scoring_distance: trial.scoring?.distance ?? '',
      reaction_time_ms: trial.reactionTimeMs,
      timestamp_ms: trial.timestampMs,
      session_time_s: trial.sessionTimeS,
    }))
  );

  const interventionRows = sessions.flatMap((s) =>
    s.interventions.map((event) => ({
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      timestamp_ms: event.timestampMs,
      session_time_s: event.sessionTimeS,
      phase: event.phase,
      type: event.type,
      outcome: event.outcome,
      cli: event.cli,
      smoothed_cli: event.smoothedCli,
      valid_frame_ratio: event.validFrameRatio,
      details: event.details ?? '',
    }))
  );

  const arithmeticRows = sessions.flatMap((s) =>
    (s.arithmeticChallenge?.trials ?? []).map((trial) => ({
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      trial_id: trial.trialId,
      problem_id: trial.problemId,
      phase: trial.phase,
      difficulty: trial.difficulty,
      practice: trial.practice,
      expression: trial.expression,
      left_operand: trial.leftOperand,
      right_operand: trial.rightOperand,
      expected_answer: trial.expectedAnswer,
      response_text: trial.responseText ?? '',
      response_value: trial.responseValue ?? '',
      correct: trial.correct,
      timed_out: trial.timedOut,
      reaction_time_ms: trial.reactionTimeMs,
      timestamp_ms: trial.timestampMs,
      session_time_s: trial.sessionTimeS,
    }))
  );

  const tlxRows = sessions
    .filter((s) => s.nasaTlx)
    .map((s) => ({
      record_id: s.recordId,
      participant_id: s.participantId,
      session_number: s.sessionNumber,
      condition: s.condition,
      form: s.form,
      mental_demand: s.nasaTlx?.mentalDemand ?? '',
      physical_demand: s.nasaTlx?.physicalDemand ?? '',
      temporal_demand: s.nasaTlx?.temporalDemand ?? '',
      performance: s.nasaTlx?.performance ?? '',
      effort: s.nasaTlx?.effort ?? '',
      frustration: s.nasaTlx?.frustration ?? '',
    }));

  const delayedRows = delayedTests.flatMap((d) =>
    d.trials.map((trial) => ({
      record_version: d.recordVersion,
      delayed_record_id: d.recordId,
      linked_session_record_id: d.linkedSessionRecordId,
      participant_id: d.participantId,
      session_number: d.sessionNumber,
      condition: d.condition,
      form: d.form,
      due_at: d.dueAtIso,
      completed_at: d.completedAtIso ?? '',
      kind: trial.kind,
      difficulty: trial.difficulty,
      block_index: trial.blockIndex,
      item_id: trial.itemId,
      cue: trial.cue,
      target: trial.target,
      correct: trial.correct,
      scoring_version: trial.scoring?.version ?? '',
      scoring_method: trial.scoring?.method ?? '',
      scoring_match_type: trial.scoring?.matchType ?? '',
      normalized_response: trial.scoring?.normalizedResponse ?? '',
      normalized_target: trial.scoring?.normalizedTarget ?? '',
      scoring_distance: trial.scoring?.distance ?? '',
      reaction_time_ms: trial.reactionTimeMs,
      selected_choice: trial.selectedChoice ?? '',
      response_text: trial.responseText ?? '',
      recognition_choices: trial.recognitionChoices?.join('|') ?? '',
    }))
  );

  return {
    session_summary: toCsv(sessionSummaryRows),
    cli_windows: toCsv(cliRows),
    feature_windows: toCsv(featureRows),
    trials: toCsv(trialRows),
    arithmetic: toCsv(arithmeticRows),
    interventions: toCsv(interventionRows),
    tlx: toCsv(tlxRows),
    delayed: toCsv(delayedRows),
  };
}

export function triggerDownload(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
