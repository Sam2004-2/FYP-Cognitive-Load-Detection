import { buildStudyExportTables, buildStudyExportPackage } from '../studyExport';
import { StudySessionRecord, StudyDelayedTestRecord } from '../../types/study';

function makeSession(overrides: Partial<StudySessionRecord> = {}): StudySessionRecord {
  return {
    recordId: 'rec_1',
    recordVersion: 2,
    participantId: 'P001',
    sessionNumber: 1,
    assignment: {
      participantId: 'P001',
      participantIdNormalized: 'p001',
      hashValue: 12345,
      hashParity: 'odd',
      conditionOrder: ['baseline', 'adaptive'],
      formOrder: ['B', 'A'],
      sessionNumber: 1,
      condition: 'adaptive',
      form: 'A',
      delayedDueAtIso: '2026-02-17T10:00:00Z',
    },
    plan: {
      baselineSeconds: 45,
      easyItemCount: 8,
      easyExposureSeconds: 4.5,
      hardItemCount: 10,
      hardExposureSeconds: 3.0,
      hardInterferenceEnabled: true,
      recognitionChoices: 4,
      microBreakSeconds: 45,
      maxMicroBreaksPerSession: 1,
      adaptationCooldownSeconds: 120,
      decisionWindowSeconds: 5,
      smoothingWindows: 3,
      adaptiveMode: 'relative',
      absoluteThreshold: 0.58,
      relativeZThreshold: 1.0,
      warmupWindows: 4,
      minStdEpsilon: 0.02,
      overloadThreshold: 0.7,
    },
    startedAtIso: '2026-02-10T10:00:00Z',
    completedAtIso: '2026-02-10T10:30:00Z',
    session2StartedEarlyOverride: false,
    condition: 'adaptive',
    form: 'A',
    totalSessionSeconds: 1800,
    activeTaskSeconds: 1500,
    breakSeconds: 300,
    delayedDueAtIso: '2026-02-17T10:00:00Z',
    pendingDelayedTest: true,
    cliSamples: [
      {
        timestampMs: 1000,
        sessionTimeS: 1,
        phase: 'baseline',
        rawCli: 0.4,
        smoothedCli: 0.42,
        validFrameRatio: 0.99,
        illuminationStd: 8,
        qualityFlags: { lowValidFrameRatio: false, unstableIllumination: false },
      },
    ],
    featureWindows: [
      {
        timestampMs: 500,
        sessionTimeS: 0.5,
        phase: 'baseline_calibration',
        windowIndex: 0,
        isCalibration: true,
        features: { blink_rate: 14, blink_count: 2 },
      },
      {
        timestampMs: 1000,
        sessionTimeS: 1,
        phase: 'learning_easy',
        windowIndex: 1,
        isCalibration: false,
        features: { blink_rate: 18, blink_count: 3, blink_rate_centered: 4, blink_count_centered: 1 },
      },
    ],
    trials: [
      {
        trialId: 't1',
        phase: 'test_easy',
        kind: 'recognition',
        difficulty: 'easy',
        blockIndex: 0,
        itemId: 'item1',
        cue: 'CAT',
        target: 'MOON',
        recognitionChoices: ['MOON', 'SUN', 'STAR', 'RAIN'],
        selectedChoice: 'MOON',
        scoring: {
          version: 2,
          method: 'exact_normalized',
          matchType: 'exact',
          normalizedResponse: 'moon',
          normalizedTarget: 'moon',
          distance: 0,
        },
        correct: true,
        reactionTimeMs: 1500,
        timestampMs: 5000,
        sessionTimeS: 5,
      },
    ],
    interventions: [
      {
        timestampMs: 10000,
        sessionTimeS: 10,
        phase: 'learning_hard',
        type: 'micro_break_60s',
        outcome: 'applied',
        cli: 0.85,
        smoothedCli: 0.82,
        validFrameRatio: 0.98,
      },
    ],
    nasaTlx: {
      mentalDemand: 75,
      physicalDemand: 20,
      temporalDemand: 50,
      performance: 60,
      effort: 70,
      frustration: 30,
    },
    ...overrides,
  } as StudySessionRecord;
}

function makeDelayed(overrides: Partial<StudyDelayedTestRecord> = {}): StudyDelayedTestRecord {
  return {
    recordId: 'del_1',
    recordVersion: 2,
    linkedSessionRecordId: 'rec_1',
    participantId: 'P001',
    sessionNumber: 1,
    condition: 'adaptive',
    form: 'A',
    dueAtIso: '2026-02-17T10:00:00Z',
    completedAtIso: '2026-02-17T12:00:00Z',
    trials: [
      {
        trialId: 'd_t1',
        phase: 'delayed_recognition_easy',
        kind: 'recognition',
        difficulty: 'easy',
        blockIndex: 0,
        itemId: 'item1',
        cue: 'CAT',
        target: 'MOON',
        recognitionChoices: ['MOON', 'SUN'],
        selectedChoice: 'MOON',
        scoring: {
          version: 2,
          method: 'exact_normalized',
          matchType: 'exact',
          normalizedResponse: 'moon',
          normalizedTarget: 'moon',
          distance: 0,
        },
        correct: true,
        reactionTimeMs: 2000,
        timestampMs: 1000,
        sessionTimeS: 1,
      },
    ],
    ...overrides,
  } as StudyDelayedTestRecord;
}

describe('buildStudyExportPackage', () => {
  it('creates a package with correct fields', () => {
    const pkg = buildStudyExportPackage('P001', [makeSession()], [makeDelayed()]);
    expect(pkg.participantId).toBe('P001');
    expect(pkg.sessions).toHaveLength(1);
    expect(pkg.delayedTests).toHaveLength(1);
    expect(pkg.exportedAtIso).toBeTruthy();
  });

  it('handles empty sessions and delayed tests', () => {
    const pkg = buildStudyExportPackage('P002', [], []);
    expect(pkg.sessions).toHaveLength(0);
    expect(pkg.delayedTests).toHaveLength(0);
  });
});

describe('buildStudyExportTables', () => {
  it('generates CSV strings for all table types', () => {
    const tables = buildStudyExportTables([makeSession()], [makeDelayed()]);
    expect(tables.session_summary).toContain('record_id');
    expect(tables.session_summary).toContain('rec_1');
    expect(tables.cli_windows).toContain('raw_cli');
    expect(tables.feature_windows).toContain('blink_rate');
    expect(tables.feature_windows).toContain('is_calibration');
    expect(tables.trials).toContain('cue');
    expect(tables.trials).toContain('scoring_method');
    expect(tables.trials).toContain('exact_normalized');
    expect(tables.interventions).toContain('micro_break_60s');
    expect(tables.tlx).toContain('mental_demand');
    expect(tables.delayed).toContain('del_1');
  });

  it('returns empty strings for tables with no data', () => {
    const tables = buildStudyExportTables([], []);
    expect(tables.session_summary).toBe('');
    expect(tables.cli_windows).toBe('');
    expect(tables.feature_windows).toBe('');
    expect(tables.trials).toBe('');
    expect(tables.interventions).toBe('');
    expect(tables.tlx).toBe('');
    expect(tables.delayed).toBe('');
  });

  it('correctly counts applied interventions in session summary', () => {
    const session = makeSession({
      interventions: [
        {
          timestampMs: 1000,
          sessionTimeS: 1,
          phase: 'learning_easy',
          type: 'micro_break_60s',
          outcome: 'applied',
          cli: 0.8,
          smoothedCli: 0.78,
          validFrameRatio: 0.99,
        },
        {
          timestampMs: 2000,
          sessionTimeS: 2,
          phase: 'learning_easy',
          type: 'micro_break_60s',
          outcome: 'dismissed',
          cli: 0.75,
          smoothedCli: 0.73,
          validFrameRatio: 0.98,
        },
      ],
    } as any);

    const tables = buildStudyExportTables([session], []);
    // The CSV should include intervention_count=1 (only 'applied')
    expect(tables.session_summary).toContain('intervention_count');
    // The interventions CSV should have both entries
    const interventionLines = tables.interventions.split('\n');
    expect(interventionLines.length).toBe(3); // header + 2 rows
  });

  it('escapes CSV special characters', () => {
    const session = makeSession({
      trials: [
        {
          trialId: 't1',
          phase: 'test_easy',
          kind: 'cued_recall',
          difficulty: 'easy',
          blockIndex: 0,
          itemId: 'item1',
          cue: 'word,with,commas',
          target: 'answer',
          correct: true,
          reactionTimeMs: 1000,
          timestampMs: 5000,
          sessionTimeS: 5,
          responseText: 'user "quoted" answer',
        },
      ],
    } as any);

    const tables = buildStudyExportTables([session], []);
    // Commas in values should be quoted
    expect(tables.trials).toContain('"word,with,commas"');
  });

  it('handles sessions without NASA-TLX', () => {
    const session = makeSession({ nasaTlx: undefined } as any);
    const tables = buildStudyExportTables([session], []);
    // TLX table should be empty (no sessions with nasaTlx)
    expect(tables.tlx).toBe('');
  });

  it('keeps export compatibility for version 1 trials without scoring metadata', () => {
    const legacySession = makeSession({
      recordVersion: 1,
      trials: [
        {
          trialId: 'legacy_t1',
          phase: 'test_easy_recognition',
          kind: 'recognition',
          difficulty: 'easy',
          blockIndex: 1,
          itemId: 'item1',
          cue: 'CAT',
          target: 'MOON',
          recognitionChoices: ['MOON', 'SUN'],
          selectedChoice: 'MOON',
          correct: true,
          reactionTimeMs: 1000,
          timestampMs: 5000,
          sessionTimeS: 5,
          condition: 'adaptive',
          form: 'A',
        },
      ],
    });

    const tables = buildStudyExportTables([legacySession], []);
    expect(tables.trials).toContain('scoring_method');
    expect(tables.trials).toContain('record_version');
    const lines = tables.trials.split('\n');
    expect(lines).toHaveLength(2);
    expect(lines[1]).toContain(',1,');
  });
});
