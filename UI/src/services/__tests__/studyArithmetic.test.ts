import { STUDY_CONFIG } from '../../config/studyConfig';
import { FEATURE_CONFIG } from '../../config/featureConfig';
import {
  estimateArithmeticAvailableSeconds,
  getArithmeticPhaseTag,
  getArithmeticPracticeProblem,
  getArithmeticProblems,
  summarizeArithmeticTrials,
} from '../studyArithmetic';
import { ArithmeticTrial } from '../../types/study';

describe('studyArithmetic', () => {
  it('selects deterministic scored problems for the same seed', () => {
    const run1 = getArithmeticProblems('medium', 4, 'P-001:1');
    const run2 = getArithmeticProblems('medium', 4, 'P-001:1');

    expect(run1.map((problem) => problem.id)).toEqual(run2.map((problem) => problem.id));
  });

  it('keeps practice distinct from scored easy problems', () => {
    const scored = getArithmeticProblems('easy', 4, 'P-001:1');
    const practice = getArithmeticPracticeProblem(
      'P-001:1',
      scored.map((problem) => problem.id)
    );

    expect(practice).not.toBeNull();
    expect(scored.map((problem) => problem.id)).not.toContain(practice?.id);
  });

  it('keeps all generated problems unique within a session', () => {
    const seed = 'P-001:2';
    const problems = [
      ...getArithmeticProblems('easy', 4, seed),
      ...getArithmeticProblems('medium', 4, seed),
      ...getArithmeticProblems('hard', 4, seed),
    ];

    expect(new Set(problems.map((problem) => problem.id)).size).toBe(problems.length);
  });

  it('enforces the easy problem rules', () => {
    const problems = getArithmeticProblems('easy', 4, 'P-001:1');

    problems.forEach((problem) => {
      expect(problem.leftOperand).toBeLessThan(10);
      expect(problem.rightOperand).toBeLessThan(10);
      expect(problem.leftOperand + problem.rightOperand).toBe(problem.answer);
      expect(problem.answer).toBeLessThanOrEqual(9);
    });
  });

  it('enforces the medium problem rules', () => {
    const problems = getArithmeticProblems('medium', 4, 'P-001:1');

    problems.forEach((problem) => {
      expect(problem.leftOperand).toBeGreaterThanOrEqual(10);
      expect(problem.rightOperand).toBeGreaterThanOrEqual(10);
      expect((problem.leftOperand % 10) + (problem.rightOperand % 10)).toBeLessThan(10);
      expect(problem.answer).toBeLessThan(100);
    });
  });

  it('enforces the hard problem rules', () => {
    const problems = getArithmeticProblems('hard', 4, 'P-001:1');

    problems.forEach((problem) => {
      expect(problem.leftOperand).toBeGreaterThanOrEqual(10);
      expect(problem.rightOperand).toBeGreaterThanOrEqual(10);
      expect((problem.leftOperand % 10) + (problem.rightOperand % 10)).toBeGreaterThanOrEqual(10);
      expect(problem.answer).toBeLessThan(100);
    });
  });

  it('summarizes scored arithmetic trials independently from practice', () => {
    const trials: ArithmeticTrial[] = [
      {
        trialId: 't-practice',
        problemId: 'arith-e-01',
        timestampMs: 1000,
        sessionTimeS: 1,
        phase: getArithmeticPhaseTag('easy'),
        difficulty: 'easy',
        leftOperand: 2,
        rightOperand: 3,
        expression: '2 + 3',
        expectedAnswer: 5,
        responseText: '5',
        responseValue: 5,
        correct: true,
        timedOut: false,
        practice: true,
        reactionTimeMs: 1500,
        condition: 'adaptive',
        form: 'A',
      },
      {
        trialId: 't-scored',
        problemId: 'arith-e-02',
        timestampMs: 2000,
        sessionTimeS: 2,
        phase: getArithmeticPhaseTag('easy'),
        difficulty: 'easy',
        leftOperand: 4,
        rightOperand: 1,
        expression: '4 + 1',
        expectedAnswer: 5,
        responseText: '4',
        responseValue: 4,
        correct: false,
        timedOut: false,
        practice: false,
        reactionTimeMs: 2200,
        condition: 'adaptive',
        form: 'A',
      },
    ];

    const summary = summarizeArithmeticTrials(trials);

    expect(summary.totalScoredCount).toBe(1);
    expect(summary.totalCorrectCount).toBe(0);
    expect(summary.summaries.find((entry) => entry.difficulty === 'easy')?.practiceCount).toBe(1);
    expect(summary.summaries.find((entry) => entry.difficulty === 'easy')?.accuracy).toBe(0);
  });

  it('offers enough total arithmetic time to cover the 10 second CLI window', () => {
    expect(estimateArithmeticAvailableSeconds(STUDY_CONFIG)).toBeGreaterThan(
      FEATURE_CONFIG.windows.length_s
    );
  });
});
