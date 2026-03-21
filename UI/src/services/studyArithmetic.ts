import {
  ArithmeticChallengeRecord,
  ArithmeticDifficulty,
  ArithmeticDifficultySummary,
  ArithmeticPhaseTag,
  ArithmeticProblem,
  ArithmeticTrial,
  StudySessionPlan,
} from '../types/study';

export const ARITHMETIC_DIFFICULTIES: ArithmeticDifficulty[] = ['easy', 'medium', 'hard'];

const PROBLEMS: Record<ArithmeticDifficulty, ArithmeticProblem[]> = {
  easy: [
    { id: 'arith-e-01', difficulty: 'easy', leftOperand: 2, rightOperand: 3, expression: '2 + 3', answer: 5 },
    { id: 'arith-e-02', difficulty: 'easy', leftOperand: 4, rightOperand: 1, expression: '4 + 1', answer: 5 },
    { id: 'arith-e-03', difficulty: 'easy', leftOperand: 3, rightOperand: 5, expression: '3 + 5', answer: 8 },
    { id: 'arith-e-04', difficulty: 'easy', leftOperand: 1, rightOperand: 7, expression: '1 + 7', answer: 8 },
    { id: 'arith-e-05', difficulty: 'easy', leftOperand: 5, rightOperand: 4, expression: '5 + 4', answer: 9 },
    { id: 'arith-e-06', difficulty: 'easy', leftOperand: 6, rightOperand: 2, expression: '6 + 2', answer: 8 },
    { id: 'arith-e-07', difficulty: 'easy', leftOperand: 2, rightOperand: 6, expression: '2 + 6', answer: 8 },
    { id: 'arith-e-08', difficulty: 'easy', leftOperand: 3, rightOperand: 4, expression: '3 + 4', answer: 7 },
    { id: 'arith-e-09', difficulty: 'easy', leftOperand: 1, rightOperand: 6, expression: '1 + 6', answer: 7 },
    { id: 'arith-e-10', difficulty: 'easy', leftOperand: 4, rightOperand: 3, expression: '4 + 3', answer: 7 },
  ],
  medium: [
    { id: 'arith-m-01', difficulty: 'medium', leftOperand: 24, rightOperand: 35, expression: '24 + 35', answer: 59 },
    { id: 'arith-m-02', difficulty: 'medium', leftOperand: 41, rightOperand: 27, expression: '41 + 27', answer: 68 },
    { id: 'arith-m-03', difficulty: 'medium', leftOperand: 53, rightOperand: 24, expression: '53 + 24', answer: 77 },
    { id: 'arith-m-04', difficulty: 'medium', leftOperand: 36, rightOperand: 42, expression: '36 + 42', answer: 78 },
    { id: 'arith-m-05', difficulty: 'medium', leftOperand: 62, rightOperand: 15, expression: '62 + 15', answer: 77 },
    { id: 'arith-m-06', difficulty: 'medium', leftOperand: 45, rightOperand: 12, expression: '45 + 12', answer: 57 },
    { id: 'arith-m-07', difficulty: 'medium', leftOperand: 31, rightOperand: 26, expression: '31 + 26', answer: 57 },
    { id: 'arith-m-08', difficulty: 'medium', leftOperand: 54, rightOperand: 13, expression: '54 + 13', answer: 67 },
    { id: 'arith-m-09', difficulty: 'medium', leftOperand: 22, rightOperand: 47, expression: '22 + 47', answer: 69 },
    { id: 'arith-m-10', difficulty: 'medium', leftOperand: 71, rightOperand: 18, expression: '71 + 18', answer: 89 },
  ],
  hard: [
    { id: 'arith-h-01', difficulty: 'hard', leftOperand: 28, rightOperand: 17, expression: '28 + 17', answer: 45 },
    { id: 'arith-h-02', difficulty: 'hard', leftOperand: 36, rightOperand: 28, expression: '36 + 28', answer: 64 },
    { id: 'arith-h-03', difficulty: 'hard', leftOperand: 47, rightOperand: 15, expression: '47 + 15', answer: 62 },
    { id: 'arith-h-04', difficulty: 'hard', leftOperand: 58, rightOperand: 14, expression: '58 + 14', answer: 72 },
    { id: 'arith-h-05', difficulty: 'hard', leftOperand: 29, rightOperand: 34, expression: '29 + 34', answer: 63 },
    { id: 'arith-h-06', difficulty: 'hard', leftOperand: 64, rightOperand: 19, expression: '64 + 19', answer: 83 },
    { id: 'arith-h-07', difficulty: 'hard', leftOperand: 75, rightOperand: 18, expression: '75 + 18', answer: 93 },
    { id: 'arith-h-08', difficulty: 'hard', leftOperand: 16, rightOperand: 27, expression: '16 + 27', answer: 43 },
    { id: 'arith-h-09', difficulty: 'hard', leftOperand: 37, rightOperand: 26, expression: '37 + 26', answer: 63 },
    { id: 'arith-h-10', difficulty: 'hard', leftOperand: 48, rightOperand: 15, expression: '48 + 15', answer: 63 },
  ],
};

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function stableHash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function shuffleWithSeed<T>(items: T[], seed: number): T[] {
  const random = seededRandom(seed);
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function getArithmeticPhaseTag(difficulty: ArithmeticDifficulty): ArithmeticPhaseTag {
  switch (difficulty) {
    case 'easy':
      return 'arithmetic_easy';
    case 'medium':
      return 'arithmetic_medium';
    case 'hard':
      return 'arithmetic_hard';
  }
}

export function getArithmeticProblems(
  difficulty: ArithmeticDifficulty,
  count: number,
  participantSeedInput: string,
  excludedIds: string[] = []
): ArithmeticProblem[] {
  const excluded = new Set(excludedIds);
  const filtered = PROBLEMS[difficulty].filter((problem) => !excluded.has(problem.id));
  const seed = stableHash(`${participantSeedInput}:${difficulty}:arithmetic`);
  return shuffleWithSeed(filtered, seed).slice(0, count);
}

export function getArithmeticPracticeProblem(
  participantSeedInput: string,
  excludedIds: string[] = []
): ArithmeticProblem | null {
  const excluded = new Set(excludedIds);
  const filtered = PROBLEMS.easy.filter((problem) => !excluded.has(problem.id));
  if (filtered.length === 0) return null;
  const seed = stableHash(`${participantSeedInput}:arithmetic:practice`);
  return shuffleWithSeed(filtered, seed)[0] ?? null;
}

export function summarizeArithmeticTrials(trials: ArithmeticTrial[]): ArithmeticChallengeRecord {
  const summaries: ArithmeticDifficultySummary[] = ARITHMETIC_DIFFICULTIES.map((difficulty) => {
    const phase = getArithmeticPhaseTag(difficulty);
    const difficultyTrials = trials.filter((trial) => trial.difficulty === difficulty);
    const practiceCount = difficultyTrials.filter((trial) => trial.practice).length;
    const scored = difficultyTrials.filter((trial) => !trial.practice);
    const correctCount = scored.filter((trial) => trial.correct).length;
    const timeoutCount = scored.filter((trial) => trial.timedOut).length;
    const meanRtMs =
      scored.length > 0
        ? scored.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / scored.length
        : 0;

    return {
      difficulty,
      phase,
      practiceCount,
      scoredCount: scored.length,
      correctCount,
      timeoutCount,
      accuracy: scored.length > 0 ? correctCount / scored.length : 0,
      meanRtMs,
    };
  });

  const scoredTrials = trials.filter((trial) => !trial.practice);
  const totalCorrectCount = scoredTrials.filter((trial) => trial.correct).length;
  const totalTimeoutCount = scoredTrials.filter((trial) => trial.timedOut).length;
  const overallMeanRtMs =
    scoredTrials.length > 0
      ? scoredTrials.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / scoredTrials.length
      : 0;

  return {
    trials,
    summaries,
    totalScoredCount: scoredTrials.length,
    totalCorrectCount,
    totalTimeoutCount,
    overallAccuracy: scoredTrials.length > 0 ? totalCorrectCount / scoredTrials.length : 0,
    overallMeanRtMs,
  };
}

export function estimateArithmeticAvailableSeconds(
  plan: Pick<
    StudySessionPlan,
    'arithmeticItemsPerDifficulty' | 'arithmeticTimeLimitSeconds' | 'arithmeticTransitionSeconds'
  >
): number {
  const scoredTrialCount = ARITHMETIC_DIFFICULTIES.length * plan.arithmeticItemsPerDifficulty;
  const transitionCount = Math.max(ARITHMETIC_DIFFICULTIES.length - 1, 0);

  return (
    scoredTrialCount * plan.arithmeticTimeLimitSeconds +
    transitionCount * plan.arithmeticTransitionSeconds
  );
}
