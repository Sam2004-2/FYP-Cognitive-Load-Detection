import React, { useCallback, useMemo, useState } from 'react';
import {
  StudyCondition,
  StudyForm,
  StudyPhaseTag,
  StudyTrialResult,
} from '../../types/study';

/* ------------------------------------------------------------------ */
/*  Arithmetic question generation with explicit difficulty ranking    */
/*  Based on dual-task paradigm research (Ashcraft & Kirk 2001;       */
/*  Galy et al. 2015) — difficulty scales via operand magnitude,      */
/*  operation type, and element interactivity.                        */
/* ------------------------------------------------------------------ */

export type ArithmeticDifficulty = 1 | 2 | 3;

export interface ArithmeticQuestion {
  id: string;
  expression: string;
  answer: number;
  difficulty: ArithmeticDifficulty;
  difficultyLabel: 'easy' | 'medium' | 'hard';
}

/** Deterministic pseudo-random number generator (mulberry32). */
function seededRng(seed: string): () => number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  }
  let s = h >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min;
}

/**
 * Generate a bank of arithmetic questions across three difficulty tiers.
 *
 * Difficulty 1 (Easy):   single-digit addition / subtraction, result 0-18
 * Difficulty 2 (Medium): two-digit +/- with carrying/borrowing, result 10-99
 * Difficulty 3 (Hard):   multi-step expressions (a × b ± c), result varies
 */
export function generateArithmeticQuestions(
  count: number,
  seed: string,
): ArithmeticQuestion[] {
  const rng = seededRng(seed);
  const questions: ArithmeticQuestion[] = [];

  const perTier = Math.ceil(count / 3);

  // --- Difficulty 1: single-digit add/subtract ---
  for (let i = 0; i < perTier && questions.length < count; i++) {
    const a = randInt(rng, 2, 9);
    const b = randInt(rng, 2, 9);
    const useAdd = rng() > 0.4;
    if (useAdd) {
      questions.push({
        id: `arith_1_${i}`,
        expression: `${a} + ${b}`,
        answer: a + b,
        difficulty: 1,
        difficultyLabel: 'easy',
      });
    } else {
      const big = Math.max(a, b);
      const small = Math.min(a, b);
      questions.push({
        id: `arith_1_${i}`,
        expression: `${big} - ${small}`,
        answer: big - small,
        difficulty: 1,
        difficultyLabel: 'easy',
      });
    }
  }

  // --- Difficulty 2: two-digit add/subtract (carrying / borrowing) ---
  for (let i = 0; i < perTier && questions.length < count; i++) {
    const a = randInt(rng, 12, 49);
    const b = randInt(rng, 12, 49);
    const useAdd = rng() > 0.5;
    if (useAdd) {
      questions.push({
        id: `arith_2_${i}`,
        expression: `${a} + ${b}`,
        answer: a + b,
        difficulty: 2,
        difficultyLabel: 'medium',
      });
    } else {
      const big = Math.max(a, b);
      const small = Math.min(a, b);
      questions.push({
        id: `arith_2_${i}`,
        expression: `${big} - ${small}`,
        answer: big - small,
        difficulty: 2,
        difficultyLabel: 'medium',
      });
    }
  }

  // --- Difficulty 3: multiplication ± constant ---
  for (let i = 0; i < perTier && questions.length < count; i++) {
    const a = randInt(rng, 3, 9);
    const b = randInt(rng, 3, 9);
    const c = randInt(rng, 3, 19);
    const usePlus = rng() > 0.5;
    const product = a * b;
    if (usePlus) {
      questions.push({
        id: `arith_3_${i}`,
        expression: `${a} \u00d7 ${b} + ${c}`,
        answer: product + c,
        difficulty: 3,
        difficultyLabel: 'hard',
      });
    } else {
      // ensure non-negative result
      const safeC = Math.min(c, product);
      questions.push({
        id: `arith_3_${i}`,
        expression: `${a} \u00d7 ${b} - ${safeC}`,
        answer: product - safeC,
        difficulty: 3,
        difficultyLabel: 'hard',
      });
    }
  }

  // Shuffle but keep deterministic
  for (let i = questions.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [questions[i], questions[j]] = [questions[j], questions[i]];
  }

  return questions.slice(0, count);
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

interface ArithmeticChallengeProps {
  questionCount: number;
  phase: StudyPhaseTag;
  condition: StudyCondition;
  form: StudyForm;
  sessionStartMs: number;
  participantSeed: string;
  onComplete: (trials: StudyTrialResult[]) => void;
}

const DIFFICULTY_COLORS: Record<ArithmeticDifficulty, string> = {
  1: 'bg-green-100 text-green-800',
  2: 'bg-yellow-100 text-yellow-800',
  3: 'bg-red-100 text-red-800',
};

const ArithmeticChallenge: React.FC<ArithmeticChallengeProps> = ({
  questionCount,
  phase,
  condition,
  form,
  sessionStartMs,
  participantSeed,
  onComplete,
}) => {
  const questions = useMemo(
    () => generateArithmeticQuestions(questionCount, participantSeed),
    [questionCount, participantSeed],
  );

  const [index, setIndex] = useState(0);
  const [answer, setAnswer] = useState('');
  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [questionStartMs, setQuestionStartMs] = useState(Date.now());

  const current = questions[index];

  const submit = useCallback(() => {
    if (!current) return;
    const numericAnswer = parseInt(answer, 10);
    const now = Date.now();

    const trial: StudyTrialResult = {
      trialId: `arith_${current.id}_${now}`,
      timestampMs: now,
      sessionTimeS: (now - sessionStartMs) / 1000,
      phase,
      kind: 'arithmetic',
      difficulty: current.difficultyLabel === 'easy' ? 'easy' : 'hard',
      blockIndex: 1, // arithmetic is a standalone interleaved block
      itemId: current.id,
      cue: current.expression,
      target: String(current.answer),
      responseText: answer,
      correct: numericAnswer === current.answer,
      reactionTimeMs: now - questionStartMs,
      condition,
      form,
      arithmeticDifficulty: current.difficulty,
    };

    const nextTrials = [...trials, trial];
    setTrials(nextTrials);

    if (index >= questions.length - 1) {
      onComplete(nextTrials);
      return;
    }

    setIndex((prev) => prev + 1);
    setAnswer('');
    setQuestionStartMs(Date.now());
  }, [answer, condition, current, form, index, onComplete, phase, questionStartMs, questions.length, sessionStartMs, trials]);

  if (!current) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center text-gray-600">
        Preparing arithmetic challenge...
      </div>
    );
  }

  const progress = ((index + 1) / questions.length) * 100;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>Arithmetic challenge</span>
        <span>
          Question {index + 1}/{questions.length}
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-amber-500 transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="rounded-xl bg-amber-50 border border-amber-100 p-6 text-center space-y-3">
        <div className="flex items-center justify-center gap-2">
          <div className="text-xs uppercase tracking-wide text-amber-700 font-semibold">
            Solve this problem
          </div>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${DIFFICULTY_COLORS[current.difficulty]}`}>
            {current.difficultyLabel}
          </span>
        </div>
        <div className="text-4xl font-semibold text-gray-800 font-mono">
          {current.expression} = ?
        </div>
      </div>

      <div className="space-y-3">
        <input
          type="number"
          inputMode="numeric"
          value={answer}
          onChange={(event) => setAnswer(event.target.value)}
          className="w-full border border-gray-300 rounded-lg px-4 py-3 text-lg text-center font-mono focus:outline-none focus:border-amber-500"
          placeholder="Enter your answer"
          autoFocus
          onKeyDown={(event) => {
            if (event.key === 'Enter' && answer.trim() !== '') {
              submit();
            }
          }}
        />
        <div className="flex justify-end">
          <button
            onClick={submit}
            disabled={answer.trim() === ''}
            className="bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg"
          >
            {index >= questions.length - 1 ? 'Finish Arithmetic' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ArithmeticChallenge;
