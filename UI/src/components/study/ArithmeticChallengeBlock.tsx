import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArithmeticPhaseTag,
  ArithmeticProblem,
  ArithmeticTrial,
  StudyCondition,
  StudyForm,
} from '../../types/study';

const DIFFICULTY_LABELS: Record<ArithmeticPhaseTag, string> = {
  arithmetic_easy: 'Easy',
  arithmetic_medium: 'Medium',
  arithmetic_hard: 'Hard',
};

interface ArithmeticChallengeBlockProps {
  phase: ArithmeticPhaseTag;
  problems: ArithmeticProblem[];
  practiceProblem?: ArithmeticProblem | null;
  transitionSeconds?: number;
  timeLimitSeconds: number;
  sessionStartMs: number;
  condition: StudyCondition;
  form: StudyForm;
  onComplete: (trials: ArithmeticTrial[]) => void;
}

const ArithmeticChallengeBlock: React.FC<ArithmeticChallengeBlockProps> = ({
  phase,
  problems,
  practiceProblem,
  transitionSeconds = 0,
  timeLimitSeconds,
  sessionStartMs,
  condition,
  form,
  onComplete,
}) => {
  const title = 'Arithmetic Challenge';
  const difficultyLabel = DIFFICULTY_LABELS[phase];
  const [showTransition, setShowTransition] = useState(transitionSeconds > 0);
  const [transitionRemaining, setTransitionRemaining] = useState(transitionSeconds);
  const [index, setIndex] = useState(0);
  const [answer, setAnswer] = useState('');
  const [countdown, setCountdown] = useState(timeLimitSeconds);
  const [trials, setTrials] = useState<ArithmeticTrial[]>([]);
  const questionStartedAtRef = useRef(Date.now());

  const sequence = useMemo(
    () => (practiceProblem ? [practiceProblem, ...problems] : problems),
    [practiceProblem, problems]
  );

  const current = sequence[index];
  const isPractice = Boolean(practiceProblem && index === 0);
  const scoredIndex = isPractice ? 0 : practiceProblem ? index : index + 1;
  const scoredCount = problems.length;

  useEffect(() => {
    if (!showTransition) return undefined;

    setTransitionRemaining(transitionSeconds);
    const startedAt = Date.now();
    const interval = window.setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      const remaining = Math.max(0, transitionSeconds - elapsed);
      setTransitionRemaining(remaining);
      if (remaining <= 0) {
        window.clearInterval(interval);
        setShowTransition(false);
        questionStartedAtRef.current = Date.now();
      }
    }, 100);

    return () => window.clearInterval(interval);
  }, [showTransition, transitionSeconds]);

  useEffect(() => {
    if (showTransition || !current || isPractice) return undefined;

    questionStartedAtRef.current = Date.now();
    setCountdown(timeLimitSeconds);
    const startedAt = Date.now();

    const timeout = window.setTimeout(() => {
      submit(true);
    }, timeLimitSeconds * 1000);

    const interval = window.setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      setCountdown(Math.max(0, timeLimitSeconds - elapsed));
    }, 100);

    return () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showTransition, current, isPractice, timeLimitSeconds, index]);

  useEffect(() => {
    if (showTransition || !current || !isPractice) return;
    questionStartedAtRef.current = Date.now();
  }, [current, isPractice, showTransition]);

  const advance = (nextTrials: ArithmeticTrial[]) => {
    if (index >= sequence.length - 1) {
      onComplete(nextTrials);
      return;
    }

    setIndex((prev) => prev + 1);
    setAnswer('');
    setCountdown(timeLimitSeconds);
    questionStartedAtRef.current = Date.now();
  };

  const submit = (timedOut: boolean) => {
    if (!current) return;

    const now = Date.now();
    const normalized = answer.trim();
    const parsed = normalized === '' ? undefined : Number.parseInt(normalized, 10);
    const reactionTimeMs = now - questionStartedAtRef.current;
    const trial: ArithmeticTrial = {
      trialId: `arith_${current.id}_${now}`,
      problemId: current.id,
      timestampMs: now,
      sessionTimeS: (now - sessionStartMs) / 1000,
      phase,
      difficulty: current.difficulty,
      leftOperand: current.leftOperand,
      rightOperand: current.rightOperand,
      expression: current.expression,
      expectedAnswer: current.answer,
      responseText: normalized || undefined,
      responseValue:
        parsed !== undefined && !Number.isNaN(parsed) ? parsed : undefined,
      correct:
        parsed !== undefined && !Number.isNaN(parsed) && parsed === current.answer,
      timedOut,
      practice: isPractice,
      reactionTimeMs,
      condition,
      form,
    };

    const nextTrials = [...trials, trial];
    setTrials(nextTrials);
    advance(nextTrials);
  };

  if (showTransition) {
    return (
      <div className="bg-white rounded-lg border border-amber-200 p-8 text-center space-y-3">
        <div className="text-xs uppercase tracking-wide text-amber-700 font-semibold">
          Next arithmetic level
        </div>
        <h2 className="text-2xl font-semibold text-gray-800">{title}</h2>
        <p className="text-gray-600">Get ready. The next level starts automatically.</p>
        <p className="text-lg text-amber-700 font-medium">{Math.ceil(transitionRemaining)}s remaining</p>
      </div>
    );
  }

  if (!current) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center text-gray-600">
        Preparing arithmetic challenge...
      </div>
    );
  }

  const progress = scoredCount > 0 ? (Math.max(scoredIndex, 0) / scoredCount) * 100 : 0;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>{title}</span>
        <span>
          {isPractice ? 'Practice' : `Problem ${scoredIndex}/${scoredCount}`}
        </span>
      </div>

      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-rose-500 transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="rounded-xl bg-rose-50 border border-rose-100 p-8 text-center space-y-3">
        <div className="text-xs uppercase tracking-wide text-rose-700 font-semibold">
          {isPractice ? 'Practice item' : `${difficultyLabel} arithmetic`}
        </div>
        <div className="text-4xl font-semibold text-gray-800">{current.expression}</div>
        <div className="text-sm text-gray-600">
          {isPractice ? 'This practice item is untimed.' : `Answer within ${timeLimitSeconds} seconds.`}
        </div>
      </div>

      {!isPractice && (
        <div className="text-center text-sm text-gray-500">
          Time remaining <span className="font-semibold text-gray-700">{countdown.toFixed(1)}s</span>
        </div>
      )}

      <div className="space-y-3">
        <input
          value={answer}
          onChange={(event) => setAnswer(event.target.value.replace(/[^0-9]/g, ''))}
          className="w-full border border-gray-300 rounded-lg px-4 py-3 text-lg focus:outline-none focus:border-rose-500"
          placeholder="Enter the total"
          inputMode="numeric"
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              submit(false);
            }
          }}
        />
        <div className="flex justify-end">
          <button
            onClick={() => submit(false)}
            className="bg-rose-600 hover:bg-rose-700 text-white px-5 py-2.5 rounded-lg"
          >
            {index >= sequence.length - 1 ? 'Finish Arithmetic' : isPractice ? 'Start Challenge' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ArithmeticChallengeBlock;
