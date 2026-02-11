import React, { useEffect, useMemo, useRef, useState } from 'react';
import { StudyCondition, StudyForm, StudyPhaseTag, StudyStimulusItem, StudyTrialResult } from '../../types/study';

interface PairedAssociateLearningBlockProps {
  items: StudyStimulusItem[];
  exposureSeconds: number;
  blockIndex: 1 | 2;
  phase: StudyPhaseTag;
  condition: StudyCondition;
  form: StudyForm;
  sessionStartMs: number;
  onComplete: (trials: StudyTrialResult[], elapsedSeconds: number) => void;
}

const PairedAssociateLearningBlock: React.FC<PairedAssociateLearningBlockProps> = ({
  items,
  exposureSeconds,
  blockIndex,
  phase,
  condition,
  form,
  sessionStartMs,
  onComplete,
}) => {
  const [index, setIndex] = useState(0);
  const [countdown, setCountdown] = useState(exposureSeconds);
  const [startedAt] = useState(Date.now());
  const trialsRef = useRef<StudyTrialResult[]>([]);

  const current = items[index];

  useEffect(() => {
    setCountdown(exposureSeconds);
  }, [index, exposureSeconds]);

  useEffect(() => {
    if (!current) return;

    const started = Date.now();
    const timer = window.setTimeout(() => {
      const now = Date.now();
      const trial: StudyTrialResult = {
        trialId: `learn_${current.id}_${now}`,
        timestampMs: now,
        sessionTimeS: (now - sessionStartMs) / 1000,
        phase,
        kind: 'learning',
        difficulty: current.difficulty,
        blockIndex,
        itemId: current.id,
        cue: current.cue,
        target: current.target,
        correct: true,
        reactionTimeMs: 0,
        condition,
        form,
      };
      trialsRef.current = [...trialsRef.current, trial];

      if (index < items.length - 1) {
        setIndex((prev) => prev + 1);
      } else {
        onComplete(trialsRef.current, (now - startedAt) / 1000);
      }
    }, exposureSeconds * 1000);

    const countdownTimer = window.setInterval(() => {
      const elapsed = (Date.now() - started) / 1000;
      setCountdown(Math.max(0, exposureSeconds - elapsed));
    }, 100);

    return () => {
      window.clearTimeout(timer);
      window.clearInterval(countdownTimer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, current, exposureSeconds]);

  const progressPct = useMemo(() => {
    if (items.length === 0) return 0;
    return ((index + 1) / items.length) * 100;
  }, [index, items.length]);

  if (!current) {
    return (
      <div className="bg-white rounded-lg p-6 text-center text-gray-600">
        Preparing learning items...
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>Learning block {blockIndex}</span>
        <span>
          Item {index + 1}/{items.length}
        </span>
      </div>

      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-blue-500 transition-all" style={{ width: `${progressPct}%` }} />
      </div>

      <div className="rounded-xl bg-blue-50 border border-blue-100 p-8 text-center space-y-3">
        <div className="text-xs uppercase tracking-wide text-blue-600 font-semibold">Memorize this pair</div>
        <div className="text-4xl font-semibold text-gray-800">{current.cue}</div>
        <div className="text-2xl text-blue-700">{current.target}</div>
      </div>

      <div className="text-center text-sm text-gray-500">
        Next item in <span className="font-semibold text-gray-700">{countdown.toFixed(1)}s</span>
      </div>
    </div>
  );
};

export default PairedAssociateLearningBlock;
