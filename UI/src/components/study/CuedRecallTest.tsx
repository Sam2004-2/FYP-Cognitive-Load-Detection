import React, { useState } from 'react';
import {
  StudyCondition,
  StudyForm,
  StudyPhaseTag,
  StudyStimulusItem,
  StudyTrialResult,
} from '../../types/study';

interface CuedRecallTestProps {
  items: StudyStimulusItem[];
  blockIndex: 1 | 2;
  phase: StudyPhaseTag;
  condition: StudyCondition;
  form: StudyForm;
  sessionStartMs: number;
  onComplete: (trials: StudyTrialResult[]) => void;
}

const CuedRecallTest: React.FC<CuedRecallTestProps> = ({
  items,
  blockIndex,
  phase,
  condition,
  form,
  sessionStartMs,
  onComplete,
}) => {
  const [index, setIndex] = useState(0);
  const [answer, setAnswer] = useState('');
  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [questionStartMs, setQuestionStartMs] = useState(Date.now());

  const current = items[index];

  const submit = () => {
    if (!current) return;
    const now = Date.now();
    const normalizedResponse = answer.trim().toLowerCase();
    const normalizedTarget = current.target.trim().toLowerCase();

    const trial: StudyTrialResult = {
      trialId: `cued_${current.id}_${now}`,
      timestampMs: now,
      sessionTimeS: (now - sessionStartMs) / 1000,
      phase,
      kind: 'cued_recall',
      difficulty: current.difficulty,
      blockIndex,
      itemId: current.id,
      cue: current.cue,
      target: current.target,
      responseText: answer,
      correct: normalizedResponse === normalizedTarget,
      reactionTimeMs: now - questionStartMs,
      condition,
      form,
    };

    const nextTrials = [...trials, trial];
    setTrials(nextTrials);

    if (index >= items.length - 1) {
      onComplete(nextTrials);
      return;
    }

    setIndex((prev) => prev + 1);
    setAnswer('');
    setQuestionStartMs(Date.now());
  };

  if (!current) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center text-gray-600">
        Preparing cued recall test...
      </div>
    );
  }

  const progress = ((index + 1) / items.length) * 100;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>Cued recall</span>
        <span>
          Item {index + 1}/{items.length}
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="rounded-xl bg-emerald-50 border border-emerald-100 p-6 text-center">
        <div className="text-xs uppercase tracking-wide text-emerald-700 font-semibold mb-2">
          Type the target paired with this cue
        </div>
        <div className="text-3xl font-semibold text-gray-800">{current.cue}</div>
      </div>

      <div className="space-y-3">
        <input
          value={answer}
          onChange={(event) => setAnswer(event.target.value)}
          className="w-full border border-gray-300 rounded-lg px-4 py-3 text-lg focus:outline-none focus:border-emerald-500"
          placeholder="Enter paired target"
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              submit();
            }
          }}
        />
        <div className="flex justify-end">
          <button
            onClick={submit}
            className="bg-emerald-600 hover:bg-emerald-700 text-white px-5 py-2.5 rounded-lg"
          >
            {index >= items.length - 1 ? 'Finish Cued Recall' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CuedRecallTest;
