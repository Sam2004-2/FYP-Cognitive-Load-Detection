import React, { useMemo, useState } from 'react';
import { buildRecognitionChoices } from '../../services/studyStimuli';
import {
  StudyCondition,
  StudyForm,
  StudyPhaseTag,
  StudyStimulusItem,
  StudyTrialResult,
} from '../../types/study';

interface RecognitionTestProps {
  items: StudyStimulusItem[];
  blockIndex: 1 | 2;
  phase: StudyPhaseTag;
  condition: StudyCondition;
  form: StudyForm;
  sessionStartMs: number;
  participantSeed: string;
  choiceCount: number;
  useInterferenceDistractors: boolean;
  onComplete: (trials: StudyTrialResult[]) => void;
}

const RecognitionTest: React.FC<RecognitionTestProps> = ({
  items,
  blockIndex,
  phase,
  condition,
  form,
  sessionStartMs,
  participantSeed,
  choiceCount,
  useInterferenceDistractors,
  onComplete,
}) => {
  const [index, setIndex] = useState(0);
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null);
  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [questionStartMs, setQuestionStartMs] = useState(Date.now());

  const current = items[index];

  const choices = useMemo(() => {
    if (!current) return [];
    return buildRecognitionChoices(
      current,
      items,
      choiceCount,
      useInterferenceDistractors,
      participantSeed
    );
  }, [choiceCount, current, items, participantSeed, useInterferenceDistractors]);

  const submit = () => {
    if (!current || !selectedChoice) return;
    const now = Date.now();
    const trial: StudyTrialResult = {
      trialId: `recognition_${current.id}_${now}`,
      timestampMs: now,
      sessionTimeS: (now - sessionStartMs) / 1000,
      phase,
      kind: 'recognition',
      difficulty: current.difficulty,
      blockIndex,
      itemId: current.id,
      cue: current.cue,
      target: current.target,
      recognitionChoices: choices.map((c) => c.value),
      selectedChoice,
      correct: selectedChoice.trim().toLowerCase() === current.target.trim().toLowerCase(),
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
    setSelectedChoice(null);
    setQuestionStartMs(Date.now());
  };

  if (!current) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center text-gray-600">
        Preparing recognition test...
      </div>
    );
  }

  const progress = ((index + 1) / items.length) * 100;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>Recognition test</span>
        <span>
          Item {index + 1}/{items.length}
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="rounded-xl bg-indigo-50 border border-indigo-100 p-6 text-center">
        <div className="text-xs uppercase tracking-wide text-indigo-700 font-semibold mb-2">
          Which target was paired with this cue?
        </div>
        <div className="text-3xl font-semibold text-gray-800">{current.cue}</div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {choices.map((choice) => (
          <button
            key={choice.value}
            onClick={() => setSelectedChoice(choice.value)}
            className={`text-left px-4 py-3 rounded-lg border transition-colors ${
              selectedChoice === choice.value
                ? 'border-indigo-500 bg-indigo-100 text-indigo-900'
                : 'border-gray-300 hover:border-indigo-300 bg-white text-gray-800'
            }`}
          >
            {choice.value}
          </button>
        ))}
      </div>

      <div className="flex justify-end">
        <button
          onClick={submit}
          disabled={!selectedChoice}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg"
        >
          {index >= items.length - 1 ? 'Finish Recognition' : 'Next'}
        </button>
      </div>
    </div>
  );
};

export default RecognitionTest;
