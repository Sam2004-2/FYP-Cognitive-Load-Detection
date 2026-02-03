import React, { useState, useEffect, useCallback } from 'react';
import { WordPair, TaskPerformance } from '../../types';

// ============================================================================
// Word Pair Sets - Two matched forms (A/B) for counterbalancing
// ============================================================================

export const WORD_PAIRS_FORM_A: WordPair[] = [
  // Easy pairs (concrete, high association)
  { cue: 'APPLE', target: 'FRUIT' },
  { cue: 'DOG', target: 'BARK' },
  { cue: 'SUN', target: 'SHINE' },
  { cue: 'BOOK', target: 'READ' },
  { cue: 'RAIN', target: 'WET' },
  { cue: 'BIRD', target: 'FLY' },
  // Hard pairs (abstract, low association)
  { cue: 'JUSTICE', target: 'SCALE' },
  { cue: 'THEORY', target: 'PROOF' },
  { cue: 'VIRTUE', target: 'MERIT' },
  { cue: 'CIPHER', target: 'CODE' },
  { cue: 'NEXUS', target: 'LINK' },
  { cue: 'AXIOM', target: 'TRUTH' },
  { cue: 'QUORUM', target: 'COUNT' },
  { cue: 'ZENITH', target: 'PEAK' },
  { cue: 'TENET', target: 'BELIEF' },
  { cue: 'SCHEMA', target: 'PLAN' },
];

export const WORD_PAIRS_FORM_B: WordPair[] = [
  // Easy pairs (concrete, high association)
  { cue: 'CAT', target: 'MEOW' },
  { cue: 'MOON', target: 'NIGHT' },
  { cue: 'TREE', target: 'LEAF' },
  { cue: 'FISH', target: 'SWIM' },
  { cue: 'SNOW', target: 'COLD' },
  { cue: 'FIRE', target: 'BURN' },
  // Hard pairs (abstract, low association)
  { cue: 'ETHICS', target: 'MORAL' },
  { cue: 'THESIS', target: 'CLAIM' },
  { cue: 'CANDOR', target: 'FRANK' },
  { cue: 'SYNTAX', target: 'RULE' },
  { cue: 'CRUX', target: 'CORE' },
  { cue: 'MAXIM', target: 'SAYING' },
  { cue: 'QUANDARY', target: 'DILEMMA' },
  { cue: 'NADIR', target: 'LOW' },
  { cue: 'DOGMA', target: 'DOCTRINE' },
  { cue: 'MOTIF', target: 'THEME' },
];

interface PairedAssociatesTaskProps {
  pairs: WordPair[];
  exposureTimeMs: number;
  mode: 'study' | 'test';
  onComplete: (performance: TaskPerformance) => void;
  onPairShown?: (pairIndex: number) => void; // Callback for CLI logging
}

const PairedAssociatesTask: React.FC<PairedAssociatesTaskProps> = ({
  pairs,
  exposureTimeMs,
  mode,
  onComplete,
  onPairShown,
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [responses, setResponses] = useState<NonNullable<TaskPerformance['responses']>>([]);
  const [testStartTime, setTestStartTime] = useState<number>(0);
  const [isComplete, setIsComplete] = useState(false);

  // Study mode: auto-advance through pairs
  useEffect(() => {
    if (mode !== 'study' || isComplete) return;

    onPairShown?.(currentIndex);

    const timer = setTimeout(() => {
      if (currentIndex < pairs.length - 1) {
        setCurrentIndex(prev => prev + 1);
      } else {
        setIsComplete(true);
        onComplete({ correct: pairs.length, total: pairs.length });
      }
    }, exposureTimeMs);

    return () => clearTimeout(timer);
  }, [mode, currentIndex, pairs.length, exposureTimeMs, isComplete, onComplete, onPairShown]);

  // Test mode: record response time start
  useEffect(() => {
    if (mode === 'test' && !isComplete) {
      setTestStartTime(Date.now());
    }
  }, [mode, currentIndex, isComplete]);

  const handleTestSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (mode !== 'test' || isComplete) return;

    const rt = Date.now() - testStartTime;
    const currentPair = pairs[currentIndex];
    const isCorrect = userInput.trim().toUpperCase() === currentPair.target.toUpperCase();

    const newResponse = {
      cue: currentPair.cue,
      expected: currentPair.target,
      given: userInput.trim().toUpperCase(),
      correct: isCorrect,
      rt_ms: rt,
    };

    const updatedResponses = [...responses, newResponse];
    setResponses(updatedResponses);
    setUserInput('');

    if (currentIndex < pairs.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else {
      // Calculate final performance
      const correct = updatedResponses.filter(r => r.correct).length;
      const totalRt = updatedResponses.reduce((sum, r) => sum + r.rt_ms, 0);
      
      setIsComplete(true);
      onComplete({
        correct,
        total: pairs.length,
        rt_mean_ms: totalRt / pairs.length,
        responses: updatedResponses,
      });
    }
  }, [mode, currentIndex, pairs, userInput, responses, testStartTime, isComplete, onComplete]);

  if (isComplete) {
    return (
      <div className="flex flex-col items-center justify-center p-8">
        <div className="text-2xl font-semibold text-green-600 mb-4">
          {mode === 'study' ? 'Study Phase Complete!' : 'Test Complete!'}
        </div>
        {mode === 'test' && (
          <div className="text-lg text-gray-600">
            Score: {responses.filter(r => r.correct).length} / {pairs.length}
          </div>
        )}
      </div>
    );
  }

  const currentPair = pairs[currentIndex];
  const progress = ((currentIndex + 1) / pairs.length) * 100;

  return (
    <div className="flex flex-col items-center justify-center p-8 min-h-[400px]">
      {/* Progress bar */}
      <div className="w-full max-w-md mb-8">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>{mode === 'study' ? 'Studying' : 'Testing'}</span>
          <span>{currentIndex + 1} / {pairs.length}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {mode === 'study' ? (
        // Study mode: show both words
        <div className="text-center">
          <div className="text-4xl font-bold text-gray-800 mb-4">
            {currentPair.cue}
          </div>
          <div className="text-3xl font-semibold text-blue-600">
            {currentPair.target}
          </div>
          <div className="mt-8 text-gray-500">
            Remember this pair...
          </div>
        </div>
      ) : (
        // Test mode: show cue, ask for target
        <form onSubmit={handleTestSubmit} className="text-center w-full max-w-md">
          <div className="text-4xl font-bold text-gray-800 mb-8">
            {currentPair.cue}
          </div>
          <div className="text-lg text-gray-600 mb-4">
            What word was paired with this?
          </div>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            className="w-full px-4 py-3 text-xl border-2 border-gray-300 rounded-lg 
                       focus:border-blue-500 focus:outline-none text-center uppercase"
            placeholder="Type your answer..."
            autoFocus
            autoComplete="off"
          />
          <button
            type="submit"
            className="mt-4 w-full bg-blue-500 hover:bg-blue-600 text-white 
                       px-6 py-3 rounded-lg font-semibold transition-colors"
          >
            Submit
          </button>
        </form>
      )}
    </div>
  );
};

export default PairedAssociatesTask;
