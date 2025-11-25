import React, { useState, useEffect, useCallback } from 'react';
import { Difficulty } from './TaskPanel';

interface MemoryTaskProps {
  difficulty: Difficulty;
  onComplete: (correct: boolean) => void;
}

type TaskMode = 'sequence' | 'nback';

const MemoryTask: React.FC<MemoryTaskProps> = ({ difficulty, onComplete }) => {
  const [mode, setMode] = useState<TaskMode>('sequence');
  const [phase, setPhase] = useState<'showing' | 'input' | 'feedback'>('showing');
  const [sequence, setSequence] = useState<number[]>([]);
  const [userInput, setUserInput] = useState<string>('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [showingDigit, setShowingDigit] = useState<number | null>(null);
  
  // N-back specific state
  const [nbackSequence, setNbackSequence] = useState<number[]>([]);
  const [nbackIndex, setNbackIndex] = useState(0);
  const [nbackN, setNbackN] = useState(1);
  const [nbackScore, setNbackScore] = useState({ hits: 0, misses: 0, falseAlarms: 0 });
  const [nbackTrials, setNbackTrials] = useState(0);
  const [nbackFeedback, setNbackFeedback] = useState<'correct' | 'wrong' | null>(null);

  const getSequenceLength = useCallback(() => {
    switch (difficulty) {
      case 'easy': return 4;
      case 'medium': return 6;
      case 'hard': return 8;
    }
  }, [difficulty]);

  const getShowTime = useCallback(() => {
    switch (difficulty) {
      case 'easy': return 1200;
      case 'medium': return 1000;
      case 'hard': return 800;
    }
  }, [difficulty]);

  const generateSequence = useCallback(() => {
    const length = getSequenceLength();
    const seq: number[] = [];
    for (let i = 0; i < length; i++) {
      seq.push(Math.floor(Math.random() * 10));
    }
    return seq;
  }, [getSequenceLength]);

  const generateNbackSequence = useCallback(() => {
    const n = difficulty === 'easy' ? 1 : difficulty === 'medium' ? 2 : 3;
    setNbackN(n);
    
    const length = 15 + (difficulty === 'hard' ? 5 : 0);
    const seq: number[] = [];
    
    // Generate sequence with ~30% matches
    for (let i = 0; i < length; i++) {
      if (i >= n && Math.random() < 0.3) {
        // Create a match
        seq.push(seq[i - n]);
      } else {
        // Random digit (avoiding accidental matches where possible)
        let digit;
        do {
          digit = Math.floor(Math.random() * 9) + 1;
        } while (i >= n && digit === seq[i - n] && Math.random() > 0.5);
        seq.push(digit);
      }
    }
    return seq;
  }, [difficulty]);

  const startSequenceTask = useCallback(() => {
    const newSequence = generateSequence();
    setSequence(newSequence);
    setCurrentIndex(0);
    setUserInput('');
    setPhase('showing');
    setIsCorrect(null);
  }, [generateSequence]);

  const startNbackTask = useCallback(() => {
    const newSequence = generateNbackSequence();
    setNbackSequence(newSequence);
    setNbackIndex(0);
    setNbackScore({ hits: 0, misses: 0, falseAlarms: 0 });
    setNbackTrials(0);
    setPhase('showing');
    setNbackFeedback(null);
  }, [generateNbackSequence]);

  // Show sequence digits one by one
  useEffect(() => {
    if (mode === 'sequence' && phase === 'showing' && sequence.length > 0) {
      if (currentIndex < sequence.length) {
        setShowingDigit(sequence[currentIndex]);
        const timer = setTimeout(() => {
          setShowingDigit(null);
          setTimeout(() => {
            setCurrentIndex(prev => prev + 1);
          }, 200);
        }, getShowTime());
        return () => clearTimeout(timer);
      } else {
        setPhase('input');
      }
    }
  }, [mode, phase, currentIndex, sequence, getShowTime]);

  // N-back digit display
  useEffect(() => {
    if (mode === 'nback' && phase === 'showing' && nbackSequence.length > 0) {
      if (nbackIndex < nbackSequence.length) {
        setShowingDigit(nbackSequence[nbackIndex]);
        
        const displayTime = difficulty === 'easy' ? 2000 : difficulty === 'medium' ? 1500 : 1200;
        const timer = setTimeout(() => {
          // Check if this was a match and user didn't respond
          const isMatch = nbackIndex >= nbackN && nbackSequence[nbackIndex] === nbackSequence[nbackIndex - nbackN];
          if (isMatch && nbackFeedback === null) {
            setNbackScore(prev => ({ ...prev, misses: prev.misses + 1 }));
          }
          
          setShowingDigit(null);
          setNbackFeedback(null);
          
          setTimeout(() => {
            setNbackIndex(prev => prev + 1);
            setNbackTrials(prev => prev + 1);
          }, 300);
        }, displayTime);
        
        return () => clearTimeout(timer);
      } else {
        // End of n-back task
        setPhase('feedback');
        const totalMatches = nbackSequence.filter((_, i) => 
          i >= nbackN && nbackSequence[i] === nbackSequence[i - nbackN]
        ).length;
        const accuracy = totalMatches > 0 ? nbackScore.hits / totalMatches : 1;
        onComplete(accuracy >= 0.6 && nbackScore.falseAlarms <= 3);
      }
    }
  }, [mode, phase, nbackIndex, nbackSequence, nbackN, difficulty, nbackFeedback, nbackScore, onComplete]);

  const handleNbackResponse = () => {
    if (phase !== 'showing' || nbackIndex < nbackN) return;
    
    const isMatch = nbackSequence[nbackIndex] === nbackSequence[nbackIndex - nbackN];
    
    if (isMatch) {
      setNbackScore(prev => ({ ...prev, hits: prev.hits + 1 }));
      setNbackFeedback('correct');
    } else {
      setNbackScore(prev => ({ ...prev, falseAlarms: prev.falseAlarms + 1 }));
      setNbackFeedback('wrong');
    }
  };

  const handleSequenceSubmit = () => {
    const userSequence = userInput.split('').map(Number);
    const correct = userSequence.length === sequence.length && 
      userSequence.every((digit, i) => digit === sequence[i]);
    
    setIsCorrect(correct);
    setPhase('feedback');
    onComplete(correct);
  };

  const handleNextTask = () => {
    if (mode === 'sequence') {
      startSequenceTask();
    } else {
      startNbackTask();
    }
  };

  // Initialize task
  useEffect(() => {
    if (mode === 'sequence') {
      startSequenceTask();
    } else {
      startNbackTask();
    }
  }, [mode, startSequenceTask, startNbackTask]);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      {/* Mode selector */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setMode('sequence')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            mode === 'sequence' 
              ? 'bg-indigo-100 text-indigo-700 border-2 border-indigo-300' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Sequence Recall
        </button>
        <button
          onClick={() => setMode('nback')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            mode === 'nback' 
              ? 'bg-indigo-100 text-indigo-700 border-2 border-indigo-300' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          {nbackN}-Back Task
        </button>
      </div>

      {/* Sequence Recall Mode */}
      {mode === 'sequence' && (
        <div className="text-center">
          {phase === 'showing' && (
            <div className="space-y-4">
              <p className="text-gray-600">Memorize the sequence</p>
              <div className="h-32 flex items-center justify-center">
                {showingDigit !== null ? (
                  <div className="text-7xl font-bold text-indigo-600 animate-pulse">
                    {showingDigit}
                  </div>
                ) : (
                  <div className="text-7xl font-bold text-gray-200">•</div>
                )}
              </div>
              <div className="flex justify-center gap-1">
                {sequence.map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-full transition-colors ${
                      i < currentIndex ? 'bg-indigo-500' : 
                      i === currentIndex ? 'bg-indigo-300 animate-pulse' : 'bg-gray-200'
                    }`}
                  />
                ))}
              </div>
            </div>
          )}

          {phase === 'input' && (
            <div className="space-y-4">
              <p className="text-gray-600">Enter the sequence you saw</p>
              <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value.replace(/[^0-9]/g, '').slice(0, sequence.length))}
                placeholder={`Enter ${sequence.length} digits`}
                className="text-center text-3xl font-mono tracking-widest border-2 border-gray-300 rounded-lg px-4 py-3 w-full max-w-xs focus:border-indigo-500 focus:outline-none"
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && userInput.length === sequence.length && handleSequenceSubmit()}
              />
              <div className="text-sm text-gray-400">
                {userInput.length}/{sequence.length} digits
              </div>
              <button
                onClick={handleSequenceSubmit}
                disabled={userInput.length !== sequence.length}
                className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-colors"
              >
                Submit
              </button>
            </div>
          )}

          {phase === 'feedback' && (
            <div className="space-y-4">
              <div className={`text-6xl ${isCorrect ? '🎉' : '😔'}`}>
                {isCorrect ? '🎉' : '😔'}
              </div>
              <p className={`text-xl font-semibold ${isCorrect ? 'text-green-600' : 'text-red-600'}`}>
                {isCorrect ? 'Correct!' : 'Incorrect'}
              </p>
              {!isCorrect && (
                <p className="text-gray-600">
                  The sequence was: <span className="font-mono font-bold">{sequence.join('')}</span>
                </p>
              )}
              <button
                onClick={handleNextTask}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-8 py-3 rounded-lg font-medium transition-colors"
              >
                Next Sequence
              </button>
            </div>
          )}
        </div>
      )}

      {/* N-Back Mode */}
      {mode === 'nback' && (
        <div className="text-center">
          <div className="mb-4 text-sm text-gray-600">
            Press SPACE or click when the current digit matches the one from <strong>{nbackN} step{nbackN > 1 ? 's' : ''} ago</strong>
          </div>

          {phase === 'showing' && (
            <div className="space-y-4">
              <div 
                className={`h-40 flex items-center justify-center rounded-xl transition-colors ${
                  nbackFeedback === 'correct' ? 'bg-green-100' :
                  nbackFeedback === 'wrong' ? 'bg-red-100' : 'bg-gray-50'
                }`}
                onClick={handleNbackResponse}
                onKeyDown={(e) => e.key === ' ' && handleNbackResponse()}
                tabIndex={0}
              >
                {showingDigit !== null ? (
                  <div className={`text-7xl font-bold transition-colors ${
                    nbackFeedback === 'correct' ? 'text-green-600' :
                    nbackFeedback === 'wrong' ? 'text-red-600' : 'text-indigo-600'
                  }`}>
                    {showingDigit}
                  </div>
                ) : (
                  <div className="text-7xl font-bold text-gray-200">•</div>
                )}
              </div>
              
              <button
                onClick={handleNbackResponse}
                disabled={nbackIndex < nbackN || nbackFeedback !== null}
                className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-colors"
              >
                Match! (or press Space)
              </button>

              <div className="flex justify-center gap-4 text-sm text-gray-500">
                <span>Trial: {nbackIndex + 1}/{nbackSequence.length}</span>
                <span>Hits: {nbackScore.hits}</span>
                <span>False Alarms: {nbackScore.falseAlarms}</span>
              </div>
            </div>
          )}

          {phase === 'feedback' && (
            <div className="space-y-4">
              <div className="text-6xl">
                {nbackScore.hits > nbackScore.misses ? '🎉' : '🤔'}
              </div>
              <p className="text-xl font-semibold text-gray-800">
                Task Complete!
              </p>
              <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Correct Matches:</span>
                  <span className="font-bold text-green-600">{nbackScore.hits}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Missed Matches:</span>
                  <span className="font-bold text-yellow-600">{nbackScore.misses}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">False Alarms:</span>
                  <span className="font-bold text-red-600">{nbackScore.falseAlarms}</span>
                </div>
              </div>
              <button
                onClick={handleNextTask}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-8 py-3 rounded-lg font-medium transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="mt-6 pt-4 border-t border-gray-100">
        <details className="text-sm text-gray-500">
          <summary className="cursor-pointer hover:text-gray-700">How to play</summary>
          <div className="mt-2 pl-4 space-y-2">
            {mode === 'sequence' ? (
              <>
                <p>• Watch the digits appear one by one</p>
                <p>• Remember the complete sequence</p>
                <p>• Type the sequence when prompted</p>
                <p>• Current difficulty: {getSequenceLength()} digits</p>
              </>
            ) : (
              <>
                <p>• Watch digits appear one at a time</p>
                <p>• Press SPACE when the current digit matches the one from {nbackN} step{nbackN > 1 ? 's' : ''} ago</p>
                <p>• Try to catch all matches without false alarms</p>
                <p>• This is a {nbackN}-back task</p>
              </>
            )}
          </div>
        </details>
      </div>
    </div>
  );
};

export default MemoryTask;

