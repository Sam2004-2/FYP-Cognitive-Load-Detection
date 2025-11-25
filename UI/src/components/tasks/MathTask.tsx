import React, { useState, useEffect, useCallback } from 'react';
import { Difficulty } from './TaskPanel';

interface MathTaskProps {
  difficulty: Difficulty;
  onComplete: (correct: boolean) => void;
}

type Operation = '+' | '-' | '×' | '÷';

interface Problem {
  expression: string;
  answer: number;
  timeLimit: number;
}

const MathTask: React.FC<MathTaskProps> = ({ difficulty, onComplete }) => {
  const [problem, setProblem] = useState<Problem | null>(null);
  const [userAnswer, setUserAnswer] = useState<string>('');
  const [phase, setPhase] = useState<'solving' | 'feedback'>('solving');
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [timeRemaining, setTimeRemaining] = useState<number>(0);
  const [timedOut, setTimedOut] = useState(false);

  const generateProblem = useCallback((): Problem => {
    const operations: Operation[] = difficulty === 'easy' 
      ? ['+', '-'] 
      : difficulty === 'medium' 
        ? ['+', '-', '×'] 
        : ['+', '-', '×', '÷'];

    const op = operations[Math.floor(Math.random() * operations.length)];
    
    let a: number, b: number, answer: number, expression: string;

    switch (difficulty) {
      case 'easy':
        // Single digit operations
        a = Math.floor(Math.random() * 9) + 1;
        b = Math.floor(Math.random() * 9) + 1;
        if (op === '-' && b > a) [a, b] = [b, a]; // Ensure positive result
        answer = op === '+' ? a + b : a - b;
        expression = `${a} ${op} ${b}`;
        return { expression, answer, timeLimit: 15 };

      case 'medium':
        if (op === '×') {
          // Multiplication: single × single or small × single
          a = Math.floor(Math.random() * 12) + 2;
          b = Math.floor(Math.random() * 9) + 2;
          answer = a * b;
          expression = `${a} × ${b}`;
        } else {
          // Two-digit +/- operations
          a = Math.floor(Math.random() * 90) + 10;
          b = Math.floor(Math.random() * 90) + 10;
          if (op === '-' && b > a) [a, b] = [b, a];
          answer = op === '+' ? a + b : a - b;
          expression = `${a} ${op} ${b}`;
        }
        return { expression, answer, timeLimit: 20 };

      case 'hard':
        const taskType = Math.floor(Math.random() * 4);
        
        switch (taskType) {
          case 0: // Multi-step: a + b × c
            a = Math.floor(Math.random() * 20) + 5;
            b = Math.floor(Math.random() * 9) + 2;
            const c = Math.floor(Math.random() * 9) + 2;
            answer = a + b * c;
            expression = `${a} + ${b} × ${c}`;
            break;
          
          case 1: // Large multiplication
            a = Math.floor(Math.random() * 20) + 10;
            b = Math.floor(Math.random() * 9) + 2;
            answer = a * b;
            expression = `${a} × ${b}`;
            break;
          
          case 2: // Division with clean result
            b = Math.floor(Math.random() * 9) + 2;
            answer = Math.floor(Math.random() * 15) + 2;
            a = b * answer;
            expression = `${a} ÷ ${b}`;
            break;
          
          default: // Three number operation: a + b - c
            a = Math.floor(Math.random() * 50) + 20;
            b = Math.floor(Math.random() * 30) + 10;
            const d = Math.floor(Math.random() * Math.min(a + b - 1, 40)) + 1;
            answer = a + b - d;
            expression = `${a} + ${b} - ${d}`;
        }
        return { expression, answer, timeLimit: 30 };
    }
  }, [difficulty]);

  const startNewProblem = useCallback(() => {
    const newProblem = generateProblem();
    setProblem(newProblem);
    setUserAnswer('');
    setPhase('solving');
    setIsCorrect(null);
    setTimedOut(false);
    setTimeRemaining(newProblem.timeLimit);
  }, [generateProblem]);

  // Timer countdown
  useEffect(() => {
    if (phase !== 'solving' || !problem) return;

    if (timeRemaining <= 0) {
      setTimedOut(true);
      setIsCorrect(false);
      setPhase('feedback');
      onComplete(false);
      return;
    }

    const timer = setTimeout(() => {
      setTimeRemaining(prev => prev - 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [phase, timeRemaining, problem, onComplete]);

  // Initialize
  useEffect(() => {
    startNewProblem();
  }, [startNewProblem]);

  const handleSubmit = () => {
    if (!problem) return;
    
    const userNum = parseFloat(userAnswer);
    const correct = !isNaN(userNum) && Math.abs(userNum - problem.answer) < 0.001;
    
    setIsCorrect(correct);
    setPhase('feedback');
    onComplete(correct);
  };

  const getTimerColor = () => {
    if (!problem) return 'bg-blue-500';
    const ratio = timeRemaining / problem.timeLimit;
    if (ratio > 0.5) return 'bg-green-500';
    if (ratio > 0.25) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getDifficultyLabel = () => {
    switch (difficulty) {
      case 'easy': return 'Single-digit arithmetic';
      case 'medium': return 'Two-digit operations & multiplication';
      case 'hard': return 'Multi-step calculations';
    }
  };

  if (!problem) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="text-sm text-gray-500">{getDifficultyLabel()}</div>
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
            timeRemaining <= 5 ? 'animate-pulse' : ''
          } ${getTimerColor()}`}>
            {timeRemaining}
          </div>
        </div>
      </div>

      {/* Timer bar */}
      <div className="h-2 bg-gray-200 rounded-full mb-6 overflow-hidden">
        <div 
          className={`h-full transition-all duration-1000 ${getTimerColor()}`}
          style={{ width: `${(timeRemaining / problem.timeLimit) * 100}%` }}
        />
      </div>

      {phase === 'solving' && (
        <div className="text-center space-y-6">
          {/* Problem display */}
          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-xl p-8">
            <div className="text-4xl md:text-5xl font-bold text-gray-800 font-mono">
              {problem.expression} = ?
            </div>
          </div>

          {/* Answer input */}
          <div className="space-y-4">
            <input
              type="text"
              value={userAnswer}
              onChange={(e) => {
                const val = e.target.value;
                // Allow numbers, negative sign, and decimal point
                if (/^-?\d*\.?\d*$/.test(val)) {
                  setUserAnswer(val);
                }
              }}
              placeholder="Your answer"
              className="text-center text-3xl font-mono border-2 border-gray-300 rounded-lg px-4 py-3 w-full max-w-xs focus:border-emerald-500 focus:outline-none"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && userAnswer && handleSubmit()}
            />
            <div>
              <button
                onClick={handleSubmit}
                disabled={!userAnswer}
                className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-colors"
              >
                Submit Answer
              </button>
            </div>
          </div>

          {/* Quick number pad for mobile */}
          <div className="grid grid-cols-5 gap-2 max-w-xs mx-auto">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 0].map((num) => (
              <button
                key={num}
                onClick={() => setUserAnswer(prev => prev + num)}
                className="aspect-square text-xl font-bold bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                {num}
              </button>
            ))}
            <button
              onClick={() => setUserAnswer(prev => prev + '-')}
              className="aspect-square text-xl font-bold bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              −
            </button>
            <button
              onClick={() => setUserAnswer(prev => prev.slice(0, -1))}
              className="aspect-square text-xl font-bold bg-red-100 hover:bg-red-200 text-red-600 rounded-lg transition-colors col-span-2"
            >
              ⌫
            </button>
            <button
              onClick={() => setUserAnswer('')}
              className="aspect-square text-sm font-medium bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {phase === 'feedback' && (
        <div className="text-center space-y-4">
          <div className="text-6xl">
            {timedOut ? '⏰' : isCorrect ? '✅' : '❌'}
          </div>
          <p className={`text-xl font-semibold ${
            timedOut ? 'text-yellow-600' : isCorrect ? 'text-green-600' : 'text-red-600'
          }`}>
            {timedOut ? 'Time\'s Up!' : isCorrect ? 'Correct!' : 'Incorrect'}
          </p>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-gray-600 mb-2">
              {problem.expression} = <span className="font-bold text-emerald-600">{problem.answer}</span>
            </p>
            {!isCorrect && userAnswer && (
              <p className="text-sm text-gray-500">
                Your answer: <span className="font-mono">{userAnswer}</span>
              </p>
            )}
          </div>

          <button
            onClick={startNewProblem}
            className="bg-emerald-500 hover:bg-emerald-600 text-white px-8 py-3 rounded-lg font-medium transition-colors"
          >
            Next Problem
          </button>
        </div>
      )}

      {/* Tips */}
      <div className="mt-6 pt-4 border-t border-gray-100">
        <details className="text-sm text-gray-500">
          <summary className="cursor-pointer hover:text-gray-700">Tips</summary>
          <div className="mt-2 pl-4 space-y-1">
            <p>• Remember order of operations: × and ÷ before + and −</p>
            <p>• Use the number pad or your keyboard</p>
            <p>• Press Enter to submit quickly</p>
            {difficulty === 'hard' && (
              <p>• Break complex problems into smaller steps</p>
            )}
          </div>
        </details>
      </div>
    </div>
  );
};

export default MathTask;

