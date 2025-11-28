import React, { useState } from 'react';
import MemoryTask from './MemoryTask';
import MathTask from './MathTask';

// Literal union types restrict values to specific strings ***
// Exported so child components (MathTask, MemoryTask) can import and use them ***
export type Difficulty = 'easy' | 'medium' | 'hard';
export type TaskType = 'memory' | 'math';

// Optional callback prop using ? - allows parent to listen for task completion ***
// Multiple params allow parent to track performance across task types and difficulties ***
interface TaskPanelProps {
  onTaskComplete?: (correct: boolean, taskType: TaskType, difficulty: Difficulty) => void;
}

const TaskPanel: React.FC<TaskPanelProps> = ({ onTaskComplete }) => {
  const [selectedTask, setSelectedTask] = useState<TaskType | null>(null);
  const [difficulty, setDifficulty] = useState<Difficulty>('easy');
  const [stats, setStats] = useState({
    totalAttempts: 0,
    correctAnswers: 0,
    streak: 0,
  });

  // Functional state update pattern - uses previous state to compute new state ***
  // The ?. optional chaining safely calls callback only if defined ***
  // The ! non-null assertion tells TS we know selectedTask is not null here ***
  const handleTaskComplete = (correct: boolean) => {
    setStats(prev => ({
      totalAttempts: prev.totalAttempts + 1,
      correctAnswers: prev.correctAnswers + (correct ? 1 : 0),
      streak: correct ? prev.streak + 1 : 0,
    }));
    onTaskComplete?.(correct, selectedTask!, difficulty);
  };

  const getDifficultyColor = (d: Difficulty) => {
    switch (d) {
      case 'easy': return 'bg-green-500 hover:bg-green-600';
      case 'medium': return 'bg-yellow-500 hover:bg-yellow-600';
      case 'hard': return 'bg-red-500 hover:bg-red-600';
    }
  };

  const getDifficultyBorder = (d: Difficulty, selected: boolean) => {
    if (!selected) return 'border-gray-200';
    switch (d) {
      case 'easy': return 'border-green-500 bg-green-50';
      case 'medium': return 'border-yellow-500 bg-yellow-50';
      case 'hard': return 'border-red-500 bg-red-50';
    }
  };

  if (!selectedTask) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-semibold text-gray-800 mb-2">Cognitive Tasks</h2>
          <p className="text-gray-600 text-sm">
            Select a task type and difficulty to begin. Higher difficulty tasks typically induce greater cognitive load.
          </p>
        </div>

        {/* Difficulty Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Difficulty Level</label>
          <div className="flex gap-3">
            {(['easy', 'medium', 'hard'] as Difficulty[]).map((d) => (
              <button
                key={d}
                onClick={() => setDifficulty(d)}
                className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all capitalize font-medium ${
                  getDifficultyBorder(d, difficulty === d)
                } ${difficulty === d ? 'shadow-md' : 'hover:border-gray-300'}`}
              >
                <span className={`inline-block w-2 h-2 rounded-full mr-2 ${getDifficultyColor(d).split(' ')[0]}`} />
                {d}
              </button>
            ))}
          </div>
        </div>

        {/* Task Type Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Memory Task Card */}
          <button
            onClick={() => setSelectedTask('memory')}
            className="text-left p-6 rounded-xl border-2 border-gray-200 hover:border-indigo-400 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center mb-3">
              <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center mr-4 group-hover:bg-indigo-200 transition-colors">
                <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">Memory Tasks</h3>
                <p className="text-sm text-gray-500">Sequence recall & n-back</p>
              </div>
            </div>
            <p className="text-sm text-gray-600">
              Test your working memory by remembering sequences of numbers, letters, or patterns.
            </p>
            <div className="mt-4 text-xs text-gray-400">
              {difficulty === 'easy' && '3-4 digit sequences'}
              {difficulty === 'medium' && '5-6 digit sequences + 1-back'}
              {difficulty === 'hard' && '7+ digits + 2-back task'}
            </div>
          </button>

          {/* Math Task Card */}
          <button
            onClick={() => setSelectedTask('math')}
            className="text-left p-6 rounded-xl border-2 border-gray-200 hover:border-emerald-400 hover:shadow-lg transition-all group"
          >
            <div className="flex items-center mb-3">
              <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center mr-4 group-hover:bg-emerald-200 transition-colors">
                <svg className="w-6 h-6 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">Math Tasks</h3>
                <p className="text-sm text-gray-500">Arithmetic challenges</p>
              </div>
            </div>
            <p className="text-sm text-gray-600">
              Solve arithmetic problems of varying complexity under time pressure.
            </p>
            <div className="mt-4 text-xs text-gray-400">
              {difficulty === 'easy' && 'Single-digit operations'}
              {difficulty === 'medium' && 'Two-digit operations'}
              {difficulty === 'hard' && 'Multi-step calculations'}
            </div>
          </button>
        </div>

        {/* Stats Display */}
        {stats.totalAttempts > 0 && (
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Session Stats</h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-gray-800">{stats.totalAttempts}</div>
                <div className="text-xs text-gray-500">Attempts</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {stats.totalAttempts > 0 ? Math.round((stats.correctAnswers / stats.totalAttempts) * 100) : 0}%
                </div>
                <div className="text-xs text-gray-500">Accuracy</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-indigo-600">{stats.streak}</div>
                <div className="text-xs text-gray-500">Streak</div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with back button */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setSelectedTask(null)}
          className="flex items-center text-gray-600 hover:text-gray-800 transition-colors"
        >
          <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Tasks
        </button>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-white text-xs font-medium ${getDifficultyColor(difficulty).split(' ')[0]}`}>
            {difficulty.toUpperCase()}
          </span>
          <span className="text-sm text-gray-500">
            Streak: {stats.streak}
          </span>
        </div>
      </div>

      {/* Task Component */}
      {selectedTask === 'memory' && (
        <MemoryTask difficulty={difficulty} onComplete={handleTaskComplete} />
      )}
      {selectedTask === 'math' && (
        <MathTask difficulty={difficulty} onComplete={handleTaskComplete} />
      )}
    </div>
  );
};

export default TaskPanel;


