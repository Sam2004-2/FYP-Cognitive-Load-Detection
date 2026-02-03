import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import PairedAssociatesTask, {
  WORD_PAIRS_FORM_A,
  WORD_PAIRS_FORM_B,
} from '../components/tasks/PairedAssociatesTask';
import { getStudySession, saveDelayedTestResult } from '../services/apiClient';
import { TaskPerformance, WordPair } from '../types';

const DelayedTest: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pairs, setPairs] = useState<WordPair[]>([]);
  const [participantId, setParticipantId] = useState('');
  const [started, setStarted] = useState(false);
  const [complete, setComplete] = useState(false);
  const [performance, setPerformance] = useState<TaskPerformance | null>(null);
  const [alreadyCompleted, setAlreadyCompleted] = useState(false);

  // Load session data
  useEffect(() => {
    const loadSession = async () => {
      if (!sessionId) {
        setError('No session ID provided');
        setLoading(false);
        return;
      }

      try {
        const session = await getStudySession(sessionId);
        
        // Check if delayed test already completed
        if (session.delayed_test) {
          setAlreadyCompleted(true);
          setPerformance(session.delayed_test);
          setLoading(false);
          return;
        }

        setParticipantId(session.participant_id);

        // Get the pairs that were studied
        const allPairs = session.form_version === 'A' ? WORD_PAIRS_FORM_A : WORD_PAIRS_FORM_B;
        const easyCount = session.task_performance.easy.total;
        const hardCount = session.task_performance.hard.total;
        
        const studiedPairs = [
          ...allPairs.slice(0, easyCount),
          ...allPairs.slice(6, 6 + hardCount),
        ];
        
        setPairs(studiedPairs);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load session');
        setLoading(false);
      }
    };

    loadSession();
  }, [sessionId]);

  const handleTestComplete = async (perf: TaskPerformance) => {
    setPerformance(perf);
    setComplete(true);

    try {
      await saveDelayedTestResult(sessionId!, perf);
    } catch (err) {
      console.error('Failed to save delayed test:', err);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading session...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
          <div className="text-6xl mb-4">❌</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Error</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  if (alreadyCompleted) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
          <div className="text-6xl mb-4">✓</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">
            Already Completed
          </h2>
          <p className="text-gray-600 mb-4">
            The delayed test for this session has already been completed.
          </p>
          {performance && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <p className="text-sm text-gray-600">
                Score: {performance.correct} / {performance.total}
              </p>
            </div>
          )}
          <button
            onClick={() => navigate('/')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  if (!started) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Delayed Recall Test
          </h1>
          <p className="text-gray-600 mb-6">
            Welcome back, <strong>{participantId}</strong>!
          </p>
          <p className="text-gray-600 mb-6">
            You will now be tested on the word pairs you learned previously.
            Type the word that was paired with each cue.
          </p>
          <div className="bg-blue-50 rounded-lg p-4 mb-6">
            <p className="text-sm text-blue-800">
              <strong>{pairs.length} pairs</strong> to recall
            </p>
          </div>
          <button
            onClick={() => setStarted(true)}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-lg font-semibold"
          >
            Start Test
          </button>
        </div>
      </div>
    );
  }

  if (complete) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
          <div className="text-6xl mb-4">✓</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Test Complete!
          </h2>
          <p className="text-gray-600 mb-6">
            Thank you for completing the delayed recall test.
          </p>
          {performance && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <p className="text-2xl font-bold text-blue-600">
                {performance.correct} / {performance.total}
              </p>
              <p className="text-sm text-gray-500">
                ({((performance.correct / performance.total) * 100).toFixed(0)}% correct)
              </p>
            </div>
          )}
          <p className="text-sm text-gray-500 mb-6">
            Your results have been saved. You may close this window.
          </p>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg"
          >
            Return Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-2">
            Delayed Recall Test
          </h2>
          <p className="text-gray-600 mb-6">
            Type the word that was paired with each cue.
          </p>
          <PairedAssociatesTask
            pairs={pairs}
            exposureTimeMs={0}
            mode="test"
            onComplete={handleTestComplete}
          />
        </div>
      </div>
    </div>
  );
};

export default DelayedTest;
