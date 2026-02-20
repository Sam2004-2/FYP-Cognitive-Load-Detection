import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getStudyPlan, computeStudyAssignment, validateSession2Timing } from '../services/studyProtocol';
import { getMostRecentSession, listPendingDelayedTests } from '../services/studyStorage';
import { StudySessionNumber, StudySetupState } from '../types/study';

const StudySetup: React.FC = () => {
  const navigate = useNavigate();
  const [participantId, setParticipantId] = useState('');
  const [sessionNumber, setSessionNumber] = useState<StudySessionNumber>(1);
  const [allowEarlySession2, setAllowEarlySession2] = useState(false);

  const previousSession1 = useMemo(() => {
    if (!participantId.trim()) return null;
    return getMostRecentSession(participantId.trim(), 1);
  }, [participantId]);

  const assignment = useMemo(() => {
    if (!participantId.trim()) return null;
    return computeStudyAssignment(participantId.trim(), sessionNumber);
  }, [participantId, sessionNumber]);

  const timingValidation = useMemo(() => {
    return validateSession2Timing(sessionNumber, previousSession1?.completedAtIso);
  }, [sessionNumber, previousSession1?.completedAtIso]);

  const pendingDelayed = useMemo(() => {
    if (!participantId.trim()) return [];
    return listPendingDelayedTests(participantId.trim());
  }, [participantId]);

  const canStart = Boolean(
    assignment &&
      (!timingValidation.tooEarly || allowEarlySession2 || sessionNumber === 1)
  );

  const handleStart = () => {
    if (!assignment) return;

    const state: StudySetupState = {
      participantId: participantId.trim(),
      assignment,
      plan: getStudyPlan(),
      session2StartedEarlyOverride: sessionNumber === 2 ? allowEarlySession2 && timingValidation.tooEarly : false,
    };

    navigate('/study/session', { state });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Study Protocol Setup</h1>
            <p className="text-gray-600 mt-1">Configure crossover session assignment and launch run.</p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="text-gray-600 hover:text-gray-800"
          >
            Close
          </button>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Participant ID</label>
            <input
              value={participantId}
              onChange={(e) => setParticipantId(e.target.value)}
              placeholder="e.g., P012"
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Session Number</label>
            <div className="flex gap-3">
              <button
                onClick={() => setSessionNumber(1)}
                className={`px-4 py-2 rounded-lg border ${sessionNumber === 1 ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300'}`}
              >
                Session 1
              </button>
              <button
                onClick={() => setSessionNumber(2)}
                className={`px-4 py-2 rounded-lg border ${sessionNumber === 2 ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300'}`}
              >
                Session 2
              </button>
            </div>
          </div>

          {assignment && (
            <div className="rounded-lg bg-blue-50 border border-blue-100 p-4 text-sm text-blue-900 space-y-1">
              <div><span className="font-semibold">Condition:</span> {assignment.condition}</div>
              <div><span className="font-semibold">Stimulus form:</span> {assignment.form}</div>
              <div><span className="font-semibold">Condition order:</span> {assignment.conditionOrder[0]} → {assignment.conditionOrder[1]}</div>
              <div><span className="font-semibold">Form order:</span> {assignment.formOrder[0]} → {assignment.formOrder[1]}</div>
              <div><span className="font-semibold">Delayed test due:</span> {new Date(assignment.delayedDueAtIso).toLocaleString()}</div>
            </div>
          )}

          {sessionNumber === 2 && timingValidation.tooEarly && (
            <div className="rounded-lg bg-amber-50 border border-amber-200 p-4 text-sm text-amber-900 space-y-2">
              <div>
                Session 2 is starting early: only {timingValidation.hoursSinceSession1?.toFixed(1)} hours since Session 1 completion.
                Recommended minimum is {timingValidation.recommendedMinimumHours} hours.
              </div>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={allowEarlySession2}
                  onChange={(e) => setAllowEarlySession2(e.target.checked)}
                />
                <span>Allow early start and log override</span>
              </label>
            </div>
          )}

          {pendingDelayed.length > 0 && (
            <div className="rounded-lg bg-purple-50 border border-purple-100 p-4 text-sm text-purple-900">
              <div className="font-medium mb-2">Pending delayed tests for this participant:</div>
              <ul className="list-disc list-inside space-y-1">
                {pendingDelayed.map((session) => (
                  <li key={session.recordId}>
                    Session {session.sessionNumber} ({session.condition}) due {new Date(session.delayedDueAtIso).toLocaleDateString()}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => navigate('/study/delayed')}
                className="mt-3 px-3 py-1.5 rounded bg-purple-600 hover:bg-purple-700 text-white"
              >
                Open Delayed Test Page
              </button>
            </div>
          )}

          <div className="flex justify-end gap-3">
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={handleStart}
              disabled={!canStart}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white"
            >
              Start Study Session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudySetup;
