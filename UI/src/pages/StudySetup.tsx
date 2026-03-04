import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { getPendingDelayedTasks } from '../services/studyApiClient';
import { ACTIVITY_PAGES, trackPageView } from '../services/studyActivityTracker';
import { getStudyPlan, computeStudyAssignment, validateSession2Timing } from '../services/studyProtocol';
import { getMostRecentSession, listPendingDelayedTests } from '../services/studyStorage';
import { PendingDelayedTask, StudySessionNumber, StudySetupState } from '../types/study';

interface StudySetupRouteState {
  participantId?: string;
  suggestedSessionNumber?: StudySessionNumber;
}

const StudySetup: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = location.state as StudySetupRouteState | undefined;

  const [participantId, setParticipantId] = useState(routeState?.participantId ?? '');
  const [sessionNumber, setSessionNumber] = useState<StudySessionNumber>(1);
  const [allowEarlySession2, setAllowEarlySession2] = useState(false);

  const [serverPending, setServerPending] = useState<PendingDelayedTask[]>([]);
  const [serverPendingStatus, setServerPendingStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [serverPendingError, setServerPendingError] = useState('');

  useEffect(() => {
    if (routeState?.participantId) {
      setParticipantId(routeState.participantId);
    }
  }, [routeState?.participantId]);

  useEffect(() => {
    const suggestedSessionNumber = routeState?.suggestedSessionNumber;
    const suggestedParticipantId = routeState?.participantId?.trim().toLowerCase();
    const currentParticipantId = participantId.trim().toLowerCase();

    if (!suggestedSessionNumber || !suggestedParticipantId) return;

    setSessionNumber(currentParticipantId === suggestedParticipantId ? suggestedSessionNumber : 1);
  }, [participantId, routeState?.participantId, routeState?.suggestedSessionNumber]);

  useEffect(() => {
    trackPageView({
      page: ACTIVITY_PAGES.STUDY_SETUP,
      participantId: routeState?.participantId,
    });
  }, [routeState?.participantId]);

  useEffect(() => {
    const id = participantId.trim();
    if (!id) {
      setServerPending([]);
      setServerPendingStatus('idle');
      setServerPendingError('');
      return;
    }

    let cancelled = false;
    const loadPending = async () => {
      setServerPendingStatus('loading');
      setServerPendingError('');

      try {
        const tasks = await getPendingDelayedTasks(id);
        if (cancelled) return;
        setServerPending(tasks);
        setServerPendingStatus('ready');
      } catch (err) {
        if (cancelled) return;
        console.error('Failed to fetch server pending delayed tasks:', err);
        setServerPending([]);
        setServerPendingStatus('error');
        setServerPendingError('Could not load server pending delayed tasks. You can still continue this session.');
      }
    };

    void loadPending();

    return () => {
      cancelled = true;
    };
  }, [participantId]);

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

  const localPendingDelayed = useMemo(() => {
    if (!participantId.trim()) return [];
    return listPendingDelayedTests(participantId.trim());
  }, [participantId]);

  const canStart = Boolean(
    assignment && (!timingValidation.tooEarly || allowEarlySession2 || sessionNumber === 1)
  );

  const handleStart = () => {
    if (!assignment) return;

    trackPageView({
      page: ACTIVITY_PAGES.STUDY_SETUP_START,
      participantId: participantId.trim(),
      sessionNumber,
      condition: assignment.condition,
    });

    const state: StudySetupState = {
      participantId: participantId.trim(),
      assignment,
      plan: getStudyPlan(),
      session2StartedEarlyOverride:
        sessionNumber === 2 ? allowEarlySession2 && timingValidation.tooEarly : false,
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
          <button onClick={() => navigate('/')} className="text-gray-600 hover:text-gray-800">
            Close
          </button>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Participant ID</label>
            <input
              value={participantId}
              onChange={(e) => setParticipantId(e.target.value)}
              placeholder="e.g., P-260222-A1B2C3"
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Use the participant ID generated on the landing page for both sessions and delayed tests.
            </p>
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

          {participantId.trim() && (
            <div className="rounded-lg bg-purple-50 border border-purple-100 p-4 text-sm text-purple-900 space-y-2">
              <div className="font-medium">Pending delayed tests (server)</div>
              {serverPendingStatus === 'loading' && <div>Checking server for pending delayed tests...</div>}
              {serverPendingStatus === 'error' && <div>{serverPendingError}</div>}
              {serverPendingStatus === 'ready' && serverPending.length === 0 && (
                <div>No server-side delayed tests are pending for this participant.</div>
              )}
              {serverPendingStatus === 'ready' && serverPending.length > 0 && (
                <ul className="list-disc list-inside space-y-1">
                  {serverPending.map((task) => (
                    <li key={`${task.linkedSessionRecordId}-${task.sessionNumber}`}>
                      Session {task.sessionNumber} ({task.condition}) due {new Date(task.dueAtIso).toLocaleString()}
                    </li>
                  ))}
                </ul>
              )}

              {localPendingDelayed.length > 0 && (
                <div className="text-xs text-purple-800">
                  Local device also has {localPendingDelayed.length} pending delayed item(s).
                </div>
              )}

              <button
                onClick={() => navigate('/study/delayed', { state: { participantId: participantId.trim() } })}
                className="mt-1 px-3 py-1.5 rounded bg-purple-600 hover:bg-purple-700 text-white"
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
