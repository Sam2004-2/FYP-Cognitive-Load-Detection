import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import CuedRecallTest from '../components/study/CuedRecallTest';
import RecognitionTest from '../components/study/RecognitionTest';
import { STUDY_RECORD_VERSION, STUDY_CONFIG } from '../config/studyConfig';
import { getPendingDelayedTasks, StudyAPIError, uploadDelayedRecord } from '../services/studyApiClient';
import { ACTIVITY_PAGES, trackPageView } from '../services/studyActivityTracker';
import { createDelayedRecordId, listPendingDelayedTests, saveDelayedResult } from '../services/studyStorage';
import { DelayedPacket } from '../services/studyStimuli';
import { triggerDownload } from '../services/studyExport';
import {
  PendingDelayedTask,
  StudyDelayedTestRecord,
  StudySessionRecord,
  StudyStimulusItem,
  StudyTrialResult,
} from '../types/study';

type DelayedPhase =
  | 'select'
  | 'test_easy_recognition'
  | 'test_easy_cued'
  | 'test_hard_recognition'
  | 'test_hard_cued'
  | 'done';

interface DelayedRouteState {
  participantId?: string;
}

interface LoadedTarget {
  source: 'server' | 'local' | 'imported';
  participantId: string;
  sessionNumber: 1 | 2;
  condition: 'adaptive' | 'baseline';
  form: 'A' | 'B';
  linkedSessionRecordId: string;
  dueAtIso: string;
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
}

function itemsFromSession(record: StudySessionRecord): {
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
} {
  const learning = record.trials.filter((trial) => trial.kind === 'learning');
  const unique = new Map<string, StudyStimulusItem>();

  for (const trial of learning) {
    if (!unique.has(trial.itemId)) {
      unique.set(trial.itemId, {
        id: trial.itemId,
        cue: trial.cue,
        target: trial.target,
        difficulty: trial.difficulty,
        interferenceGroup: trial.itemId.split('-').slice(0, 2).join('-'),
      });
    }
  }

  const items = Array.from(unique.values());
  return {
    easyItems: items.filter((item) => item.difficulty === 'easy'),
    hardItems: items.filter((item) => item.difficulty === 'hard'),
  };
}

const StudyDelayedTest: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = location.state as DelayedRouteState | undefined;

  const [participantLookupId, setParticipantLookupId] = useState(routeState?.participantId ?? '');
  const [serverPending, setServerPending] = useState<PendingDelayedTask[]>([]);
  const [serverStatus, setServerStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [serverError, setServerError] = useState('');

  const localPending = useMemo(() => {
    const participantId = participantLookupId.trim();
    if (!participantId) {
      return listPendingDelayedTests();
    }
    return listPendingDelayedTests(participantId);
  }, [participantLookupId]);

  const [loaded, setLoaded] = useState<LoadedTarget | null>(null);
  const [phase, setPhase] = useState<DelayedPhase>('select');
  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [completedRecord, setCompletedRecord] = useState<StudyDelayedTestRecord | null>(null);
  const [sessionStartMs] = useState(Date.now());

  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState('');

  useEffect(() => {
    trackPageView({
      page: ACTIVITY_PAGES.STUDY_DELAYED,
      participantId: routeState?.participantId,
    });
  }, [routeState?.participantId]);

  const cancelledRef = useRef(false);

  const refreshServerPending = useCallback(async (participantId: string) => {
    const trimmed = participantId.trim();
    if (!trimmed) {
      setServerPending([]);
      setServerStatus('idle');
      setServerError('');
      return;
    }

    setServerStatus('loading');
    setServerError('');

    try {
      const pending = await getPendingDelayedTasks(trimmed);
      if (cancelledRef.current) return;
      setServerPending(pending);
      setServerStatus('ready');
    } catch (err) {
      if (cancelledRef.current) return;
      console.error('Failed to fetch server pending delayed tests:', err);
      setServerPending([]);
      setServerStatus('error');
      if (err instanceof StudyAPIError) {
        setServerError(`Failed to load server pending tasks (${err.status ?? 'unknown'}).`);
      } else {
        setServerError('Failed to load server pending tasks.');
      }
    }
  }, []);

  useEffect(() => {
    cancelledRef.current = false;
    void refreshServerPending(participantLookupId);
    return () => {
      cancelledRef.current = true;
    };
  }, [participantLookupId, refreshServerPending]);

  const startFromLocalPending = (record: StudySessionRecord) => {
    const { easyItems, hardItems } = itemsFromSession(record);
    setLoaded({
      source: 'local',
      participantId: record.participantId,
      sessionNumber: record.sessionNumber,
      condition: record.condition,
      form: record.form,
      linkedSessionRecordId: record.recordId,
      dueAtIso: record.delayedDueAtIso,
      easyItems,
      hardItems,
    });
    setTrials([]);
    setCompletedRecord(null);
    setUploadStatus('idle');
    setUploadMessage('');
    setPhase('test_easy_recognition');
  };

  const startFromServerPending = (task: PendingDelayedTask) => {
    setLoaded({
      source: 'server',
      participantId: task.participantId,
      sessionNumber: task.sessionNumber,
      condition: task.condition,
      form: task.form,
      linkedSessionRecordId: task.linkedSessionRecordId,
      dueAtIso: task.dueAtIso,
      easyItems: task.easyItems,
      hardItems: task.hardItems,
    });
    setTrials([]);
    setCompletedRecord(null);
    setUploadStatus('idle');
    setUploadMessage('');
    setPhase('test_easy_recognition');
  };

  const handlePacketImport = async (file: File) => {
    const text = await file.text();
    const packet = JSON.parse(text) as DelayedPacket;

    setLoaded({
      source: 'imported',
      participantId: packet.participantId,
      sessionNumber: packet.sessionNumber,
      condition: 'baseline',
      form: packet.form,
      linkedSessionRecordId: '',
      dueAtIso: new Date().toISOString(),
      easyItems: packet.easyItems,
      hardItems: packet.hardItems,
    });
    setTrials([]);
    setCompletedRecord(null);
    setUploadStatus('idle');
    setUploadMessage('');
    setPhase('test_easy_recognition');
  };

  const appendTrials = (newTrials: StudyTrialResult[]) => {
    setTrials((prev) => [...prev, ...newTrials]);
  };

  const finish = async (finalTrials: StudyTrialResult[] = []) => {
    if (!loaded) return;

    const merged = [...trials, ...finalTrials];
    const recognition = merged.filter((trial) => trial.kind === 'recognition');
    const cued = merged.filter((trial) => trial.kind === 'cued_recall');

    const record: StudyDelayedTestRecord = {
      recordVersion: STUDY_RECORD_VERSION,
      recordId: createDelayedRecordId(loaded.participantId, loaded.sessionNumber, loaded.condition),
      linkedSessionRecordId: loaded.linkedSessionRecordId,
      participantId: loaded.participantId,
      sessionNumber: loaded.sessionNumber,
      condition: loaded.condition,
      form: loaded.form,
      dueAtIso: loaded.dueAtIso,
      completedAtIso: new Date().toISOString(),
      trials: merged,
      recognitionAccuracy:
        recognition.length > 0 ? recognition.filter((trial) => trial.correct).length / recognition.length : 0,
      recognitionMeanRtMs:
        recognition.length > 0
          ? recognition.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / recognition.length
          : 0,
      cuedRecallAccuracy: cued.length > 0 ? cued.filter((trial) => trial.correct).length / cued.length : 0,
      cuedRecallMeanRtMs:
        cued.length > 0 ? cued.reduce((sum, trial) => sum + trial.reactionTimeMs, 0) / cued.length : 0,
    };

    saveDelayedResult(record);
    setCompletedRecord(record);
    setPhase('done');

    setUploadStatus('uploading');
    setUploadMessage('Uploading delayed test record to study server...');

    try {
      const result = await uploadDelayedRecord(record);
      setUploadStatus('success');
      setUploadMessage(`Upload complete. Record ${result.recordId} stored at ${new Date(result.storedAtIso).toLocaleString()}.`);
      await refreshServerPending(record.participantId);
    } catch (err) {
      console.error('Delayed upload failed:', err);
      setUploadStatus('error');
      if (err instanceof StudyAPIError) {
        setUploadMessage(`Upload failed (${err.status ?? 'unknown'}). Download fallback JSON below.`);
      } else {
        setUploadMessage('Upload failed. Download fallback JSON below.');
      }
    }
  };

  const exportDelayedJson = () => {
    if (!completedRecord) return;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    triggerDownload(
      JSON.stringify(completedRecord, null, 2),
      `study_${completedRecord.participantId}_delayed_session${completedRecord.sessionNumber}_${stamp}.json`,
      'application/json'
    );
  };

  return (
    <div className="min-h-screen">
      <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">Delayed Test</h1>
            <p className="text-gray-600 mt-1">Session-linked delayed recognition and cued recall.</p>
          </div>
          <button onClick={() => navigate('/study/setup')} className="text-gray-600 hover:text-gray-800">
            Close
          </button>
        </div>

        {phase === 'select' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
              <h2 className="text-xl font-semibold text-gray-800">Load Pending Tests by Participant ID</h2>
              <div className="flex flex-col sm:flex-row gap-3">
                <input
                  value={participantLookupId}
                  onChange={(event) => setParticipantLookupId(event.target.value)}
                  placeholder="Participant ID"
                  className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:border-gray-400"
                />
                <button
                  onClick={() => {
                    void refreshServerPending(participantLookupId);
                  }}
                  className="px-4 py-2 rounded-lg bg-gray-900 hover:bg-gray-800 text-white"
                >
                  Refresh
                </button>
              </div>

              {serverStatus === 'loading' && (
                <p className="text-sm text-gray-600">Checking server for delayed tests...</p>
              )}
              {serverStatus === 'error' && (
                <p className="text-sm text-red-700">{serverError}</p>
              )}
              {serverStatus === 'ready' && serverPending.length === 0 && (
                <p className="text-sm text-gray-600">No server-side delayed tests found for this participant.</p>
              )}
              {serverPending.length > 0 && (
                <div className="space-y-3">
                  {serverPending.map((task) => (
                    <button
                      key={`${task.linkedSessionRecordId}-${task.sessionNumber}`}
                      onClick={() => startFromServerPending(task)}
                      className="w-full text-left px-4 py-3 rounded-lg border border-gray-300 hover:border-gray-400 bg-white"
                    >
                      <div className="font-medium text-gray-800">
                        {task.participantId} • Session {task.sessionNumber} • {task.condition}
                      </div>
                      <div className="text-sm text-gray-500">Due: {new Date(task.dueAtIso).toLocaleString()}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Local fallback pending tests</h2>
              {localPending.length === 0 ? (
                <p className="text-gray-600">No pending delayed tests found in local storage.</p>
              ) : (
                <div className="space-y-3">
                  {localPending.map((record) => (
                    <button
                      key={record.recordId}
                      onClick={() => startFromLocalPending(record)}
                      className="w-full text-left px-4 py-3 rounded-lg border border-gray-300 hover:border-gray-400 bg-white"
                    >
                      <div className="font-medium text-gray-800">
                        {record.participantId} • Session {record.sessionNumber} • {record.condition}
                      </div>
                      <div className="text-sm text-gray-500">Due: {new Date(record.delayedDueAtIso).toLocaleString()}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Import delayed packet (fallback)</h2>
              <input
                type="file"
                accept="application/json"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void handlePacketImport(file);
                  }
                }}
              />
              <p className="text-sm text-gray-500 mt-2">
                Use this only if server lookup is unavailable and the researcher provided a delayed packet JSON.
              </p>
            </div>
          </div>
        )}

        {loaded && phase === 'test_easy_recognition' && (
          <RecognitionTest
            items={loaded.easyItems}
            blockIndex={1}
            phase="test_easy_recognition"
            condition={loaded.condition}
            form={loaded.form}
            sessionStartMs={sessionStartMs}
            participantSeed={`${loaded.participantId}:delayed:easy`}
            choiceCount={STUDY_CONFIG.recognitionChoices}
            useInterferenceDistractors={false}
            onComplete={(newTrials) => {
              appendTrials(newTrials);
              setPhase('test_easy_cued');
            }}
          />
        )}

        {loaded && phase === 'test_easy_cued' && (
          <CuedRecallTest
            items={loaded.easyItems}
            blockIndex={1}
            phase="test_easy_cued_recall"
            condition={loaded.condition}
            form={loaded.form}
            sessionStartMs={sessionStartMs}
            onComplete={(newTrials) => {
              appendTrials(newTrials);
              setPhase('test_hard_recognition');
            }}
          />
        )}

        {loaded && phase === 'test_hard_recognition' && (
          <RecognitionTest
            items={loaded.hardItems}
            blockIndex={2}
            phase="test_hard_recognition"
            condition={loaded.condition}
            form={loaded.form}
            sessionStartMs={sessionStartMs}
            participantSeed={`${loaded.participantId}:delayed:hard`}
            choiceCount={STUDY_CONFIG.recognitionChoices}
            useInterferenceDistractors={true}
            onComplete={(newTrials) => {
              appendTrials(newTrials);
              setPhase('test_hard_cued');
            }}
          />
        )}

        {loaded && phase === 'test_hard_cued' && (
          <CuedRecallTest
            items={loaded.hardItems}
            blockIndex={2}
            phase="test_hard_cued_recall"
            condition={loaded.condition}
            form={loaded.form}
            sessionStartMs={sessionStartMs}
            onComplete={(newTrials) => {
              void finish(newTrials);
            }}
          />
        )}

        {phase === 'done' && completedRecord && (
          <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
            <h2 className="text-2xl font-semibold text-gray-800">Delayed test complete</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-lg bg-gray-50 p-4">
                <div className="text-sm text-gray-500">Recognition accuracy</div>
                <div className="text-2xl font-semibold text-gray-800">
                  {(completedRecord.recognitionAccuracy * 100).toFixed(1)}%
                </div>
              </div>
              <div className="rounded-lg bg-gray-50 p-4">
                <div className="text-sm text-gray-500">Cued recall accuracy</div>
                <div className="text-2xl font-semibold text-gray-800">
                  {(completedRecord.cuedRecallAccuracy * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {uploadStatus !== 'idle' && (
              <div
                className={`text-sm rounded px-3 py-2 border ${
                  uploadStatus === 'error'
                    ? 'text-red-700 bg-red-50 border-red-100'
                    : uploadStatus === 'success'
                    ? 'text-green-700 bg-green-50 border-green-200'
                    : 'text-gray-700 bg-gray-50 border-gray-200'
                }`}
              >
                {uploadMessage}
              </div>
            )}

            <div className="flex gap-3">
              {uploadStatus === 'error' && (
                <button
                  onClick={exportDelayedJson}
                  className="px-4 py-2 rounded-lg bg-gray-900 hover:bg-gray-800 text-white"
                >
                  Export Delayed JSON (Fallback)
                </button>
              )}
              <button
                onClick={() => navigate('/study/setup', { state: { participantId: completedRecord.participantId } })}
                className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-800"
              >
                Back to Study Setup
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StudyDelayedTest;
