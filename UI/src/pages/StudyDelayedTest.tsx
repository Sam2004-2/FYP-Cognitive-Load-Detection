import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CuedRecallTest from '../components/study/CuedRecallTest';
import RecognitionTest from '../components/study/RecognitionTest';
import { STUDY_RECORD_VERSION, STUDY_CONFIG } from '../config/studyConfig';
import { createDelayedRecordId, listPendingDelayedTests, saveDelayedResult } from '../services/studyStorage';
import { DelayedPacket } from '../services/studyStimuli';
import { triggerDownload } from '../services/studyExport';
import { StudyDelayedTestRecord, StudySessionRecord, StudyStimulusItem, StudyTrialResult } from '../types/study';

type DelayedPhase =
  | 'select'
  | 'test_easy_recognition'
  | 'test_easy_cued'
  | 'test_hard_recognition'
  | 'test_hard_cued'
  | 'done';

interface LoadedTarget {
  source: 'pending' | 'imported';
  participantId: string;
  sessionNumber: 1 | 2;
  condition: 'adaptive' | 'baseline';
  form: 'A' | 'B';
  linkedSessionRecordId: string;
  dueAtIso: string;
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
}

function itemsFromSession(record: StudySessionRecord): { easyItems: StudyStimulusItem[]; hardItems: StudyStimulusItem[] } {
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
  const pending = useMemo(() => listPendingDelayedTests(), []);

  const [loaded, setLoaded] = useState<LoadedTarget | null>(null);
  const [phase, setPhase] = useState<DelayedPhase>('select');
  const [trials, setTrials] = useState<StudyTrialResult[]>([]);
  const [completedRecord, setCompletedRecord] = useState<StudyDelayedTestRecord | null>(null);
  const [sessionStartMs] = useState(Date.now());

  const startFromPending = (record: StudySessionRecord) => {
    const { easyItems, hardItems } = itemsFromSession(record);
    setLoaded({
      source: 'pending',
      participantId: record.participantId,
      sessionNumber: record.sessionNumber,
      condition: record.condition,
      form: record.form,
      linkedSessionRecordId: record.recordId,
      dueAtIso: record.delayedDueAtIso,
      easyItems,
      hardItems,
    });
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
    setPhase('test_easy_recognition');
  };

  const appendTrials = (newTrials: StudyTrialResult[]) => {
    setTrials((prev) => [...prev, ...newTrials]);
  };

  const finish = (finalTrials: StudyTrialResult[] = []) => {
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
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Delayed Test</h1>
            <p className="text-gray-600 mt-1">Session-linked delayed recognition and cued recall.</p>
          </div>
          <button onClick={() => navigate('/study/setup')} className="text-gray-600 hover:text-gray-800">
            Close
          </button>
        </div>

        {phase === 'select' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Pending local delayed tests</h2>
              {pending.length === 0 ? (
                <p className="text-gray-600">No pending delayed tests found in local storage.</p>
              ) : (
                <div className="space-y-3">
                  {pending.map((record) => (
                    <button
                      key={record.recordId}
                      onClick={() => startFromPending(record)}
                      className="w-full text-left px-4 py-3 rounded-lg border border-gray-300 hover:border-blue-400 bg-white"
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
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Import delayed packet</h2>
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
                Use this if delayed test is being run on a different device.
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
              finish(newTrials);
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

            <div className="flex gap-3">
              <button
                onClick={exportDelayedJson}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
              >
                Export Delayed JSON
              </button>
              <button
                onClick={() => navigate('/study/setup')}
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
