import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import NasaTLXForm from '../components/NasaTLXForm';
import { finalizeSession, exportStudyPackage } from '../services/studyStorage';
import { createDelayedPacket } from '../services/studyStimuli';
import { triggerDownload } from '../services/studyExport';
import { StudySessionRecord, StudySummaryState, StudyStimulusItem } from '../types/study';
import { NASATLXScores } from '../types';

function learningItemsFromRecord(record: StudySessionRecord): {
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
} {
  const learning = record.trials.filter((trial) => trial.kind === 'learning');
  const seen = new Set<string>();
  const items: StudyStimulusItem[] = [];

  for (const trial of learning) {
    if (seen.has(trial.itemId)) continue;
    seen.add(trial.itemId);
    items.push({
      id: trial.itemId,
      cue: trial.cue,
      target: trial.target,
      difficulty: trial.difficulty,
      interferenceGroup: trial.itemId.split('-').slice(0, 2).join('-'),
    });
  }

  return {
    easyItems: items.filter((item) => item.difficulty === 'easy'),
    hardItems: items.filter((item) => item.difficulty === 'hard'),
  };
}

const StudySummary: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as StudySummaryState | undefined;

  const [record, setRecord] = useState<StudySessionRecord | null>(state?.record ?? null);
  const [saved, setSaved] = useState(false);

  const summary = useMemo(() => {
    if (!record) return null;

    const immediateRecognition = record.trials.filter((t) => t.kind === 'recognition');
    const immediateCued = record.trials.filter((t) => t.kind === 'cued_recall');

    const recognitionAccuracy =
      immediateRecognition.length > 0
        ? immediateRecognition.filter((t) => t.correct).length / immediateRecognition.length
        : 0;
    const cuedAccuracy =
      immediateCued.length > 0
        ? immediateCued.filter((t) => t.correct).length / immediateCued.length
        : 0;

    return {
      recognitionAccuracy,
      cuedAccuracy,
      interventionCount: record.interventions.filter((e) => e.outcome === 'applied').length,
    };
  }, [record]);

  if (!record || !summary) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-lg w-full text-center space-y-3">
          <h2 className="text-xl font-semibold text-gray-800">No study record found</h2>
          <p className="text-gray-600">Complete a study session first.</p>
          <button
            onClick={() => navigate('/study/setup')}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
          >
            Go to Study Setup
          </button>
        </div>
      </div>
    );
  }

  const submitTlx = (scores: NASATLXScores) => {
    const updated: StudySessionRecord = {
      ...record,
      nasaTlx: scores,
      completedAtIso: record.completedAtIso ?? new Date().toISOString(),
    };
    setRecord(updated);
    finalizeSession(updated);
    setSaved(true);
  };

  const exportSessionJson = () => {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    triggerDownload(
      JSON.stringify(record, null, 2),
      `study_${record.participantId}_session${record.sessionNumber}_${record.condition}_${stamp}.json`,
      'application/json'
    );
  };

  const exportDelayedPacket = () => {
    const { easyItems, hardItems } = learningItemsFromRecord(record);
    const packet = createDelayedPacket(
      record.participantId,
      record.sessionNumber,
      record.form,
      easyItems,
      hardItems
    );

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    triggerDownload(
      JSON.stringify(packet, null, 2),
      `study_${record.participantId}_session${record.sessionNumber}_delayed_packet_${stamp}.json`,
      'application/json'
    );
  };

  const exportBundle = () => {
    exportStudyPackage(record.participantId, {
      downloadCanonicalJson: true,
      downloadTables: true,
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Study Session Summary</h1>
            <p className="text-gray-600 mt-1">
              Participant {record.participantId} • Session {record.sessionNumber} • {record.condition}
            </p>
          </div>
          <button
            onClick={() => navigate('/study/setup')}
            className="text-gray-600 hover:text-gray-800"
          >
            Close
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm text-gray-500">Recognition accuracy</div>
            <div className="text-2xl font-semibold text-gray-800">{(summary.recognitionAccuracy * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm text-gray-500">Cued recall accuracy</div>
            <div className="text-2xl font-semibold text-gray-800">{(summary.cuedAccuracy * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm text-gray-500">Interventions applied</div>
            <div className="text-2xl font-semibold text-gray-800">{summary.interventionCount}</div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm text-gray-500">CLI windows</div>
            <div className="text-2xl font-semibold text-gray-800">{record.cliSamples.length}</div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <NasaTLXForm
            title="NASA-TLX (Session-level)"
            submitLabel={saved ? 'NASA-TLX Saved' : 'Save NASA-TLX'}
            onSubmit={submitTlx}
          />
          {saved && (
            <div className="mt-3 text-sm text-green-700 bg-green-50 border border-green-100 rounded px-3 py-2">
              Session record finalized and saved locally.
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-3">Exports</h2>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={exportSessionJson}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
            >
              Export Session JSON
            </button>
            <button
              onClick={exportDelayedPacket}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white"
            >
              Export Delayed Packet
            </button>
            <button
              onClick={exportBundle}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              Export Participant Bundle (JSON + CSV)
            </button>
            <button
              onClick={() => navigate('/study/delayed')}
              className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white"
            >
              Open Delayed Test Page
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudySummary;
