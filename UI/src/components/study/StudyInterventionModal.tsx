import React from 'react';
import { StudyInterventionType } from '../../types/study';

interface StudyInterventionModalProps {
  open: boolean;
  type: StudyInterventionType | null;
  details?: string;
  onAccept: () => void;
  onDismiss: () => void;
}

const titleMap: Record<StudyInterventionType, string> = {
  micro_break_60s: 'Micro-break recommended',
  pacing_change: 'Pacing adjusted',
  difficulty_step_down: 'Difficulty adjusted',
  suppressed_trigger: 'Baseline trigger logged',
  low_confidence_pause: 'Low confidence mode',
};

const bodyMap: Record<StudyInterventionType, string> = {
  micro_break_60s: 'Cognitive load has remained high. Take a 60-second micro-break to recover.',
  pacing_change: 'Exposure time has been increased for upcoming items to reduce overload.',
  difficulty_step_down: 'Interference intensity has been reduced for the remaining hard-block items.',
  suppressed_trigger: 'A high-load trigger was detected and logged, but no adaptation is applied in baseline condition.',
  low_confidence_pause: 'Signal quality is low. Adaptive actions are temporarily paused until quality recovers.',
};

const StudyInterventionModal: React.FC<StudyInterventionModalProps> = ({
  open,
  type,
  details,
  onAccept,
  onDismiss,
}) => {
  if (!open || !type) return null;

  const requiresChoice = type === 'micro_break_60s';

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-6 space-y-4">
        <h3 className="text-xl font-semibold text-gray-800">{titleMap[type]}</h3>
        <p className="text-gray-700">{bodyMap[type]}</p>
        {details && <p className="text-sm text-gray-500">{details}</p>}

        <div className="flex justify-end gap-3 pt-2">
          {requiresChoice ? (
            <>
              <button
                onClick={onDismiss}
                className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-800"
              >
                Dismiss
              </button>
              <button
                onClick={onAccept}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
              >
                Start 60s Break
              </button>
            </>
          ) : (
            <button
              onClick={onAccept}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
            >
              Acknowledge
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default StudyInterventionModal;
