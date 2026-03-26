import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createParticipantIdentity, StudyAPIError } from '../services/studyApiClient';
import { ACTIVITY_PAGES, trackPageView } from '../services/studyActivityTracker';
import { StudyParticipantIdentity } from '../types/study';

interface StudySetupRouteState {
  participantId: string;
}

const SessionSetup: React.FC = () => {
  const navigate = useNavigate();

  const [consentAccepted, setConsentAccepted] = useState(false);
  const [instructionsAccepted, setInstructionsAccepted] = useState(false);

  const [participant, setParticipant] = useState<StudyParticipantIdentity | null>(null);
  const [participantAcknowledged, setParticipantAcknowledged] = useState(false);
  const [participantLoading, setParticipantLoading] = useState(false);

  const [cameraPermission, setCameraPermission] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    trackPageView({ page: ACTIVITY_PAGES.SESSION_SETUP });
  }, []);

  const requestCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((track) => track.stop());
      setCameraPermission(true);
      setError('');
    } catch (err) {
      console.error('Camera access denied:', err);
      setError('Camera access denied. Please enable camera permissions to continue.');
    }
  };

  const generateParticipantId = async () => {
    setParticipantLoading(true);
    setError('');

    try {
      const identity = await createParticipantIdentity();
      setParticipant(identity);
      setParticipantAcknowledged(false);
    } catch (err) {
      console.error('Participant ID generation failed:', err);
      if (err instanceof StudyAPIError) {
        setError(`Participant ID generation failed: ${err.message}`);
      } else {
        setError('Participant ID generation failed. Please refresh and try again.');
      }
    } finally {
      setParticipantLoading(false);
    }
  };

  const startSession = () => {
    if (!participant) return;
    const state: StudySetupRouteState = {
      participantId: participant.participantId,
    };
    navigate('/study/setup', { state });
  };

  const canStart =
    consentAccepted &&
    instructionsAccepted &&
    cameraPermission &&
    Boolean(participant) &&
    participantAcknowledged;

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-3xl w-full">
        <div className="bg-white rounded-lg border border-gray-200 p-8 space-y-6">
          <div className="text-center">
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">Cognitive Load Study</h1>
            <p className="text-gray-600">Please complete setup before starting your study session.</p>
          </div>

          <div className="border rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold text-gray-800">Consent and Instructions</h2>
            <p className="text-gray-700 text-sm">
              Requirements: laptop/desktop with webcam, stable internet, quiet environment, and latest Chrome/Edge.
              Keep your face centered with steady lighting. Session duration is approximately 10-15 minutes.
              You will need your participant ID again for Session 2 and delayed follow-up.
            </p>
            <a
              href="/study-instructions.md"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-gray-900 underline hover:text-gray-600"
            >
              Open full participant instructions
            </a>
            <label className="flex items-start gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={consentAccepted}
                onChange={(event) => setConsentAccepted(event.target.checked)}
                className="mt-1"
              />
              <span>I consent to participate and allow webcam-based data collection for this study.</span>
            </label>
            <label className="flex items-start gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={instructionsAccepted}
                onChange={(event) => setInstructionsAccepted(event.target.checked)}
                className="mt-1"
              />
              <span>I have read and understood the participant instructions.</span>
            </label>
          </div>

          <div className="border rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold text-gray-800">Participant ID</h2>
            <p className="text-sm text-gray-600">
              Generate your participant ID now and keep it for all follow-up sessions.
            </p>
            <button
              onClick={generateParticipantId}
              disabled={participantLoading}
              className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white px-5 py-2 rounded-lg"
            >
              {participantLoading ? 'Generating...' : participant ? 'Regenerate Participant ID' : 'Generate Participant ID'}
            </button>

            {participant && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-2">
                <div className="text-sm text-gray-900">
                  <span className="font-semibold">Participant ID:</span>{' '}
                  <span className="font-mono text-base">{participant.participantId}</span>
                </div>
                <div className="text-xs text-gray-500">
                  Created at: {new Date(participant.createdAtIso).toLocaleString()}
                </div>
                <label className="flex items-start gap-2 text-sm text-gray-900">
                  <input
                    type="checkbox"
                    checked={participantAcknowledged}
                    onChange={(event) => setParticipantAcknowledged(event.target.checked)}
                    className="mt-1"
                  />
                  <span>I saved this participant ID and will reuse it for future sessions.</span>
                </label>
              </div>
            )}
          </div>

          <div className="border rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold text-gray-800">Camera Access</h2>
            <p className="text-gray-600">
              Camera access is required for real-time facial feature processing during the study.
            </p>
            {!cameraPermission ? (
              <button
                onClick={requestCameraPermission}
                className="bg-gray-900 hover:bg-gray-800 text-white px-6 py-2 rounded-lg"
              >
                Enable Camera
              </button>
            ) : (
              <div className="text-green-700 font-medium">Camera access granted.</div>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
              {error}
            </div>
          )}

          <button
            onClick={startSession}
            disabled={!canStart}
            className="w-full bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white px-8 py-3 rounded-lg text-lg font-semibold"
          >
            Start Study Setup
          </button>
        </div>
      </div>
    </div>
  );
};

export default SessionSetup;
