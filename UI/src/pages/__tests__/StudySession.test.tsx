import { act, fireEvent, render, screen } from '@testing-library/react';
import StudySession from '../StudySession';
import { StudySetupState } from '../../types/study';

const mockNavigate = jest.fn();

const setupState: StudySetupState = {
  participantId: 'P-TEST-001',
  assignment: {
    participantId: 'P-TEST-001',
    participantIdNormalized: 'p-test-001',
    hashValue: 42,
    hashParity: 'even',
    conditionOrder: ['adaptive', 'baseline'],
    formOrder: ['A', 'B'],
    sessionNumber: 1,
    condition: 'adaptive',
    form: 'A',
    delayedDueAtIso: '2026-03-19T10:00:00Z',
  },
  plan: {
    baselineSeconds: 45,
    easyItemCount: 8,
    easyExposureSeconds: 4.5,
    hardItemCount: 10,
    hardExposureSeconds: 3,
    hardInterferenceEnabled: true,
    recognitionChoices: 4,
    microBreakSeconds: 45,
    maxMicroBreaksPerSession: 1,
    adaptationCooldownSeconds: 120,
    decisionWindowSeconds: 5,
    smoothingWindows: 3,
    adaptiveMode: 'relative',
    absoluteThreshold: 0.45,
    relativeZThreshold: 1,
    warmupWindows: 4,
    minStdEpsilon: 0.02,
    overloadThreshold: 0.7,
    arithmeticPracticeCount: 1,
    arithmeticItemsPerDifficulty: 4,
    arithmeticTimeLimitSeconds: 8,
    arithmeticTransitionSeconds: 3,
  },
  session2StartedEarlyOverride: false,
};

jest.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ state: setupState }),
}), { virtual: true });

jest.mock('../../components/WebcamFeed', () => () => <div>WebcamFeed</div>);
jest.mock('../../components/LiveFeaturePanel', () => () => <div>LiveFeaturePanel</div>);
jest.mock('../../components/NasaTLXForm', () => () => <div>NASA TLX Preview</div>);
jest.mock('../../components/study/StudyInterventionModal', () => () => null);

jest.mock('../../components/study/PairedAssociateLearningBlock', () => (props: any) => (
  <button onClick={() => props.onComplete([], 0)}>Complete learning {props.phase}</button>
));

jest.mock('../../components/study/RecognitionTest', () => (props: any) => (
  <button onClick={() => props.onComplete([])}>Complete recognition {props.phase}</button>
));

jest.mock('../../components/study/CuedRecallTest', () => (props: any) => (
  <button onClick={() => props.onComplete([])}>Complete cued {props.phase}</button>
));

jest.mock('../../components/study/ArithmeticChallengeBlock', () => (props: any) => (
  <button onClick={() => props.onComplete([])}>Complete arithmetic {props.phase}</button>
));

jest.mock('../../services/apiClient', () => ({
  testConnection: jest.fn(async () => true),
  predictCognitiveLoad: jest.fn(),
}));

jest.mock('../../services/studyActivityTracker', () => ({
  ACTIVITY_PAGES: {
    STUDY_SESSION: 'study_session',
    STUDY_SESSION_COMPLETE: 'study_session_complete',
  },
  trackPageView: jest.fn(),
}));

jest.mock('../../services/studyStorage', () => ({
  createSessionRecordId: jest.fn(() => 'record-1'),
  saveSessionDraft: jest.fn(),
}));

describe('StudySession', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockNavigate.mockReset();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('transitions from hard cued recall through arithmetic blocks to NASA-TLX', async () => {
    render(<StudySession />);

    expect(screen.getByText('Baseline Calibration')).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(45000);
    });

    fireEvent.click(await screen.findByRole('button', { name: 'Complete learning learning_easy' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete recognition test_easy_recognition' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete cued test_easy_cued_recall' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete learning learning_hard' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete recognition test_hard_recognition' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete cued test_hard_cued_recall' }));

    fireEvent.click(screen.getByRole('button', { name: 'Complete arithmetic arithmetic_easy' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete arithmetic arithmetic_medium' }));
    fireEvent.click(screen.getByRole('button', { name: 'Complete arithmetic arithmetic_hard' }));

    expect(screen.getByText('Session complete')).toBeInTheDocument();
    expect(screen.getByText(/including arithmetic performance/i)).toBeInTheDocument();
  });
});
