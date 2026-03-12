import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { createParticipantIdentity } from '../../services/studyApiClient';

jest.mock('react-router-dom', () => ({
  useNavigate: () => jest.fn(),
}), { virtual: true });

import SessionSetup from '../SessionSetup';

jest.mock('../../services/studyApiClient', () => ({
  createParticipantIdentity: jest.fn(),
  postStudyActivity: jest.fn().mockResolvedValue({}),
}));

jest.mock('../../services/studyActivityTracker', () => ({
  trackPageView: jest.fn(),
  ACTIVITY_PAGES: { SESSION_SETUP: 'session_setup' },
}));

const mockedCreateParticipantIdentity = createParticipantIdentity as jest.MockedFunction<typeof createParticipantIdentity>;

describe('SessionSetup', () => {
  beforeEach(() => {
    mockedCreateParticipantIdentity.mockReset();

    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: jest.fn().mockResolvedValue({
          getTracks: () => [{ stop: jest.fn() }],
        }),
      },
      configurable: true,
    });
  });

  it('requires consent, participant ID acknowledgement, and camera before start', async () => {
    mockedCreateParticipantIdentity.mockResolvedValue({
      participantId: 'P-260222-ABC123',
      createdAtIso: '2026-02-22T12:00:00Z',
    });

    render(<SessionSetup />);

    const startButton = screen.getByRole('button', { name: 'Start Study Setup' });
    expect(startButton).toBeDisabled();

    fireEvent.click(screen.getByLabelText(/I consent to participate/i));
    fireEvent.click(screen.getByLabelText(/I have read and understood/i));

    fireEvent.click(screen.getByRole('button', { name: 'Generate Participant ID' }));

    await waitFor(() => {
      expect(screen.getByText(/Participant ID:/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText(/I saved this participant ID/i));
    fireEvent.click(screen.getByRole('button', { name: 'Enable Camera' }));

    await waitFor(() => {
      expect(startButton).toBeEnabled();
    });
  });
});
