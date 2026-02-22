import {
  buildAdminExportUrl,
  createParticipantIdentity,
  getPendingDelayedTasks,
  uploadDelayedRecord,
  uploadSessionRecord,
} from '../studyApiClient';

const mockFetch = jest.fn();

global.fetch = mockFetch as unknown as typeof fetch;

describe('studyApiClient', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('creates participant identity', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        participant_id: 'P-260222-ABC123',
        created_at_iso: '2026-02-22T12:00:00Z',
      }),
    });

    const result = await createParticipantIdentity();

    expect(result.participantId).toBe('P-260222-ABC123');
    expect(result.createdAtIso).toBe('2026-02-22T12:00:00Z');
  });

  it('uploads session and delayed records', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          record_id: 'session-1',
          stored_at_iso: '2026-02-22T12:00:00Z',
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          record_id: 'delayed-1',
          stored_at_iso: '2026-02-22T12:05:00Z',
        }),
      });

    const sessionResult = await uploadSessionRecord({
      recordId: 'session-1',
      participantId: 'P-1',
    } as any);

    const delayedResult = await uploadDelayedRecord({
      recordId: 'delayed-1',
      linkedSessionRecordId: 'session-1',
      participantId: 'P-1',
    } as any);

    expect(sessionResult.success).toBe(true);
    expect(sessionResult.recordId).toBe('session-1');
    expect(delayedResult.success).toBe(true);
    expect(delayedResult.recordId).toBe('delayed-1');
  });

  it('maps server pending delayed tasks', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        pending: [
          {
            linked_session_record_id: 'session-1',
            participant_id: 'P-1',
            session_number: 2,
            condition: 'adaptive',
            form: 'B',
            due_at_iso: '2026-02-23T10:00:00Z',
            easy_items: [{ id: 'e1' }],
            hard_items: [{ id: 'h1' }],
          },
        ],
      }),
    });

    const pending = await getPendingDelayedTasks('P-1');

    expect(pending).toHaveLength(1);
    expect(pending[0].linkedSessionRecordId).toBe('session-1');
    expect(pending[0].sessionNumber).toBe(2);
    expect(pending[0].condition).toBe('adaptive');
    expect(pending[0].form).toBe('B');
    expect(pending[0].easyItems).toHaveLength(1);
    expect(pending[0].hardItems).toHaveLength(1);
  });

  it('builds admin export URLs', () => {
    const url = buildAdminExportUrl({
      participantId: 'P-1',
      fromIso: '2026-02-01T00:00:00Z',
      toIso: '2026-02-28T23:59:59Z',
      format: 'zip',
    });

    expect(url).toContain('/admin/reports/export?');
    expect(url).toContain('participant_id=P-1');
    expect(url).toContain('format=zip');
  });
});
