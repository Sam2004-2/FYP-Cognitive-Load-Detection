import {
  buildAdminExportUrl,
  createParticipantIdentity,
  downloadAdminReports,
  getAdminMonitoringSummary,
  getAdminReportIndex,
  getPendingDelayedTasks,
  postStudyActivity,
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

  it('posts activity events', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
    });

    await postStudyActivity({
      eventType: 'page_view',
      page: 'study_setup',
      participantId: 'P-1',
      visitorId: 'V-1',
      sessionNumber: 1,
      condition: 'adaptive',
      metadata: { source: 'test' },
    });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/study/activity'),
      expect.objectContaining({
        method: 'POST',
      })
    );
  });

  it('maps admin report index response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        generated_at_iso: '2026-03-04T10:00:00Z',
        count: 1,
        records: [
          {
            participant_id: 'P-1',
            kind: 'sessions',
            record_id: 'r1',
            event_time_iso: '2026-03-04T09:00:00Z',
            stored_at_iso: '2026-03-04T09:05:00Z',
            path: '/opt/cle-data/reports/sessions/P-1/r1.json',
          },
        ],
      }),
    });

    const result = await getAdminReportIndex('token-1');

    expect(result.count).toBe(1);
    expect(result.records[0].participantId).toBe('P-1');
    expect(result.records[0].kind).toBe('sessions');
  });

  it('maps admin monitoring summary response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        generated_at_iso: '2026-03-04T10:00:00Z',
        totals: { session_records: 2 },
        condition_counts: { adaptive: 1, baseline: 1 },
        intervention_counts: { micro_break_60s: 2 },
        daily_uploads: [
          {
            date: '2026-03-04',
            session_records: 2,
            delayed_records: 1,
            total_records: 3,
          },
        ],
        recent_records: [
          {
            participant_id: 'P-1',
            kind: 'sessions',
            record_id: 'r1',
            condition: 'adaptive',
            session_number: 1,
            stored_at_iso: '2026-03-04T09:05:00Z',
          },
        ],
        activity: {
          active_last_15m: 1,
          active_last_60m: 2,
          visitors_last_24h: 3,
          page_views_last_24h: 6,
          page_view_counts: { study_setup: 4 },
          recent_events: [
            {
              occurred_at_iso: '2026-03-04T09:05:00Z',
              event_type: 'page_view',
              page: 'study_setup',
            },
          ],
        },
      }),
    });

    const summary = await getAdminMonitoringSummary('token-1');

    expect(summary.totals.session_records).toBe(2);
    expect(summary.dailyUploads[0].totalRecords).toBe(3);
    expect(summary.activity.activeLast15m).toBe(1);
    expect(summary.activity.recentEvents[0].eventType).toBe('page_view');
  });

  it('downloads admin exports as blob', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      blob: async () => new Blob(['hello'], { type: 'application/zip' }),
    });

    const blob = await downloadAdminReports('token-1', { format: 'zip' });
    expect(blob.size).toBeGreaterThan(0);
  });
});
