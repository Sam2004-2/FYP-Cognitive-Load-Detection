import { FEATURE_CONFIG } from '../config/featureConfig';
import {
  AdminExportQuery,
  AdminMonitoringSummary,
  AdminReportIndexResponse,
  PendingDelayedTask,
  StudyActivityEventInput,
  StudyDelayedTestRecord,
  StudyDelayedUploadResponse,
  StudyParticipantIdentity,
  StudySessionRecord,
  StudySessionUploadResponse,
} from '../types/study';
import { APIError } from './apiClient';

/** @deprecated Use APIError from apiClient instead */
export { APIError as StudyAPIError } from './apiClient';

const API_BASE_URL = FEATURE_CONFIG.api.base_url;

function buildUrl(path: string): string {
  if (!API_BASE_URL) return path;
  return `${API_BASE_URL}${path}`;
}

async function parseError(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function parseSessionNumber(value: unknown): 1 | 2 {
  return value === 2 ? 2 : 1;
}

function parseCondition(value: unknown): 'adaptive' | 'baseline' {
  return value === 'adaptive' ? 'adaptive' : 'baseline';
}

function parseForm(value: unknown): 'A' | 'B' {
  return value === 'B' ? 'B' : 'A';
}

function buildAdminQueryString(query: AdminExportQuery): string {
  const params = new URLSearchParams();

  if (query.participantId) params.set('participant_id', query.participantId);
  if (query.fromIso) params.set('from_iso', query.fromIso);
  if (query.toIso) params.set('to_iso', query.toIso);
  if (query.format) params.set('format', query.format);

  const serialized = params.toString();
  return serialized ? `?${serialized}` : '';
}

function getAdminHeaders(token: string, includeJsonContentType = false): HeadersInit {
  const trimmed = token.trim();
  return includeJsonContentType
    ? {
        Authorization: `Bearer ${trimmed}`,
        'Content-Type': 'application/json',
      }
    : {
        Authorization: `Bearer ${trimmed}`,
      };
}

export async function createParticipantIdentity(): Promise<StudyParticipantIdentity> {
  const response = await fetch(buildUrl('/study/participants'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to create participant ID: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  return {
    participantId: String(data.participant_id ?? ''),
    createdAtIso: String(data.created_at_iso ?? ''),
  };
}

export async function uploadSessionRecord(
  record: StudySessionRecord
): Promise<StudySessionUploadResponse> {
  const response = await fetch(buildUrl('/study/session-records'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ record }),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to upload session record: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  return {
    success: Boolean(data.success),
    recordId: String(data.record_id ?? record.recordId),
    storedAtIso: String(data.stored_at_iso ?? ''),
  };
}

export async function uploadDelayedRecord(
  record: StudyDelayedTestRecord
): Promise<StudyDelayedUploadResponse> {
  const response = await fetch(buildUrl('/study/delayed-records'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ record }),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to upload delayed record: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  return {
    success: Boolean(data.success),
    recordId: String(data.record_id ?? record.recordId),
    storedAtIso: String(data.stored_at_iso ?? ''),
  };
}

export async function getPendingDelayedTasks(participantId: string): Promise<PendingDelayedTask[]> {
  const response = await fetch(buildUrl(`/study/pending-delayed/${encodeURIComponent(participantId.trim())}`), {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to fetch pending delayed tasks: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  const tasks = Array.isArray(data.pending) ? data.pending : [];

  return tasks.map((task: any) => ({
    linkedSessionRecordId: String(task.linked_session_record_id ?? ''),
    participantId: String(task.participant_id ?? participantId),
    sessionNumber: parseSessionNumber(task.session_number),
    condition: parseCondition(task.condition),
    form: parseForm(task.form),
    dueAtIso: String(task.due_at_iso ?? ''),
    easyItems: Array.isArray(task.easy_items) ? task.easy_items : [],
    hardItems: Array.isArray(task.hard_items) ? task.hard_items : [],
  }));
}

export async function postStudyActivity(event: StudyActivityEventInput): Promise<void> {
  const response = await fetch(buildUrl('/study/activity'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      event_type: event.eventType,
      page: event.page,
      participant_id: event.participantId,
      visitor_id: event.visitorId,
      session_number: event.sessionNumber,
      condition: event.condition,
      metadata: event.metadata ?? {},
    }),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to track study activity: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }
}

export async function getAdminReportIndex(
  token: string,
  query: AdminExportQuery = {}
): Promise<AdminReportIndexResponse> {
  const response = await fetch(buildUrl(`/admin/reports/index${buildAdminQueryString(query)}`), {
    method: 'GET',
    headers: getAdminHeaders(token),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to load report index: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  const records = Array.isArray(data.records) ? data.records : [];

  return {
    generatedAtIso: String(data.generated_at_iso ?? ''),
    count: Number(data.count ?? records.length),
    records: records.map((record: any) => ({
      participantId: String(record.participant_id ?? ''),
      kind: record.kind === 'delayed' ? 'delayed' : 'sessions',
      recordId: String(record.record_id ?? ''),
      eventTimeIso: record.event_time_iso ? String(record.event_time_iso) : null,
      storedAtIso: String(record.stored_at_iso ?? ''),
      path: String(record.path ?? ''),
    })),
  };
}

export async function getAdminMonitoringSummary(token: string): Promise<AdminMonitoringSummary> {
  const response = await fetch(buildUrl('/admin/monitoring/summary'), {
    method: 'GET',
    headers: getAdminHeaders(token),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to load monitoring summary: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  const data = await response.json();
  const dailyUploads = Array.isArray(data.daily_uploads) ? data.daily_uploads : [];
  const recentRecords = Array.isArray(data.recent_records) ? data.recent_records : [];
  const activity = data.activity && typeof data.activity === 'object' ? data.activity : {};
  const recentEvents = Array.isArray(activity.recent_events) ? activity.recent_events : [];

  return {
    generatedAtIso: String(data.generated_at_iso ?? ''),
    totals: data.totals && typeof data.totals === 'object' ? data.totals : {},
    conditionCounts: data.condition_counts && typeof data.condition_counts === 'object' ? data.condition_counts : {},
    interventionCounts:
      data.intervention_counts && typeof data.intervention_counts === 'object'
        ? data.intervention_counts
        : {},
    dailyUploads: dailyUploads.map((point: any) => ({
      date: String(point.date ?? ''),
      sessionRecords: Number(point.session_records ?? 0),
      delayedRecords: Number(point.delayed_records ?? 0),
      totalRecords: Number(point.total_records ?? 0),
    })),
    recentRecords: recentRecords.map((record: any) => ({
      participantId: String(record.participant_id ?? ''),
      kind: record.kind === 'delayed' ? 'delayed' : 'sessions',
      recordId: String(record.record_id ?? ''),
      condition: record.condition ? String(record.condition) : undefined,
      sessionNumber:
        typeof record.session_number === 'number' ? Number(record.session_number) : undefined,
      storedAtIso: String(record.stored_at_iso ?? ''),
    })),
    activity: {
      activeLast15m: Number(activity.active_last_15m ?? 0),
      activeLast60m: Number(activity.active_last_60m ?? 0),
      visitorsLast24h: Number(activity.visitors_last_24h ?? 0),
      pageViewsLast24h: Number(activity.page_views_last_24h ?? 0),
      pageViewCounts:
        activity.page_view_counts && typeof activity.page_view_counts === 'object'
          ? activity.page_view_counts
          : {},
      recentEvents: recentEvents.map((event: any) => ({
        occurredAtIso: String(event.occurred_at_iso ?? ''),
        eventType: String(event.event_type ?? 'unknown'),
        page: String(event.page ?? 'unknown'),
        participantId: event.participant_id ? String(event.participant_id) : undefined,
        visitorId: event.visitor_id ? String(event.visitor_id) : undefined,
        sessionNumber:
          typeof event.session_number === 'number' ? Number(event.session_number) : undefined,
        condition: event.condition ? String(event.condition) : undefined,
      })),
    },
  };
}

export async function downloadAdminReports(
  token: string,
  query: AdminExportQuery = {}
): Promise<Blob> {
  const response = await fetch(buildUrl(`/admin/reports/export${buildAdminQueryString(query)}`), {
    method: 'GET',
    headers: getAdminHeaders(token),
  });

  if (!response.ok) {
    throw new APIError(
      `Failed to export reports: ${response.statusText}`,
      response.status,
      await parseError(response)
    );
  }

  return response.blob();
}

export function buildAdminExportUrl(query: AdminExportQuery): string {
  return buildUrl(`/admin/reports/export${buildAdminQueryString(query)}`);
}
