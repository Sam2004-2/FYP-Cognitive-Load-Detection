import { FEATURE_CONFIG } from '../config/featureConfig';
import {
  AdminExportQuery,
  PendingDelayedTask,
  StudyDelayedTestRecord,
  StudyDelayedUploadResponse,
  StudyParticipantIdentity,
  StudySessionRecord,
  StudySessionUploadResponse,
} from '../types/study';

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

export class StudyAPIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'StudyAPIError';
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

export async function createParticipantIdentity(): Promise<StudyParticipantIdentity> {
  const response = await fetch(buildUrl('/study/participants'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new StudyAPIError(
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
    throw new StudyAPIError(
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
    throw new StudyAPIError(
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
    throw new StudyAPIError(
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

export function buildAdminExportUrl(query: AdminExportQuery): string {
  const params = new URLSearchParams();

  if (query.participantId) params.set('participant_id', query.participantId);
  if (query.fromIso) params.set('from_iso', query.fromIso);
  if (query.toIso) params.set('to_iso', query.toIso);
  if (query.format) params.set('format', query.format);

  const serialized = params.toString();
  return buildUrl(`/admin/reports/export${serialized ? `?${serialized}` : ''}`);
}
