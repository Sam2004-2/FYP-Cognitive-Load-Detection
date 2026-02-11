import { STUDY_RECORD_VERSION, STUDY_STORAGE_KEYS } from '../config/studyConfig';
import {
  StudyDelayedTestRecord,
  StudySessionNumber,
  StudySessionRecord,
} from '../types/study';
import {
  buildStudyExportPackage,
  buildStudyExportTables,
  StudyExportPackage,
  triggerDownload,
} from './studyExport';

function sessionDraftKey(recordId: string): string {
  return `${STUDY_STORAGE_KEYS.sessionDraftPrefix}${recordId}`;
}

function sessionFinalKey(recordId: string): string {
  return `${STUDY_STORAGE_KEYS.sessionFinalPrefix}${recordId}`;
}

function delayedKey(recordId: string): string {
  return `${STUDY_STORAGE_KEYS.delayedPrefix}${recordId}`;
}

function safeParse<T>(raw: string | null): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function keysWithPrefix(prefix: string): string[] {
  const keys: string[] = [];
  for (let i = 0; i < localStorage.length; i += 1) {
    const key = localStorage.key(i);
    if (key && key.startsWith(prefix)) {
      keys.push(key);
    }
  }
  return keys;
}

export function createSessionRecordId(
  participantId: string,
  sessionNumber: StudySessionNumber,
  condition: string
): string {
  const timestamp = Date.now();
  return `${participantId.trim()}_s${sessionNumber}_${condition}_${timestamp}`;
}

export function createDelayedRecordId(
  participantId: string,
  sessionNumber: StudySessionNumber,
  condition: string
): string {
  const timestamp = Date.now();
  return `${participantId.trim()}_d${sessionNumber}_${condition}_${timestamp}`;
}

export function saveSessionDraft(record: StudySessionRecord): void {
  const payload = {
    ...record,
    recordVersion: STUDY_RECORD_VERSION,
  };
  localStorage.setItem(sessionDraftKey(record.recordId), JSON.stringify(payload));
}

export function loadSessionDraft(recordId: string): StudySessionRecord | null {
  return safeParse<StudySessionRecord>(localStorage.getItem(sessionDraftKey(recordId)));
}

export function finalizeSession(record: StudySessionRecord): void {
  const payload = {
    ...record,
    recordVersion: STUDY_RECORD_VERSION,
  };
  localStorage.setItem(sessionFinalKey(record.recordId), JSON.stringify(payload));
  localStorage.removeItem(sessionDraftKey(record.recordId));
}

export function listSessionDrafts(participantId?: string): StudySessionRecord[] {
  return keysWithPrefix(STUDY_STORAGE_KEYS.sessionDraftPrefix)
    .map((key) => safeParse<StudySessionRecord>(localStorage.getItem(key)))
    .filter((entry): entry is StudySessionRecord => Boolean(entry))
    .filter((entry) => (participantId ? entry.participantId === participantId : true))
    .sort((a, b) => a.startedAtIso.localeCompare(b.startedAtIso));
}

export function listFinalSessions(participantId?: string): StudySessionRecord[] {
  return keysWithPrefix(STUDY_STORAGE_KEYS.sessionFinalPrefix)
    .map((key) => safeParse<StudySessionRecord>(localStorage.getItem(key)))
    .filter((entry): entry is StudySessionRecord => Boolean(entry))
    .filter((entry) => (participantId ? entry.participantId === participantId : true))
    .sort((a, b) => a.startedAtIso.localeCompare(b.startedAtIso));
}

export function getMostRecentSession(
  participantId: string,
  sessionNumber?: StudySessionNumber
): StudySessionRecord | null {
  const matches = listFinalSessions(participantId)
    .filter((session) => (sessionNumber ? session.sessionNumber === sessionNumber : true))
    .sort((a, b) => b.startedAtIso.localeCompare(a.startedAtIso));

  return matches[0] ?? null;
}

export function listPendingDelayedTests(participantId?: string): StudySessionRecord[] {
  return listFinalSessions(participantId)
    .filter((session) => session.pendingDelayedTest)
    .sort((a, b) => a.delayedDueAtIso.localeCompare(b.delayedDueAtIso));
}

export function saveDelayedResult(record: StudyDelayedTestRecord): void {
  const payload = {
    ...record,
    recordVersion: STUDY_RECORD_VERSION,
  };
  localStorage.setItem(delayedKey(record.recordId), JSON.stringify(payload));

  const linked = safeParse<StudySessionRecord>(
    localStorage.getItem(sessionFinalKey(record.linkedSessionRecordId))
  );
  if (linked) {
    const updated = {
      ...linked,
      pendingDelayedTest: false,
    };
    localStorage.setItem(sessionFinalKey(updated.recordId), JSON.stringify(updated));
  }
}

export function listDelayedResults(participantId?: string): StudyDelayedTestRecord[] {
  return keysWithPrefix(STUDY_STORAGE_KEYS.delayedPrefix)
    .map((key) => safeParse<StudyDelayedTestRecord>(localStorage.getItem(key)))
    .filter((entry): entry is StudyDelayedTestRecord => Boolean(entry))
    .filter((entry) => (participantId ? entry.participantId === participantId : true))
    .sort((a, b) => a.dueAtIso.localeCompare(b.dueAtIso));
}

interface ExportOptions {
  downloadCanonicalJson?: boolean;
  downloadTables?: boolean;
}

export function exportStudyPackage(
  participantId: string,
  options: ExportOptions = { downloadCanonicalJson: true, downloadTables: true }
): StudyExportPackage {
  const sessions = listFinalSessions(participantId);
  const delayed = listDelayedResults(participantId);

  const pkg = buildStudyExportPackage(participantId, sessions, delayed);
  const exportedAtDate = new Date(pkg.exportedAtIso).toISOString().slice(0, 19).replace(/[:T]/g, '-');

  if (options.downloadCanonicalJson) {
    triggerDownload(
      JSON.stringify(pkg, null, 2),
      `study_${participantId}_bundle_${exportedAtDate}.json`,
      'application/json'
    );
  }

  if (options.downloadTables) {
    const tables = buildStudyExportTables(sessions, delayed);
    Object.entries(tables).forEach(([name, content]) => {
      if (!content) return;
      triggerDownload(content, `study_${participantId}_${name}_${exportedAtDate}.csv`, 'text/csv');
    });
  }

  return pkg;
}
