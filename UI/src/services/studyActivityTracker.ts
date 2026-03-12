import { StudyCondition } from '../types/study';
import { postStudyActivity } from './studyApiClient';

const VISITOR_ID_KEY = 'cle_study_visitor_id';

export const ACTIVITY_PAGES = {
  SESSION_SETUP: 'session_setup',
  STUDY_SETUP: 'study_setup',
  STUDY_SETUP_START: 'study_setup_start_session',
  STUDY_SESSION: 'study_session',
  STUDY_SESSION_COMPLETE: 'study_session_complete',
  STUDY_SUMMARY: 'study_summary',
  STUDY_DELAYED: 'study_delayed',
} as const;

export type ActivityPage = (typeof ACTIVITY_PAGES)[keyof typeof ACTIVITY_PAGES];

function randomSegment(length: number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let out = '';
  for (let i = 0; i < length; i += 1) {
    out += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return out;
}

let cachedVisitorId: string | null = null;

export function getOrCreateVisitorId(): string {
  if (cachedVisitorId) return cachedVisitorId;

  try {
    const existing = window.localStorage.getItem(VISITOR_ID_KEY);
    if (existing && existing.trim()) {
      cachedVisitorId = existing.trim();
      return cachedVisitorId;
    }
  } catch {
    // Ignore localStorage access issues and fall back to in-memory identifier.
  }

  const generated = `V-${randomSegment(8)}`;
  cachedVisitorId = generated;
  try {
    window.localStorage.setItem(VISITOR_ID_KEY, generated);
  } catch {
    // Ignore if storage is unavailable.
  }
  return generated;
}

interface TrackPageViewInput {
  page: ActivityPage;
  participantId?: string;
  sessionNumber?: number;
  condition?: StudyCondition;
  metadata?: Record<string, unknown>;
}

export function trackPageView(input: TrackPageViewInput): void {
  const visitorId = getOrCreateVisitorId();
  void postStudyActivity({
    eventType: 'page_view',
    page: input.page,
    visitorId,
    participantId: input.participantId,
    sessionNumber: input.sessionNumber,
    condition: input.condition,
    metadata: input.metadata ?? {},
  }).catch((err) => {
    if (process.env.NODE_ENV === 'development') {
      console.warn('[tracker]', err);
    }
  });
}
