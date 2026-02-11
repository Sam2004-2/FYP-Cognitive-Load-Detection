import { STUDY_CONFIG } from '../config/studyConfig';
import { StudyAssignment, StudyCondition, StudyForm, StudySessionNumber } from '../types/study';

const DAY_MS = 24 * 60 * 60 * 1000;
const DELAYED_TEST_DAYS = 7;

function normalizeParticipantId(participantId: string): string {
  return participantId.trim().toLowerCase();
}

function stableHash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function orderFromParity(parity: 'even' | 'odd'): {
  conditionOrder: [StudyCondition, StudyCondition];
  formOrder: [StudyForm, StudyForm];
} {
  if (parity === 'even') {
    return {
      conditionOrder: ['adaptive', 'baseline'],
      formOrder: ['A', 'B'],
    };
  }

  return {
    conditionOrder: ['baseline', 'adaptive'],
    formOrder: ['B', 'A'],
  };
}

export function computeStudyAssignment(
  participantId: string,
  sessionNumber: StudySessionNumber,
  now: Date = new Date()
): StudyAssignment {
  const participantIdNormalized = normalizeParticipantId(participantId);
  const hashValue = stableHash(participantIdNormalized);
  const hashParity: 'even' | 'odd' = hashValue % 2 === 0 ? 'even' : 'odd';

  const { conditionOrder, formOrder } = orderFromParity(hashParity);
  const index = sessionNumber === 1 ? 0 : 1;
  const condition = conditionOrder[index];
  const form = formOrder[index];

  const delayedDueAt = new Date(now.getTime() + DELAYED_TEST_DAYS * DAY_MS);

  return {
    participantId,
    participantIdNormalized,
    hashValue,
    hashParity,
    conditionOrder,
    formOrder,
    sessionNumber,
    condition,
    form,
    delayedDueAtIso: delayedDueAt.toISOString(),
  };
}

export interface Session2TimingValidation {
  tooEarly: boolean;
  hoursSinceSession1: number | null;
  recommendedMinimumHours: number;
}

export function validateSession2Timing(
  sessionNumber: StudySessionNumber,
  previousSessionCompletedAtIso?: string,
  now: Date = new Date()
): Session2TimingValidation {
  if (sessionNumber !== 2 || !previousSessionCompletedAtIso) {
    return {
      tooEarly: false,
      hoursSinceSession1: null,
      recommendedMinimumHours: 24,
    };
  }

  const completedAt = new Date(previousSessionCompletedAtIso).getTime();
  const diffMs = now.getTime() - completedAt;
  const hoursSinceSession1 = diffMs / (60 * 60 * 1000);

  return {
    tooEarly: hoursSinceSession1 < 24,
    hoursSinceSession1,
    recommendedMinimumHours: 24,
  };
}

export function getStudyPlan() {
  return { ...STUDY_CONFIG };
}
