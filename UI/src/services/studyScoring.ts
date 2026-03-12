import { StudyRecallScoring } from '../types/study';

const DIACRITIC_REGEX = /[\u0300-\u036f]/g;
const NON_ALPHANUMERIC_REGEX = /[^a-z0-9\s]/g;

export interface RecallScoreResult {
  correct: boolean;
  scoring: StudyRecallScoring;
}

export function normalizeRecallText(value: string): string {
  return value
    .normalize('NFD')
    .replace(DIACRITIC_REGEX, '')
    .toLowerCase()
    .replace(NON_ALPHANUMERIC_REGEX, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function damerauLevenshteinDistance(source: string, target: string): number {
  const sourceLength = source.length;
  const targetLength = target.length;

  if (sourceLength === 0) return targetLength;
  if (targetLength === 0) return sourceLength;

  const matrix: number[][] = Array.from({ length: sourceLength + 1 }, (_, row) =>
    Array.from({ length: targetLength + 1 }, (_, col) => (row === 0 ? col : col === 0 ? row : 0))
  );

  for (let row = 1; row <= sourceLength; row += 1) {
    for (let col = 1; col <= targetLength; col += 1) {
      const cost = source[row - 1] === target[col - 1] ? 0 : 1;
      let value = Math.min(
        matrix[row - 1][col] + 1,
        matrix[row][col - 1] + 1,
        matrix[row - 1][col - 1] + cost
      );

      if (
        row > 1 &&
        col > 1 &&
        source[row - 1] === target[col - 2] &&
        source[row - 2] === target[col - 1]
      ) {
        value = Math.min(value, matrix[row - 2][col - 2] + 1);
      }

      matrix[row][col] = value;
    }
  }

  return matrix[sourceLength][targetLength];
}

export function scoreCuedRecallResponse(
  response: string,
  expectedTarget: string,
  candidateTargets: string[]
): RecallScoreResult {
  const normalizedResponse = normalizeRecallText(response);
  const normalizedTarget = normalizeRecallText(expectedTarget);
  const distance = damerauLevenshteinDistance(normalizedResponse, normalizedTarget);

  if (normalizedResponse === normalizedTarget) {
    return {
      correct: true,
      scoring: {
        version: 2,
        method: 'exact_normalized',
        matchType: 'exact',
        normalizedResponse,
        normalizedTarget,
        distance: 0,
      },
    };
  }

  const normalizedCandidates = candidateTargets
    .map((target) => normalizeRecallText(target))
    .filter((candidate) => candidate && candidate !== normalizedTarget);

  if (distance !== 1) {
    return {
      correct: false,
      scoring: {
        version: 2,
        method: 'tolerant_damerau_1',
        matchType: 'mismatch',
        normalizedResponse,
        normalizedTarget,
        distance,
      },
    };
  }

  const competingDistances = normalizedCandidates.map((candidate) =>
    damerauLevenshteinDistance(normalizedResponse, candidate)
  );
  const hasAmbiguousCompetingTarget = competingDistances.some((candidateDistance) => candidateDistance <= distance);

  if (hasAmbiguousCompetingTarget) {
    return {
      correct: false,
      scoring: {
        version: 2,
        method: 'tolerant_damerau_1',
        matchType: 'ambiguous_near_match',
        normalizedResponse,
        normalizedTarget,
        distance,
      },
    };
  }

  return {
    correct: true,
    scoring: {
      version: 2,
      method: 'tolerant_damerau_1',
      matchType: 'near_match',
      normalizedResponse,
      normalizedTarget,
      distance,
    },
  };
}
