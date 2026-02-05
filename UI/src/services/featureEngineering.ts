/**
 * Baseline + temporal feature engineering for real-time inference.
 *
 * Produces engineered features:
 *  - N base window features (see FEATURE_NAMES)
 *  - N baseline-centered features
 *  - N deltas of centered features
 */

import { FEATURE_NAMES } from '../config/featureConfig';
import { WindowFeatures } from '../types/features';

export type BaselineVector = Record<string, number>;
export type CenteredVector = Record<string, number>;

function isFiniteNumber(x: unknown): x is number {
  return typeof x === 'number' && Number.isFinite(x);
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

export function computeBaseline(samples: WindowFeatures[]): BaselineVector {
  const baseline: BaselineVector = {};

  for (const key of FEATURE_NAMES) {
    const vals = samples
      .map((s) => (s as any)[key])
      .filter((v): v is number => isFiniteNumber(v));
    baseline[key] = median(vals);
  }

  return baseline;
}

export function engineerFeatures(
  window: WindowFeatures,
  baseline: BaselineVector,
  prevCentered: CenteredVector | null
): { featureMap: Record<string, number>; nextPrevCentered: CenteredVector } {
  const featureMap: Record<string, number> = {};
  const nextPrevCentered: CenteredVector = {};

  for (const key of FEATURE_NAMES) {
    const baseVal = (window as any)[key];
    const safeBase = isFiniteNumber(baseVal) ? baseVal : 0;
    const baseBaseline = isFiniteNumber(baseline[key]) ? baseline[key] : 0;

    const centered = safeBase - baseBaseline;
    const prev = prevCentered && isFiniteNumber(prevCentered[key]) ? prevCentered[key] : null;
    const delta = prev === null ? 0 : centered - prev;

    featureMap[key] = safeBase;
    featureMap[`${key}_centered`] = centered;
    featureMap[`${key}_delta`] = delta;

    nextPrevCentered[key] = centered;
  }

  return { featureMap, nextPrevCentered };
}
