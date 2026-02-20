import { median, computeBaseline, engineerFeatures } from '../featureEngineering';
import { WindowFeatures } from '../../types/features';

function makeWindow(overrides: Partial<WindowFeatures> = {}): WindowFeatures {
  return {
    blink_rate: 15,
    blink_count: 3,
    mean_blink_duration: 200,
    ear_std: 0.02,
    perclos: 0.1,
    mouth_open_mean: 0.3,
    mouth_open_std: 0.05,
    roll_std: 0.01,
    pitch_std: 0.02,
    yaw_std: 0.015,
    motion_mean: 0.005,
    motion_std: 0.002,
    mean_brightness: 120,
    std_brightness: 10,
    mean_quality: 0.95,
    valid_frame_ratio: 0.98,
    ...overrides,
  };
}

describe('median', () => {
  it('returns 0 for empty array', () => {
    expect(median([])).toBe(0);
  });

  it('returns the middle value for odd-length array', () => {
    expect(median([3, 1, 2])).toBe(2);
  });

  it('returns the average of two middle values for even-length array', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });

  it('handles single element', () => {
    expect(median([42])).toBe(42);
  });

  it('does not mutate the input array', () => {
    const arr = [3, 1, 2];
    median(arr);
    expect(arr).toEqual([3, 1, 2]);
  });
});

describe('computeBaseline', () => {
  it('computes median for each feature across samples', () => {
    const samples = [
      makeWindow({ blink_rate: 10, perclos: 0.1 }),
      makeWindow({ blink_rate: 20, perclos: 0.3 }),
      makeWindow({ blink_rate: 15, perclos: 0.2 }),
    ];

    const baseline = computeBaseline(samples);
    expect(baseline['blink_rate']).toBe(15);
    expect(baseline['perclos']).toBe(0.2);
  });

  it('returns 0 for features with no finite values', () => {
    const samples = [makeWindow({ blink_rate: NaN }), makeWindow({ blink_rate: NaN })];
    const baseline = computeBaseline(samples);
    expect(baseline['blink_rate']).toBe(0);
  });

  it('handles empty samples array', () => {
    const baseline = computeBaseline([]);
    expect(baseline['blink_rate']).toBe(0);
  });
});

describe('engineerFeatures', () => {
  it('produces base, centered, and delta features', () => {
    const window = makeWindow({ blink_rate: 20, perclos: 0.3 });
    const baseline = { blink_rate: 15, perclos: 0.2 } as Record<string, number>;

    const { featureMap, nextPrevCentered } = engineerFeatures(window, baseline, null);

    expect(featureMap['blink_rate']).toBe(20);
    expect(featureMap['blink_rate_centered']).toBe(5);
    expect(featureMap['blink_rate_delta']).toBe(0); // no prev => delta = 0
    expect(featureMap['perclos_centered']).toBeCloseTo(0.1);
    expect(nextPrevCentered['blink_rate']).toBe(5);
  });

  it('computes correct delta from previous centered', () => {
    const window = makeWindow({ blink_rate: 20 });
    const baseline = { blink_rate: 15 } as Record<string, number>;
    const prevCentered = { blink_rate: 3 } as Record<string, number>;

    const { featureMap } = engineerFeatures(window, baseline, prevCentered);

    // centered = 20 - 15 = 5; delta = 5 - 3 = 2
    expect(featureMap['blink_rate_centered']).toBe(5);
    expect(featureMap['blink_rate_delta']).toBe(2);
  });

  it('handles NaN/undefined base values gracefully', () => {
    const window = makeWindow({ blink_rate: NaN });
    const baseline = { blink_rate: 10 } as Record<string, number>;

    const { featureMap } = engineerFeatures(window, baseline, null);

    // NaN base => safeBase = 0; centered = 0 - 10 = -10
    expect(featureMap['blink_rate']).toBe(0);
    expect(featureMap['blink_rate_centered']).toBe(-10);
  });

  it('handles NaN baseline gracefully', () => {
    const window = makeWindow({ blink_rate: 20 });
    const baseline = { blink_rate: NaN } as Record<string, number>;

    const { featureMap } = engineerFeatures(window, baseline, null);

    // NaN baseline => safeBaseline = 0; centered = 20 - 0 = 20
    expect(featureMap['blink_rate_centered']).toBe(20);
  });
});
