import { validateWindowQuality, WindowBuffer } from '../windowBuffer';
import { FrameFeatures } from '../../types/features';

function makeFrame(valid = true, overrides: Partial<FrameFeatures> = {}): FrameFeatures {
  return {
    ear_left: 0.3,
    ear_right: 0.3,
    ear_mean: 0.3,
    brightness: 120,
    quality: 1.0,
    eye_center_x: 0.5,
    eye_center_y: 0.5,
    mouth_mar: 0.3,
    roll: 0.0,
    pitch: 0.0,
    yaw: 0.0,
    valid,
    ...overrides,
  };
}

describe('validateWindowQuality', () => {
  it('returns [false, 1.0] for empty array', () => {
    const [isValid, badRatio] = validateWindowQuality([]);
    expect(isValid).toBe(false);
    expect(badRatio).toBe(1.0);
  });

  it('returns [true, 0.0] when all frames are valid', () => {
    const frames = Array.from({ length: 100 }, () => makeFrame(true));
    const [isValid, badRatio] = validateWindowQuality(frames);
    expect(isValid).toBe(true);
    expect(badRatio).toBe(0.0);
  });

  it('returns [false, ratio] when bad ratio exceeds threshold', () => {
    const frames = [
      ...Array.from({ length: 90 }, () => makeFrame(true)),
      ...Array.from({ length: 10 }, () => makeFrame(false)),
    ];
    const [isValid, badRatio] = validateWindowQuality(frames, 0.05);
    expect(isValid).toBe(false);
    expect(badRatio).toBeCloseTo(0.1);
  });

  it('returns [true, ratio] when bad ratio is at threshold', () => {
    const frames = [
      ...Array.from({ length: 95 }, () => makeFrame(true)),
      ...Array.from({ length: 5 }, () => makeFrame(false)),
    ];
    const [isValid, badRatio] = validateWindowQuality(frames, 0.05);
    expect(isValid).toBe(true);
    expect(badRatio).toBeCloseTo(0.05);
  });
});

describe('WindowBuffer', () => {
  it('initializes with correct capacity', () => {
    const buf = new WindowBuffer(10, 30);
    expect(buf.capacity).toBe(300);
    expect(buf.length).toBe(0);
    expect(buf.fillRatio).toBe(0);
  });

  it('adds frames and tracks length', () => {
    const buf = new WindowBuffer(1, 10); // capacity 10
    for (let i = 0; i < 5; i++) {
      buf.addFrame(makeFrame());
    }
    expect(buf.length).toBe(5);
    expect(buf.fillRatio).toBeCloseTo(0.5);
  });

  it('reports ready when buffer is full', () => {
    const buf = new WindowBuffer(1, 5); // capacity 5
    for (let i = 0; i < 5; i++) {
      buf.addFrame(makeFrame());
    }
    expect(buf.isReady()).toBe(true);
  });

  it('reports not ready when buffer is not full', () => {
    const buf = new WindowBuffer(1, 10);
    buf.addFrame(makeFrame());
    expect(buf.isReady()).toBe(false);
  });

  it('evicts oldest frames when capacity is exceeded', () => {
    const buf = new WindowBuffer(1, 3); // capacity 3

    buf.addFrame(makeFrame(true, { ear_mean: 0.1 }));
    buf.addFrame(makeFrame(true, { ear_mean: 0.2 }));
    buf.addFrame(makeFrame(true, { ear_mean: 0.3 }));
    buf.addFrame(makeFrame(true, { ear_mean: 0.4 }));

    expect(buf.length).toBe(3);

    const window = buf.getWindow();
    expect(window[0].ear_mean).toBeCloseTo(0.2); // oldest should be evicted
    expect(window[2].ear_mean).toBeCloseTo(0.4);
  });

  it('getWindow returns a copy (not internal reference)', () => {
    const buf = new WindowBuffer(1, 5);
    buf.addFrame(makeFrame());
    const w1 = buf.getWindow();
    const w2 = buf.getWindow();
    expect(w1).not.toBe(w2);
    expect(w1).toEqual(w2);
  });

  it('getWindowTimes returns [0, 0] for empty buffer', () => {
    const buf = new WindowBuffer(1, 10);
    const [start, end] = buf.getWindowTimes();
    expect(start).toBe(0.0);
    expect(end).toBe(0.0);
  });

  it('getWindowTimes returns increasing times', () => {
    const buf = new WindowBuffer(1, 10);
    for (let i = 0; i < 5; i++) {
      buf.addFrame(makeFrame());
    }
    const [start, end] = buf.getWindowTimes();
    expect(end).toBeGreaterThan(0);
    expect(start).toBeLessThanOrEqual(end);
  });

  it('reset clears buffer and frame count', () => {
    const buf = new WindowBuffer(1, 10);
    for (let i = 0; i < 5; i++) {
      buf.addFrame(makeFrame());
    }
    expect(buf.length).toBe(5);

    buf.reset();
    expect(buf.length).toBe(0);
    expect(buf.fillRatio).toBe(0);
    expect(buf.isReady()).toBe(false);
  });
});
