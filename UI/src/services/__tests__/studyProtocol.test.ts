import { computeStudyAssignment, validateSession2Timing } from '../studyProtocol';

describe('studyProtocol', () => {
  it('assigns deterministic condition/form order by parity', () => {
    const s1 = computeStudyAssignment('P001', 1, new Date('2026-02-10T10:00:00Z'));
    const s2 = computeStudyAssignment('P001', 2, new Date('2026-02-10T10:00:00Z'));

    expect(s1.conditionOrder).toHaveLength(2);
    expect(s1.formOrder).toHaveLength(2);
    expect(s1.condition).toBe(s1.conditionOrder[0]);
    expect(s2.condition).toBe(s1.conditionOrder[1]);
    expect(s1.form).toBe(s1.formOrder[0]);
    expect(s2.form).toBe(s1.formOrder[1]);
    expect(s1.hashValue).toBe(s2.hashValue);
  });

  it('flags early session 2 starts', () => {
    const result = validateSession2Timing(2, '2026-02-10T00:00:00Z', new Date('2026-02-10T12:00:00Z'));
    expect(result.tooEarly).toBe(true);
    expect(result.hoursSinceSession1).not.toBeNull();
    expect(result.recommendedMinimumHours).toBe(24);
  });

  it('allows session 2 after 24h', () => {
    const result = validateSession2Timing(2, '2026-02-10T00:00:00Z', new Date('2026-02-11T02:00:00Z'));
    expect(result.tooEarly).toBe(false);
  });
});
