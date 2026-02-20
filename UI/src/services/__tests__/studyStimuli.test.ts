import {
  getStimulusItemsForBlock,
  buildRecognitionChoices,
  createDelayedPacket,
} from '../studyStimuli';

describe('getStimulusItemsForBlock', () => {
  it('returns requested number of items', () => {
    const items = getStimulusItemsForBlock('A', 'easy', 4, 'test-seed');
    expect(items).toHaveLength(4);
  });

  it('filters by difficulty', () => {
    const easyItems = getStimulusItemsForBlock('A', 'easy', 50, 'test-seed');
    const hardItems = getStimulusItemsForBlock('A', 'hard', 50, 'test-seed');

    easyItems.forEach((item) => expect(item.difficulty).toBe('easy'));
    hardItems.forEach((item) => expect(item.difficulty).toBe('hard'));
  });

  it('produces deterministic order for same seed', () => {
    const run1 = getStimulusItemsForBlock('A', 'easy', 5, 'seed-abc');
    const run2 = getStimulusItemsForBlock('A', 'easy', 5, 'seed-abc');
    expect(run1.map((i) => i.id)).toEqual(run2.map((i) => i.id));
  });

  it('produces different order for different seeds', () => {
    const run1 = getStimulusItemsForBlock('A', 'easy', 5, 'seed-abc');
    const run2 = getStimulusItemsForBlock('A', 'easy', 5, 'seed-xyz');
    // Very unlikely to be identical — test for at least one difference
    const ids1 = run1.map((i) => i.id);
    const ids2 = run2.map((i) => i.id);
    expect(ids1).not.toEqual(ids2);
  });

  it('returns items from form B when requested', () => {
    const items = getStimulusItemsForBlock('B', 'easy', 3, 'test-seed');
    expect(items).toHaveLength(3);
    items.forEach((item) => expect(item.difficulty).toBe('easy'));
  });
});

describe('buildRecognitionChoices', () => {
  it('includes the correct answer', () => {
    const items = getStimulusItemsForBlock('A', 'easy', 6, 'test-seed');
    const target = items[0];
    const choices = buildRecognitionChoices(target, items, 4, false, 'test-seed');

    const correct = choices.filter((c) => c.isCorrect);
    expect(correct).toHaveLength(1);
    expect(correct[0].value).toBe(target.target);
  });

  it('returns the requested number of choices', () => {
    const items = getStimulusItemsForBlock('A', 'easy', 6, 'test-seed');
    const target = items[0];
    const choices = buildRecognitionChoices(target, items, 4, false, 'test-seed');
    expect(choices.length).toBeLessThanOrEqual(4);
    expect(choices.length).toBeGreaterThanOrEqual(2);
  });

  it('has deterministic output for same seed', () => {
    const items = getStimulusItemsForBlock('A', 'easy', 6, 'test-seed');
    const target = items[0];
    const run1 = buildRecognitionChoices(target, items, 4, false, 'seed-a');
    const run2 = buildRecognitionChoices(target, items, 4, false, 'seed-a');
    expect(run1.map((c) => c.value)).toEqual(run2.map((c) => c.value));
  });

  it('does not include the target as a distractor', () => {
    const items = getStimulusItemsForBlock('A', 'easy', 10, 'test-seed');
    const target = items[0];
    const choices = buildRecognitionChoices(target, items, 4, false, 'test-seed');

    const matching = choices.filter((c) => c.value === target.target);
    // Only the correct answer should match
    expect(matching).toHaveLength(1);
    expect(matching[0].isCorrect).toBe(true);
  });
});

describe('createDelayedPacket', () => {
  it('creates a packet with correct fields', () => {
    const easyItems = getStimulusItemsForBlock('A', 'easy', 3, 'test');
    const hardItems = getStimulusItemsForBlock('A', 'hard', 3, 'test');

    const packet = createDelayedPacket('P001', 1, 'A', easyItems, hardItems);

    expect(packet.participantId).toBe('P001');
    expect(packet.sessionNumber).toBe(1);
    expect(packet.form).toBe('A');
    expect(packet.easyItems).toHaveLength(3);
    expect(packet.hardItems).toHaveLength(3);
    expect(packet.generatedAtIso).toBeTruthy();
  });

  it('includes an ISO timestamp', () => {
    const packet = createDelayedPacket('P002', 2, 'B', [], []);
    // ISO string should be parseable
    const date = new Date(packet.generatedAtIso);
    expect(date.getTime()).not.toBeNaN();
  });
});
