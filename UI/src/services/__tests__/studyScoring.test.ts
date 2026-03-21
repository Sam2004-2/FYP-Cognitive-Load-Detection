import { normalizeRecallText, scoreCuedRecallResponse } from '../studyScoring';

describe('normalizeRecallText', () => {
  it('normalizes casing, punctuation, whitespace, and diacritics', () => {
    expect(normalizeRecallText('  Café!!!  au---lait  ')).toBe('cafe au lait');
  });
});

describe('scoreCuedRecallResponse', () => {
  it('accepts exact normalized matches', () => {
    const result = scoreCuedRecallResponse('  MURMURED!! ', 'murmured', ['murmured', 'murmur']);
    expect(result.correct).toBe(true);
    expect(result.scoring.method).toBe('exact_normalized');
    expect(result.scoring.matchType).toBe('exact');
    expect(result.scoring.distance).toBe(0);
  });

  it('accepts a single insertion as a near match when unique', () => {
    const result = scoreCuedRecallResponse('tableeau', 'tableau', ['tableau', 'fabled']);
    expect(result.correct).toBe(true);
    expect(result.scoring.method).toBe('tolerant_damerau_1');
    expect(result.scoring.matchType).toBe('near_match');
    expect(result.scoring.distance).toBe(1);
  });

  it('accepts a single deletion as a near match when unique', () => {
    const result = scoreCuedRecallResponse('tablau', 'tableau', ['tableau', 'fabled']);
    expect(result.correct).toBe(true);
    expect(result.scoring.matchType).toBe('near_match');
    expect(result.scoring.distance).toBe(1);
  });

  it('accepts a single substitution as a near match when unique', () => {
    const result = scoreCuedRecallResponse('tableou', 'tableau', ['tableau', 'fabled']);
    expect(result.correct).toBe(true);
    expect(result.scoring.matchType).toBe('near_match');
    expect(result.scoring.distance).toBe(1);
  });

  it('accepts a single adjacent transposition as a near match when unique', () => {
    const result = scoreCuedRecallResponse('tabelau', 'tableau', ['tableau', 'fabled']);
    expect(result.correct).toBe(true);
    expect(result.scoring.matchType).toBe('near_match');
    expect(result.scoring.distance).toBe(1);
  });

  it('rejects clear misses', () => {
    const result = scoreCuedRecallResponse('harbor', 'tableau', ['tableau', 'fabled']);
    expect(result.correct).toBe(false);
    expect(result.scoring.matchType).toBe('mismatch');
  });

  it('rejects an ambiguous response that exactly matches another active target', () => {
    const result = scoreCuedRecallResponse('ankle', 'anklet', ['anklet', 'ankle']);
    expect(result.correct).toBe(false);
    expect(result.scoring.matchType).toBe('ambiguous_near_match');
  });

  it('rejects an ambiguous response that is equally close to another active target', () => {
    const result = scoreCuedRecallResponse('cpher', 'cipher', ['cipher', 'cypher']);
    expect(result.correct).toBe(false);
    expect(result.scoring.matchType).toBe('ambiguous_near_match');
  });
});
