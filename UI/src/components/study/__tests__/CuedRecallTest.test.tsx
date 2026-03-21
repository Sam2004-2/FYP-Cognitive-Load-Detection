import { fireEvent, render, screen } from '@testing-library/react';
import CuedRecallTest from '../CuedRecallTest';
import { StudyStimulusItem, StudyTrialResult } from '../../../types/study';

describe('CuedRecallTest', () => {
  it('stores scoring metadata for accepted near matches', () => {
    const items: StudyStimulusItem[] = [
      { id: 'B-H18', cue: 'postal', target: 'tableau', difficulty: 'hard', interferenceGroup: 'hard-g9' },
      { id: 'B-H17', cue: 'dorsal', target: 'fabled', difficulty: 'hard', interferenceGroup: 'hard-g9' },
    ];
    const onComplete = jest.fn<void, [StudyTrialResult[]]>();

    render(
      <CuedRecallTest
        items={items}
        blockIndex={2}
        phase="test_hard_cued_recall"
        condition="adaptive"
        form="B"
        sessionStartMs={Date.now()}
        onComplete={onComplete}
      />
    );

    fireEvent.change(screen.getByPlaceholderText('Enter paired target'), {
      target: { value: 'tabelau' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));

    fireEvent.change(screen.getByPlaceholderText('Enter paired target'), {
      target: { value: 'fabled' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Finish Cued Recall' }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    const trials = onComplete.mock.calls[0][0];
    expect(trials[0].correct).toBe(true);
    expect(trials[0].scoring).toMatchObject({
      version: 2,
      method: 'tolerant_damerau_1',
      matchType: 'near_match',
      normalizedTarget: 'tableau',
    });
    expect(trials[1].correct).toBe(true);
    expect(trials[1].scoring).toMatchObject({
      version: 2,
      method: 'exact_normalized',
      matchType: 'exact',
      normalizedTarget: 'fabled',
    });
  });
});
