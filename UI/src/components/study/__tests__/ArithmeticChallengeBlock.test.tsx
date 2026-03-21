import { act, fireEvent, render, screen } from '@testing-library/react';
import ArithmeticChallengeBlock from '../ArithmeticChallengeBlock';
import { ArithmeticProblem, ArithmeticTrial } from '../../../types/study';

const easyProblem: ArithmeticProblem = {
  id: 'arith-e-01',
  difficulty: 'easy',
  leftOperand: 2,
  rightOperand: 3,
  expression: '2 + 3',
  answer: 5,
};

describe('ArithmeticChallengeBlock', () => {
  afterEach(() => {
    jest.useRealTimers();
  });

  it('records a correct submitted answer', () => {
    const onComplete = jest.fn<void, [ArithmeticTrial[]]>();

    render(
      <ArithmeticChallengeBlock
        phase="arithmetic_easy"

        problems={[easyProblem]}
        timeLimitSeconds={8}
        sessionStartMs={Date.now()}
        condition="adaptive"
        form="A"
        onComplete={onComplete}
      />
    );

    fireEvent.change(screen.getByPlaceholderText('Enter the total'), {
      target: { value: '5' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Finish Arithmetic' }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0][0]).toMatchObject({
      expectedAnswer: 5,
      responseText: '5',
      responseValue: 5,
      correct: true,
      timedOut: false,
      practice: false,
    });
  });

  it('records a blank manual submission as incorrect without timeout', () => {
    const onComplete = jest.fn<void, [ArithmeticTrial[]]>();

    render(
      <ArithmeticChallengeBlock
        phase="arithmetic_medium"

        problems={[
          {
            id: 'arith-m-01',
            difficulty: 'medium',
            leftOperand: 24,
            rightOperand: 35,
            expression: '24 + 35',
            answer: 59,
          },
        ]}
        timeLimitSeconds={8}
        sessionStartMs={Date.now()}
        condition="baseline"
        form="B"
        onComplete={onComplete}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Finish Arithmetic' }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0][0]).toMatchObject({
      responseText: undefined,
      responseValue: undefined,
      correct: false,
      timedOut: false,
      practice: false,
      condition: 'baseline',
      form: 'B',
    });
  });

  it('records timeout when no answer is submitted before the limit', () => {
    jest.useFakeTimers();
    const onComplete = jest.fn<void, [ArithmeticTrial[]]>();

    render(
      <ArithmeticChallengeBlock
        phase="arithmetic_hard"

        problems={[
          {
            id: 'arith-h-01',
            difficulty: 'hard',
            leftOperand: 28,
            rightOperand: 17,
            expression: '28 + 17',
            answer: 45,
          },
        ]}
        timeLimitSeconds={8}
        sessionStartMs={Date.now()}
        condition="adaptive"
        form="A"
        onComplete={onComplete}
      />
    );

    act(() => {
      jest.advanceTimersByTime(8000);
    });

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0][0]).toMatchObject({
      responseText: undefined,
      responseValue: undefined,
      correct: false,
      timedOut: true,
      practice: false,
    });
  });
});
