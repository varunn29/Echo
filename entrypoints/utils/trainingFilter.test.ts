import { describe, expect, it } from 'vitest';
import {
  isObviousNonContent,
  isPotentialStyleSample,
  isEligibleTrainingSample,
} from './trainingFilter';

describe('isObviousNonContent', function () {
  it('should reject empty messages', function () {
    expect(isObviousNonContent('   ')).toBe(true);
  });

  it('should reject terminal commands', function () {
    expect(isObviousNonContent('npm install express')).toBe(true);
  });

  it('should reject Git commands', function () {
    expect(isObviousNonContent('git status')).toBe(true);
  });

  it('should reject messages consisting entirely of code blocks', function () {
    const text = '```javascript\nconsole.log("Hello");\n```';

    expect(isObviousNonContent(text)).toBe(true);
  });

  it('should accept natural language', function () {
    expect(
      isObviousNonContent('I enjoy building projects with React.')
    ).toBe(false);
  });
});

describe('isPotentialStyleSample', function () {
  it('should accept natural personal writing', function () {
    expect(
      isPotentialStyleSample(
        'I started learning React recently and I enjoy building projects.'
      )
    ).toBe(true);
  });

  it('should reject terminal commands', function () {
    expect(isPotentialStyleSample('npm install express')).toBe(false);
  });

  it('should reject empty messages', function () {
    expect(isPotentialStyleSample('   ')).toBe(false);
  });
});

describe('isEligibleTrainingSample', function () {
  it('should accept natural personal prose with strong style signal', function () {
    const text =
      'I started learning web development by building simple UI widgets and now I prefer TypeScript with React for larger fullstack projects.';

    expect(isEligibleTrainingSample(text)).toBe(true);
  });

  it('should accept mixed prose and code when prose provides strong style signal', function () {
    const text = `I am trying to implement a debounce hook in React, but the timer keeps resetting on each render:

\`\`\`tsx
function useDebounce(value, delay) {
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
}
\`\`\`

How can I persist the timer reference across re-renders without triggering extra updates?`;

    expect(isEligibleTrainingSample(text)).toBe(true);
  });

  it('should reject empty or whitespace text', function () {
    expect(isEligibleTrainingSample('')).toBe(false);
    expect(isEligibleTrainingSample('   \n  ')).toBe(false);
  });

  it('should reject pure fenced code blocks', function () {
    const text = '```python\ndef solve():\n    return 42\n```';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject unfenced source code', function () {
    const text = 'function calculateSum(a: number, b: number): number { return a + b; }';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject terminal commands', function () {
    expect(isEligibleTrainingSample('npm run build')).toBe(false);
    expect(isEligibleTrainingSample('git checkout -b feature/auth')).toBe(false);
  });

  it('should reject structured JSON data', function () {
    const text = '{"service": "echo", "status": "active", "version": "1.0.0"}';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject short conversational fragments', function () {
    expect(isEligibleTrainingSample('ok')).toBe(false);
    expect(isEligibleTrainingSample('sounds good')).toBe(false);
    expect(isEligibleTrainingSample('thanks!')).toBe(false);
  });

  it('should reject weak style samples with insufficient prose length or signal', function () {
    // 5 words without personal context
    expect(isEligibleTrainingSample('How does CSS flexbox work?')).toBe(false);

    // 9 words generic prose without personal context
    expect(isEligibleTrainingSample('The quick brown fox jumps over the lazy dog.')).toBe(false);
  });

  it('should reject mixed prose and code when prose signal is weak', function () {
    const text = `Here is my code:

\`\`\`ts
const x = 1;
\`\`\`

Fix this.`;

    expect(isEligibleTrainingSample(text)).toBe(false);
  });
});
