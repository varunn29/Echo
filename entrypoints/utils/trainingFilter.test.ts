import { describe, expect, it } from 'vitest';
import {
  isObviousNonContent,
  isPotentialStyleSample,
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