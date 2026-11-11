import 'fake-indexeddb/auto';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  createEchoDatabase,
  getTrainingReadiness,
  DEFAULT_TRAINING_READINESS_THRESHOLD,
  type EchoDatabase,
  type WritingMessage,
} from './database';

function createSampleMessage(
  text: string,
  trainingStatus: 'pending' | 'trained' = 'pending',
  platform = 'ChatGPT',
  capturedAt = new Date()
): WritingMessage {
  return {
    platform,
    text,
    createdAt: new Date(),
    capturedAt,
    trainingStatus,
  };
}

describe('Training Readiness Foundation (getTrainingReadiness)', function () {
  let testDb: EchoDatabase;

  beforeEach(async function () {
    const dbName = `TrainingReadinessTest_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    testDb = createEchoDatabase(dbName);
    await testDb.open();
  });

  afterEach(async function () {
    if (testDb && testDb.isOpen()) {
      await testDb.delete();
      testDb.close();
    }
  });

  describe('1. Empty Database', function () {
    it('should return zero counts and not ready when database has no messages', async function () {
      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(0);
      expect(result.eligibleCount).toBe(0);
      expect(result.threshold).toBe(DEFAULT_TRAINING_READINESS_THRESHOLD);
      expect(result.isReady).toBe(false);
      expect(result.eligibleSamples).toEqual([]);
      expect(result.breakdown).toEqual({
        nonContent: 0,
        code: 0,
        terminal: 0,
        structuredData: 0,
        shortFragment: 0,
        weakOrNoStyle: 0,
        totalExcluded: 0,
      });
    });
  });

  describe('2. Pending Eligible Prose', function () {
    it('should identify natural personal prose with strong style signal as eligible', async function () {
      const sample1 = createSampleMessage(
        'I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today\'s time.'
      );
      const sample2 = createSampleMessage(
        'I have not yet decided but planning to try with AI and some ML, also with cloud computing, will think about other technologies in the coming future.'
      );

      await testDb.writingMessages.bulkAdd([sample1, sample2]);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(2);
      expect(result.eligibleCount).toBe(2);
      expect(result.eligibleSamples).toHaveLength(2);
      expect(result.eligibleSamples[0]?.text).toBe(sample1.text);
      expect(result.eligibleSamples[1]?.text).toBe(sample2.text);
      expect(result.breakdown.totalExcluded).toBe(0);
    });
  });

  describe('3. Already-Trained Messages', function () {
    it('should ignore already-trained messages and only count pending messages', async function () {
      const trainedSample1 = createSampleMessage(
        'I spent several years developing frontend architectures and exploring distributed systems in depth.',
        'trained'
      );
      const trainedSample2 = createSampleMessage(
        'When building backend APIs, I generally prefer clear relational schemas with strong typing over dynamic schemas.',
        'trained'
      );
      const pendingSample = createSampleMessage(
        'Recently I have been exploring small parameter open models for personalized writing assistance on local devices.'
      );

      await testDb.writingMessages.bulkAdd([trainedSample1, trainedSample2, pendingSample]);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(1);
      expect(result.eligibleCount).toBe(1);
      expect(result.eligibleSamples).toHaveLength(1);
      expect(result.eligibleSamples[0]?.text).toBe(pendingSample.text);
      expect(result.breakdown.totalExcluded).toBe(0);
    });
  });

  describe('4. Code and Terminal Content Exclusion', function () {
    it('should exclude fenced code, unfenced code, terminal commands, and compiler logs', async function () {
      const pureFencedCode = createSampleMessage(
        '```typescript\ninterface Config { port: number; host: string; }\n```'
      );
      const unfencedCode = createSampleMessage(
        'function calculateTotal(items) {\n  return items.reduce((sum, item) => sum + item.price, 0);\n}'
      );
      const terminalCommand = createSampleMessage(
        'npm install @tanstack/react-query dexie vitest'
      );
      const gitCommand = createSampleMessage(
        'git checkout -b feature/training-readiness'
      );
      const compilerError = createSampleMessage(
        'TypeError: Cannot read properties of undefined (reading "map")\n    at WritingData.tsx:180:24\n    at Component.render'
      );

      await testDb.writingMessages.bulkAdd([
        pureFencedCode,
        unfencedCode,
        terminalCommand,
        gitCommand,
        compilerError,
      ]);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(5);
      expect(result.eligibleCount).toBe(0);
      expect(result.eligibleSamples).toHaveLength(0);
      expect(result.breakdown.totalExcluded).toBe(5);
      expect(result.breakdown.code).toBeGreaterThanOrEqual(1);
      expect(result.breakdown.terminal).toBeGreaterThanOrEqual(1);
    });
  });

  describe('5. Short Fragments Exclusion', function () {
    it('should exclude conversational acknowledgements and short fragments', async function () {
      const fragment1 = createSampleMessage('ok');
      const fragment2 = createSampleMessage('sounds good');
      const fragment3 = createSampleMessage('thanks a lot');
      const fragment4 = createSampleMessage('yes please');

      await testDb.writingMessages.bulkAdd([fragment1, fragment2, fragment3, fragment4]);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(4);
      expect(result.eligibleCount).toBe(0);
      expect(result.breakdown.shortFragment).toBe(4);
      expect(result.breakdown.totalExcluded).toBe(4);
    });
  });

  describe('6. Weak / No-Style Samples Exclusion', function () {
    it('should exclude short questions and generic prose lacking personal style signal', async function () {
      // 5 words without personal context
      const weak1 = createSampleMessage('How does CSS flexbox work?');
      // 9 words generic prose without personal context
      const weak2 = createSampleMessage('The quick brown fox jumps over the lazy dog.');

      await testDb.writingMessages.bulkAdd([weak1, weak2]);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(2);
      expect(result.eligibleCount).toBe(0);
      expect(result.breakdown.weakOrNoStyle).toBe(2);
      expect(result.breakdown.totalExcluded).toBe(2);
    });
  });

  describe('7. Mixed Content Rules', function () {
    it('should accept mixed content when the prose portion has a strong style signal (14+ words)', async function () {
      const strongMixed = createSampleMessage(
        `I am trying to implement a debounce hook in React, but the timer keeps resetting on each render:

\`\`\`tsx
function useDebounce(value, delay) {
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
}
\`\`\`

How can I persist the timer reference across re-renders without triggering extra updates?`
      );

      await testDb.writingMessages.add(strongMixed);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(1);
      expect(result.eligibleCount).toBe(1);
      expect(result.eligibleSamples).toHaveLength(1);
      expect(result.breakdown.totalExcluded).toBe(0);
    });

    it('should exclude mixed content when the prose portion has a weak style signal (<14 words)', async function () {
      const weakMixed = createSampleMessage(
        `Here is my component code:

\`\`\`tsx
const [count, setCount] = useState(0);
\`\`\`

Please fix this issue.`
      );

      await testDb.writingMessages.add(weakMixed);

      const result = await getTrainingReadiness(testDb);

      expect(result.totalPending).toBe(1);
      expect(result.eligibleCount).toBe(0);
      expect(result.breakdown.weakOrNoStyle).toBe(1);
      expect(result.breakdown.totalExcluded).toBe(1);
    });
  });

  describe('8. Readiness Threshold Calculation', function () {
    it('should accurately calculate isReady against the default threshold of 25', async function () {
      // Generate 24 eligible samples
      const messages24 = Array.from({ length: 24 }, (_, i) =>
        createSampleMessage(
          `Sample ${i + 1}: I enjoy building distributed applications with TypeScript and React because the typing system catches subtle architectural bugs early.`
        )
      );

      await testDb.writingMessages.bulkAdd(messages24);

      let result = await getTrainingReadiness(testDb);

      expect(result.eligibleCount).toBe(24);
      expect(result.threshold).toBe(25);
      expect(result.isReady).toBe(false);

      // Add the 25th sample
      const sample25 = createSampleMessage(
        'Sample 25: Exploring fine-tuning on local CPU environments requires balancing memory constraints with token batch sizes carefully.'
      );
      await testDb.writingMessages.add(sample25);

      result = await getTrainingReadiness(testDb);

      expect(result.eligibleCount).toBe(25);
      expect(result.threshold).toBe(25);
      expect(result.isReady).toBe(true);

      // Display string format verification
      const displayString = `${result.eligibleCount} / ${result.threshold} eligible samples collected`;
      expect(displayString).toBe('25 / 25 eligible samples collected');
    });

    it('should support configurable threshold parameter', async function () {
      const messages = Array.from({ length: 12 }, (_, i) =>
        createSampleMessage(
          `Sample ${i + 1}: Working with IndexedDB in Chrome extensions requires handling service worker lifecycle events and connection upgrades smoothly.`
        )
      );

      await testDb.writingMessages.bulkAdd(messages);

      // Custom threshold of 10 -> should be ready (12 >= 10)
      const readyResult = await getTrainingReadiness(10, testDb);
      expect(readyResult.eligibleCount).toBe(12);
      expect(readyResult.threshold).toBe(10);
      expect(readyResult.isReady).toBe(true);

      // Custom threshold of 20 -> should not be ready (12 < 20)
      const notReadyResult = await getTrainingReadiness({ threshold: 20 }, testDb);
      expect(notReadyResult.eligibleCount).toBe(12);
      expect(notReadyResult.threshold).toBe(20);
      expect(notReadyResult.isReady).toBe(false);

      // Display formatting matches dashboard requirement
      const displayString = `${notReadyResult.eligibleCount} / ${notReadyResult.threshold} eligible samples collected`;
      expect(displayString).toBe('12 / 20 eligible samples collected');
    });
  });

  describe('9. Read-Only Behaviour (No Modification of Training Status)', function () {
    it('should leave all database records and their training statuses completely unmodified', async function () {
      const pendingEligible = createSampleMessage(
        'I am designing a modular architecture for personalized local writing assistance that decouples background collection from local model training.',
        'pending'
      );
      const pendingTerminal = createSampleMessage(
        'npm run build && npm test',
        'pending'
      );
      const trainedMessage = createSampleMessage(
        'I previously completed fullstack training modules covering relational database optimization and schema design.',
        'trained'
      );

      const id1 = await testDb.writingMessages.add(pendingEligible);
      const id2 = await testDb.writingMessages.add(pendingTerminal);
      const id3 = await testDb.writingMessages.add(trainedMessage);

      // Run training readiness evaluation multiple times
      const firstRun = await getTrainingReadiness(testDb);
      const secondRun = await getTrainingReadiness(testDb);

      expect(firstRun.eligibleCount).toBe(1);
      expect(secondRun.eligibleCount).toBe(1);

      // Verify all raw records in the database remain unchanged
      const rawRecords = await testDb.writingMessages.toArray();

      expect(rawRecords).toHaveLength(3);

      const record1 = rawRecords.find((r) => r.id === id1);
      const record2 = rawRecords.find((r) => r.id === id2);
      const record3 = rawRecords.find((r) => r.id === id3);

      expect(record1?.trainingStatus).toBe('pending');
      expect(record1?.text).toBe(pendingEligible.text);

      expect(record2?.trainingStatus).toBe('pending');
      expect(record2?.text).toBe(pendingTerminal.text);

      expect(record3?.trainingStatus).toBe('trained');
      expect(record3?.text).toBe(trainedMessage.text);
    });
  });
});
