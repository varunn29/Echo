import 'fake-indexeddb/auto';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  db,
  saveWritingMessage,
  getWritingMessages,
  DEFAULT_DUPLICATE_WINDOW_MS,
  type WritingMessage,
} from './database';

describe('Message Collection Integration Suite', function () {
  beforeEach(async function () {
    await db.writingMessages.clear();
  });

  describe('1. Consecutive Message Capture & Persistence', function () {
    it('should reliably persist consecutive distinct messages', async function () {
      const messages = [
        'First message: How do I build a Chrome extension?',
        'Second message: How does Manifest V3 handle service workers?',
        'Third message: Explain IndexedDB transactions.',
      ];

      const savedIds: number[] = [];

      for (let i = 0; i < messages.length; i++) {
        const messageText = messages[i]!;
        const time = new Date(Date.now() + i * 5000);
        const id = await saveWritingMessage({
          platform: 'ChatGPT',
          text: messageText,
          createdAt: time,
          capturedAt: time,
          trainingStatus: 'pending',
        });

        expect(id).toBeTypeOf('number');
        savedIds.push(id!);
      }

      const allRecords = await getWritingMessages();
      expect(allRecords).toHaveLength(3);
      // getWritingMessages returns sorted by capturedAt descending (newest first)
      expect(allRecords[0]?.text).toBe(messages[2]);
      expect(allRecords[1]?.text).toBe(messages[1]);
      expect(allRecords[2]?.text).toBe(messages[0]);
    });
  });

  describe('2. Identical Messages: Accidental vs. Intentional Repetitions', function () {
    it('should reject accidental duplicates sent within the deduplication window', async function () {
      const baseTime = new Date('2026-11-07T10:00:00.000Z');
      const rapidTime = new Date('2026-11-07T10:00:00.500Z'); // 500ms later (e.g. Enter + submit)

      const id1 = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'What is WebAssembly?',
        createdAt: baseTime,
        capturedAt: baseTime,
        trainingStatus: 'pending',
      });

      const id2 = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'What is WebAssembly?',
        createdAt: rapidTime,
        capturedAt: rapidTime,
        trainingStatus: 'pending',
      });

      expect(id1).toBeTypeOf('number');
      expect(id2).toBeUndefined(); // Ignored as duplicate

      const records = await getWritingMessages();
      expect(records).toHaveLength(1);
    });

    it('should preserve intentionally repeated messages sent outside the deduplication window', async function () {
      const mondayTime = new Date('2026-11-02T10:00:00.000Z');
      const fridayTime = new Date('2026-11-06T15:00:00.000Z'); // 4 days later

      const id1 = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'Can you summarize this article for me?',
        createdAt: mondayTime,
        capturedAt: mondayTime,
        trainingStatus: 'pending',
      });

      const id2 = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'Can you summarize this article for me?',
        createdAt: fridayTime,
        capturedAt: fridayTime,
        trainingStatus: 'pending',
      });

      expect(id1).toBeTypeOf('number');
      expect(id2).toBeTypeOf('number');
      expect(id2).not.toBe(id1);

      const records = await getWritingMessages();
      expect(records).toHaveLength(2);
      expect(records[0]?.id).toBe(id2);
      expect(records[1]?.id).toBe(id1);
    });
  });

  describe('3. Service Worker Suspension & Restart Simulation', function () {
    it('should maintain data integrity across database disconnect and reconnect', async function () {
      // Step 1: Save initial records
      const initialId = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'Message saved before service worker suspension',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });
      expect(initialId).toBeTypeOf('number');

      // Step 2: Simulate service worker suspension by closing the Dexie database connection
      db.close();
      expect(db.isOpen()).toBe(false);

      // Step 3: Simulate wake-up and save new message (Dexie auto-reopens on subsequent transaction)
      const afterWakeupId = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'Message saved after service worker wake-up',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      expect(afterWakeupId).toBeTypeOf('number');
      expect(afterWakeupId).not.toBe(initialId);

      // Step 4: Verify all records exist intact
      const records = await getWritingMessages();
      expect(records).toHaveLength(2);
      expect(records.map((r) => r.text)).toContain(
        'Message saved before service worker suspension'
      );
      expect(records.map((r) => r.text)).toContain(
        'Message saved after service worker wake-up'
      );
    });
  });

  describe('4. Preservation of Existing Records During Schema Operations', function () {
    it('should preserve all 102 existing records and correctly append new ones', async function () {
      // Seed 102 synthetic records representing the user's existing database
      const existingRecords: WritingMessage[] = [];
      const startTime = new Date('2026-10-01T00:00:00.000Z').getTime();

      for (let i = 1; i <= 102; i++) {
        const recordTime = new Date(startTime + i * 3600 * 1000);
        existingRecords.push({
          id: i,
          platform: 'ChatGPT',
          text: `Historical collected user message #${i}`,
          createdAt: recordTime,
          capturedAt: recordTime,
          trainingStatus: 'pending',
        });
      }

      await db.writingMessages.bulkAdd(existingRecords);
      const initialCount = await db.writingMessages.count();
      expect(initialCount).toBe(102);

      // Save a new message via the updated pipeline
      const newTime = new Date();
      const newId = await saveWritingMessage({
        platform: 'ChatGPT',
        text: 'Newly captured user message #103',
        createdAt: newTime,
        capturedAt: newTime,
        trainingStatus: 'pending',
      });

      expect(newId).toBe(103);

      // Verify all 102 historical records remain unchanged and intact
      const totalCount = await db.writingMessages.count();
      expect(totalCount).toBe(103);

      const record1 = await db.writingMessages.get(1);
      expect(record1?.text).toBe('Historical collected user message #1');

      const record102 = await db.writingMessages.get(102);
      expect(record102?.text).toBe('Historical collected user message #102');

      const record103 = await db.writingMessages.get(103);
      expect(record103?.text).toBe('Newly captured user message #103');
    });
  });

  describe('5. JSON Export Integrity Verification', function () {
    it('should generate a complete, valid JSON backup matching all stored records', async function () {
      // Seed 5 sample records
      for (let i = 1; i <= 5; i++) {
        await saveWritingMessage({
          platform: 'ChatGPT',
          text: `Sample message #${i}`,
          createdAt: new Date(`2026-11-0${i}T10:00:00.000Z`),
          capturedAt: new Date(`2026-11-0${i}T10:00:00.000Z`),
          trainingStatus: i % 2 === 0 ? 'trained' : 'pending',
        });
      }

      const storedMessages = await getWritingMessages();
      expect(storedMessages).toHaveLength(5);

      // Simulate export transformation
      const exportData = {
        exportedAt: new Date().toISOString(),
        totalMessages: storedMessages.length,
        messages: storedMessages.map(function (message) {
          return {
            id: message.id,
            platform: message.platform,
            text: message.text,
            createdAt:
              message.createdAt instanceof Date
                ? message.createdAt.toISOString()
                : new Date(message.createdAt).toISOString(),
            capturedAt:
              message.capturedAt instanceof Date
                ? message.capturedAt.toISOString()
                : new Date(message.capturedAt).toISOString(),
            trainingStatus: message.trainingStatus,
          };
        }),
      };

      const jsonString = JSON.stringify(exportData, null, 2);
      const restored = JSON.parse(jsonString);

      expect(restored.totalMessages).toBe(5);
      expect(restored.messages).toHaveLength(5);
      expect(restored.messages[0].id).toBeTypeOf('number');
      expect(restored.messages[0].platform).toBe('ChatGPT');
      expect(restored.messages[0].createdAt).toMatch(
        /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/
      );
    });
  });
});
