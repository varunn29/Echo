import 'fake-indexeddb/auto';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Dexie from 'dexie';
import {
  createEchoDatabase,
  deleteWritingMessage,
  saveWritingMessage,
  type EchoDatabase,
} from './database';
import {
  classifyExtractionError,
  clearKnowledgeRetryAlarm,
  COORDINATOR_STORAGE_KEY,
  createDefaultCoordinatorState,
  enqueueKnowledgeMessage,
  getCoordinatorState,
  getKnowledgeCoordinatorStatus,
  KNOWLEDGE_RETRY_ALARM_NAME,
  mutateCoordinatorState,
  processKnowledgeQueue,
  reconsiderSkippedKnowledgeMessages,
  recoverKnowledgeCoordinator,
  resetCoordinatorStateForTesting,
  retryFailedKnowledgeMessages,
  sanitizeErrorMessage,
  scheduleKnowledgeRetryAlarm,
  syncUntrackedKnowledgeMessages,
  type KnowledgeCoordinatorState,
  type StorageLike,
} from './knowledgeCoordinator';
import * as knowledgeExtractor from './knowledgeExtractor';
import * as ollamaClient from './ollamaClient';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Knowledge Coordinator Suite', function () {
  const openDatabases: (Dexie | EchoDatabase)[] = [];
  const dbNamesToDelete: string[] = [];

  function trackDb<T extends Dexie>(dbInstance: T): T {
    openDatabases.push(dbInstance);
    if (!dbNamesToDelete.includes(dbInstance.name)) {
      dbNamesToDelete.push(dbInstance.name);
    }
    return dbInstance;
  }

  // In-memory mock storage for tests
  let mockStorageStore: Record<string, unknown> = {};
  const testStorage: StorageLike = {
    async get(key: string) {
      return { [key]: mockStorageStore[key] };
    },
    async set(items: Record<string, unknown>) {
      Object.assign(mockStorageStore, items);
    },
    async remove(key: string) {
      delete mockStorageStore[key];
    },
  };

  // Mock alarms
  let registeredAlarms: Record<string, { delayInMinutes?: number; periodInMinutes?: number }> = {};
  const mockAlarms = {
    get: vi.fn(async (name: string) => registeredAlarms[name] || null),
    create: vi.fn((name: string, info: { delayInMinutes?: number; periodInMinutes?: number }) => {
      registeredAlarms[name] = info;
    }),
    clear: vi.fn(async (name: string) => {
      const existed = Boolean(registeredAlarms[name]);
      delete registeredAlarms[name];
      return existed;
    }),
  };

  beforeEach(function () {
    mockStorageStore = {};
    registeredAlarms = {};
    resetCoordinatorStateForTesting();
    vi.restoreAllMocks();

    // Stub global browser.alarms
    (globalThis as unknown as { browser: { alarms: typeof mockAlarms } }).browser = {
      alarms: mockAlarms,
    };
  });

  afterEach(async function () {
    resetCoordinatorStateForTesting();
    vi.restoreAllMocks();

    for (const dbInstance of openDatabases) {
      if (dbInstance.isOpen()) {
        dbInstance.close();
      }
    }
    for (const name of dbNamesToDelete) {
      await Dexie.delete(name);
    }
    openDatabases.length = 0;
    dbNamesToDelete.length = 0;
  });

  describe('1. State Initialization and Versioning', function () {
    it('should initialize default state with version 1', async function () {
      const state = await getCoordinatorState(testStorage);
      expect(state.version).toBe(1);
      expect(state.pendingQueue).toEqual([]);
      expect(state.inFlight).toBeNull();
      expect(state.registry).toEqual({});
      expect(state.serviceStatus).toBe('unknown');
    });

    it('should reset corrupted or invalid state to default', async function () {
      await testStorage.set({ [COORDINATOR_STORAGE_KEY]: { invalid: true } });
      const state = await getCoordinatorState(testStorage);
      expect(state.version).toBe(1);
      expect(state.pendingQueue).toEqual([]);
    });
  });

  describe('2. Concurrent State Mutations without Lost Updates', function () {
    it('should serialize concurrent mutations without dropping updates', async function () {
      // Run 20 concurrent mutations incrementing a counter
      const promises: Promise<void>[] = [];

      for (let i = 1; i <= 20; i++) {
        promises.push(
          mutateCoordinatorState((state) => {
            state.registry[i] = {
              messageId: i,
              status: 'pending',
              attempts: 0,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            };
          }, testStorage)
        );
      }

      await Promise.all(promises);

      const finalState = await getCoordinatorState(testStorage);
      // All 20 entries must be present without any race condition loss
      expect(Object.keys(finalState.registry)).toHaveLength(20);
    });
  });

  describe('3. Enqueueing and Pre-Classification', function () {
    it('should skip ineligible code and terminal messages without enqueuing', async function () {
      const codeMsg = {
        id: 101,
        platform: 'ChatGPT',
        text: '```python\ndef test(): pass\n```',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending' as const,
      };

      const result = await enqueueKnowledgeMessage(codeMsg, { storage: testStorage });
      expect(result.enqueued).toBe(false);
      expect(result.status).toBe('skipped');
      expect(result.reason).toBe('ineligible_content');

      const state = await getCoordinatorState(testStorage);
      expect(state.pendingQueue).toHaveLength(0);
      expect(state.registry[101]?.status).toBe('skipped');
    });

    it('should enqueue eligible prose messages into pendingQueue', async function () {
      const proseMsg = {
        id: 102,
        platform: 'ChatGPT',
        text: 'I am building an automated trading platform using Rust and WebSockets.',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending' as const,
      };

      const result = await enqueueKnowledgeMessage(proseMsg, { storage: testStorage });
      expect(result.enqueued).toBe(true);
      expect(result.status).toBe('pending');

      const state = await getCoordinatorState(testStorage);
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(102);
      expect(state.registry[102]?.status).toBe('pending');
    });

    it('should prevent duplicate enqueueing of already queued or extracted messages', async function () {
      const proseMsg = {
        id: 103,
        platform: 'ChatGPT',
        text: 'I learned React for building interactive single page applications.',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending' as const,
      };

      const first = await enqueueKnowledgeMessage(proseMsg, { storage: testStorage });
      expect(first.enqueued).toBe(true);

      const second = await enqueueKnowledgeMessage(proseMsg, { storage: testStorage });
      expect(second.enqueued).toBe(false);
      expect(second.reason).toBe('already_queued_or_extracted');

      const state = await getCoordinatorState(testStorage);
      expect(state.pendingQueue).toHaveLength(1);
    });
  });

  describe('4. Execution Mutex & Prevention of Concurrent Queue Runs', function () {
    it('should return already_running if a queue process is already active in this instance', async function () {
      const dbName = createUniqueDbName('MutexTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Seed 1 message
      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I designed a metrics ingestion pipeline in Go.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      await enqueueKnowledgeMessage(
        (await testDb.writingMessages.get(msgId))!,
        { storage: testStorage }
      );

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      // Keep extractKnowledgeFromMessage pending
      let resolveExtraction!: () => void;
      const extractionPromise = new Promise<never>((resolve) => {
        resolveExtraction = () => resolve({
          messageId: msgId,
          outcome: 'extracted',
          extractedCount: 1,
          duplicateCount: 0,
        } as never);
      });

      vi.spyOn(knowledgeExtractor, 'extractKnowledgeFromMessage').mockReturnValue(extractionPromise as never);

      // Start first process
      const firstRunPromise = processKnowledgeQueue({
        storage: testStorage,
        targetDb: testDb,
      });

      // Try starting second process simultaneously
      const secondRun = await processKnowledgeQueue({
        storage: testStorage,
        targetDb: testDb,
      });

      expect(secondRun.status).toBe('already_running');
      expect(secondRun.processedCount).toBe(0);

      resolveExtraction();
      await firstRunPromise;
    });
  });

  describe('5. Sequential Processing: Facts, Empty Results, and Status Updates', function () {
    it('should process eligible messages sequentially and record outcome extracted with facts', async function () {
      const dbName = createUniqueDbName('SuccessFacts');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Recently I built an offline note-taking app with SQLite.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg = (await testDb.writingMessages.get(msgId))!;
      await enqueueKnowledgeMessage(msg, { storage: testStorage });

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(knowledgeExtractor, 'extractKnowledgeFromMessage').mockResolvedValue({
        messageId: msgId,
        outcome: 'extracted',
        extractedCount: 1,
        duplicateCount: 0,
      });

      const res = await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      expect(res.status).toBe('completed');
      expect(res.processedCount).toBe(1);

      const state = await getCoordinatorState(testStorage);
      expect(state.pendingQueue).toHaveLength(0);
      expect(state.inFlight).toBeNull();
      expect(state.registry[msgId]?.status).toBe('extracted');
      expect(state.registry[msgId]?.extractedCount).toBe(1);
    });

    it('should record empty evaluation results as extracted without error', async function () {
      const dbName = createUniqueDbName('SuccessEmpty');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'In my opinion software architecture choices involve many trade-offs.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg = (await testDb.writingMessages.get(msgId))!;
      await enqueueKnowledgeMessage(msg, { storage: testStorage });

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(knowledgeExtractor, 'extractKnowledgeFromMessage').mockResolvedValue({
        messageId: msgId,
        outcome: 'empty',
        extractedCount: 0,
        duplicateCount: 0,
      });

      const res = await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      expect(res.status).toBe('completed');
      expect(res.processedCount).toBe(1);

      const state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('extracted');
      expect(state.registry[msgId]?.extractedCount).toBe(0);
    });
  });

  describe('6. Error Classification: Outages vs Actionable Blocks vs Message Failures', function () {
    it('should classify temporary service outages, preserve pending message without incrementing attempts, and schedule alarm', async function () {
      const dbName = createUniqueDbName('OutageTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am learning Kotlin for Android development.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg = (await testDb.writingMessages.get(msgId))!;
      await enqueueKnowledgeMessage(msg, { storage: testStorage });

      // Ollama is offline
      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: false,
        isModelReady: false,
        modelName: 'qwen3.5:2b',
        error: 'Ollama is offline. Start Ollama.',
      });

      const res = await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      expect(res.status).toBe('service_offline');

      const state = await getCoordinatorState(testStorage);
      expect(state.serviceStatus).toBe('offline');
      expect(state.pendingQueue).toHaveLength(1);
      // Attempt count must NOT be incremented for service outage
      expect(state.pendingQueue[0]?.attempts).toBe(0);
      expect(state.registry[msgId]?.attempts).toBe(0);
      expect(state.registry[msgId]?.status).toBe('pending');

      // Alarm must be scheduled
      expect(mockAlarms.create).toHaveBeenCalledWith(
        KNOWLEDGE_RETRY_ALARM_NAME,
        expect.objectContaining({ delayInMinutes: 1, periodInMinutes: 1 })
      );
    });

    it('should classify actionable configuration errors (403 origin / 404 model), halt queue, clear alarm, and preserve attempts', async function () {
      const dbName = createUniqueDbName('ConfigBlocked');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am building an authentication microservice in Node.js.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg = (await testDb.writingMessages.get(msgId))!;
      await enqueueKnowledgeMessage(msg, { storage: testStorage });

      // Ollama running but model missing (404/not ready)
      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: false,
        modelName: 'qwen3.5:2b',
        error: 'Model "qwen3.5:2b" is not installed in Ollama. Run: ollama pull qwen3.5:2b',
      });

      const res = await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      expect(res.status).toBe('service_blocked');

      const state = await getCoordinatorState(testStorage);
      expect(state.serviceStatus).toBe('blocked_configuration');
      expect(state.serviceError).toContain('Model "qwen3.5:2b" is not installed');
      expect(state.pendingQueue[0]?.attempts).toBe(0);

      // Must NOT schedule endless retry alarm
      expect(mockAlarms.clear).toHaveBeenCalledWith(KNOWLEDGE_RETRY_ALARM_NAME);
    });

    it('should retry message-level failure up to 3 times before transitioning to failed_permanent', async function () {
      const dbName = createUniqueDbName('RetryPolicyTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built an analytics service in Python.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg = (await testDb.writingMessages.get(msgId))!;
      await enqueueKnowledgeMessage(msg, { storage: testStorage });

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      // Mock message-level inference failure (e.g. JSON parse failure)
      vi.spyOn(knowledgeExtractor, 'extractKnowledgeFromMessage').mockRejectedValue(
        new Error('Failed to parse Ollama response as valid JSON.')
      );

      // First execution: attempt 1 -> failed_temporary
      await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      let state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('failed_temporary');
      expect(state.registry[msgId]?.attempts).toBe(1);
      expect(state.pendingQueue).toHaveLength(1);

      // Second execution: attempt 2 -> failed_temporary
      await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('failed_temporary');
      expect(state.registry[msgId]?.attempts).toBe(2);
      expect(state.pendingQueue).toHaveLength(1);

      // Third execution: attempt 3 -> failed_permanent dead-letter state
      await processKnowledgeQueue({ storage: testStorage, targetDb: testDb });
      state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('failed_permanent');
      expect(state.registry[msgId]?.attempts).toBe(3);
      expect(state.registry[msgId]?.lastError).toContain('Failed to parse Ollama response');
      // Must be removed from pendingQueue
      expect(state.pendingQueue).toHaveLength(0);
    });
  });

  describe('7. Service Worker Restart & In-Flight Recovery', function () {
    it('should requeue in-flight message if worker restarted mid-request and evidence does not exist', async function () {
      const dbName = createUniqueDbName('RecoveryRequeue');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am creating a desktop utility in Electron.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Simulate crash: state has inFlight recorded
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: {
            messageId: msgId,
            startedAt: new Date().toISOString(),
            attempt: 1,
          },
          registry: {
            [msgId]: {
              messageId: msgId,
              status: 'pending',
              attempts: 1,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      const recovery = await recoverKnowledgeCoordinator({
        storage: testStorage,
        targetDb: testDb,
      });

      expect(recovery.recovered).toBe(true);
      expect(recovery.action).toBe('requeued');
      expect(recovery.messageId).toBe(msgId);

      const state = await getCoordinatorState(testStorage);
      expect(state.inFlight).toBeNull();
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(msgId);
      expect(state.registry[msgId]?.status).toBe('pending');
    });

    it('should discard in-flight record if source message was deleted while worker was stopped', async function () {
      const dbName = createUniqueDbName('RecoveryDeleted');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Message 999 does not exist in testDb.writingMessages
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: {
            messageId: 999,
            startedAt: new Date().toISOString(),
            attempt: 1,
          },
          registry: {
            [999]: {
              messageId: 999,
              status: 'pending',
              attempts: 1,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      const recovery = await recoverKnowledgeCoordinator({
        storage: testStorage,
        targetDb: testDb,
      });

      expect(recovery.recovered).toBe(true);
      expect(recovery.action).toBe('discarded_deleted');

      const state = await getCoordinatorState(testStorage);
      expect(state.inFlight).toBeNull();
      expect(state.pendingQueue).toHaveLength(0);
      expect(state.registry[999]).toBeUndefined();
    });

    it('should also requeue in-flight message with pre-existing evidence instead of incorrectly marking extracted', async function () {
      const dbName = createUniqueDbName('RecoveryEvidenceRequeue');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built a video transcoder using FFmpeg and C.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Pre-existing evidence exists in IndexedDB from an earlier run
      await testDb.knowledgeEvidence.add({
        knowledgeId: 1,
        messageId: msgId,
        sourceExcerpt: 'built a video transcoder',
        isValid: true,
        extractedAt: new Date(),
      });

      // Worker died while re-extracting this message (e.g. manual retry on attempt 2)
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: {
            messageId: msgId,
            startedAt: new Date().toISOString(),
            attempt: 2,
          },
          registry: {
            [msgId]: {
              messageId: msgId,
              status: 'pending',
              attempts: 2,
              enqueuedAt: '2026-11-12T00:00:00.000Z',
              updatedAt: '2026-11-12T00:00:00.000Z',
            },
          },
          serviceStatus: 'online',
        },
      });

      const recovery = await recoverKnowledgeCoordinator({
        storage: testStorage,
        targetDb: testDb,
      });

      // Must be requeued at front of queue, NOT marked extracted from evidence
      expect(recovery.recovered).toBe(true);
      expect(recovery.action).toBe('requeued');
      expect(recovery.messageId).toBe(msgId);

      const state = await getCoordinatorState(testStorage);
      expect(state.inFlight).toBeNull();
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(msgId);
      // Attempt history preserved: attempt 2 in flight resets to previous attempt count (1)
      expect(state.pendingQueue[0]?.attempts).toBe(1);
      expect(state.registry[msgId]?.status).toBe('pending');
      expect(state.registry[msgId]?.attempts).toBe(1);
      expect(state.registry[msgId]?.enqueuedAt).toBe('2026-11-12T00:00:00.000Z');
      expect(state.registry[msgId]?.status).not.toBe('extracted');
    });
  });

  describe('8. Manual Recovery Operations', function () {
    it('should reset failed_permanent messages to pending with 0 attempts on retryFailedKnowledgeMessages', async function () {
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: null,
          registry: {
            [201]: {
              messageId: 201,
              status: 'failed_permanent',
              attempts: 3,
              lastError: 'Inference timeout',
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
            [202]: {
              messageId: 202,
              status: 'extracted',
              attempts: 1,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      const count = await retryFailedKnowledgeMessages(undefined, { storage: testStorage });
      expect(count).toBe(1);

      const state = await getCoordinatorState(testStorage);
      expect(state.registry[201]?.status).toBe('pending');
      expect(state.registry[201]?.attempts).toBe(0);
      expect(state.registry[201]?.lastError).toBeUndefined();
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(201);
      // Other messages remain untouched
      expect(state.registry[202]?.status).toBe('extracted');
    });

    it('should re-evaluate eligibility on reconsiderSkippedKnowledgeMessages without blindly enqueueing', async function () {
      const dbName = createUniqueDbName('ReconsiderSkipped');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Msg 301 is now eligible prose (e.g. user updated or prompt had knowledge)
      const msg1Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Recently I built an open-source Markdown editor in Svelte.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Msg 302 is still pure code (ineligible)
      const msg2Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: '```bash\nnpm install express\n```',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: null,
          registry: {
            [msg1Id]: {
              messageId: msg1Id,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
            [msg2Id]: {
              messageId: msg2Id,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      const enqueuedCount = await reconsiderSkippedKnowledgeMessages(undefined, {
        storage: testStorage,
        targetDb: testDb,
      });

      // Only msg1 should be enqueued; msg2 remains skipped!
      expect(enqueuedCount).toBe(1);

      const state = await getCoordinatorState(testStorage);
      expect(state.registry[msg1Id]?.status).toBe('pending');
      expect(state.registry[msg2Id]?.status).toBe('skipped');
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(msg1Id);
    });
  });

  describe('9. Coordinator Status Reporting & Alarm Cleanup', function () {
    it('should report comprehensive coordinator status metrics', async function () {
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [{ messageId: 401, enqueuedAt: '...', attempts: 0 }],
          inFlight: null,
          registry: {
            [401]: { messageId: 401, status: 'pending', attempts: 0, enqueuedAt: '...', updatedAt: '...' },
            [402]: { messageId: 402, status: 'extracted', attempts: 1, enqueuedAt: '...', updatedAt: '...' },
            [403]: { messageId: 403, status: 'skipped', attempts: 0, enqueuedAt: '...', updatedAt: '...' },
            [404]: { messageId: 404, status: 'failed_temporary', attempts: 1, enqueuedAt: '...', updatedAt: '...' },
            [405]: { messageId: 405, status: 'failed_permanent', attempts: 3, enqueuedAt: '...', updatedAt: '...' },
          },
          serviceStatus: 'online',
        },
      });

      const status = await getKnowledgeCoordinatorStatus(testStorage);
      expect(status.pendingCount).toBe(1);
      expect(status.extractedCount).toBe(1);
      expect(status.skippedCount).toBe(1);
      expect(status.failedTemporaryCount).toBe(1);
      expect(status.failedPermanentCount).toBe(1);
      expect(status.serviceStatus).toBe('online');
      expect(status.isProcessing).toBe(false);
    });

    it('should clear retry alarm when queue is completely empty', async function () {
      await scheduleKnowledgeRetryAlarm();
      expect(mockAlarms.create).toHaveBeenCalled();

      await clearKnowledgeRetryAlarm();
      expect(mockAlarms.clear).toHaveBeenCalledWith(KNOWLEDGE_RETRY_ALARM_NAME);
    });
  });

  describe('10. Batch Synchronization and Existing State Preservation', function () {
    it('should preserve existing extracted, skipped, failed_permanent, and pending records during batch synchronization', async function () {
      const fixedEnqueuedAt = '2026-11-01T10:00:00.000Z';
      const fixedUpdatedAt = '2026-11-01T10:05:00.000Z';

      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [{ messageId: 104, enqueuedAt: fixedEnqueuedAt, attempts: 0 }],
          inFlight: null,
          registry: {
            [101]: {
              messageId: 101,
              status: 'extracted',
              attempts: 1,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
            [102]: {
              messageId: 102,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
            [103]: {
              messageId: 103,
              status: 'failed_permanent',
              attempts: 3,
              lastError: 'Simulated LLM failure',
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
            [104]: {
              messageId: 104,
              status: 'pending',
              attempts: 0,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
          },
          serviceStatus: 'online',
        },
      });

      const batchMessages = [
        { id: 101, text: 'This text is now eligible prose but message is already extracted.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 102, text: 'This text is also eligible prose but message is already skipped.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 103, text: 'This text is eligible prose but message failed permanently.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 104, text: 'This message is already pending in queue.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
      ];

      const syncResult = await syncUntrackedKnowledgeMessages(batchMessages, { storage: testStorage });

      expect(syncResult.totalChecked).toBe(4);
      expect(syncResult.alreadyTrackedCount).toBe(4);
      expect(syncResult.newlyQueuedCount).toBe(0);
      expect(syncResult.newlySkippedCount).toBe(0);

      const state = await getCoordinatorState(testStorage);

      // 101 extracted preserved
      expect(state.registry[101]?.status).toBe('extracted');
      expect(state.registry[101]?.attempts).toBe(1);
      expect(state.registry[101]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[101]?.updatedAt).toBe(fixedUpdatedAt);

      // 102 skipped preserved
      expect(state.registry[102]?.status).toBe('skipped');
      expect(state.registry[102]?.attempts).toBe(0);
      expect(state.registry[102]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[102]?.updatedAt).toBe(fixedUpdatedAt);

      // 103 failed_permanent preserved
      expect(state.registry[103]?.status).toBe('failed_permanent');
      expect(state.registry[103]?.attempts).toBe(3);
      expect(state.registry[103]?.lastError).toBe('Simulated LLM failure');
      expect(state.registry[103]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[103]?.updatedAt).toBe(fixedUpdatedAt);

      // 104 pending preserved without duplicate
      expect(state.registry[104]?.status).toBe('pending');
      expect(state.registry[104]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[104]?.updatedAt).toBe(fixedUpdatedAt);
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(104);
    });

    it('should preserve existing registry records during ordinary enqueueKnowledgeMessage', async function () {
      const fixedEnqueuedAt = '2026-11-01T10:00:00.000Z';
      const fixedUpdatedAt = '2026-11-01T10:05:00.000Z';

      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: null,
          registry: {
            [201]: {
              messageId: 201,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
          },
          serviceStatus: 'online',
        },
      });

      // Even with eligible prose text, ordinary enqueue must NOT overwrite skipped record
      const res = await enqueueKnowledgeMessage(
        {
          id: 201,
          text: 'I specialize in distributed databases and stream processing architecture.',
          platform: 'ChatGPT',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        { storage: testStorage }
      );

      expect(res.enqueued).toBe(false);
      expect(res.status).toBe('skipped');
      expect(res.reason).toBe('already_queued_or_extracted');

      const state = await getCoordinatorState(testStorage);
      expect(state.registry[201]?.status).toBe('skipped');
      expect(state.registry[201]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[201]?.updatedAt).toBe(fixedUpdatedAt);
      expect(state.pendingQueue).toHaveLength(0);
    });

    it('should require explicit reconsiderSkippedKnowledgeMessages to move skipped messages back to pending', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('ExplicitReconsider')));
      await testDb.open();

      const fixedEnqueuedAt = '2026-11-01T10:00:00.000Z';
      const fixedUpdatedAt = '2026-11-01T10:05:00.000Z';

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built and developed distributed task scheduling engines using TypeScript.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: null,
          registry: {
            [msgId]: {
              messageId: msgId,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
          },
          serviceStatus: 'online',
        },
      });

      // 1. Batch sync cannot reconsider skipped message
      const dbMsg = (await testDb.writingMessages.get(msgId))!;
      const syncResult = await syncUntrackedKnowledgeMessages([dbMsg], {
        storage: testStorage,
        targetDb: testDb,
      });
      expect(syncResult.alreadyTrackedCount).toBe(1);
      expect(syncResult.newlyQueuedCount).toBe(0);

      let state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('skipped');
      expect(state.pendingQueue).toHaveLength(0);

      // 2. Explicit reconsideration workflow DOES reconsider it
      const reconsideredCount = await reconsiderSkippedKnowledgeMessages([msgId], {
        storage: testStorage,
        targetDb: testDb,
      });
      expect(reconsideredCount).toBe(1);

      state = await getCoordinatorState(testStorage);
      expect(state.registry[msgId]?.status).toBe('pending');
      expect(state.registry[msgId]?.attempts).toBe(0);
      expect(state.pendingQueue).toHaveLength(1);
      expect(state.pendingQueue[0]?.messageId).toBe(msgId);
    });

    it('should classify and batch-sync untracked messages into pending and skipped in a single mutation', async function () {
      let storageSetCalls = 0;
      const trackingStorage: StorageLike = {
        async get(key: string) {
          return { [key]: mockStorageStore[key] };
        },
        async set(items: Record<string, unknown>) {
          storageSetCalls++;
          Object.assign(mockStorageStore, items);
        },
        async remove(key: string) {
          delete mockStorageStore[key];
        },
      };

      const messages = [
        { id: 301, text: 'I built and deployed full-stack web applications using React and Node.js.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 302, text: 'const a = 123; console.log(a);', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 303, text: 'I designed and implemented distributed caching engines with Raft consensus.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: 301, text: 'Duplicate 301 in same batch', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
        { id: NaN, text: 'Invalid id message', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const },
      ];

      const result = await syncUntrackedKnowledgeMessages(messages, { storage: trackingStorage });

      expect(result.totalChecked).toBe(5);
      expect(result.newlyQueuedCount).toBe(2); // 301, 303
      expect(result.newlySkippedCount).toBe(1); // 302 (code)
      expect(result.alreadyTrackedCount).toBe(1); // duplicate 301
      expect(result.invalidCount).toBe(1); // NaN

      // Crucial: exactly ONE storage set call was performed for the entire batch
      expect(storageSetCalls).toBe(1);

      const state = await getCoordinatorState(trackingStorage);
      expect(state.registry[301]?.status).toBe('pending');
      expect(state.registry[302]?.status).toBe('skipped');
      expect(state.registry[303]?.status).toBe('pending');
      expect(state.pendingQueue).toHaveLength(2);
      expect(state.pendingQueue.map((q) => q.messageId)).toEqual([301, 303]);
    });

    it('should handle empty, null, or invalid message batches gracefully', async function () {
      const emptyRes = await syncUntrackedKnowledgeMessages([], { storage: testStorage });
      expect(emptyRes.totalChecked).toBe(0);
      expect(emptyRes.newlyQueuedCount).toBe(0);

      const nullRes = await syncUntrackedKnowledgeMessages(null as any, { storage: testStorage });
      expect(nullRes.totalChecked).toBe(0);
      expect(nullRes.newlyQueuedCount).toBe(0);
    });

    it('should serialize concurrent enqueueKnowledgeMessage and syncUntrackedKnowledgeMessages safely', async function () {
      const msg1 = { id: 401, text: 'I created and designed distributed event brokers using TypeScript.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const };
      const msg2 = { id: 402, text: 'I built and deployed real-time data streaming pipelines in Python.', platform: 'ChatGPT', createdAt: new Date(), capturedAt: new Date(), trainingStatus: 'pending' as const };

      // Dispatch both concurrently
      const [res1, res2] = await Promise.all([
        enqueueKnowledgeMessage(msg1, { storage: testStorage }),
        syncUntrackedKnowledgeMessages([msg2], { storage: testStorage }),
      ]);

      expect(res1.enqueued).toBe(true);
      expect(res2.newlyQueuedCount).toBe(1);

      const state = await getCoordinatorState(testStorage);
      expect(state.registry[401]?.status).toBe('pending');
      expect(state.registry[402]?.status).toBe('pending');
      expect(state.pendingQueue).toHaveLength(2);
    });
  });
});

