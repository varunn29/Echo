import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Dexie from 'dexie';

// Polyfill defineBackground on globalThis BEFORE importing background.ts
(globalThis as any).defineBackground = (fn: () => void) => fn;

import {
  createEchoDatabase,
  type EchoDatabase,
} from './database';
import * as databaseModule from './database';
import * as trainingTrigger from './trainingTrigger';
import * as knowledgeCoordinator from './knowledgeCoordinator';
import {
  COORDINATOR_STORAGE_KEY,
  KNOWLEDGE_RETRY_ALARM_NAME,
  type StorageLike,
} from './knowledgeCoordinator';
import { RETRY_ALARM_NAME } from './trainingTrigger';
import defaultBackgroundExport, {
  handleBackgroundAlarm,
  handleRuntimeMessage,
  initializeBackgroundServices,
  resetBackgroundStateForTesting,
  runBackgroundEntrypoint,
  type SaveMessageResponse,
} from '../background';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Background Service Worker Integration Suite (Phase 3, Step 3B)', () => {
  const openDatabases: (Dexie | EchoDatabase)[] = [];
  const dbNamesToDelete: string[] = [];

  function trackDb<T extends Dexie>(dbInstance: T): T {
    openDatabases.push(dbInstance);
    if (!dbNamesToDelete.includes(dbInstance.name)) {
      dbNamesToDelete.push(dbInstance.name);
    }
    return dbInstance;
  }

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

  let registeredAlarms: Record<string, any> = {};
  const mockAlarms = {
    get: vi.fn(async (name: string) => registeredAlarms[name] || null),
    create: vi.fn((name: string, info: any) => {
      registeredAlarms[name] = info;
    }),
    clear: vi.fn(async (name: string) => {
      const existed = Boolean(registeredAlarms[name]);
      delete registeredAlarms[name];
      return existed;
    }),
    onAlarm: {
      addListener: vi.fn(),
    },
  };

  const mockRuntime = {
    id: 'test-echo-extension-id',
    onMessage: {
      addListener: vi.fn(),
    },
    sendMessage: vi.fn(),
  };

  let testDb: EchoDatabase;

  beforeEach(async () => {
    mockStorageStore = {};
    registeredAlarms = {};
    resetBackgroundStateForTesting();
    knowledgeCoordinator.resetCoordinatorStateForTesting();
    vi.restoreAllMocks();

    (globalThis as any).browser = {
      alarms: mockAlarms,
      runtime: mockRuntime,
      storage: {
        local: testStorage,
      },
    };

    testDb = trackDb(createEchoDatabase(createUniqueDbName('BgIntegrationTest')));
    await testDb.open();
  });

  afterEach(async () => {
    resetBackgroundStateForTesting();
    knowledgeCoordinator.resetCoordinatorStateForTesting();
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

  describe('1. Automatic Enqueueing on Message Capture', () => {
    it('1. should automatically enqueue successfully saved writing messages for knowledge extraction', async () => {
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 1,
      });

      let response: SaveMessageResponse | null = null;
      const handled = handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'I am creating a Rust compiler frontend with LLVM.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled).toBe(true);

      // Wait for database and coordinator promises to resolve
      await new Promise((r) => setTimeout(r, 50));

      expect(response).toEqual({
        success: true,
        status: 'saved',
        id: expect.any(Number),
      });

      const savedId = (response as any).id;
      expect(savedId).toBeGreaterThan(0);

      expect(enqueueSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          id: savedId,
          text: 'I am creating a Rust compiler frontend with LLVM.',
          platform: 'ChatGPT',
        }),
        expect.anything()
      );

      expect(processSpy).toHaveBeenCalled();

      // Verify persistent coordinator state
      const state = await knowledgeCoordinator.getCoordinatorState(testStorage);
      expect(state.registry[savedId]).toBeDefined();
    });
  });

  describe('2. Failed Database Saves', () => {
    it('2. should not enqueue if database save fails with an error', async () => {
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      vi.spyOn(databaseModule, 'saveWritingMessage').mockRejectedValue(
        new Error('IndexedDB disk quota exceeded')
      );

      let response: any = null;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'Valid prose text that cannot be saved.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(response).toEqual({
        success: false,
        status: 'error',
        error: 'IndexedDB disk quota exceeded',
      });

      expect(enqueueSpy).not.toHaveBeenCalled();

      const state = await knowledgeCoordinator.getCoordinatorState(testStorage);
      expect(state.pendingQueue).toHaveLength(0);
    });
  });

  describe('3. Invalid or Missing Message IDs', () => {
    it('3a. should not enqueue if message is a duplicate within deduplication window', async () => {
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 1,
      });

      const messagePayload = {
        type: 'NEW_USER_MESSAGE',
        text: 'Duplicate testing sentence.',
      };

      let firstResponse: any = null;
      handleRuntimeMessage(
        messagePayload,
        {},
        (res) => {
          firstResponse = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));
      expect(firstResponse.status).toBe('saved');
      expect(enqueueSpy).toHaveBeenCalledTimes(1);

      enqueueSpy.mockClear();

      // Send duplicate
      let secondResponse: any = null;
      handleRuntimeMessage(
        messagePayload,
        {},
        (res) => {
          secondResponse = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));
      expect(secondResponse.status).toBe('duplicate');
      expect(enqueueSpy).not.toHaveBeenCalled();
    });

    it('3b. should not enqueue if saveWritingMessage returns non-numeric or missing id', async () => {
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      vi.spyOn(databaseModule, 'saveWritingMessage').mockResolvedValue(NaN as any);

      let response: any = null;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'Some test writing message.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));
      expect(response.status).toBe('saved');
      expect(enqueueSpy).not.toHaveBeenCalled();
    });

    it('3c. should reject empty or whitespace message without saving or enqueueing', async () => {
      const saveSpy = vi.spyOn(databaseModule, 'saveWritingMessage');
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');

      let response: any = null;
      const handled = handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: '    ',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled).toBe(false);
      expect(response).toEqual({
        success: false,
        status: 'invalid',
        error: 'Message text must not be empty.',
      });

      expect(saveSpy).not.toHaveBeenCalled();
      expect(enqueueSpy).not.toHaveBeenCalled();
    });
  });

  describe('4. Independence: Knowledge Enqueue Failures do not Break Message Collection', () => {
    it('4. should successfully save message even if knowledge enqueue rejects', async () => {
      vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage').mockRejectedValue(
        new Error('Storage quota exceeded')
      );

      let response: any = null;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'I built an analytics service in Python.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(response).toEqual({
        success: true,
        status: 'saved',
        id: expect.any(Number),
      });

      // Verify writing message was indeed saved in DB
      const messages = await testDb.writingMessages.toArray();
      expect(messages).toHaveLength(1);
      expect(messages[0]!.text).toBe('I built an analytics service in Python.');
    });
  });

  describe('5. Independence: Knowledge Failures do not Prevent Training Trigger', () => {
    it('5. should continue training trigger check even if knowledge processing fails', async () => {
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockRejectedValue(
        new Error('Ollama service offline')
      );

      const trainingSpy = vi
        .spyOn(trainingTrigger, 'checkAndTriggerTraining')
        .mockResolvedValue({
          action: 'insufficient_samples',
          eligibleCount: 1,
          threshold: 25,
        });

      let response: any = null;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'My engineering philosophy emphasizes clean code.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(response.status).toBe('saved');
      expect(trainingSpy).toHaveBeenCalled();
    });
  });

  describe('6. Independence: Training Trigger Failures do not Prevent Knowledge Enqueueing', () => {
    it('6. should continue knowledge enqueueing even if training trigger rejects', async () => {
      vi.spyOn(trainingTrigger, 'checkAndTriggerTraining').mockRejectedValue(
        new Error('Training service 500 error')
      );

      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 1,
      });

      let response: any = null;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'I developed an open-source tool for metrics collection.',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(response.status).toBe('saved');
      expect(enqueueSpy).toHaveBeenCalled();
      expect(processSpy).toHaveBeenCalled();
    });
  });

  describe('7. Service Worker Startup Recovery', () => {
    it('7. should invoke recoverKnowledgeCoordinator during background startup', async () => {
      const recoverSpy = vi
        .spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator')
        .mockResolvedValue({ recovered: false, action: 'idle' });

      await initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      expect(recoverSpy).toHaveBeenCalled();
    });

    it('8. should ensure startup recovery is idempotent and does not run concurrently', async () => {
      let runCount = 0;
      const recoverSpy = vi
        .spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator')
        .mockImplementation(async () => {
          runCount++;
          await new Promise((r) => setTimeout(r, 20));
          return { recovered: false, action: 'idle' };
        });

      await Promise.all([
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb }),
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb }),
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb }),
      ]);

      expect(runCount).toBe(1);
      expect(recoverSpy).toHaveBeenCalledTimes(1);
    });

    it('9. should resume pending knowledge extraction after recovery completes', async () => {
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockResolvedValue({
        recovered: true,
        action: 'requeued',
        messageId: 99,
      });

      const processSpy = vi
        .spyOn(knowledgeCoordinator, 'processKnowledgeQueue')
        .mockResolvedValue({
          status: 'completed',
          processedCount: 1,
        });

      await initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      expect(processSpy).toHaveBeenCalled();
    });
  });

  describe('8. Alarm Integration & Separation', () => {
    it('10. should invoke processKnowledgeQueue when KNOWLEDGE_RETRY_ALARM fires', async () => {
      const processSpy = vi
        .spyOn(knowledgeCoordinator, 'processKnowledgeQueue')
        .mockResolvedValue({
          status: 'completed',
          processedCount: 1,
        });

      handleBackgroundAlarm(
        { name: KNOWLEDGE_RETRY_ALARM_NAME },
        { storage: testStorage, targetDb: testDb }
      );

      expect(processSpy).toHaveBeenCalled();
    });

    it('11. should ignore unrelated alarms without executing training or knowledge handlers', async () => {
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue');
      const trainSpy = vi.spyOn(trainingTrigger, 'checkAndTriggerTraining');

      handleBackgroundAlarm(
        { name: 'some_third_party_alarm' },
        { storage: testStorage, targetDb: testDb }
      );

      expect(processSpy).not.toHaveBeenCalled();
      expect(trainSpy).not.toHaveBeenCalled();
    });

    it('12. should handle RETRY_ALARM_NAME independently for training without triggering knowledge', async () => {
      const trainSpy = vi
        .spyOn(trainingTrigger, 'checkAndTriggerTraining')
        .mockResolvedValue({
          action: 'insufficient_samples',
          eligibleCount: 5,
          threshold: 25,
        });

      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue');

      handleBackgroundAlarm(
        { name: RETRY_ALARM_NAME },
        { storage: testStorage, targetDb: testDb }
      );

      expect(trainSpy).toHaveBeenCalled();
      expect(processSpy).not.toHaveBeenCalled();
    });
  });

  describe('9. Runtime Message Handling & Validation', () => {
    it('13a. should validate RETRY_FAILED_KNOWLEDGE_MESSAGES payload and reject malformed inputs', async () => {
      let response: any = null;

      // Malformed: messageIds is a string
      const handled1 = handleRuntimeMessage(
        {
          type: 'RETRY_FAILED_KNOWLEDGE_MESSAGES',
          messageIds: 'not-an-array',
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled1).toBe(false);
      expect(response).toEqual({
        success: false,
        error: 'Invalid messageIds: must be an array of positive integers.',
      });

      // Malformed: messageIds contains negative or non-integer
      let response2: any = null;
      handleRuntimeMessage(
        {
          type: 'RETRY_FAILED_KNOWLEDGE_MESSAGES',
          messageIds: [1, -5, 3.14],
        },
        {},
        (res) => {
          response2 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(response2).toEqual({
        success: false,
        error: 'Invalid messageIds: must be an array of positive integers.',
      });

      // Valid: empty/undefined messageIds
      const retrySpy = vi
        .spyOn(knowledgeCoordinator, 'retryFailedKnowledgeMessages')
        .mockResolvedValue(2);
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 2,
      });

      let response3: any = null;
      const handled3 = handleRuntimeMessage(
        {
          type: 'RETRY_FAILED_KNOWLEDGE_MESSAGES',
        },
        {},
        (res) => {
          response3 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled3).toBe(true);
      await new Promise((r) => setTimeout(r, 20));
      expect(response3).toEqual({ success: true, requeuedCount: 2 });
      expect(retrySpy).toHaveBeenCalledWith(undefined, expect.anything());
    });

    it('13b. should validate RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES payload', async () => {
      let response: any = null;

      handleRuntimeMessage(
        {
          type: 'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES',
          messageIds: ['abc' as any],
        },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(response).toEqual({
        success: false,
        error: 'Invalid messageIds: must be an array of positive integers.',
      });

      const reconsiderSpy = vi
        .spyOn(knowledgeCoordinator, 'reconsiderSkippedKnowledgeMessages')
        .mockResolvedValue(1);
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 1,
      });

      let response2: any = null;
      handleRuntimeMessage(
        {
          type: 'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES',
          messageIds: [42],
        },
        {},
        (res) => {
          response2 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 20));
      expect(response2).toEqual({ success: true, requeuedCount: 1 });
      expect(reconsiderSpy).toHaveBeenCalledWith([42], expect.anything());
    });

    it('13c. should return coordinator status on GET_KNOWLEDGE_COORDINATOR_STATUS', async () => {
      vi.spyOn(knowledgeCoordinator, 'getKnowledgeCoordinatorStatus').mockResolvedValue({
        pendingCount: 3,
        inFlightMessageId: null,
        extractedCount: 10,
        skippedCount: 2,
        failedTemporaryCount: 1,
        failedPermanentCount: 0,
        serviceStatus: 'online',
        isProcessing: false,
      });

      let response: any = null;
      const handled = handleRuntimeMessage(
        { type: 'GET_KNOWLEDGE_COORDINATOR_STATUS' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled).toBe(true);
      await new Promise((r) => setTimeout(r, 20));

      expect(response).toEqual({
        success: true,
        status: expect.objectContaining({
          pendingCount: 3,
          extractedCount: 10,
          serviceStatus: 'online',
        }),
      });
    });

    it('13d. should trigger batch synchronization and processing on START_KNOWLEDGE_EXTRACTION', async () => {
      const syncSpy = vi.spyOn(knowledgeCoordinator, 'syncUntrackedKnowledgeMessages');
      const enqueueSpy = vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage');
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 3,
      });

      // Seed 2 messages into testDb
      await databaseModule.saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'First sample writing text.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      );
      await databaseModule.saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Second sample writing text.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      );

      let response: any = null;
      const handled = handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled).toBe(true);
      await new Promise((r) => setTimeout(r, 40));

      expect(response).toEqual({
        success: true,
        result: {
          status: 'completed',
          processedCount: 3,
        },
      });

      // Crucial: calls batch sync ONCE, and does NOT loop calling enqueueKnowledgeMessage individually
      expect(syncSpy).toHaveBeenCalledTimes(1);
      expect(enqueueSpy).not.toHaveBeenCalled();
      expect(processSpy).toHaveBeenCalledTimes(1);
    });

    it('13d-2. should return error response if batch sync fails during START_KNOWLEDGE_EXTRACTION', async () => {
      vi.spyOn(knowledgeCoordinator, 'syncUntrackedKnowledgeMessages').mockRejectedValue(
        new Error('Batch storage quota exceeded')
      );

      let response: any = null;
      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 40));

      expect(response).toEqual({
        success: false,
        error: 'Batch storage quota exceeded',
      });
    });

    it('13e. should return false for unknown message types or null message', async () => {
      let response: any = null;
      const handled1 = handleRuntimeMessage(
        { type: 'UNKNOWN_ACTION_TYPE' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      expect(handled1).toBe(false);
      expect(response).toBeNull();

      const handled2 = handleRuntimeMessage(null, {}, () => {}, {
        storage: testStorage,
        targetDb: testDb,
      });
      expect(handled2).toBe(false);
    });

    it('13f. should handle CHECK_TRAINING_STATUS and TRIGGER_TRAINING_CHECK', async () => {
      vi.spyOn(trainingTrigger, 'pollActiveTrainingJob').mockResolvedValue(null);
      vi.spyOn(trainingTrigger, 'checkAndTriggerTraining').mockResolvedValue({
        action: 'insufficient_samples',
        eligibleCount: 2,
        threshold: 25,
      });

      let response1: any = null;
      handleRuntimeMessage(
        { type: 'CHECK_TRAINING_STATUS' },
        {},
        (res) => {
          response1 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 20));
      expect(response1).toEqual({ success: true, activeJob: null });

      let response2: any = null;
      handleRuntimeMessage(
        { type: 'TRIGGER_TRAINING_CHECK' },
        {},
        (res) => {
          response2 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 20));
      expect(response2).toEqual({
        success: true,
        result: expect.objectContaining({ action: 'insufficient_samples' }),
      });
    });
  });

  describe('10. Concurrency Prevention on Extraction Triggers', () => {
    it('14. should not run concurrent extractions if a trigger arrives while one is already running', async () => {
      // Simulate mutex: second call returns already_running
      let callIndex = 0;
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockImplementation(async () => {
        callIndex++;
        if (callIndex > 1) {
          return { status: 'already_running', processedCount: 0 };
        }
        await new Promise((r) => setTimeout(r, 40));
        return { status: 'completed', processedCount: 1 };
      });

      let res1: any = null;
      let res2: any = null;

      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          res1 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          res2 = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 60));

      expect(res1.result.status).toBe('completed');
      expect(res2.result.status).toBe('already_running');
    });
  });

  describe('11. Service Worker Initialization Hook', () => {
    it('15. should register alarm and message listeners when runBackgroundEntrypoint runs', () => {
      runBackgroundEntrypoint();

      expect(mockAlarms.onAlarm.addListener).toHaveBeenCalled();
      expect(mockRuntime.onMessage.addListener).toHaveBeenCalled();
    });
  });
});

