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
import * as offscreenManager from './offscreenManager';
import * as knowledgeExtractor from './knowledgeExtractor';
import * as ollamaClient from './ollamaClient';
import {
  COORDINATOR_STORAGE_KEY,
  getCoordinatorState,
  KNOWLEDGE_RETRY_ALARM_NAME,
  type StorageLike,
} from './knowledgeCoordinator';
import { RETRY_ALARM_NAME } from './trainingTrigger';
import defaultBackgroundExport, {
  handleBackgroundAlarm,
  handleRuntimeMessage,
  initializeBackgroundServices,
  resetBackgroundStateForTesting,
  setIsRecoverySucceededForTesting,
  getIsRecoverySucceededForTesting,
  getStartupPromiseForTesting,
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
    const pendingStartup = getStartupPromiseForTesting();
    if (pendingStartup) {
      await Promise.race([
        pendingStartup.catch(() => {}),
        new Promise((resolve) => setTimeout(resolve, 500)),
      ]);
    }
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

      await handleBackgroundAlarm(
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
    beforeEach(() => {
      setIsRecoverySucceededForTesting(true);
    });

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
    beforeEach(() => {
      setIsRecoverySucceededForTesting(true);
    });

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
    it('15. should register alarm and message listeners when runBackgroundEntrypoint runs', async () => {
      runBackgroundEntrypoint();

      expect(mockAlarms.onAlarm.addListener).toHaveBeenCalled();
      expect(mockRuntime.onMessage.addListener).toHaveBeenCalled();

      const pending = getStartupPromiseForTesting();
      if (pending) {
        await pending.catch(() => {});
      }
    });
  });

  describe('12. Startup Initialization Barrier & Concurrency Safety', () => {
    it('16. should prevent alarm-triggered queue dispatch while startup recovery is pending', async () => {
      let resolveRecovery!: (val: any) => void;
      const recoveryPromise = new Promise((resolve) => {
        resolveRecovery = resolve;
      });
      const recoverySpy = vi
        .spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator')
        .mockReturnValue(recoveryPromise as any);
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      // 1. Start background services initialization (recovery remains pending)
      initializeBackgroundServices({ storage: testStorage, targetDb: testDb });
      expect(recoverySpy).toHaveBeenCalled();

      // 2. Alarm fires while startup recovery is still in flight
      const alarmPromise = handleBackgroundAlarm(
        { name: KNOWLEDGE_RETRY_ALARM_NAME },
        { storage: testStorage, targetDb: testDb }
      );

      // Give event loop time to advance microtasks
      await new Promise((r) => setTimeout(r, 25));

      // processKnowledgeQueue must NOT be called while recovery is pending
      expect(processSpy).not.toHaveBeenCalled();

      // 3. Resolve recovery
      resolveRecovery({ recovered: false, action: 'idle' });
      await alarmPromise;

      // processKnowledgeQueue is now dispatched safely
      expect(processSpy).toHaveBeenCalled();
    });

    it('17. should prevent runtime-message-triggered queue dispatch while startup recovery is pending', async () => {
      let resolveRecovery!: (val: any) => void;
      const recoveryPromise = new Promise((resolve) => {
        resolveRecovery = resolve;
      });
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockReturnValue(recoveryPromise as any);
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      // 1. Start background services initialization
      initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      // 2. Dashboard requests START_KNOWLEDGE_EXTRACTION while recovery is pending
      let response: any = null;
      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 25));
      expect(processSpy).not.toHaveBeenCalled();
      expect(response).toBeNull();

      // 3. Complete recovery
      resolveRecovery({ recovered: false, action: 'idle' });
      await new Promise((r) => setTimeout(r, 40));

      expect(processSpy).toHaveBeenCalled();
      expect(response).toEqual({
        success: true,
        result: { status: 'completed', processedCount: 0 },
      });
    });

    it('18. should ensure recovery completes before NEW_USER_MESSAGE triggers queue dispatch', async () => {
      const executionOrder: string[] = [];

      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockImplementation(async () => {
        await new Promise((r) => setTimeout(r, 20));
        executionOrder.push('startup_recovery');
        return { recovered: false, action: 'idle' };
      });

      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockImplementation(async () => {
        executionOrder.push('queue_dispatch');
        return { status: 'completed', processedCount: 0 };
      });

      // Start initialization and receive new message concurrently
      initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      let captured = false;
      handleRuntimeMessage(
        {
          type: 'NEW_USER_MESSAGE',
          text: 'Message checking barrier execution ordering',
        },
        {},
        (res) => {
          captured = res?.success;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 60));

      expect(captured).toBe(true);
      expect(executionOrder.length).toBeGreaterThanOrEqual(1);
      // Startup recovery must precede any queue dispatch
      expect(executionOrder[0]).toBe('startup_recovery');
      expect(executionOrder).toContain('queue_dispatch');
    });

    it('19. should ensure multiple concurrent events waiting for startup initialization do not cause duplicate initialization', async () => {
      let recoveryCalls = 0;
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockImplementation(async () => {
        recoveryCalls++;
        await new Promise((r) => setTimeout(r, 25));
        return { recovered: false, action: 'idle' };
      });

      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      // Kick off background startup
      initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      // Concurrently fire alarm and runtime messages
      await Promise.all([
        handleBackgroundAlarm(
          { name: KNOWLEDGE_RETRY_ALARM_NAME },
          { storage: testStorage, targetDb: testDb }
        ),
        new Promise((resolve) => {
          handleRuntimeMessage(
            { type: 'START_KNOWLEDGE_EXTRACTION' },
            {},
            resolve,
            { storage: testStorage, targetDb: testDb }
          );
        }),
        new Promise((resolve) => {
          handleRuntimeMessage(
            { type: 'NEW_USER_MESSAGE', text: 'Concurrent event test' },
            {},
            resolve,
            { storage: testStorage, targetDb: testDb }
          );
        }),
      ]);

      await new Promise((r) => setTimeout(r, 50));

      // Recovery must only execute once across all events
      expect(recoveryCalls).toBe(1);
    });

    it('20. should ensure recovery failure leaves recovery incomplete, clears startupPromise, and schedules retry alarm', async () => {
      const alarmSpy = vi.spyOn(mockAlarms, 'create');
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockRejectedValue(
        new Error('Database locked during startup')
      );

      // Attempt startup initialization
      await expect(
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb })
      ).rejects.toThrow('Database locked during startup');

      // Assertions
      expect(getIsRecoverySucceededForTesting()).toBe(false);
      expect(getStartupPromiseForTesting()).toBeNull();
      expect(alarmSpy).toHaveBeenCalledWith(
        KNOWLEDGE_RETRY_ALARM_NAME,
        expect.objectContaining({ periodInMinutes: 1 })
      );
    });

    it('21. should retry recovery on subsequent alarm without service worker restart after initial recovery failure', async () => {
      let recoveryAttempts = 0;
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockImplementation(async () => {
        recoveryAttempts++;
        if (recoveryAttempts === 1) {
          throw new Error('Initial transient failure');
        }
        return { recovered: false, action: 'idle' };
      });
      const processSpy = vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      // 1. Initial startup fails
      await expect(
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb })
      ).rejects.toThrow('Initial transient failure');
      expect(recoveryAttempts).toBe(1);
      expect(getIsRecoverySucceededForTesting()).toBe(false);

      // 2. Alarm fires without service worker restarting
      await handleBackgroundAlarm(
        { name: KNOWLEDGE_RETRY_ALARM_NAME },
        { storage: testStorage, targetDb: testDb }
      );

      // Assertions: recovery retried and succeeded
      expect(recoveryAttempts).toBe(2);
      expect(getIsRecoverySucceededForTesting()).toBe(true);
      expect(processSpy).toHaveBeenCalled();
    });

    it('22. should ensure multiple concurrent events during recovery retry share the same recovery attempt', async () => {
      // Start in unrecovered state
      setIsRecoverySucceededForTesting(false);

      let recoveryCalls = 0;
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockImplementation(async () => {
        recoveryCalls++;
        await new Promise((r) => setTimeout(r, 25));
        return { recovered: false, action: 'idle' };
      });
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      // Two concurrent events trigger retry
      await Promise.all([
        handleBackgroundAlarm(
          { name: KNOWLEDGE_RETRY_ALARM_NAME },
          { storage: testStorage, targetDb: testDb }
        ),
        new Promise((resolve) => {
          handleRuntimeMessage(
            { type: 'START_KNOWLEDGE_EXTRACTION' },
            {},
            resolve,
            { storage: testStorage, targetDb: testDb }
          );
        }),
      ]);

      // Exactly 1 recovery attempt across both concurrent events
      expect(recoveryCalls).toBe(1);
      expect(getIsRecoverySucceededForTesting()).toBe(true);
    });

    it('23. should ensure events arriving during startup queue draining wait until the startup promise settles', async () => {
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockResolvedValue({
        recovered: false,
        action: 'idle',
      });

      let resolveStartupQueue!: () => void;
      const startupQueuePromise = new Promise<void>((resolve) => {
        resolveStartupQueue = resolve;
      });

      let queueCalls = 0;
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockImplementation(async () => {
        queueCalls++;
        if (queueCalls === 1) {
          // Startup queue draining is in progress
          await startupQueuePromise;
          return { status: 'completed', processedCount: 0 };
        }
        return { status: 'completed', processedCount: 1 };
      });

      try {
        // 1. Kick off startup
        const initPromise = initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

        // Give microtasks time to complete recovery and enter processKnowledgeQueue
        await new Promise((r) => setTimeout(r, 15));
        expect(getIsRecoverySucceededForTesting()).toBe(true);
        expect(queueCalls).toBe(1); // Startup queue call in flight

        // 2. Alarm fires while startup queue draining is still in progress
        let alarmFinished = false;
        const alarmPromise = handleBackgroundAlarm(
          { name: KNOWLEDGE_RETRY_ALARM_NAME },
          { storage: testStorage, targetDb: testDb }
        )?.then(() => {
          alarmFinished = true;
        });

        await new Promise((r) => setTimeout(r, 20));
        // Alarm must be waiting for startup queue processing to settle
        expect(queueCalls).toBe(1);
        expect(alarmFinished).toBe(false);

        // 3. Startup queue settles
        resolveStartupQueue();
        await initPromise;
        await alarmPromise;

        // Alarm dispatches safely after startup promise settled
        expect(queueCalls).toBe(2);
        expect(alarmFinished).toBe(true);
      } finally {
        resolveStartupQueue();
      }
    });

    it('24. should process a NEW_USER_MESSAGE enqueued during startup queue completion after startup promise settles', async () => {
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockResolvedValue({
        recovered: false,
        action: 'idle',
      });

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'llama3:latest',
      });

      vi.spyOn(offscreenManager, 'ensureOffscreenDocument').mockResolvedValue();

      let resolveTeardown!: () => void;
      const teardownGate = new Promise<void>((resolve) => {
        resolveTeardown = resolve;
      });

      let closeCalls = 0;
      vi.spyOn(offscreenManager, 'closeOffscreenDocument').mockImplementation(async () => {
        closeCalls++;
        if (closeCalls === 1) {
          // Gate the startup queue's completion teardown
          await teardownGate;
        }
      });

      let savedMessageId: number | null = null;
      let resolveExtraction!: (id: number) => void;
      let rejectExtraction!: (err: unknown) => void;
      const extractionDone = new Promise<number>((resolve, reject) => {
        resolveExtraction = resolve;
        rejectExtraction = reject;
      });

      const processedMessageIds: number[] = [];
      vi.spyOn(knowledgeExtractor, 'extractKnowledgeFromMessage').mockImplementation(async (msg) => {
        try {
          if (msg.id) {
            processedMessageIds.push(msg.id);
            if (savedMessageId !== null && msg.id === savedMessageId) {
              resolveExtraction(msg.id);
            }
          }
          return {
            messageId: msg.id!,
            outcome: 'extracted',
            extractedCount: 1,
            duplicateCount: 0,
          };
        } catch (err) {
          rejectExtraction(err);
          throw err;
        }
      });

      const originalEnqueue = knowledgeCoordinator.enqueueKnowledgeMessage;
      let resolveEnqueue!: () => void;
      let rejectEnqueue!: (err: unknown) => void;
      const enqueueDone = new Promise<void>((resolve, reject) => {
        resolveEnqueue = resolve;
        rejectEnqueue = reject;
      });
      vi.spyOn(knowledgeCoordinator, 'enqueueKnowledgeMessage').mockImplementation(async (...args) => {
        try {
          const res = await originalEnqueue(...args);
          resolveEnqueue();
          return res;
        } catch (err) {
          rejectEnqueue(err);
          throw err;
        }
      });

      try {
        // 1. Start background initialization
        const initPromise = initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

        // Wait until startup queue starts and blocks inside closeOffscreenDocument teardown
        await new Promise((r) => setTimeout(r, 15));
        expect(closeCalls).toBe(1);
        expect(getIsRecoverySucceededForTesting()).toBe(true);

        // 2. While startup teardown is in flight, a NEW_USER_MESSAGE arrives
        handleRuntimeMessage(
          {
            type: 'NEW_USER_MESSAGE',
            text: 'I have been reflecting deeply on our product strategy and architecture over the past several months, and I realize we need a more decoupled approach for long-term scalability and reliability.',
          },
          {},
          (response: any) => {
            savedMessageId = response?.id;
          },
          { storage: testStorage, targetDb: testDb }
        );

        // Wait for enqueue operation to complete with a bounded timeout
        let enqueueTimer!: ReturnType<typeof setTimeout>;
        const enqueueTimeout = new Promise<never>((_, reject) => {
          enqueueTimer = setTimeout(() => {
            reject(new Error('Timed out waiting for message to be enqueued in coordinator'));
          }, 2000);
        });
        try {
          await Promise.race([enqueueDone, enqueueTimeout]);
        } finally {
          clearTimeout(enqueueTimer);
        }

        expect(savedMessageId).not.toBeNull();
        expect(typeof savedMessageId).toBe('number');
        expect(savedMessageId!).toBeGreaterThan(0);

        // Read actual coordinator state from test storage and assert newly persisted message exists in pendingQueue
        const stateDuringTeardown = await getCoordinatorState(testStorage);
        expect(
          stateDuringTeardown.pendingQueue.some((item) => item.messageId === savedMessageId)
        ).toBe(true);

        // While teardown is still blocked, no extraction processing should have run yet
        expect(processedMessageIds).toHaveLength(0);

        // 3. Unblock startup queue teardown so startup promise settles
        resolveTeardown();
        await initPromise;

        // 4. Await extraction completion for the expected message ID with a reasonable timeout
        let extractionTimer!: ReturnType<typeof setTimeout>;
        const extractionTimeout = new Promise<never>((_, reject) => {
          extractionTimer = setTimeout(() => {
            reject(new Error(`Timed out waiting for message ${savedMessageId} to be extracted`));
          }, 2000);
        });
        try {
          await Promise.race([extractionDone, extractionTimeout]);
        } finally {
          clearTimeout(extractionTimer);
        }

        // Verify that the subsequent queue dispatch processed that actual message ID (not hardcoded)
        expect(processedMessageIds).toEqual([savedMessageId]);

        // Verify coordinator state in storage reflects successful extraction
        let finalState = await getCoordinatorState(testStorage);
        if (finalState.registry[savedMessageId!]?.status !== 'extracted') {
          for (let i = 0; i < 20 && finalState.registry[savedMessageId!]?.status !== 'extracted'; i++) {
            await new Promise((r) => setTimeout(r, 5));
            finalState = await getCoordinatorState(testStorage);
          }
        }
        expect(finalState.pendingQueue).toHaveLength(0);
        expect(finalState.registry[savedMessageId!]?.status).toBe('extracted');
      } finally {
        resolveTeardown();
      }
    });

    it('25. should not trigger another recovery attempt on subsequent event if queue processing failed after recovery', async () => {
      let recoveryCount = 0;
      vi.spyOn(knowledgeCoordinator, 'recoverKnowledgeCoordinator').mockImplementation(async () => {
        recoveryCount++;
        return { recovered: false, action: 'idle' };
      });

      // Initial queue processing fails (e.g. Ollama service offline)
      let queueRun = 0;
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockImplementation(async () => {
        queueRun++;
        if (queueRun === 1) {
          throw new Error('Ollama service connection refused');
        }
        return { status: 'completed', processedCount: 0 };
      });

      // 1. Startup initialization completes recovery, queue fails
      await initializeBackgroundServices({ storage: testStorage, targetDb: testDb });

      expect(recoveryCount).toBe(1);
      expect(getIsRecoverySucceededForTesting()).toBe(true);
      expect(getStartupPromiseForTesting()).toBeNull();

      // 2. Subsequent alarm fires
      await handleBackgroundAlarm(
        { name: KNOWLEDGE_RETRY_ALARM_NAME },
        { storage: testStorage, targetDb: testDb }
      );

      // Recovery was NOT called again; queue was dispatched directly
      expect(recoveryCount).toBe(1);
      expect(queueRun).toBe(2);
    });

    it('26. should preserve inFlight state in storage and schedule retry if offscreen teardown fails during recovery', async () => {
      const alarmSpy = vi.spyOn(mockAlarms, 'create');
      vi.spyOn(offscreenManager, 'closeOffscreenDocument').mockRejectedValue(
        new Error('Offscreen renderer unresponsive')
      );

      const msgId = 999;
      const initialInFlight = {
        messageId: msgId,
        startedAt: '2026-11-12T10:00:00.000Z',
        attempt: 1,
      };

      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: initialInFlight,
          registry: {
            [msgId]: {
              messageId: msgId,
              status: 'pending',
              attempts: 1,
              enqueuedAt: '2026-11-12T10:00:00.000Z',
              updatedAt: '2026-11-12T10:00:00.000Z',
            },
          },
          serviceStatus: 'online',
        },
      });

      // Initialize background services (recovery should fail because closeOffscreenDocument throws)
      await expect(
        initializeBackgroundServices({ storage: testStorage, targetDb: testDb })
      ).rejects.toThrow('Offscreen renderer unresponsive');

      // Verify inFlight is STILL present in storage
      const storageState = (await testStorage.get(COORDINATOR_STORAGE_KEY)) as any;
      const state = storageState[COORDINATOR_STORAGE_KEY];
      expect(state.inFlight).toEqual(initialInFlight);
      expect(state.pendingQueue).toHaveLength(0);

      // Verify recovery is not marked succeeded and retry alarm was scheduled
      expect(getIsRecoverySucceededForTesting()).toBe(false);
      expect(getStartupPromiseForTesting()).toBeNull();
      expect(alarmSpy).toHaveBeenCalledWith(
        KNOWLEDGE_RETRY_ALARM_NAME,
        expect.objectContaining({ periodInMinutes: 1 })
      );
    });
  });
});

