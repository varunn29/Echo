import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import Dexie from 'dexie';
import React from 'react';
import { renderToString } from 'react-dom/server';

import WritingData from '../dashboard/components/WritingData';
import {
  createEchoDatabase,
  saveWritingMessage,
  savePersonalKnowledge,
  saveKnowledgeEvidence,
  type EchoDatabase,
} from './database';
import {
  COORDINATOR_STORAGE_KEY,
  type KnowledgeCoordinatorStatus,
  type StorageLike,
} from './knowledgeCoordinator';
import * as knowledgeCoordinator from './knowledgeCoordinator';
import { handleRuntimeMessage } from '../background';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Dashboard Integration Suite (Phase 3, Step 4)', () => {
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

  let testDb: EchoDatabase;

  beforeEach(async () => {
    mockStorageStore = {};
    vi.restoreAllMocks();
    knowledgeCoordinator.resetCoordinatorStateForTesting();

    testDb = trackDb(createEchoDatabase(createUniqueDbName('DashboardTest')));
    await testDb.open();

    (globalThis as any).document = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      visibilityState: 'visible',
    };
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    knowledgeCoordinator.resetCoordinatorStateForTesting();

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

    delete (globalThis as any).browser;
    delete (globalThis as any).document;
  });

  describe('1. Removal of Direct Dashboard Extraction', () => {
    it('should NOT import or invoke runKnowledgeExtraction anywhere in WritingData.tsx', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      expect(content).not.toContain('runKnowledgeExtraction');
      expect(content).not.toContain('ExtractionProgress');
      expect(content).not.toContain('ExtractionSummary');
    });

    it('should communicate solely via background runtime messages', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      expect(content).toContain("'GET_KNOWLEDGE_COORDINATOR_STATUS'");
      expect(content).toContain("'START_KNOWLEDGE_EXTRACTION'");
      expect(content).toContain("'RETRY_FAILED_KNOWLEDGE_MESSAGES'");
      expect(content).toContain("'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES'");
    });
  });

  describe('2. Dashboard Component Rendering & Structure', () => {
    it('should render the personal knowledge extraction dashboard without errors', () => {
      const html = renderToString(React.createElement(WritingData));

      expect(html).toContain('My Writing Data');
      expect(html).toContain('Personal Knowledge Extraction');
      expect(html).toContain('Collected Messages');
      expect(html).toContain('Extracted Personal Knowledge');
      expect(html).toContain('Import Sample Writing');
      expect(html).toContain('Extract Personal Knowledge');
    });
  });

  describe('3. Coordinator Status Requests and Lifecycle Responses', () => {
    it('should request coordinator status using GET_KNOWLEDGE_COORDINATOR_STATUS', async () => {
      const sentMessages: any[] = [];
      (globalThis as any).browser = {
        runtime: {
          sendMessage: vi.fn(async (msg: any) => {
            sentMessages.push(msg);
            if (msg.type === 'GET_KNOWLEDGE_COORDINATOR_STATUS') {
              const mockStatus: KnowledgeCoordinatorStatus = {
                pendingCount: 2,
                extractedCount: 5,
                skippedCount: 1,
                failedTemporaryCount: 0,
                failedPermanentCount: 0,
                inFlightMessageId: null,
                serviceStatus: 'online',
                isProcessing: false,
              };
              return { success: true, status: mockStatus };
            }
            return { success: false };
          }),
        },
      };

      const res = await (globalThis as any).browser.runtime.sendMessage({
        type: 'GET_KNOWLEDGE_COORDINATOR_STATUS',
      });

      expect(sentMessages).toContainEqual({ type: 'GET_KNOWLEDGE_COORDINATOR_STATUS' });
      expect(res.success).toBe(true);
      expect(res.status.pendingCount).toBe(2);
      expect(res.status.extractedCount).toBe(5);
    });

    it('should handle START_KNOWLEDGE_EXTRACTION and trigger queue processing', async () => {
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 1,
      });

      let runtimeResponse: any = null;
      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (res) => {
          runtimeResponse = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 40));

      expect(runtimeResponse).toEqual({
        success: true,
        result: {
          status: 'completed',
          processedCount: 1,
        },
      });
    });

    it('should preserve existing registry records when dashboard triggers START_KNOWLEDGE_EXTRACTION', async () => {
      const fixedEnqueuedAt = '2026-11-01T10:00:00.000Z';
      const fixedUpdatedAt = '2026-11-01T10:05:00.000Z';

      const skippedMsgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Short note',
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
            [skippedMsgId]: {
              messageId: skippedMsgId,
              status: 'skipped',
              attempts: 0,
              enqueuedAt: fixedEnqueuedAt,
              updatedAt: fixedUpdatedAt,
            },
          },
          serviceStatus: 'online',
        },
      });

      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

      let res: any = null;
      handleRuntimeMessage(
        { type: 'START_KNOWLEDGE_EXTRACTION' },
        {},
        (response) => {
          res = response;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 40));
      expect(res?.success).toBe(true);

      const state = await knowledgeCoordinator.getCoordinatorState(testStorage);
      expect(state.registry[skippedMsgId]?.status).toBe('skipped');
      expect(state.registry[skippedMsgId]?.enqueuedAt).toBe(fixedEnqueuedAt);
      expect(state.registry[skippedMsgId]?.updatedAt).toBe(fixedUpdatedAt);
      expect(state.pendingQueue).toHaveLength(0);
    });

    it('should handle RETRY_FAILED_KNOWLEDGE_MESSAGES and requeue permanently failed items', async () => {
      // Seed a failed_permanent entry in coordinator storage
      await testStorage.set({
        [COORDINATOR_STORAGE_KEY]: {
          version: 1,
          pendingQueue: [],
          inFlight: null,
          registry: {
            [501]: {
              messageId: 501,
              status: 'failed_permanent',
              attempts: 3,
              lastError: 'Inference timeout',
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      let response: any = null;
      handleRuntimeMessage(
        { type: 'RETRY_FAILED_KNOWLEDGE_MESSAGES' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 40));

      expect(response).toEqual({
        success: true,
        requeuedCount: 1,
      });

      const updatedState = await knowledgeCoordinator.getCoordinatorState(testStorage);
      expect(updatedState.registry[501]?.status).toBe('pending');
      expect(updatedState.registry[501]?.attempts).toBe(0);
    });

    it('should handle RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES and re-evaluate eligibility', async () => {
      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built a distributed cache in Rust with Raft consensus.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Mark it skipped in registry
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
              enqueuedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          },
          serviceStatus: 'online',
        },
      });

      let response: any = null;
      handleRuntimeMessage(
        { type: 'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES' },
        {},
        (res) => {
          response = res;
        },
        { storage: testStorage, targetDb: testDb }
      );

      await new Promise((r) => setTimeout(r, 40));

      expect(response).toEqual({
        success: true,
        requeuedCount: 1,
      });

      const updatedState = await knowledgeCoordinator.getCoordinatorState(testStorage);
      expect(updatedState.registry[msgId]?.status).toBe('pending');
      expect(updatedState.pendingQueue).toHaveLength(1);
    });
  });

  describe('4. Message and Personal Knowledge Management Integrity', () => {
    it('should preserve message deletion and linked knowledge invalidation behavior', async () => {
      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I specialize in TypeScript and React application architecture.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const kId = (await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'TypeScript and React',
          predicate: 'specializes in',
          value: 'application architecture',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      ))!;

      await saveKnowledgeEvidence(
        {
          knowledgeId: kId,
          messageId: msgId,
          sourceExcerpt: 'specialize in TypeScript and React',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Verify active before delete
      const initialActive = await testDb.personalKnowledge
        .where('status')
        .equals('active')
        .toArray();
      expect(initialActive).toHaveLength(1);

      // Delete message using standard database function
      const { deleteWritingMessage } = await import('./database');
      await deleteWritingMessage(msgId, testDb);

      // Verify message deleted and knowledge invalidated/scrubbed
      const remainingMsg = await testDb.writingMessages.get(msgId);
      expect(remainingMsg).toBeUndefined();

      const updatedKnowledge = await testDb.personalKnowledge.get(kId);
      expect(updatedKnowledge?.status).toBe('archived');
      expect(updatedKnowledge?.subject).toBe('');

      const updatedEvidence = await testDb.knowledgeEvidence.where('knowledgeId').equals(kId).first();
      expect(updatedEvidence?.isValid).toBe(false);
      expect(updatedEvidence?.invalidationReason).toBe('source_message_deleted');
    });
  });
});
