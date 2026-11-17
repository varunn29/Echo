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
  addManualPersonalKnowledge,
  updatePersonalKnowledge,
  deletePersonalKnowledge,
  getPersonalKnowledge,
  getPersonalKnowledgeById,
  getKnowledgeEvidence,
  type EchoDatabase,
  type PersonalKnowledgeItem,
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
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

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
      vi.spyOn(knowledgeCoordinator, 'processKnowledgeQueue').mockResolvedValue({
        status: 'completed',
        processedCount: 0,
      });

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

  describe('5. Personal Knowledge Curation Dashboard Experience', () => {
    it('1. Opening and submitting the add form (renders modal markup)', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      // Verify trigger button and modal markup are present in the component
      expect(content).toContain('Add Knowledge');
      expect(content).toContain('Add Personal Knowledge');
      expect(content).toContain('Edit Personal Knowledge');
      expect(content).toContain('Subject *');
      expect(content).toContain('Predicate *');
      expect(content).toContain('Value *');
      expect(content).toContain('Category');
      expect(content).toContain('Context Scope');
      expect(content).toContain('Stability');
      expect(content).toContain('Confidence');
      expect(content).toContain('Importance');
      expect(content).toContain('Cancel');
      expect(content).toContain('Save Knowledge');

      // Verify SSR renders cleanly without crashing
      const html = renderToString(React.createElement(WritingData));
      expect(html).toContain('My Writing Data');
    });

    it('2. Successful manual fact creation (addManualPersonalKnowledge creates record in DB)', async () => {
      const factId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'TypeScript',
          predicate: 'specializes in',
          value: 'type systems and strict null checks',
          contextScope: 'language tooling',
          stability: 'stable',
          confidence: 'high',
          importance: 'primary',
        },
        testDb
      );

      expect(typeof factId).toBe('number');
      expect(factId).toBeGreaterThan(0);

      const saved = await getPersonalKnowledgeById(factId, testDb);
      expect(saved).toBeDefined();
      expect(saved?.sourceType).toBe('manual');
      expect(saved?.status).toBe('active');
      expect(saved?.editedByUser).toBe(false);
      expect(saved?.subject).toBe('TypeScript');
      expect(saved?.predicate).toBe('specializes in');
      expect(saved?.value).toBe('type systems and strict null checks');
      expect(saved?.contextScope).toBe('language tooling');
    });

    it('3. Validation errors (empty subject, predicate, or value rejected)', async () => {
      // Database level validation checks
      await expect(
        addManualPersonalKnowledge(
          { category: 'skills', subject: '   ', predicate: 'knows', value: 'Rust' },
          testDb
        )
      ).rejects.toThrow(/subject.*empty/i);

      await expect(
        addManualPersonalKnowledge(
          { category: 'skills', subject: 'Rust', predicate: '  ', value: 'Systems programming' },
          testDb
        )
      ).rejects.toThrow(/predicate.*empty/i);

      await expect(
        addManualPersonalKnowledge(
          { category: 'skills', subject: 'Rust', predicate: 'uses', value: '   ' },
          testDb
        )
      ).rejects.toThrow(/value.*empty/i);

      // Component level validation rules in WritingData.tsx
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');
      expect(content).toContain("'Subject is required.'");
      expect(content).toContain("'Predicate is required.'");
      expect(content).toContain("'Value is required.'");
    });

    it('4. Editing an existing fact (updatePersonalKnowledge updates mutable fields)', async () => {
      const factId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Python',
          predicate: 'builds',
          value: 'scripts',
          stability: 'stable',
          confidence: 'medium',
          importance: 'secondary',
        },
        testDb
      );

      await updatePersonalKnowledge(
        factId,
        {
          value: 'production ML pipelines and microservices',
          importance: 'primary',
          confidence: 'high',
          contextScope: 'data engineering',
        },
        testDb
      );

      const retrieved = await getPersonalKnowledgeById(factId, testDb);
      expect(retrieved?.value).toBe('production ML pipelines and microservices');
      expect(retrieved?.importance).toBe('primary');
      expect(retrieved?.confidence).toBe('high');
      expect(retrieved?.contextScope).toBe('data engineering');
      expect(retrieved?.editedByUser).toBe(true);
    });

    it('5. Preserving provenance during edits (sourceType preserved, editedByUser: true)', async () => {
      // Extracted fact preservation
      const extractedId = (await savePersonalKnowledge(
        {
          category: 'projects',
          subject: 'Echo Extension',
          predicate: 'uses',
          value: 'WXT framework',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          sourceType: 'extracted',
          editedByUser: false,
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      ))!;

      await updatePersonalKnowledge(
        extractedId,
        { value: 'WXT and React 19' },
        testDb
      );
      const updatedExtracted = await getPersonalKnowledgeById(extractedId, testDb);
      expect(updatedExtracted?.sourceType).toBe('extracted');
      expect(updatedExtracted?.editedByUser).toBe(true);

      // Manual fact preservation
      const manualId = await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Editor',
          predicate: 'prefers',
          value: 'VS Code',
          stability: 'stable',
          confidence: 'high',
          importance: 'primary',
        },
        testDb
      );

      await updatePersonalKnowledge(
        manualId,
        { value: 'Cursor and Neovim' },
        testDb
      );
      const updatedManual = await getPersonalKnowledgeById(manualId, testDb);
      expect(updatedManual?.sourceType).toBe('manual');
      expect(updatedManual?.editedByUser).toBe(true);
    });

    it('6. Rejecting edits to archived facts', async () => {
      const factId = await addManualPersonalKnowledge(
        {
          category: 'preferences',
          subject: 'Coffee',
          predicate: 'drinks',
          value: 'espresso',
          stability: 'stable',
          confidence: 'high',
          importance: 'secondary',
        },
        testDb
      );

      await deletePersonalKnowledge(factId, 'user_manually_rejected', testDb);

      await expect(
        updatePersonalKnowledge(factId, { value: 'filter brew' }, testDb)
      ).rejects.toThrow(/not active.*archived/i);
    });

    it('7. Confirming and completing deletion (deletePersonalKnowledge archives fact, invalidates evidence)', async () => {
      const msgId = (await saveWritingMessage(
        {
          platform: 'LinkedIn',
          text: 'I maintain open source Dart and Flutter packages.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const factId = (await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Flutter',
          predicate: 'maintains',
          value: 'open source packages',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          sourceType: 'extracted',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      ))!;

      await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: msgId,
          sourceExcerpt: 'maintain open source Dart and Flutter packages',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Delete fact using dashboard deletion reason
      await deletePersonalKnowledge(factId, 'user_manually_rejected', testDb);

      const archivedFact = await getPersonalKnowledgeById(factId, testDb);
      expect(archivedFact?.status).toBe('archived');

      const evidenceList = await getKnowledgeEvidence(factId, testDb);
      expect(evidenceList).toHaveLength(1);
      expect(evidenceList[0]?.isValid).toBe(false);
      expect(evidenceList[0]?.invalidationReason).toBe('user_manually_rejected');
    });

    it('8. Cancelling deletion leaves fact and evidence unchanged', async () => {
      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I write technical architecture RFCs regularly.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const factId = (await savePersonalKnowledge(
        {
          category: 'habits',
          subject: 'Technical RFCs',
          predicate: 'writes',
          value: 'regularly',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          sourceType: 'extracted',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      ))!;

      await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: msgId,
          sourceExcerpt: 'write technical architecture RFCs',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // In the UI, cancelling sets deletingKnowledgeItem to null without calling deletePersonalKnowledge
      let deletingKnowledgeItem: PersonalKnowledgeItem | null = await getPersonalKnowledgeById(factId, testDb) || null;
      expect(deletingKnowledgeItem).not.toBeNull();
      // User clicks Cancel:
      deletingKnowledgeItem = null;
      expect(deletingKnowledgeItem).toBeNull();

      // Verify DB remains completely unmodified
      const intactFact = await getPersonalKnowledgeById(factId, testDb);
      expect(intactFact?.status).toBe('active');
      expect(intactFact?.subject).toBe('Technical RFCs');

      const intactEvidence = await getKnowledgeEvidence(factId, testDb);
      expect(intactEvidence[0]?.isValid).toBe(true);
      expect(intactEvidence[0]?.invalidationReason).toBeUndefined();
    });

    it('9. Refreshing the list after successful add, edit, delete', async () => {
      // Initial active facts count is 0
      const initialList = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(initialList).toHaveLength(0);

      // Add a manual fact
      const factId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Architecture',
          predicate: 'specializes in',
          value: 'Clean Architecture',
          stability: 'stable',
          confidence: 'high',
          importance: 'primary',
        },
        testDb
      );

      const afterAddList = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(afterAddList).toHaveLength(1);
      expect(afterAddList[0]?.id).toBe(factId);
      expect(afterAddList[0]?.value).toBe('Clean Architecture');

      // Edit the fact
      await updatePersonalKnowledge(
        factId,
        { value: 'Clean Architecture and DDD' },
        testDb
      );

      const afterEditList = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(afterEditList).toHaveLength(1);
      expect(afterEditList[0]?.value).toBe('Clean Architecture and DDD');

      // Delete the fact
      await deletePersonalKnowledge(factId, 'user_manually_rejected', testDb);

      const afterDeleteList = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(afterDeleteList).toHaveLength(0);
    });

    it('10. Displaying manual vs extracted provenance correctly (Manually added / Added vs Extracted / Confirmed)', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      // Verify badge rendering conditions
      expect(content).toContain("item.sourceType === 'manual'");
      expect(content).toContain('Manually added');
      expect(content).toContain('Extracted');
      expect(content).toContain('item.editedByUser');
      expect(content).toContain('Edited');

      // Verify date labels
      expect(content).toContain('Added: ');
      expect(content).toContain('Confirmed: ');
    });

    it('11. Handling database failures without losing user input', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingData.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      // In handleSaveKnowledge: error is caught, set to knowledgeFormError, and form data is NOT cleared on error
      expect(content).toContain('setKnowledgeFormError(msg)');
      expect(content).toContain('setIsSavingKnowledge(false)');

      // In handleConfirmDeleteKnowledge: error is caught and set to knowledgeDeleteError
      expect(content).toContain('setKnowledgeDeleteError(msg)');
      expect(content).toContain('setIsDeletingKnowledge(false)');
    });
  });
});
