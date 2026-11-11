import 'fake-indexeddb/auto';

import { afterEach, describe, expect, it } from 'vitest';
import Dexie from 'dexie';
import {
  createEchoDatabase,
  saveWritingMessage,
  getWritingMessages,
  deleteWritingMessage,
  savePersonalKnowledge,
  getPersonalKnowledge,
  getPersonalKnowledgeById,
  saveKnowledgeEvidence,
  getKnowledgeEvidence,
  getEvidenceByMessageId,
  type EchoDatabase,
  type PersonalKnowledgeItem,
  type KnowledgeEvidence,
  type WritingMessage,
} from './database';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Dexie V1 to V2 Migration and Knowledge Storage Suite', function () {
  const openDatabases: (Dexie | EchoDatabase)[] = [];
  const dbNamesToDelete: string[] = [];

  function trackDb<T extends Dexie>(dbInstance: T): T {
    openDatabases.push(dbInstance);
    if (!dbNamesToDelete.includes(dbInstance.name)) {
      dbNamesToDelete.push(dbInstance.name);
    }
    return dbInstance;
  }

  afterEach(async function () {
    for (const dbInstance of openDatabases) {
      if (dbInstance.isOpen()) {
        dbInstance.close();
      }
    }
    openDatabases.length = 0;

    for (const dbName of dbNamesToDelete) {
      try {
        await Dexie.delete(dbName);
      } catch {
        // Ignore deletion cleanup failures in test teardown
      }
    }
    dbNamesToDelete.length = 0;
  });

  describe('1. V1 to V2 Database Migration & Historical Preservation', function () {
    it('should upgrade a Version 1 database to Version 2 while preserving all historical records and indexes', async function () {
      const dbName = createUniqueDbName('MigrationV1toV2');

      // Step 1: Initialize database strictly at Version 1
      const v1Db = trackDb(new Dexie(dbName));
      v1Db.version(1).stores({
        writingMessages: '++id, platform, capturedAt, trainingStatus',
      });
      await v1Db.open();

      expect(v1Db.verno).toBe(1);

      // Seed historical writing messages with varied platforms, timestamps, and training statuses
      const historicalTime1 = new Date('2026-10-01T10:15:30.000Z');
      const historicalTime2 = new Date('2026-10-05T14:20:00.000Z');
      const historicalTime3 = new Date('2026-10-10T18:45:10.000Z');

      const msg1Id = await v1Db.table('writingMessages').add({
        platform: 'ChatGPT',
        text: 'Historical user prompt #1 about TypeScript architecture.',
        createdAt: historicalTime1,
        capturedAt: historicalTime1,
        trainingStatus: 'pending',
      });

      const msg2Id = await v1Db.table('writingMessages').add({
        platform: 'ChatGPT',
        text: 'Historical user prompt #2 about React components.',
        createdAt: historicalTime2,
        capturedAt: historicalTime2,
        trainingStatus: 'trained',
      });

      const msg3Id = await v1Db.table('writingMessages').add({
        platform: 'Claude',
        text: 'Historical prompt #3 from another platform.',
        createdAt: historicalTime3,
        capturedAt: historicalTime3,
        trainingStatus: 'pending',
      });

      expect(msg1Id).toBe(1);
      expect(msg2Id).toBe(2);
      expect(msg3Id).toBe(3);

      const countBeforeMigration = await v1Db.table('writingMessages').count();
      expect(countBeforeMigration).toBe(3);

      // Step 2: Close Version 1 connection cleanly before upgrading
      v1Db.close();

      // Step 3: Open the same database using the updated Echo database factory (Version 2)
      const v2Db = trackDb(createEchoDatabase(dbName));
      await v2Db.open();

      // Verify schema version upgraded to 2
      expect(v2Db.verno).toBe(2);

      // Verify all 3 historical messages survived intact
      const countAfterMigration = await v2Db.writingMessages.count();
      expect(countAfterMigration).toBe(3);

      const restored1 = await v2Db.writingMessages.get(1);
      expect(restored1).toBeDefined();
      expect(restored1?.id).toBe(1);
      expect(restored1?.platform).toBe('ChatGPT');
      expect(restored1?.text).toBe(
        'Historical user prompt #1 about TypeScript architecture.'
      );
      expect(new Date(restored1!.capturedAt).toISOString()).toBe(
        historicalTime1.toISOString()
      );
      expect(restored1?.trainingStatus).toBe('pending');

      const restored2 = await v2Db.writingMessages.get(2);
      expect(restored2).toBeDefined();
      expect(restored2?.id).toBe(2);
      expect(restored2?.platform).toBe('ChatGPT');
      expect(restored2?.text).toBe(
        'Historical user prompt #2 about React components.'
      );
      expect(new Date(restored2!.capturedAt).toISOString()).toBe(
        historicalTime2.toISOString()
      );
      expect(restored2?.trainingStatus).toBe('trained');

      const restored3 = await v2Db.writingMessages.get(3);
      expect(restored3).toBeDefined();
      expect(restored3?.id).toBe(3);
      expect(restored3?.platform).toBe('Claude');
      expect(restored3?.trainingStatus).toBe('pending');

      // Verify that the new V2 stores are available and initialized empty
      expect(v2Db.personalKnowledge).toBeDefined();
      expect(v2Db.knowledgeEvidence).toBeDefined();

      const initialKnowledgeCount = await v2Db.personalKnowledge.count();
      const initialEvidenceCount = await v2Db.knowledgeEvidence.count();
      expect(initialKnowledgeCount).toBe(0);
      expect(initialEvidenceCount).toBe(0);

      // Verify saving a new message in V2 works alongside surviving V1 messages
      const newMsgId = await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'New message captured after V2 migration.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        v2Db
      );

      expect(newMsgId).toBe(4);
      expect(await v2Db.writingMessages.count()).toBe(4);
    });
  });

  describe('2. Personal Knowledge Storage Operations', function () {
    it('should save and retrieve personal knowledge items', async function () {
      const dbName = createUniqueDbName('KnowledgeOps');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const item1: PersonalKnowledgeItem = {
        category: 'skills',
        subject: 'TypeScript',
        predicate: 'proficient_in',
        value: 'advanced',
        contextScope: 'fullstack web',
        stability: 'stable',
        status: 'active',
        freshnessState: 'evergreen',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date('2026-11-01T10:00:00Z'),
        updatedAt: new Date('2026-11-01T10:00:00Z'),
        lastConfirmedAt: new Date('2026-11-01T10:00:00Z'),
      };

      const item2: PersonalKnowledgeItem = {
        category: 'habits',
        subject: 'AI Assisting',
        predicate: 'uses_ai_when_stuck',
        value: 'true',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'medium',
        importance: 'secondary',
        createdAt: new Date('2026-11-02T10:00:00Z'),
        updatedAt: new Date('2026-11-02T10:00:00Z'),
        lastConfirmedAt: new Date('2026-11-02T10:00:00Z'),
      };

      const id1 = await savePersonalKnowledge(item1, testDb);
      const id2 = await savePersonalKnowledge(item2, testDb);

      expect(id1).toBe(1);
      expect(id2).toBe(2);

      // Retrieve all
      const allItems = await getPersonalKnowledge(undefined, testDb);
      expect(allItems).toHaveLength(2);

      // Retrieve by category
      const skillItems = await getPersonalKnowledge(
        { category: 'skills' },
        testDb
      );
      expect(skillItems).toHaveLength(1);
      expect(skillItems[0]?.subject).toBe('TypeScript');

      const habitItems = await getPersonalKnowledge(
        { category: 'habits' },
        testDb
      );
      expect(habitItems).toHaveLength(1);
      expect(habitItems[0]?.subject).toBe('AI Assisting');

      // Retrieve by status
      const activeItems = await getPersonalKnowledge(
        { status: 'active' },
        testDb
      );
      expect(activeItems).toHaveLength(2);

      const archivedItems = await getPersonalKnowledge(
        { status: 'archived' },
        testDb
      );
      expect(archivedItems).toHaveLength(0);

      // Retrieve by ID
      const fetched1 = await getPersonalKnowledgeById(id1, testDb);
      expect(fetched1?.subject).toBe('TypeScript');
      expect(fetched1?.confidence).toBe('high');
      expect(fetched1?.contextScope).toBe('fullstack web');
    });

    it('should reject invalid personal knowledge items', async function () {
      const dbName = createUniqueDbName('InvalidKnowledge');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Missing required subject
      const invalidItem = {
        category: 'skills',
        subject: '   ',
        predicate: 'known',
        value: 'yes',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      } as PersonalKnowledgeItem;

      await expect(
        savePersonalKnowledge(invalidItem, testDb)
      ).rejects.toThrow('Invalid personal knowledge item');
    });
  });

  describe('3. Knowledge Evidence Storage and Provenance Linking', function () {
    it('should save and retrieve evidence linked to knowledge and messages', async function () {
      const dbName = createUniqueDbName('EvidenceOps');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const messageId = 10;
      const knowledgeId = 20;

      const evidence1: KnowledgeEvidence = {
        knowledgeId,
        messageId,
        sourceExcerpt: 'I have been using React and TypeScript for my web projects.',
        relevanceNote: 'Explicit declaration of frontend technical skills.',
        isValid: true,
        extractedAt: new Date('2026-11-09T08:00:00Z'),
      };

      const evId = await saveKnowledgeEvidence(evidence1, testDb);
      expect(evId).toBe(1);

      // Retrieve by knowledgeId
      const evidenceList = await getKnowledgeEvidence(knowledgeId, testDb);
      expect(evidenceList).toHaveLength(1);
      expect(evidenceList[0]?.sourceExcerpt).toBe(
        'I have been using React and TypeScript for my web projects.'
      );
      expect(evidenceList[0]?.isValid).toBe(true);

      // Retrieve by messageId
      const msgEvidenceList = await getEvidenceByMessageId(messageId, testDb);
      expect(msgEvidenceList).toHaveLength(1);
      expect(msgEvidenceList[0]?.knowledgeId).toBe(knowledgeId);
    });

    it('should reject invalid evidence with non-positive IDs', async function () {
      const dbName = createUniqueDbName('InvalidEvidence');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const invalidEvidence: KnowledgeEvidence = {
        knowledgeId: 0,
        messageId: -1,
        sourceExcerpt: 'test',
        isValid: true,
        extractedAt: new Date(),
      };

      await expect(
        saveKnowledgeEvidence(invalidEvidence, testDb)
      ).rejects.toThrow('Invalid knowledge evidence');
    });
  });

  describe('4. Message Deletion and Evidence Privacy', function () {
    it('should atomically delete message, scrub evidence excerpt and relevanceNote, and scrub/archive orphaned knowledge', async function () {
      const dbName = createUniqueDbName('DeletionPrivacy');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // 1. Create a writing message
      const messageId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am currently building an extension called Echo with Vite and React.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;
      expect(messageId).toBeTypeOf('number');

      // 2. Create a personal knowledge item supported solely by this message
      const createdAt = new Date('2026-11-08T10:00:00Z');
      const knowledgeId = await savePersonalKnowledge(
        {
          category: 'projects',
          subject: 'Echo Extension',
          predicate: 'is_building',
          value: 'active',
          contextScope: 'browser development',
          stability: 'evolving',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt,
          updatedAt: createdAt,
          lastConfirmedAt: createdAt,
        },
        testDb
      );

      // 3. Link evidence to this message with both sourceExcerpt and relevanceNote
      const evidenceId = await saveKnowledgeEvidence(
        {
          knowledgeId,
          messageId,
          sourceExcerpt: 'I am currently building an extension called Echo',
          relevanceNote: 'Explicit declaration of user project and tooling',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // 4. Delete the writing message
      await deleteWritingMessage(messageId, testDb);

      // 5. Verify the writing message is deleted
      const deletedMessage = await testDb.writingMessages.get(messageId);
      expect(deletedMessage).toBeUndefined();

      // 6. Verify evidence record was retained for provenance but BOTH sourceExcerpt and relevanceNote are scrubbed
      const scrubbedEvidence = await testDb.knowledgeEvidence.get(evidenceId);
      expect(scrubbedEvidence).toBeDefined();
      expect(scrubbedEvidence?.isValid).toBe(false);
      expect(scrubbedEvidence?.sourceExcerpt).toBe(''); // Excerpt scrubbed!
      expect(scrubbedEvidence?.relevanceNote).toBe(''); // Relevance note scrubbed!
      expect(scrubbedEvidence?.invalidationReason).toBe('source_message_deleted');
      expect(scrubbedEvidence?.invalidatedAt).toBeInstanceOf(Date);

      // 7. Verify knowledge item has ALL four content fields cleared and is archived
      const archivedKnowledge = await testDb.personalKnowledge.get(knowledgeId);
      expect(archivedKnowledge).toBeDefined();
      expect(archivedKnowledge?.status).toBe('archived');
      expect(archivedKnowledge?.subject).toBe(''); // Cleared
      expect(archivedKnowledge?.predicate).toBe(''); // Cleared
      expect(archivedKnowledge?.value).toBe(''); // Cleared
      expect(archivedKnowledge?.contextScope).toBe(''); // Cleared

      // Non-content lifecycle metadata is preserved
      expect(archivedKnowledge?.category).toBe('projects');
      expect(archivedKnowledge?.stability).toBe('evolving');
      expect(archivedKnowledge?.freshnessState).toBe('fresh');
      expect(archivedKnowledge?.confidence).toBe('high');
      expect(archivedKnowledge?.importance).toBe('primary');
      expect(new Date(archivedKnowledge!.createdAt).toISOString()).toBe(createdAt.toISOString());
      expect(new Date(archivedKnowledge!.updatedAt).getTime()).toBeGreaterThan(createdAt.getTime());
    });

    it('should archive and scrub knowledge with multiple evidence records when one source is deleted, while preserving surviving evidence records', async function () {
      const dbName = createUniqueDbName('MultiEvidence');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // 1. Create two distinct writing messages
      const msg1Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I prefer using Python for backend data analysis.',
          createdAt: new Date('2026-11-01T10:00:00Z'),
          capturedAt: new Date('2026-11-01T10:00:00Z'),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg2Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'For data analysis pipelines, Python is my primary language.',
          createdAt: new Date('2026-11-05T10:00:00Z'),
          capturedAt: new Date('2026-11-05T10:00:00Z'),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // 2. Create one knowledge item supported by both messages
      const knowledgeId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Python',
          predicate: 'prefers_for',
          value: 'backend data analysis and pipelines',
          contextScope: 'backend development',
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
      );

      // 3. Link evidence from both messages
      const ev1Id = await saveKnowledgeEvidence(
        {
          knowledgeId,
          messageId: msg1Id,
          sourceExcerpt: 'I prefer using Python for backend data analysis.',
          relevanceNote: 'First source mentioning Python preferences',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const ev2Id = await saveKnowledgeEvidence(
        {
          knowledgeId,
          messageId: msg2Id,
          sourceExcerpt: 'Python is my primary language.',
          relevanceNote: 'Second source corroborating Python preference',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // 4. Delete only message 1
      await deleteWritingMessage(msg1Id, testDb);

      // Message 1 is deleted, Message 2 remains
      expect(await testDb.writingMessages.get(msg1Id)).toBeUndefined();
      expect(await testDb.writingMessages.get(msg2Id)).toBeDefined();

      // Evidence 1 is scrubbed and invalidated
      const ev1 = await testDb.knowledgeEvidence.get(ev1Id);
      expect(ev1?.isValid).toBe(false);
      expect(ev1?.sourceExcerpt).toBe('');
      expect(ev1?.relevanceNote).toBe('');
      expect(ev1?.invalidationReason).toBe('source_message_deleted');

      // SURVIVING EVIDENCE 2: remains valid, excerpt intact, relevanceNote intact, and linked to knowledgeId
      const ev2 = await testDb.knowledgeEvidence.get(ev2Id);
      expect(ev2?.isValid).toBe(true);
      expect(ev2?.sourceExcerpt).toBe('Python is my primary language.');
      expect(ev2?.relevanceNote).toBe('Second source corroborating Python preference');
      expect(ev2?.knowledgeId).toBe(knowledgeId);

      // CONSERVATIVE PRIVACY: Knowledge item is archived and all content fields scrubbed
      const knowledgeAfterFirstDelete = await testDb.personalKnowledge.get(knowledgeId);
      expect(knowledgeAfterFirstDelete?.status).toBe('archived');
      expect(knowledgeAfterFirstDelete?.subject).toBe('');
      expect(knowledgeAfterFirstDelete?.predicate).toBe('');
      expect(knowledgeAfterFirstDelete?.value).toBe('');
      expect(knowledgeAfterFirstDelete?.contextScope).toBe('');

      // 5. When message 2 is subsequently deleted
      await deleteWritingMessage(msg2Id, testDb);

      // Message 2 is deleted
      expect(await testDb.writingMessages.get(msg2Id)).toBeUndefined();

      // Evidence 2 is now also scrubbed
      const ev2After = await testDb.knowledgeEvidence.get(ev2Id);
      expect(ev2After?.isValid).toBe(false);
      expect(ev2After?.sourceExcerpt).toBe('');
      expect(ev2After?.relevanceNote).toBe('');

      // Knowledge item remains archived with content scrubbed
      const knowledgeAfterSecondDelete = await testDb.personalKnowledge.get(knowledgeId);
      expect(knowledgeAfterSecondDelete?.status).toBe('archived');
      expect(knowledgeAfterSecondDelete?.value).toBe('');
    });

    it('should scrub and archive superseded knowledge when its source message is deleted', async function () {
      const dbName = createUniqueDbName('SupersededPrivacy');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I used to develop in Angular AngularJS.',
          createdAt: new Date('2026-10-01T10:00:00Z'),
          capturedAt: new Date('2026-10-01T10:00:00Z'),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Knowledge item was already marked as superseded (e.g. by learning React)
      const knowledgeId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'AngularJS',
          predicate: 'legacy_experience',
          value: 'deprecated knowledge',
          contextScope: 'legacy frontend',
          stability: 'evolving',
          status: 'superseded',
          freshnessState: 'stale',
          confidence: 'high',
          importance: 'secondary',
          createdAt: new Date('2026-10-01T10:00:00Z'),
          updatedAt: new Date('2026-10-01T10:00:00Z'),
          lastConfirmedAt: new Date('2026-10-01T10:00:00Z'),
        },
        testDb
      );

      const evidenceId = await saveKnowledgeEvidence(
        {
          knowledgeId,
          messageId: msgId,
          sourceExcerpt: 'I used to develop in Angular AngularJS.',
          relevanceNote: 'Historical stack knowledge',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // User deletes the original source message
      await deleteWritingMessage(msgId, testDb);

      // Verify message is deleted
      expect(await testDb.writingMessages.get(msgId)).toBeUndefined();

      // Verify evidence is invalidated and scrubbed
      const ev = await testDb.knowledgeEvidence.get(evidenceId);
      expect(ev?.isValid).toBe(false);
      expect(ev?.sourceExcerpt).toBe('');
      expect(ev?.relevanceNote).toBe('');
      expect(ev?.invalidationReason).toBe('source_message_deleted');

      // Verify superseded knowledge item is now archived and its content fields cleared
      const k = await testDb.personalKnowledge.get(knowledgeId);
      expect(k?.status).toBe('archived');
      expect(k?.subject).toBe('');
      expect(k?.predicate).toBe('');
      expect(k?.value).toBe('');
      expect(k?.contextScope).toBe('');
    });

    it('should roll back all modifications across stores if a deletion transaction encounters an error', async function () {
      const dbName = createUniqueDbName('RollbackSafety');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const messageId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Crucial user message that must not be lost if transaction fails.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const knowledgeId = await savePersonalKnowledge(
        {
          category: 'goals',
          subject: 'Personal Project',
          predicate: 'aims_to_finish',
          value: 'Q4',
          contextScope: 'personal work',
          stability: 'temporary',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'medium',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      );

      const evidenceId = await saveKnowledgeEvidence(
        {
          knowledgeId,
          messageId,
          sourceExcerpt: 'Crucial user message excerpt',
          relevanceNote: 'Crucial note',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Simulate an atomic transaction failure
      let transactionError: Error | undefined;
      try {
        await testDb.transaction(
          'rw',
          [testDb.writingMessages, testDb.knowledgeEvidence, testDb.personalKnowledge],
          async function () {
            // Modify evidence
            await testDb.knowledgeEvidence.update(evidenceId, {
              isValid: false,
              sourceExcerpt: '',
              relevanceNote: '',
            });

            // Modify knowledge
            await testDb.personalKnowledge.update(knowledgeId, {
              status: 'archived',
              subject: '',
              predicate: '',
              value: '',
              contextScope: '',
            });

            // Delete message
            await testDb.writingMessages.delete(messageId);

            // Deliberate mid-transaction failure
            throw new Error('Simulated disk/database failure');
          }
        );
      } catch (error) {
        transactionError = error as Error;
      }

      expect(transactionError).toBeDefined();
      expect(transactionError?.message).toBe('Simulated disk/database failure');

      // Verify complete rollback: message NOT deleted, evidence NOT scrubbed, knowledge NOT archived
      const messageAfterRollback = await testDb.writingMessages.get(messageId);
      expect(messageAfterRollback).toBeDefined();
      expect(messageAfterRollback?.text).toBe(
        'Crucial user message that must not be lost if transaction fails.'
      );

      const evidenceAfterRollback = await testDb.knowledgeEvidence.get(evidenceId);
      expect(evidenceAfterRollback?.isValid).toBe(true);
      expect(evidenceAfterRollback?.sourceExcerpt).toBe('Crucial user message excerpt');
      expect(evidenceAfterRollback?.relevanceNote).toBe('Crucial note');

      const knowledgeAfterRollback = await testDb.personalKnowledge.get(knowledgeId);
      expect(knowledgeAfterRollback?.status).toBe('active');
      expect(knowledgeAfterRollback?.subject).toBe('Personal Project');
      expect(knowledgeAfterRollback?.predicate).toBe('aims_to_finish');
      expect(knowledgeAfterRollback?.value).toBe('Q4');
      expect(knowledgeAfterRollback?.contextScope).toBe('personal work');
    });

    it('should delete a standalone writing message cleanly without error when no evidence is linked', async function () {
      const dbName = createUniqueDbName('StandaloneDelete');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Standalone message with no linked knowledge.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      await deleteWritingMessage(msgId, testDb);

      const deleted = await testDb.writingMessages.get(msgId);
      expect(deleted).toBeUndefined();
    });
  });

  describe('5. Lifecycle and Connection Management', function () {
    it('should close database connection when versionchange is triggered', async function () {
      const dbName = createUniqueDbName('LifecycleTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();
      expect(testDb.isOpen()).toBe(true);

      // Simulate a versionchange event as fired by Dexie when another tab upgrades
      testDb.on('versionchange').fire(new Event('versionchange'));

      // The listener should have closed testDb
      expect(testDb.isOpen()).toBe(false);
    });
  });
});

