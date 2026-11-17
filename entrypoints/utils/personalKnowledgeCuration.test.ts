import 'fake-indexeddb/auto';

import { afterEach, describe, expect, it } from 'vitest';
import Dexie from 'dexie';
import {
  createEchoDatabase,
  addManualPersonalKnowledge,
  updatePersonalKnowledge,
  deletePersonalKnowledge,
  savePersonalKnowledge,
  getPersonalKnowledge,
  getPersonalKnowledgeById,
  saveKnowledgeEvidence,
  getKnowledgeEvidence,
  deleteWritingMessage,
  saveWritingMessage,
  VALID_INVALIDATION_REASONS,
  type EchoDatabase,
  type PersonalKnowledgeItem,
  type KnowledgeEvidence,
  type InvalidationReason,
  type ManualPersonalKnowledgeInput,
  type UpdatePersonalKnowledgeInput,
} from './database';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Personal Knowledge Curation Database Operations', function () {
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

  describe('1. Manual Personal Knowledge Creation', function () {
    it('should create a manual fact with all fields and explicit manual provenance', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('manual_create_all')));
      const beforeTime = Date.now() - 50;

      const input: ManualPersonalKnowledgeInput = {
        category: 'skills',
        subject: 'TypeScript',
        predicate: 'mastered building',
        value: 'type-safe compiler passes',
        contextScope: 'compilers',
        stability: 'stable',
        confidence: 'high',
        importance: 'primary',
        freshnessState: 'fresh',
      };

      const id = await addManualPersonalKnowledge(input, testDb);

      expect(typeof id).toBe('number');
      expect(id).toBeGreaterThan(0);

      const saved = await getPersonalKnowledgeById(id, testDb);
      expect(saved).toBeDefined();
      expect(saved?.id).toBe(id);
      expect(saved?.category).toBe('skills');
      expect(saved?.subject).toBe('TypeScript');
      expect(saved?.predicate).toBe('mastered building');
      expect(saved?.value).toBe('type-safe compiler passes');
      expect(saved?.contextScope).toBe('compilers');
      expect(saved?.stability).toBe('stable');
      expect(saved?.status).toBe('active');
      expect(saved?.confidence).toBe('high');
      expect(saved?.importance).toBe('primary');
      expect(saved?.freshnessState).toBe('fresh');
      expect(saved?.sourceType).toBe('manual');
      expect(saved?.editedByUser).toBe(false);

      expect(saved?.createdAt.getTime()).toBeGreaterThanOrEqual(beforeTime);
      expect(saved?.updatedAt.getTime()).toBeGreaterThanOrEqual(beforeTime);
      expect(saved?.lastConfirmedAt.getTime()).toBeGreaterThanOrEqual(beforeTime);

      // Verify no evidence rows were created
      const evidence = await getKnowledgeEvidence(id, testDb);
      expect(evidence).toHaveLength(0);
    });

    it('should use safe default values when optional fields are omitted', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('manual_create_defaults')));

      const input: ManualPersonalKnowledgeInput = {
        category: 'preferences',
        subject: 'Code Review',
        predicate: 'prefers',
        value: 'small focused pull requests',
      };

      const id = await addManualPersonalKnowledge(input, testDb);
      const saved = await getPersonalKnowledgeById(id, testDb);

      expect(saved).toBeDefined();
      expect(saved?.category).toBe('preferences');
      expect(saved?.stability).toBe('stable');
      expect(saved?.confidence).toBe('high');
      expect(saved?.importance).toBe('primary');
      expect(saved?.freshnessState).toBe('fresh');
      expect(saved?.status).toBe('active');
      expect(saved?.sourceType).toBe('manual');
      expect(saved?.contextScope).toBeUndefined();
    });

    it('should trim whitespace from text fields', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('manual_create_trim')));

      const input: ManualPersonalKnowledgeInput = {
        category: 'goals',
        subject: '   Distributed Systems   ',
        predicate: '   aims to architect   ',
        value: '   resilient Raft clusters   ',
        contextScope: '   infrastructure   ',
      };

      const id = await addManualPersonalKnowledge(input, testDb);
      const saved = await getPersonalKnowledgeById(id, testDb);

      expect(saved?.subject).toBe('Distributed Systems');
      expect(saved?.predicate).toBe('aims to architect');
      expect(saved?.value).toBe('resilient Raft clusters');
      expect(saved?.contextScope).toBe('infrastructure');
    });

    it('should reject invalid or empty inputs', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('manual_create_rejections')));

      // Null or non-object input
      await expect(
        addManualPersonalKnowledge(null as unknown as ManualPersonalKnowledgeInput, testDb)
      ).rejects.toThrow('Invalid manual personal knowledge: input must be an object.');

      // Invalid category
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'unsupported_category' as unknown as ManualPersonalKnowledgeInput['category'],
            subject: 'Subject',
            predicate: 'predicate',
            value: 'value',
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge category');

      // Empty subject
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: '   ',
            predicate: 'predicate',
            value: 'value',
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge subject: subject must not be empty.');

      // Empty predicate
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'Subject',
            predicate: '',
            value: 'value',
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge predicate: predicate must not be empty.');

      // Empty value
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'Subject',
            predicate: 'predicate',
            value: '   ',
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge value: value must not be empty.');

      // Invalid stability
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'Subject',
            predicate: 'predicate',
            value: 'value',
            stability: 'infinite' as unknown as ManualPersonalKnowledgeInput['stability'],
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge stability');

      // Invalid confidence
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'Subject',
            predicate: 'predicate',
            value: 'value',
            confidence: 'maximum' as unknown as ManualPersonalKnowledgeInput['confidence'],
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge confidence');

      // Invalid importance
      await expect(
        addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'Subject',
            predicate: 'predicate',
            value: 'value',
            importance: 'critical' as unknown as ManualPersonalKnowledgeInput['importance'],
          },
          testDb
        )
      ).rejects.toThrow('Invalid personal knowledge importance');
    });
  });

  describe('2. Personal Knowledge Fact Editing', function () {
    it('should update mutable fields, advance updatedAt, and set editedByUser: true', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('edit_success')));

      const initialId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Python',
          predicate: 'uses',
          value: 'Flask',
        },
        testDb
      );

      const beforeUpdate = await getPersonalKnowledgeById(initialId, testDb);
      expect(beforeUpdate?.editedByUser).toBe(false);

      // Brief delay to ensure timestamp progression
      await new Promise((resolve) => setTimeout(resolve, 15));

      const updates: UpdatePersonalKnowledgeInput = {
        predicate: 'specializes in',
        value: 'FastAPI and async IO',
        confidence: 'high',
        importance: 'primary',
      };

      await updatePersonalKnowledge(initialId, updates, testDb);

      const updated = await getPersonalKnowledgeById(initialId, testDb);
      expect(updated).toBeDefined();
      expect(updated?.subject).toBe('Python'); // Unmodified
      expect(updated?.predicate).toBe('specializes in'); // Updated
      expect(updated?.value).toBe('FastAPI and async IO'); // Updated
      expect(updated?.confidence).toBe('high');
      expect(updated?.importance).toBe('primary');
      expect(updated?.editedByUser).toBe(true);
      expect(updated?.sourceType).toBe('manual'); // Provenance retained
      expect(updated?.createdAt.getTime()).toBe(beforeUpdate?.createdAt.getTime()); // Immutable createdAt
      expect(updated?.updatedAt.getTime()).toBeGreaterThan(beforeUpdate!.updatedAt.getTime());
    });

    it('should reject modification of immutable fields (id, createdAt, sourceType, status)', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('edit_immutable')));

      const id = await addManualPersonalKnowledge(
        {
          category: 'habits',
          subject: 'Deep Work',
          predicate: 'blocks',
          value: 'morning hours for coding',
        },
        testDb
      );

      // Attempt to modify id
      await expect(
        updatePersonalKnowledge(id, { id: 9999 } as unknown as UpdatePersonalKnowledgeInput, testDb)
      ).rejects.toThrow('Cannot modify immutable field: "id".');

      // Attempt to modify createdAt
      await expect(
        updatePersonalKnowledge(id, { createdAt: new Date() } as unknown as UpdatePersonalKnowledgeInput, testDb)
      ).rejects.toThrow('Cannot modify immutable field: "createdAt".');

      // Attempt to modify sourceType
      await expect(
        updatePersonalKnowledge(id, { sourceType: 'extracted' } as unknown as UpdatePersonalKnowledgeInput, testDb)
      ).rejects.toThrow('Cannot modify immutable field: "sourceType".');

      // Attempt to modify status (must use deletePersonalKnowledge)
      await expect(
        updatePersonalKnowledge(id, { status: 'archived' } as unknown as UpdatePersonalKnowledgeInput, testDb)
      ).rejects.toThrow('Cannot modify immutable field: "status".');
    });

    it('should reject updates to non-existent or archived facts', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('edit_non_existent_and_archived')));

      // Non-existent fact
      await expect(
        updatePersonalKnowledge(8888, { value: 'New Value' }, testDb)
      ).rejects.toThrow('Personal knowledge item #8888 not found.');

      // Archived fact
      const id = await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Echo Extension',
          predicate: 'implements',
          value: 'on-device intelligence',
        },
        testDb
      );

      await deletePersonalKnowledge(id, 'user_manually_rejected', testDb);

      await expect(
        updatePersonalKnowledge(id, { value: 'Different Value' }, testDb)
      ).rejects.toThrow('Cannot update personal knowledge item');
    });

    it('should reject invalid or empty update values', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('edit_invalid_values')));

      const id = await addManualPersonalKnowledge(
        {
          category: 'opinions',
          subject: 'Architecture',
          predicate: 'believes',
          value: 'modular monoliths are underrated',
        },
        testDb
      );

      // Empty payload
      await expect(
        updatePersonalKnowledge(id, {}, testDb)
      ).rejects.toThrow('Invalid update payload: at least one mutable field must be provided.');

      // Empty trimmed subject
      await expect(
        updatePersonalKnowledge(id, { subject: '   ' }, testDb)
      ).rejects.toThrow('Invalid personal knowledge subject');

      // Empty trimmed predicate
      await expect(
        updatePersonalKnowledge(id, { predicate: '' }, testDb)
      ).rejects.toThrow('Invalid personal knowledge predicate');

      // Empty trimmed value
      await expect(
        updatePersonalKnowledge(id, { value: '   ' }, testDb)
      ).rejects.toThrow('Invalid personal knowledge value');

      // Invalid category
      await expect(
        updatePersonalKnowledge(id, { category: 'non_existent' as unknown as UpdatePersonalKnowledgeInput['category'] }, testDb)
      ).rejects.toThrow('Invalid personal knowledge category');
    });

    it('should reject unknown properties and internally controlled fields, keeping original record unchanged', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('edit_strict_allowlist')));

      const id = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'TypeScript',
          predicate: 'builds',
          value: 'type systems',
        },
        testDb
      );

      const beforeAttempt = await getPersonalKnowledgeById(id, testDb);

      // 1. Reject an unknown property alongside an otherwise valid update
      await expect(
        updatePersonalKnowledge(
          id,
          { value: 'advanced type systems', foo: 'bar' } as unknown as UpdatePersonalKnowledgeInput,
          testDb
        )
      ).rejects.toThrow('Cannot update unsupported field: "foo".');

      // 2. Reject editedByUser supplied by caller
      await expect(
        updatePersonalKnowledge(
          id,
          { value: 'advanced type systems', editedByUser: true } as unknown as UpdatePersonalKnowledgeInput,
          testDb
        )
      ).rejects.toThrow('Cannot modify internal field: "editedByUser".');

      // 3. Reject updatedAt supplied by caller
      await expect(
        updatePersonalKnowledge(
          id,
          { value: 'advanced type systems', updatedAt: new Date() } as unknown as UpdatePersonalKnowledgeInput,
          testDb
        )
      ).rejects.toThrow('Cannot modify internal field: "updatedAt".');

      // 4. Reject lastConfirmedAt supplied by caller
      await expect(
        updatePersonalKnowledge(
          id,
          { value: 'advanced type systems', lastConfirmedAt: new Date() } as unknown as UpdatePersonalKnowledgeInput,
          testDb
        )
      ).rejects.toThrow('Cannot modify internal field: "lastConfirmedAt".');

      // 5. Confirm original database record remains unchanged after all rejected attempts
      const afterAttempts = await getPersonalKnowledgeById(id, testDb);
      expect(afterAttempts?.value).toBe('type systems');
      expect(afterAttempts?.editedByUser).toBe(false);
      expect(afterAttempts?.updatedAt.getTime()).toBe(beforeAttempt!.updatedAt.getTime());
    });
  });

  describe('3. Safe Fact Soft-Deletion and Evidence Invalidation', function () {
    it('should soft-delete an extracted fact and invalidate all linked evidence with user_manually_rejected', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('delete_extracted_with_evidence')));

      // 1. Create an extracted fact
      const factId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Rust',
          predicate: 'writes',
          value: 'systems software',
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
      );

      // 2. Link two evidence rows
      const ev1Id = await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: 101,
          sourceExcerpt: 'I write Rust for our database engine daily.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const ev2Id = await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: 102,
          sourceExcerpt: 'Rust memory safety prevents data races.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Also create an unrelated fact with evidence to verify non-interference
      const unrelatedFactId = await savePersonalKnowledge(
        {
          category: 'projects',
          subject: 'Side Project',
          predicate: 'builds',
          value: 'CLI tools',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'secondary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      );

      const unrelatedEvId = await saveKnowledgeEvidence(
        {
          knowledgeId: unrelatedFactId,
          messageId: 201,
          sourceExcerpt: 'Working on my CLI side project.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // 3. Soft-delete the first fact
      const beforeDelete = Date.now() - 20;
      await deletePersonalKnowledge(factId, 'user_manually_rejected', testDb);

      // 4. Verify fact status is archived and text is preserved
      const factAfter = await getPersonalKnowledgeById(factId, testDb);
      expect(factAfter).toBeDefined();
      expect(factAfter?.status).toBe('archived');
      expect(factAfter?.subject).toBe('Rust');
      expect(factAfter?.value).toBe('systems software');
      expect(factAfter?.updatedAt.getTime()).toBeGreaterThanOrEqual(beforeDelete);

      // 5. Verify linked evidence is invalidated
      const linkedEvidence = await getKnowledgeEvidence(factId, testDb);
      expect(linkedEvidence).toHaveLength(2);
      for (const ev of linkedEvidence) {
        expect(ev.isValid).toBe(false);
        expect(ev.invalidationReason).toBe('user_manually_rejected');
        expect(ev.invalidatedAt).toBeDefined();
        expect(ev.invalidatedAt!.getTime()).toBeGreaterThanOrEqual(beforeDelete);
      }

      // 6. Verify unrelated fact and evidence are completely untouched
      const unrelatedAfter = await getPersonalKnowledgeById(unrelatedFactId, testDb);
      expect(unrelatedAfter?.status).toBe('active');
      const unrelatedEv = await testDb.knowledgeEvidence.get(unrelatedEvId);
      expect(unrelatedEv?.isValid).toBe(true);
      expect(unrelatedEv?.invalidatedAt).toBeUndefined();
    });

    it('should safely soft-delete a manual fact that has zero evidence records', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('delete_manual_no_evidence')));

      const manualId = await addManualPersonalKnowledge(
        {
          category: 'habits',
          subject: 'Hydration',
          predicate: 'drinks',
          value: '3 liters of water daily',
        },
        testDb
      );

      const evidenceBefore = await getKnowledgeEvidence(manualId, testDb);
      expect(evidenceBefore).toHaveLength(0);

      // Delete should succeed without errors
      await deletePersonalKnowledge(manualId, 'user_manually_rejected', testDb);

      const archived = await getPersonalKnowledgeById(manualId, testDb);
      expect(archived?.status).toBe('archived');
    });

    it('should reject deletion of non-existent fact ID', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('delete_non_existent')));

      await expect(
        deletePersonalKnowledge(9999, 'user_manually_rejected', testDb)
      ).rejects.toThrow('Personal knowledge item #9999 not found.');
    });

    it('should maintain transaction atomicity if an error occurs', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('delete_atomicity')));

      const factId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Testing',
          predicate: 'writes',
          value: 'atomic unit tests',
        },
        testDb
      );

      // Test with invalid ID
      await expect(
        deletePersonalKnowledge(-1, 'user_manually_rejected', testDb)
      ).rejects.toThrow('Invalid personal knowledge ID');

      // Original fact remains active
      const fact = await getPersonalKnowledgeById(factId, testDb);
      expect(fact?.status).toBe('active');
    });

    it('should reject unsupported invalidation reasons and leave fact and evidence unchanged', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('delete_invalid_reason')));

      const factId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Go',
          predicate: 'programs',
          value: 'network daemons',
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
      );

      const evidenceId = await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: 55,
          sourceExcerpt: 'I write network daemons in Go.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // 1. Reject unsupported invalidation reasons
      await expect(
        deletePersonalKnowledge(factId, 'unsupported_reason' as unknown as InvalidationReason, testDb)
      ).rejects.toThrow('Invalid invalidation reason: "unsupported_reason".');

      await expect(
        deletePersonalKnowledge(factId, '' as unknown as InvalidationReason, testDb)
      ).rejects.toThrow('Invalid invalidation reason: "".');

      await expect(
        deletePersonalKnowledge(factId, null as unknown as InvalidationReason, testDb)
      ).rejects.toThrow('Invalid invalidation reason: "null".');

      // 2. Confirm the fact remains active
      const fact = await getPersonalKnowledgeById(factId, testDb);
      expect(fact?.status).toBe('active');

      // 3. Confirm linked evidence remains valid and unchanged
      const evidence = await testDb.knowledgeEvidence.get(evidenceId);
      expect(evidence?.isValid).toBe(true);
      expect(evidence?.invalidationReason).toBeUndefined();
      expect(evidence?.invalidatedAt).toBeUndefined();
    });
  });

  describe('4. Filter and Backward Compatibility Support', function () {
    it('should filter facts by sourceType (manual vs extracted)', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('filter_source_type')));

      await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Kotlin',
          predicate: 'develops',
          value: 'Android apps',
        },
        testDb
      );

      await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Swift',
          predicate: 'develops',
          value: 'iOS apps',
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
      );

      const manualOnly = await getPersonalKnowledge({ sourceType: 'manual' }, testDb);
      expect(manualOnly).toHaveLength(1);
      expect(manualOnly[0]?.subject).toBe('Kotlin');
      expect(manualOnly[0]?.sourceType).toBe('manual');

      const extractedOnly = await getPersonalKnowledge({ sourceType: 'extracted' }, testDb);
      expect(extractedOnly).toHaveLength(1);
      expect(extractedOnly[0]?.subject).toBe('Swift');
      expect(extractedOnly[0]?.sourceType).toBe('extracted');
    });

    it('should treat legacy records with undefined sourceType as extracted', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('legacy_source_type')));

      // Insert raw record mimicking legacy pre-Step 1 record
      const rawId = await testDb.personalKnowledge.add({
        category: 'projects',
        subject: 'Legacy Project',
        predicate: 'started in',
        value: '2025',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      } as PersonalKnowledgeItem);

      const legacyItem = await getPersonalKnowledgeById(rawId as number, testDb);
      expect(legacyItem?.sourceType).toBeUndefined();

      // Filter by extracted should match legacy record via fallback
      const extractedList = await getPersonalKnowledge({ sourceType: 'extracted' }, testDb);
      expect(extractedList.some((item) => item.id === rawId)).toBe(true);

      const manualList = await getPersonalKnowledge({ sourceType: 'manual' }, testDb);
      expect(manualList.some((item) => item.id === rawId)).toBe(false);
    });

    it('should ensure existing deleteWritingMessage continues to cascade to evidence and facts', async function () {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('cascade_writing_message')));

      const messageId = await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I have been studying machine learning models in depth.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        60000,
        testDb
      );

      expect(messageId).toBeDefined();

      const factId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Machine Learning',
          predicate: 'studies',
          value: 'deep learning architectures',
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
      );

      await saveKnowledgeEvidence(
        {
          knowledgeId: factId,
          messageId: messageId!,
          sourceExcerpt: 'I have been studying machine learning models in depth.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Delete writing message
      await deleteWritingMessage(messageId!, testDb);

      // Verify evidence is invalidated with source_message_deleted
      const evidence = await getKnowledgeEvidence(factId, testDb);
      expect(evidence[0]?.isValid).toBe(false);
      expect(evidence[0]?.invalidationReason).toBe('source_message_deleted');

      // Verify knowledge is archived
      const fact = await getPersonalKnowledgeById(factId, testDb);
      expect(fact?.status).toBe('archived');
    });
  });
});
