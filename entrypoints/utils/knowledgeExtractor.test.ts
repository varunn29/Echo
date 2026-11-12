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
  clearProcessedMessagesCache,
  extractKnowledgeFromMessage,
  getProcessedMessageIds,
  parseExtractionResponse,
  persistGroundedKnowledge,
  runKnowledgeExtraction,
  validateAndGroundItem,
  type ExtractSingleMessageResult,
  type ExtractionOutcome,
  type PersistGroundedKnowledgeResult,
  type RawExtractedItem,
  type ValidatedKnowledgeDraft,
} from './knowledgeExtractor';
import * as ollamaClient from './ollamaClient';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Knowledge Extraction Suite', function () {
  const openDatabases: (Dexie | EchoDatabase)[] = [];
  const dbNamesToDelete: string[] = [];

  function trackDb<T extends Dexie>(dbInstance: T): T {
    openDatabases.push(dbInstance);
    if (!dbNamesToDelete.includes(dbInstance.name)) {
      dbNamesToDelete.push(dbInstance.name);
    }
    return dbInstance;
  }

  beforeEach(async function () {
    await clearProcessedMessagesCache();
  });

  afterEach(async function () {
    await clearProcessedMessagesCache();
    vi.restoreAllMocks();

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
        // Ignore teardown cleanup errors
      }
    }
    dbNamesToDelete.length = 0;
  });

  describe('1. Grounding and Validation Logic', function () {
    const sourceMessage =
      'Recently I built an analytics dashboard, a data tool using React and Node.js.';

    it('should validate and ground authentic fact with verbatim excerpt', function () {
      const rawItem: RawExtractedItem = {
        category: 'projects',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: 'a data tool',
        contextScope: 'web development',
        sourceExcerpt: 'Recently I built an analytics dashboard, a data tool',
        confidence: 'high',
        importance: 'primary',
      };

      const result = validateAndGroundItem(rawItem, sourceMessage);
      expect(result).not.toBeNull();
      expect(result?.category).toBe('projects');
      expect(result?.subject).toBe('analytics dashboard');
      expect(result?.predicate).toBe('built');
      expect(result?.value).toBe('a data tool');
      expect(result?.sourceExcerpt).toBe(
        'Recently I built an analytics dashboard, a data tool'
      );
      expect(result?.confidence).toBe('high');
      expect(result?.importance).toBe('primary');
      expect(result?.stability).toBe('evolving');
    });

    it('should reject fabricated or hallucinated source excerpts', function () {
      const hallucinatedItem: RawExtractedItem = {
        category: 'skills',
        subject: 'Python',
        predicate: 'proficient_in',
        value: 'advanced Python',
        sourceExcerpt: 'I have 10 years of Python experience', // NOT in sourceMessage!
      };

      const result = validateAndGroundItem(hallucinatedItem, sourceMessage);
      expect(result).toBeNull();
    });

    it('should reject invalid categories', function () {
      const invalidCatItem: RawExtractedItem = {
        category: 'random_category',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: 'a platform',
        sourceExcerpt: 'built an analytics dashboard',
      };

      const result = validateAndGroundItem(invalidCatItem, sourceMessage);
      expect(result).toBeNull();
    });

    it('should reject items with missing required fields', function () {
      // Missing value
      const missingValue: RawExtractedItem = {
        category: 'projects',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: '   ',
        sourceExcerpt: 'built an analytics dashboard',
      };
      expect(validateAndGroundItem(missingValue, sourceMessage)).toBeNull();

      // Missing subject
      const missingSubject: RawExtractedItem = {
        category: 'projects',
        subject: '',
        predicate: 'built',
        value: 'platform',
        sourceExcerpt: 'built an analytics dashboard',
      };
      expect(validateAndGroundItem(missingSubject, sourceMessage)).toBeNull();

      // Missing excerpt
      const missingExcerpt: RawExtractedItem = {
        category: 'projects',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: 'platform',
        sourceExcerpt: '',
      };
      expect(validateAndGroundItem(missingExcerpt, sourceMessage)).toBeNull();
    });

    it('should default confidence to medium and importance to secondary', function () {
      const minimalItem: RawExtractedItem = {
        category: 'habits',
        subject: 'coding',
        predicate: 'uses',
        value: 'React and Node.js',
        sourceExcerpt: 'using React and Node.js',
      };

      const result = validateAndGroundItem(minimalItem, sourceMessage);
      expect(result).not.toBeNull();
      expect(result?.confidence).toBe('medium');
      expect(result?.importance).toBe('secondary');
    });

    it('should parse valid JSON response and return array of grounded drafts', function () {
      const jsonResponse = JSON.stringify({
        knowledge: [
          {
            category: 'projects',
            subject: 'analytics dashboard',
            predicate: 'built',
            value: 'data tool',
            sourceExcerpt: 'built an analytics dashboard, a data tool',
          },
          {
            category: 'skills',
            subject: 'backend',
            predicate: 'uses',
            value: 'Node.js',
            sourceExcerpt: 'using React and Node.js',
          },
        ],
      });

      const parsed = parseExtractionResponse(jsonResponse, sourceMessage);
      expect(parsed).toHaveLength(2);
      expect(parsed[0]?.subject).toBe('analytics dashboard');
      expect(parsed[1]?.subject).toBe('backend');
    });

    it('should handle empty knowledge list', function () {
      const jsonResponse = JSON.stringify({ knowledge: [] });
      const parsed = parseExtractionResponse(jsonResponse, sourceMessage);
      expect(parsed).toEqual([]);
    });

    it('should throw clear error on malformed JSON', function () {
      const badJson = 'Not JSON at all';
      expect(() => parseExtractionResponse(badJson, sourceMessage)).toThrow(
        'Failed to parse Ollama response as valid JSON.'
      );
    });

    it('should reject project limitations and workarounds mistakenly classified as skills', function () {
      const dashboardSample =
        'the main difficulty which I faced while building it is that I was not able to get the live data of events and so I have to build a static one where the events never end, the users only view simulations';

      const limitationSkill: RawExtractedItem = {
        category: 'skills',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: 'a static one where the events never end',
        sourceExcerpt:
          'so I have to build a static one where the events never end, the users only view simulations',
      };

      expect(validateAndGroundItem(limitationSkill, dashboardSample)).toBeNull();
    });

    it('should reject skills with project development verbs (built, created, developed)', function () {
      const devSkill: RawExtractedItem = {
        category: 'skills',
        subject: 'React and Node.js',
        predicate: 'built',
        value: 'a data tool',
        sourceExcerpt: 'built an analytics dashboard, a data tool',
      };

      expect(validateAndGroundItem(devSkill, sourceMessage)).toBeNull();
    });

    it('should reject goals with completed past actions', function () {
      const pastGoal: RawExtractedItem = {
        category: 'goals',
        subject: 'analytics dashboard',
        predicate: 'built',
        value: 'a data tool',
        sourceExcerpt: 'built an analytics dashboard, a data tool',
      };

      expect(validateAndGroundItem(pastGoal, sourceMessage)).toBeNull();
    });

    it('should reject generic pronouns as subjects', function () {
      const pronounItem: RawExtractedItem = {
        category: 'habits',
        subject: 'it',
        predicate: 'provides',
        value: 'concise explanation',
        sourceExcerpt: 'provides a concise explanation',
      };

      const msg = 'Whenever I encounter an error I use AI tools because it provides a concise explanation';
      expect(validateAndGroundItem(pronounItem, msg)).toBeNull();
    });

    it('should reject hallucinated values where content words do not appear in source text', function () {
      const hallucinatedValue: RawExtractedItem = {
        category: 'skills',
        subject: 'TypeScript',
        predicate: 'uses',
        value: 'cryptography and blockchain smart contracts',
        sourceExcerpt: 'using React and Node.js',
      };

      expect(validateAndGroundItem(hallucinatedValue, sourceMessage)).toBeNull();
    });

    it('should reject items whose sourceExcerpt comes from an unrelated clause and does not substantiate the claim', function () {
      const fullText =
        "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";

      // Mismatched extraction: claim about "today's time" paired with unrelated plan excerpt
      const mismatchedItem: RawExtractedItem = {
        category: 'states',
        subject: 'web development',
        predicate: 'feel that there is not much left for',
        value: "today's time",
        sourceExcerpt: 'so planning to move ahead and learn some latest technologies',
        confidence: 'high',
        importance: 'secondary',
      };

      expect(validateAndGroundItem(mismatchedItem, fullText)).toBeNull();
    });

    it('should accept items when the sourceExcerpt directly contains and substantiates the claim', function () {
      const fullText =
        "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";

      const correctlyGroundedItem: RawExtractedItem = {
        category: 'opinions',
        subject: 'web development',
        predicate: 'feels',
        value: 'not much left for web development',
        sourceExcerpt:
          "now I feel that there is not much left for web development in today's time",
        confidence: 'high',
        importance: 'secondary',
      };

      const result = validateAndGroundItem(correctlyGroundedItem, fullText);
      expect(result).not.toBeNull();
      expect(result?.subject).toBe('web development');
      expect(result?.value).toBe('not much left for web development');
    });

    it('should calibrate confidence for tentative goals and plans to medium', function () {
      const fullText =
        "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";

      const tentativeGoal: RawExtractedItem = {
        category: 'goals',
        subject: 'learning path',
        predicate: 'planning to learn',
        value: 'latest technologies',
        sourceExcerpt: 'so planning to move ahead and learn some latest technologies',
        confidence: 'high', // Model output high
        importance: 'secondary',
      };

      const result = validateAndGroundItem(tentativeGoal, fullText);
      expect(result).not.toBeNull();
      // Epistemic calibration: forward-looking plans/goals capped at medium
      expect(result?.confidence).toBe('medium');
    });

    it('should preserve high confidence for definitive completed facts', function () {
      const fullText =
        "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";

      const definitiveFact: RawExtractedItem = {
        category: 'skills',
        subject: 'frontend stack',
        predicate: 'learned',
        value: 'React',
        sourceExcerpt: 'I also learned React',
        confidence: 'high',
        importance: 'primary',
      };

      const result = validateAndGroundItem(definitiveFact, fullText);
      expect(result).not.toBeNull();
      expect(result?.confidence).toBe('high');
    });
  });

  describe('2. Atomic Persistence & Deduplication', function () {
    it('should persist a new knowledge item and linked evidence atomically', async function () {
      const dbName = createUniqueDbName('AtomicSave');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const messageId = 100;
      await testDb.writingMessages.put({
        id: messageId,
        platform: 'ChatGPT',
        text: 'building an extension called Echo',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      const draft: ValidatedKnowledgeDraft = {
        category: 'projects',
        subject: 'Echo Extension',
        predicate: 'building',
        value: 'Chrome MV3 AI assistant',
        contextScope: 'browser extension',
        sourceExcerpt: 'building an extension called Echo',
        confidence: 'high',
        importance: 'primary',
        stability: 'evolving',
      };

      const result = await persistGroundedKnowledge(draft, messageId, testDb);
      expect(result.isNewKnowledge).toBe(true);
      expect(result.abortedDueToDeletion).toBe(false);
      expect(result.knowledgeId).toBeTypeOf('number');

      // Verify personalKnowledge row
      const savedItem = await testDb.personalKnowledge.get(result.knowledgeId);
      expect(savedItem).toBeDefined();
      expect(savedItem?.subject).toBe('Echo Extension');
      expect(savedItem?.status).toBe('active');

      // Verify knowledgeEvidence row
      const evidence = await testDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(result.knowledgeId)
        .toArray();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.messageId).toBe(messageId);
      expect(evidence[0]?.sourceExcerpt).toBe(
        'building an extension called Echo'
      );
      expect(evidence[0]?.isValid).toBe(true);
    });

    it('should abort persistence when the source message does not exist in writingMessages', async function () {
      const dbName = createUniqueDbName('MissingSourceMessage');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const draft: ValidatedKnowledgeDraft = {
        category: 'projects',
        subject: 'Ghost Project',
        predicate: 'created',
        value: 'does not exist',
        sourceExcerpt: 'created a ghost project',
        confidence: 'high',
        importance: 'primary',
        stability: 'evolving',
      };

      // ID 9999 was never saved in writingMessages
      const result = await persistGroundedKnowledge(draft, 9999, testDb);
      expect(result.abortedDueToDeletion).toBe(true);
      expect(result.isNewKnowledge).toBe(false);
      expect(result.knowledgeId).toBe(0);

      // Verify nothing was inserted in personalKnowledge or knowledgeEvidence
      expect(await testDb.personalKnowledge.count()).toBe(0);
      expect(await testDb.knowledgeEvidence.count()).toBe(0);
    });

    it('should deduplicate existing active knowledge and link new evidence without duplicate knowledge', async function () {
      const dbName = createUniqueDbName('DeduplicationTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      await testDb.writingMessages.put({
        id: 1,
        platform: 'ChatGPT',
        text: 'I am proficient in TypeScript for frontend',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });
      await testDb.writingMessages.put({
        id: 2,
        platform: 'ChatGPT',
        text: 'TypeScript is my daily language',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      const draft1: ValidatedKnowledgeDraft = {
        category: 'skills',
        subject: 'TypeScript',
        predicate: 'proficient_in',
        value: 'frontend architecture',
        sourceExcerpt: 'I am proficient in TypeScript for frontend',
        confidence: 'high',
        importance: 'primary',
        stability: 'stable',
      };

      // Message 1 creates the item
      const res1 = await persistGroundedKnowledge(draft1, 1, testDb);
      expect(res1.isNewKnowledge).toBe(true);
      expect(res1.abortedDueToDeletion).toBe(false);

      // Message 2 mentions the exact same fact (case-insensitive subject/predicate match)
      const draft2: ValidatedKnowledgeDraft = {
        category: 'skills',
        subject: ' typescript ',
        predicate: ' PROFICIENT_IN ',
        value: 'frontend architecture updated',
        sourceExcerpt: 'TypeScript is my daily language',
        confidence: 'high',
        importance: 'secondary',
        stability: 'stable',
      };

      const res2 = await persistGroundedKnowledge(draft2, 2, testDb);
      expect(res2.isNewKnowledge).toBe(false);
      expect(res2.abortedDueToDeletion).toBe(false);
      expect(res2.knowledgeId).toBe(res1.knowledgeId);

      // Total knowledge count must still be 1
      const allKnowledge = await testDb.personalKnowledge.toArray();
      expect(allKnowledge).toHaveLength(1);

      // Both evidence records must be linked to the single knowledge item
      const allEvidence = await testDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(res1.knowledgeId)
        .toArray();
      expect(allEvidence).toHaveLength(2);
      expect(allEvidence[0]?.messageId).toBe(1);
      expect(allEvidence[1]?.messageId).toBe(2);
    });

    it('should not duplicate evidence if the same message is processed twice', async function () {
      const dbName = createUniqueDbName('DuplicateEvidenceTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      await testDb.writingMessages.put({
        id: 5,
        platform: 'ChatGPT',
        text: 'Whenever I am stuck I use AI',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      const draft: ValidatedKnowledgeDraft = {
        category: 'habits',
        subject: 'Problem Solving',
        predicate: 'uses_ai',
        value: 'when stuck',
        sourceExcerpt: 'Whenever I am stuck I use AI',
        confidence: 'medium',
        importance: 'secondary',
        stability: 'stable',
      };

      // Process message 5 first time
      const res1 = await persistGroundedKnowledge(draft, 5, testDb);
      expect(res1.isNewKnowledge).toBe(true);

      // Process message 5 second time
      const res2 = await persistGroundedKnowledge(draft, 5, testDb);
      expect(res2.isNewKnowledge).toBe(false);

      const evidenceCount = await testDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(res1.knowledgeId)
        .count();
      expect(evidenceCount).toBe(1); // Evidence not duplicated!
    });

    it('should never increase confidence or importance when corroborating repeated mentions', async function () {
      const dbName = createUniqueDbName('ConfidencePreservation');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      await testDb.writingMessages.put({
        id: 10,
        platform: 'ChatGPT',
        text: 'I am learning Python for data science',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });
      await testDb.writingMessages.put({
        id: 11,
        platform: 'ChatGPT',
        text: 'learning Python for data science',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      const draft1: ValidatedKnowledgeDraft = {
        category: 'skills',
        subject: 'Python',
        predicate: 'learning',
        value: 'data science fundamentals',
        sourceExcerpt: 'I am learning Python for data science',
        confidence: 'medium', // tentative/learning
        importance: 'secondary',
        stability: 'stable',
      };

      // Message 1 saves initial item with medium confidence
      const res1 = await persistGroundedKnowledge(draft1, 10, testDb);
      const item1 = await testDb.personalKnowledge.get(res1.knowledgeId);
      expect(item1?.confidence).toBe('medium');
      expect(item1?.importance).toBe('secondary');

      // Message 2 repeats the exact same fact with another draft
      const draft2: ValidatedKnowledgeDraft = {
        ...draft1,
        sourceExcerpt: 'learning Python for data science',
        confidence: 'high', // model returns high, but repeated mention must NOT inflate stored confidence
        importance: 'primary',
      };

      const res2 = await persistGroundedKnowledge(draft2, 11, testDb);
      expect(res2.isNewKnowledge).toBe(false);

      const itemAfterCorroboration = await testDb.personalKnowledge.get(res1.knowledgeId);
      // Stored confidence and importance must remain as originally assigned based on the source fact
      expect(itemAfterCorroboration?.confidence).toBe('medium');
      expect(itemAfterCorroboration?.importance).toBe('secondary');
    });

    it('should check evidence inside Dexie transaction before inserting to guarantee idempotency', async function () {
      const dbName = createUniqueDbName('TransactionIdempotency');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      await testDb.writingMessages.put({
        id: 20,
        platform: 'ChatGPT',
        text: 'created an interactive portfolio website',
        createdAt: new Date(),
        capturedAt: new Date(),
        trainingStatus: 'pending',
      });

      const draft: ValidatedKnowledgeDraft = {
        category: 'projects',
        subject: 'Portfolio',
        predicate: 'created',
        value: 'interactive portfolio website',
        sourceExcerpt: 'created an interactive portfolio website',
        confidence: 'high',
        importance: 'primary',
        stability: 'evolving',
      };

      // Call persistGroundedKnowledge for message 20 twice
      const r1 = await persistGroundedKnowledge(draft, 20, testDb);
      const r2 = await persistGroundedKnowledge(draft, 20, testDb);

      expect(r1.isNewKnowledge).toBe(true);
      expect(r2.isNewKnowledge).toBe(false);

      // Verify only 1 knowledge item and 1 evidence record exist
      const items = await testDb.personalKnowledge.toArray();
      const evidence = await testDb.knowledgeEvidence.toArray();
      expect(items).toHaveLength(1);
      expect(evidence).toHaveLength(1);
    });
  });

  describe('3. End-to-End Extraction Pipeline (Mocked Ollama)', function () {
    it('should pre-filter code and run extraction on eligible prose messages', async function () {
      const dbName = createUniqueDbName('FullPipeline');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Seed messages
      const msg1Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Recently I built an analytics dashboard, a data tool using React.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Message 2: Pure source code -> should be skipped by classifier
      const msg2Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: '```typescript\nfunction sum(a: number, b: number) { return a + b; }\n```',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Mock Ollama status & chat completion
      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'analytics dashboard',
              predicate: 'built',
              value: 'a data tool using React',
              sourceExcerpt: 'Recently I built an analytics dashboard, a data tool',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      const summary = await runKnowledgeExtraction({
        targetDb: testDb,
        forceAll: true,
      });

      expect(summary.totalAnalyzed).toBe(2);
      expect(summary.candidateCount).toBe(1); // msg1 was candidate
      expect(summary.skippedCount).toBe(1); // msg2 was skipped
      expect(summary.extractedCount).toBe(1);

      // Verify knowledge was persisted
      const saved = await testDb.personalKnowledge.toArray();
      expect(saved).toHaveLength(1);
      expect(saved[0]?.subject).toBe('analytics dashboard');

      // Verify evidence provenance
      const evidence = await testDb.knowledgeEvidence.toArray();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.messageId).toBe(msg1Id);
    });

    it('should fail with clear error when Ollama is offline', async function () {
      const dbName = createUniqueDbName('OfflineTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: false,
        isModelReady: false,
        modelName: 'qwen3.5:2b',
        error: 'Ollama is offline.',
      });

      await expect(
        runKnowledgeExtraction({
          targetDb: testDb,
        })
      ).rejects.toThrow('Ollama is offline.');
    });

    it('should fail with clear error when model is not available', async function () {
      const dbName = createUniqueDbName('ModelMissingTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: false,
        modelName: 'qwen3.5:2b',
        error: 'Model "qwen3.5:2b" is not installed in Ollama.',
      });

      await expect(
        runKnowledgeExtraction({
          targetDb: testDb,
        })
      ).rejects.toThrow('Model "qwen3.5:2b" is not installed in Ollama.');
    });

    it('should maintain privacy deletion behavior for newly extracted knowledge', async function () {
      const dbName = createUniqueDbName('PrivacyWithExtraction');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am struggling with complex backend state architecture.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const draft: ValidatedKnowledgeDraft = {
        category: 'states',
        subject: 'backend state architecture',
        predicate: 'struggling_with',
        value: 'complex state management',
        sourceExcerpt: 'I am struggling with complex backend state architecture',
        confidence: 'medium',
        importance: 'secondary',
        stability: 'temporary',
      };

      const result = await persistGroundedKnowledge(draft, msgId, testDb);

      // Verify created
      expect(await testDb.personalKnowledge.get(result.knowledgeId)).toBeDefined();

      // Now execute message deletion
      await deleteWritingMessage(msgId, testDb);

      // Verify message deleted
      expect(await testDb.writingMessages.get(msgId)).toBeUndefined();

      // Verify knowledge content was cleared and status archived
      const archived = await testDb.personalKnowledge.get(result.knowledgeId);
      expect(archived?.status).toBe('archived');
      expect(archived?.subject).toBe('');
      expect(archived?.value).toBe('');

      // Verify evidence excerpt and note were scrubbed
      const ev = await testDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(result.knowledgeId)
        .first();
      expect(ev?.isValid).toBe(false);
      expect(ev?.sourceExcerpt).toBe('');
      expect(ev?.relevanceNote).toBe('');
    });

    it('should track empty-result messages distinctly from failed extractions so failures can be retried', async function () {
      const dbName = createUniqueDbName('EmptyVsFailed');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      // Seed candidate 1: personal prose candidate that passes pre-filter but yields no personal facts from LLM
      const msg1Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'In my opinion technology has evolved rapidly, but it is hard to predict what will happen next.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      // Seed candidate 2: valid fact, but extraction will fail (e.g. timeout)
      const msg2Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built an e-commerce platform using Next.js and Prisma.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      // Mock queryOllamaJson: msg1 returns empty knowledge, msg2 throws an error
      vi.spyOn(ollamaClient, 'queryOllamaJson').mockImplementation(async (text) => {
        if (text.toLowerCase().includes('in my opinion technology')) {
          return JSON.stringify({ knowledge: [] });
        }
        throw new Error('Connection timeout to Ollama daemon');
      });

      const summary = await runKnowledgeExtraction({
        targetDb: testDb,
        forceAll: true,
      });

      expect(summary.totalAnalyzed).toBe(2);
      expect(summary.emptyResultCount).toBe(1); // msg1 was empty result
      expect(summary.errors).toHaveLength(1); // msg2 failed
      expect(summary.errors[0]).toContain('Connection timeout');

      // msg1 should be recorded in storage as processed
      // msg2 failed, so it was NOT recorded and can be retried
      const storageIds = await getProcessedMessageIds();
      expect(storageIds.has(msg1Id)).toBe(true);
      expect(storageIds.has(msg2Id)).toBe(false);
    });

    it('should recover gracefully and remain idempotent if storage was not recorded after Dexie write', async function () {
      const dbName = createUniqueDbName('RecoveryFromInterruption');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Recently I built an analytics dashboard, a data tool using React.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'analytics dashboard',
              predicate: 'built',
              value: 'a data tool using React',
              sourceExcerpt: 'Recently I built an analytics dashboard, a data tool',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      // Run 1: processes msgId and persists in Dexie
      const summary1 = await runKnowledgeExtraction({
        targetDb: testDb,
      });
      expect(summary1.extractedCount).toBe(1);

      // Simulate interruption where storage cache was cleared / lost
      await clearProcessedMessagesCache();
      expect((await getProcessedMessageIds()).size).toBe(0);

      // Run 2: Even with storage cache cleared, Dexie knowledgeEvidence identifies msgId as already processed!
      const summary2 = await runKnowledgeExtraction({
        targetDb: testDb,
      });

      // No new extraction occurred because Dexie knowledgeEvidence deduplicated it!
      expect(summary2.candidateCount).toBe(0);
      expect(summary2.extractedCount).toBe(0);

      // Knowledge and evidence rows in Dexie are not duplicated
      expect(await testDb.personalKnowledge.count()).toBe(1);
      expect(await testDb.knowledgeEvidence.count()).toBe(1);
    });
  });

  describe('4. Extraction V2 Accuracy Improvements (16 Audit Scenarios & Regression Suite)', function () {
    const msg52 =
      'ok then we can go with this proposed plan then, and when are we going to start building now';
    const msg97 =
      "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";
    const msg98 =
      'I have not yet decided but planning to try with AI and some ML, also with cloud computing, will think about other technologies in the coming future';
    const msg108 =
      'so will we have to do it everytime even when our project is  complete';
    const msg157 =
      'In technical discussions, if I am confident in the design then I will advocate my perspective, but if others have more context then I listen to theirs as well';
    const msg158 =
      "I have thought carefully about it but it's not really simple to come up with a novel product idea, I don't know about any such product idea yet";
    const msg160 =
      'One thing that really frustrates me is premature microservice complexity when a clean monolithic architecture would be much simpler and more maintainable';
    const msg161 =
      'Recently I built the analytics dashboard, a data tool and I built it because I was applying to a software company and it was my first complete project, the main difficulty which I faced while building it is that I was not able to get the live data of events and so I have to build a static one where the events never end, the users only view simulations';
    const msg162 =
      'Right now I am currently learning Rust syntax and exploring memory management models for system programming';

    describe('A. Procedural Conversation Question Rejection', function () {
      it('should reject procedural question: "when are we going to start building now" (#9 / Msg #52)', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'planning to',
          value: 'start building',
          sourceExcerpt:
            'ok then we can go with this proposed plan then, and when are we going to start building now',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(item, msg52);
        expect(result).toBeNull();
      });

      it('should reject procedural question: "so will we have to do it everytime even when our project is complete" (#16 / Msg #108)', function () {
        const item: RawExtractedItem = {
          category: 'projects',
          subject: 'project',
          predicate: 'will be done',
          value: 'every time',
          sourceExcerpt:
            'so will we have to do it everytime even when our project is  complete',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg108);
        expect(result).toBeNull();
      });

      it('should reject conversational planning question with question mark', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'planning to',
          value: 'start building',
          sourceExcerpt: 'When are we going to start building now?',
        };
        expect(
          validateAndGroundItem(item, 'When are we going to start building now?')
        ).toBeNull();
      });

      it('should reject workflow coordination query without personal fact', function () {
        const text = 'What should we do next for our database schema?';
        const item: RawExtractedItem = {
          category: 'projects',
          subject: 'database schema',
          predicate: 'should do',
          value: 'next steps',
          sourceExcerpt: 'What should we do next for our database schema?',
        };
        expect(validateAndGroundItem(item, text)).toBeNull();
      });
    });

    describe('B. Tautological Triples Rejection', function () {
      it('should reject tautological triples where subject equals value: "HTML | learned | HTML" (#10 / Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'HTML',
          predicate: 'learned',
          value: 'HTML',
          sourceExcerpt: 'So I started by learning HTML, CSS and JavaScript',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).toBeNull();
      });

      it('should reject tautological triples for JavaScript: "JavaScript | learned | JavaScript" (#11 / Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'JavaScript',
          predicate: 'learned',
          value: 'JavaScript',
          sourceExcerpt: 'So I started by learning HTML, CSS and JavaScript',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).toBeNull();
      });

      it('should reject tautological triples for React: "React | learned | React" (#12 / Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'React',
          predicate: 'learned',
          value: 'React',
          sourceExcerpt:
            'for that I also learned React, typescript, tailwind, node js, express js, postgres',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).toBeNull();
      });

      it('should reject tautological triples for goals: "latest technologies | planning to learn | latest technologies" (#5 / Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'latest technologies',
          predicate: 'planning to learn',
          value: 'latest technologies',
          sourceExcerpt:
            'so planning to move ahead and learn some latest technologies',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).toBeNull();
      });
    });

    describe('C. Vague Conversational Filler Rejection', function () {
      it('should reject vague filler values: "user | want to learn | many things" (#7 / Msg #162)', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'want to learn',
          value: 'many things',
          sourceExcerpt: 'I want to learn many things',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(item, msg162);
        expect(result).toBeNull();
      });

      it('should reject vague filler values: "user | will think about | other technologies in the coming future" (#15 / Msg #98)', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'will think about',
          value: 'other technologies in the coming future',
          sourceExcerpt:
            'will think about other technologies in the coming future',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(item, msg98);
        expect(result).toBeNull();
      });

      it('should reject non-informative filler: "user | thought too much | about a product idea" (#3 / Msg #158)', function () {
        const item: RawExtractedItem = {
          category: 'states',
          subject: 'user',
          predicate: 'thought too much',
          value: 'about a product idea',
          sourceExcerpt: 'I have thought too much about it',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg158);
        expect(result).toBeNull();
      });

      it('should reject trivial filler like "this and that" or "something"', function () {
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'building',
          value: 'this and that',
          sourceExcerpt: 'building this and that',
        };
        expect(validateAndGroundItem(item, msg162)).toBeNull();
      });
    });

    describe('D. Skills Completeness & Non-Inflation', function () {
      it('should preserve complete 9-technology stack when extracted from learning journey (Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'user',
          predicate: 'learned',
          value:
            'HTML, CSS, JavaScript, React, TypeScript, Tailwind, Node.js, Express.js, PostgreSQL',
          sourceExcerpt:
            'So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres',
          confidence: 'high',
          importance: 'primary',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('skills');
        expect(result?.subject).toBe('user');
        expect(result?.predicate).toBe('learned');
        expect(result?.value).toBe(
          'HTML, CSS, JavaScript, React, TypeScript, Tailwind, Node.js, Express.js, PostgreSQL'
        );
        // Excerpt contains all learning tokens; no unwarranted inflation to proficient_in
        expect(result?.confidence).toBe('medium');
        expect(result?.stability).toBe('stable');
      });

      it('should preserve complete technology list when multiple skills are listed in learning path (Msg #97)', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'web development stack',
          predicate: 'learned',
          value:
            'HTML, CSS, JavaScript, React, typescript, tailwind, node js, express js, postgres',
          sourceExcerpt:
            'So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres',
          confidence: 'high',
          importance: 'primary',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('skills');
        expect(result?.subject).toBe('web development stack');
        expect(result?.predicate).toBe('learned');
        expect(result?.value).toContain('HTML, CSS, JavaScript');
        expect(result?.value).toContain('React, typescript');
        expect(result?.value).toContain('postgres');
        expect(result?.confidence).toBe('medium');
        expect(result?.stability).toBe('stable');
      });

      it('should not inflate "learned" to "proficient_in" or "expert" unless user explicitly claimed expertise', function () {
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'web technologies',
          predicate: 'proficient_in',
          value: 'React and Node.js',
          sourceExcerpt:
            'for that I also learned React, typescript, tailwind, node js',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg97);
        expect(result).not.toBeNull();
        // Automatically normalized from "proficient_in" to "learned" because excerpt does not mention proficiency/expertise!
        expect(result?.predicate).toBe('learned');
      });

      it('should permit "proficient_in" when user explicitly states proficiency or expertise', function () {
        const expertText =
          'I am highly proficient in TypeScript and design patterns after years of production experience.';
        const item: RawExtractedItem = {
          category: 'skills',
          subject: 'TypeScript',
          predicate: 'proficient_in',
          value: 'design patterns and production architecture',
          sourceExcerpt:
            'I am highly proficient in TypeScript and design patterns',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, expertText);
        expect(result).not.toBeNull();
        expect(result?.predicate).toBe('proficient_in');
      });
    });

    describe('E. Project Constraints & Limitations Preservation', function () {
      it('should preserve technical constraints and workarounds under projects', function () {
        const item: RawExtractedItem = {
          category: 'projects',
          subject: 'analytics dashboard',
          predicate: 'built',
          value:
            'a static platform where events never end because live event data was unavailable',
          sourceExcerpt:
            'the main difficulty which I faced while building it is that I was not able to get the live data of events and so I have to build a static one where the events never end',
          confidence: 'high',
          importance: 'primary',
        };
        const result = validateAndGroundItem(item, msg161);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('projects');
        expect(result?.subject).toBe('analytics dashboard');
        expect(result?.value).toContain('events never end');
        expect(result?.confidence).toBe('high');
        expect(result?.stability).toBe('evolving');
      });
    });

    describe('F. Category Semantic Calibration & Stability', function () {
      it('should reclassify debate/discussion approaches from goals to habits', function () {
        const item: RawExtractedItem = {
          category: 'goals', // Model mistakenly returned goals
          subject: 'technical discussions',
          predicate: 'communicates',
          value:
            'shares architectural perspective when confident and listens to other viewpoints',
          sourceExcerpt:
            'In technical discussions, if I am confident in the design then I will advocate my perspective, but if others have more context then I listen to theirs as well',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg157);
        expect(result).not.toBeNull();
        // Reclassified to habits!
        expect(result?.category).toBe('habits');
        expect(result?.stability).toBe('stable');
      });

      it('should reclassify durable technical convictions from states to opinions', function () {
        const item: RawExtractedItem = {
          category: 'states', // Model mistakenly returned states
          subject: 'system architecture',
          predicate: 'frustrated_by',
          value:
            'premature microservice complexity compared to a well-structured monolith',
          sourceExcerpt:
            'One thing that really frustrates me is premature microservice complexity when a clean monolithic architecture would be much simpler and more maintainable',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg160);
        expect(result).not.toBeNull();
        // Reclassified to opinions!
        expect(result?.category).toBe('opinions');
        expect(result?.stability).toBe('stable');
      });

      it('should preserve valid temporary states as temporary stability', function () {
        const item: RawExtractedItem = {
          category: 'states',
          subject: 'learning focus',
          predicate: 'currently exploring',
          value: 'Rust syntax and memory management models',
          sourceExcerpt:
            'Right now I am currently learning Rust syntax and exploring memory management models',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, msg162);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('states');
        expect(result?.stability).toBe('temporary');
      });

      it('should NOT force explicit goals into habits simply because they contain "debate"', function () {
        const text = 'My goal is to participate in a debate tournament.';
        const item: RawExtractedItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'participate in',
          value: 'a debate tournament',
          sourceExcerpt: 'My goal is to participate in a debate tournament',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(item, text);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('goals'); // Must remain goals!
        expect(result?.stability).toBe('evolving');
      });

      it('should NOT force ephemeral compiler error frustrations into opinions', function () {
        const text = 'This compiler error frustrates me.';
        const item: RawExtractedItem = {
          category: 'states',
          subject: 'user',
          predicate: 'feels frustrated by',
          value: 'this compiler error',
          sourceExcerpt: 'This compiler error frustrates me',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, text);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('states'); // Must remain states!
        expect(result?.stability).toBe('temporary');
      });

      it('should preserve durable tutorial critique when extracted as opinions', function () {
        const text = 'I dislike tutorials that skip explaining why.';
        const item: RawExtractedItem = {
          category: 'opinions',
          subject: 'programming tutorials',
          predicate: 'dislikes',
          value: 'tutorials that skip explaining why',
          sourceExcerpt: 'I dislike tutorials that skip explaining why',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, text);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('opinions');
        expect(result?.stability).toBe('stable');
      });

      it('should preserve durable tutorial critique when extracted as preferences', function () {
        const text = 'I dislike tutorials that skip explaining why.';
        const item: RawExtractedItem = {
          category: 'preferences',
          subject: 'learning preference',
          predicate: 'dislikes',
          value: 'tutorials that skip explaining why',
          sourceExcerpt: 'I dislike tutorials that skip explaining why',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, text);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('preferences');
        expect(result?.stability).toBe('stable');
      });

      it('should recalibrate tutorial critique from states to opinions when phrased as frustration', function () {
        const text = 'It frustrates me when tutorials skip explaining why.';
        const item: RawExtractedItem = {
          category: 'states',
          subject: 'programming tutorials',
          predicate: 'frustrated_by',
          value: 'tutorials that skip explaining why',
          sourceExcerpt: 'It frustrates me when tutorials skip explaining why',
          confidence: 'high',
        };
        const result = validateAndGroundItem(item, text);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('opinions'); // Recalibrated to opinions!
        expect(result?.stability).toBe('stable');
      });
    });

    describe('G. Mixed Declarative Statements & Procedural Questions', function () {
      const mixedProjectMsg =
        'The main difficulty which I faced while building the analytics platform was not being able to get live event data, so I built a static simulation. Will we have to do it every time even when our project is complete?';

      it('should preserve genuine personal difficulty & workaround from mixed message', function () {
        const item: RawExtractedItem = {
          category: 'projects',
          subject: 'analytics platform',
          predicate: 'built',
          value:
            'static simulation because live event data was unavailable',
          sourceExcerpt:
            'The main difficulty which I faced while building the analytics platform was not being able to get live event data, so I built a static simulation',
          confidence: 'high',
          importance: 'primary',
        };
        const result = validateAndGroundItem(item, mixedProjectMsg);
        expect(result).not.toBeNull();
        expect(result?.category).toBe('projects');
        expect(result?.subject).toBe('analytics platform');
        expect(result?.predicate).toBe('built');
        expect(result?.value).toContain('static simulation');
        expect(result?.stability).toBe('evolving');
      });

      it('should reject procedural question extracted from mixed message', function () {
        const questionItem: RawExtractedItem = {
          category: 'projects',
          subject: 'the project',
          predicate: 'will have to do',
          value: 'every time when complete',
          sourceExcerpt:
            'Will we have to do it every time even when our project is complete?',
          confidence: 'medium',
        };
        const result = validateAndGroundItem(questionItem, mixedProjectMsg);
        expect(result).toBeNull();
      });

      it('should reject pure procedural question without declarative fact', function () {
        const pureQuestion =
          'Will we have to do it every time even when our project is complete?';
        const questionItem: RawExtractedItem = {
          category: 'projects',
          subject: 'project',
          predicate: 'will have to do',
          value: 'every time',
          sourceExcerpt:
            'Will we have to do it every time even when our project is complete?',
        };
        expect(validateAndGroundItem(questionItem, pureQuestion)).toBeNull();
      });
    });
  });

  describe('5. Deletion Race & Concurrent Operations', function () {
    it('should abort persistence and create zero knowledge/evidence when message is deleted during Ollama inference', async function () {
      const dbName = createUniqueDbName('DeletionDuringInference');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Recently I built an automated reporting engine using Python.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      let resolveOllama!: (val: string) => void;
      const ollamaPromise = new Promise<string>((resolve) => {
        resolveOllama = resolve;
      });

      const ollamaSpy = vi.spyOn(ollamaClient, 'queryOllamaJson').mockReturnValue(ollamaPromise);

      const savedMsg = (await testDb.writingMessages.get(msgId))!;
      const extractPromise = extractKnowledgeFromMessage(savedMsg, { targetDb: testDb });

      // While Ollama is still processing, the user deletes the source message
      await deleteWritingMessage(msgId, testDb);
      expect(await testDb.writingMessages.get(msgId)).toBeUndefined();

      // Now Ollama returns with a valid extracted knowledge payload
      resolveOllama(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'automated reporting engine',
              predicate: 'built',
              value: 'using Python',
              sourceExcerpt: 'Recently I built an automated reporting engine using Python',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      const result = await extractPromise;
      expect(result.outcome).toBe('aborted_deleted');
      expect(result.extractedCount).toBe(0);
      expect(result.duplicateCount).toBe(0);

      // Verify no knowledge items and no evidence rows exist
      expect(await testDb.personalKnowledge.count()).toBe(0);
      expect(await testDb.knowledgeEvidence.count()).toBe(0);
      expect(ollamaSpy).toHaveBeenCalled();
    });

    it('should scrub evidence and archive knowledge when source message is deleted after persistence', async function () {
      const dbName = createUniqueDbName('DeletionAfterPersistence');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I designed a distributed cache in Rust for low latency.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'distributed cache',
              predicate: 'designed',
              value: 'in Rust for low latency',
              sourceExcerpt: 'I designed a distributed cache in Rust for low latency',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      const savedMsg = (await testDb.writingMessages.get(msgId))!;
      const result = await extractKnowledgeFromMessage(savedMsg, { targetDb: testDb });

      expect(result.outcome).toBe('extracted');
      expect(result.extractedCount).toBe(1);

      // Verify knowledge and evidence were created
      const knowledgeItems = await testDb.personalKnowledge.toArray();
      expect(knowledgeItems).toHaveLength(1);
      const kId = knowledgeItems[0]!.id!;
      const evidenceRows = await testDb.knowledgeEvidence.where('knowledgeId').equals(kId).toArray();
      expect(evidenceRows).toHaveLength(1);

      // Now delete message
      await deleteWritingMessage(msgId, testDb);

      // Verify message deleted
      expect(await testDb.writingMessages.get(msgId)).toBeUndefined();

      // Verify knowledge archived and scrubbed
      const archivedK = await testDb.personalKnowledge.get(kId);
      expect(archivedK?.status).toBe('archived');
      expect(archivedK?.subject).toBe('');
      expect(archivedK?.value).toBe('');

      // Verify evidence invalidated and scrubbed
      const scrubbedEv = await testDb.knowledgeEvidence.where('knowledgeId').equals(kId).first();
      expect(scrubbedEv?.isValid).toBe(false);
      expect(scrubbedEv?.sourceExcerpt).toBe('');
      expect(scrubbedEv?.relevanceNote).toBe('');
    });
  });

  describe('6. Reusable extractKnowledgeFromMessage()', function () {
    it('should require a persisted message with a valid numeric id', async function () {
      await expect(
        extractKnowledgeFromMessage({
          platform: 'ChatGPT',
          text: 'I am learning Go for systems programming.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        })
      ).rejects.toThrow(
        'extractKnowledgeFromMessage requires a persisted message with a valid numeric id.'
      );
    });

    it('should return aborted_deleted if source message does not exist before calling Ollama', async function () {
      const dbName = createUniqueDbName('NonExistentPreCheck');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const ollamaSpy = vi.spyOn(ollamaClient, 'queryOllamaJson');

      const result = await extractKnowledgeFromMessage(
        {
          id: 8888,
          platform: 'ChatGPT',
          text: 'I built an iOS app using Swift.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        { targetDb: testDb }
      );

      expect(result.outcome).toBe('aborted_deleted');
      expect(result.extractedCount).toBe(0);
      expect(result.duplicateCount).toBe(0);
      // Ollama should never have been invoked
      expect(ollamaSpy).not.toHaveBeenCalled();
    });

    it('should extract valid grounded knowledge and return outcome extracted with counts', async function () {
      const dbName = createUniqueDbName('ValidSingleExtract');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built a modern CLI tool in Go for container management.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'CLI tool',
              predicate: 'built',
              value: 'in Go for container management',
              sourceExcerpt: 'I built a modern CLI tool in Go for container management',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      const msg = (await testDb.writingMessages.get(msgId))!;
      const result = await extractKnowledgeFromMessage(msg, { targetDb: testDb });

      expect(result.outcome).toBe('extracted');
      expect(result.extractedCount).toBe(1);
      expect(result.duplicateCount).toBe(0);

      const items = await testDb.personalKnowledge.toArray();
      expect(items).toHaveLength(1);
      expect(items[0]?.subject).toBe('CLI tool');
    });

    it('should handle empty knowledge response and return outcome empty', async function () {
      const dbName = createUniqueDbName('EmptySingleExtract');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'Technology today changes fast and nobody knows what is next.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [],
        })
      );

      const msg = (await testDb.writingMessages.get(msgId))!;
      const result = await extractKnowledgeFromMessage(msg, { targetDb: testDb });

      expect(result.outcome).toBe('empty');
      expect(result.extractedCount).toBe(0);
      expect(result.duplicateCount).toBe(0);
      expect(await testDb.personalKnowledge.count()).toBe(0);
    });

    it('should handle duplicate knowledge response and increment duplicateCount', async function () {
      const dbName = createUniqueDbName('DuplicateSingleExtract');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msg1Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I learned React for building interactive frontend apps.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      const msg2Id = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I also learned React for frontend earlier this year.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockImplementation(async (prompt) => {
        if (prompt.includes('earlier this year')) {
          return JSON.stringify({
            knowledge: [
              {
                category: 'skills',
                subject: 'React',
                predicate: 'learned',
                value: 'for frontend',
                sourceExcerpt: 'I also learned React for frontend earlier this year',
                confidence: 'high',
                importance: 'primary',
              },
            ],
          });
        }
        return JSON.stringify({
          knowledge: [
            {
              category: 'skills',
              subject: 'React',
              predicate: 'learned',
              value: 'for frontend',
              sourceExcerpt: 'I learned React for building interactive frontend apps',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        });
      });

      const msg1 = (await testDb.writingMessages.get(msg1Id))!;
      const res1 = await extractKnowledgeFromMessage(msg1, { targetDb: testDb });
      expect(res1.outcome).toBe('extracted');
      expect(res1.extractedCount).toBe(1);
      expect(res1.duplicateCount).toBe(0);

      const msg2 = (await testDb.writingMessages.get(msg2Id))!;
      const res2 = await extractKnowledgeFromMessage(msg2, { targetDb: testDb });
      expect(res2.outcome).toBe('extracted');
      expect(res2.extractedCount).toBe(0);
      expect(res2.duplicateCount).toBe(1); // Corroborated!

      // Only 1 knowledge item exists
      expect(await testDb.personalKnowledge.count()).toBe(1);
      // 2 evidence links exist
      expect(await testDb.knowledgeEvidence.count()).toBe(2);
    });

    it('should propagate Ollama network or timeout errors to the caller', async function () {
      const dbName = createUniqueDbName('OllamaErrorPropagate');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I am building an embedded firmware driver in C.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockRejectedValue(
        new Error('Ollama request timed out after 60s.')
      );

      const msg = (await testDb.writingMessages.get(msgId))!;
      await expect(
        extractKnowledgeFromMessage(msg, { targetDb: testDb })
      ).rejects.toThrow('Ollama request timed out after 60s.');
    });
  });

  describe('7. runKnowledgeExtraction() Compatibility & Processed Cache', function () {
    it('should mark successfully evaluated empty-result messages as processed in storage', async function () {
      const dbName = createUniqueDbName('EmptyResultProcessed');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'In my opinion software is always evolving, which is very interesting to observe.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [],
        })
      );

      const summary = await runKnowledgeExtraction({ targetDb: testDb });
      expect(summary.emptyResultCount).toBe(1);
      expect(summary.extractedCount).toBe(0);

      // Successfully evaluated empty results must be marked processed
      const processed = await getProcessedMessageIds();
      expect(processed.has(msgId)).toBe(true);
    });

    it('should NOT mark failed extraction messages as processed so they can be retried', async function () {
      const dbName = createUniqueDbName('FailedNotProcessed');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built an inventory service using Go and MongoDB.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockRejectedValue(
        new Error('Ollama request timed out after 60s.')
      );

      const summary = await runKnowledgeExtraction({ targetDb: testDb });
      expect(summary.errors).toHaveLength(1);
      expect(summary.errors[0]).toContain(`Message #${msgId}`);

      // Failed message must NOT be in processed cache
      const processed = await getProcessedMessageIds();
      expect(processed.has(msgId)).toBe(false);
    });

    it('should NOT mark messages aborted due to deletion as processed', async function () {
      const dbName = createUniqueDbName('AbortedDeletedNotProcessed');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I created a custom game engine in C++ from scratch.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      // While queryOllamaJson is called, delete the message
      vi.spyOn(ollamaClient, 'queryOllamaJson').mockImplementation(async () => {
        await deleteWritingMessage(msgId, testDb);
        return JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'game engine',
              predicate: 'created',
              value: 'in C++ from scratch',
              sourceExcerpt: 'I created a custom game engine in C++ from scratch',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        });
      });

      const summary = await runKnowledgeExtraction({ targetDb: testDb });
      expect(summary.skippedCount).toBe(1);
      expect(summary.extractedCount).toBe(0);

      // Aborted deleted message must NOT be marked processed
      const processed = await getProcessedMessageIds();
      expect(processed.has(msgId)).toBe(false);
    });

    it('should re-evaluate messages when forceAll is true', async function () {
      const dbName = createUniqueDbName('ForceAllTest');
      const testDb = trackDb(createEchoDatabase(dbName));
      await testDb.open();

      const msgId = (await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: 'I built an online marketplace using React and Node.js.',
          createdAt: new Date(),
          capturedAt: new Date(),
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      ))!;

      vi.spyOn(ollamaClient, 'checkOllamaStatus').mockResolvedValue({
        isRunning: true,
        isModelReady: true,
        modelName: 'qwen3.5:2b',
      });

      vi.spyOn(ollamaClient, 'queryOllamaJson').mockResolvedValue(
        JSON.stringify({
          knowledge: [
            {
              category: 'projects',
              subject: 'online marketplace',
              predicate: 'built',
              value: 'using React and Node.js',
              sourceExcerpt: 'I built an online marketplace using React and Node.js',
              confidence: 'high',
              importance: 'primary',
            },
          ],
        })
      );

      // First run: extracts
      const summary1 = await runKnowledgeExtraction({ targetDb: testDb });
      expect(summary1.extractedCount).toBe(1);

      // Second run without forceAll: skips because already processed
      const summary2 = await runKnowledgeExtraction({ targetDb: testDb });
      expect(summary2.candidateCount).toBe(0);

      // Third run with forceAll: true: candidate evaluated again
      const summary3 = await runKnowledgeExtraction({ targetDb: testDb, forceAll: true });
      expect(summary3.candidateCount).toBe(1);
      expect(summary3.duplicateCount).toBe(1); // Corroborated without duplicating
    });
  });
});

