import 'fake-indexeddb/auto';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  createEchoDatabase,
  savePersonalKnowledge,
  saveKnowledgeEvidence,
  addManualPersonalKnowledge,
  deletePersonalKnowledge,
  type EchoDatabase,
  type PersonalKnowledgeItem,
  type KnowledgeEvidence,
} from './database';
import {
  extractTopicKeywords,
  formatKnowledgeItemForGeneration,
  formatAffirmativeConstraint,
  isAutobiographicalTopic,
  retrieveRelevantKnowledge,
  scoreKnowledgeItem,
  stemWord,
  isAuthorSubject,
  toFirstPersonVerb,
} from './knowledgeRetrieval';

describe('Personal Knowledge Retrieval Engine V1', () => {
  let testDb: EchoDatabase;
  let dbCounter = 0;

  beforeEach(() => {
    dbCounter++;
    testDb = createEchoDatabase(`TestRetrievalDb_${Date.now()}_${dbCounter}`);
  });

  afterEach(async () => {
    if (testDb.isOpen()) {
      await testDb.delete();
    }
  });

  describe('extractTopicKeywords', () => {
    it('extracts meaningful lowercase keywords and filters out common stop words', () => {
      const topic = 'Why and how I built the Task Tracker task platform with React';
      const keywords = extractTopicKeywords(topic);

      expect(keywords).toContain('built');
      expect(keywords).toContain('task');
      expect(keywords).toContain('tracker');
      expect(keywords).toContain('platform');
      expect(keywords).toContain('react');

      expect(keywords).not.toContain('why');
      expect(keywords).not.toContain('and');
      expect(keywords).not.toContain('how');
      expect(keywords).not.toContain('i');
      expect(keywords).not.toContain('the');
      expect(keywords).not.toContain('with');
    });

    it('returns empty array for empty, whitespace, or stop-word-only input', () => {
      expect(extractTopicKeywords('')).toEqual([]);
      expect(extractTopicKeywords('   ')).toEqual([]);
      expect(extractTopicKeywords('this is what it is')).toEqual([]);
    });

    it('deduplicates repetitive keywords', () => {
      const keywords = extractTopicKeywords('React React react REACT development');
      expect(keywords).toEqual(['react', 'development']);
    });

    it('filters out prompt intent words like casual, social, media, post, write when substantive keywords are present', () => {
      const topic = 'write a casual social media post about Match Point';
      const keywords = extractTopicKeywords(topic);
      expect(keywords).toEqual(['match', 'point']);
    });

    it('falls back to non-stop keywords if the prompt consists only of intent words', () => {
      const topic = 'social media post';
      const keywords = extractTopicKeywords(topic);
      expect(keywords).toEqual(['social', 'media', 'post']);
    });
  });

  describe('formatKnowledgeItemForGeneration', () => {
    it('formats into concise, grounded first-person factual statement for projects', () => {
      const item: PersonalKnowledgeItem = {
        category: 'projects',
        subject: 'Task Tracker',
        predicate: 'built',
        value: 'a task management platform for developers',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };

      const formatted = formatKnowledgeItemForGeneration(item);
      expect(formatted).toBe(
        '[Completed Project]: I built Task Tracker, a task management platform for developers.'
      );
    });

    it('de-duplicates repeated verb phrases in project fact formatting', () => {
      const item: PersonalKnowledgeItem = {
        category: 'projects',
        subject: 'Match Point',
        predicate: 'built',
        value: 'a betting platform I built while applying to some iGaming company; it was my first complete project',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };

      const formatted = formatKnowledgeItemForGeneration(item);
      expect(formatted).toBe(
        '[Completed Project]: I built Match Point, a betting platform while applying to some iGaming company; it was my first complete project.'
      );
      expect(formatted).not.toContain('I built Match Point, a betting platform I built');
    });

    it('formats project limitations into affirmative descriptions of actual reality', () => {
      const item: PersonalKnowledgeItem = {
        category: 'projects',
        subject: 'Task Tracker',
        predicate: 'constraint',
        value: 'could not get real-time websocket access, so built an offline sync queue',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };

      const formatted = formatKnowledgeItemForGeneration(item);
      expect(formatted).toBe(
        '[Project Architecture & Reality]: Task Tracker currently operates as an offline sync queue. It does not have real-time websocket access or live external sync.'
      );
    });

    it('formats skills, habits, and goals with authentic first-person framing', () => {
      const skillItem: PersonalKnowledgeItem = {
        category: 'skills',
        subject: 'React and TypeScript',
        predicate: 'uses',
        value: 'for web applications',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };
      expect(formatKnowledgeItemForGeneration(skillItem)).toBe(
        '[Current Skill]: I learned and actively use React and TypeScript (for web applications).'
      );

      const habitItem: PersonalKnowledgeItem = {
        category: 'habits',
        subject: 'coding',
        predicate: 'uses',
        value: 'AI when stuck to get concise answers',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };
      expect(formatKnowledgeItemForGeneration(habitItem)).toBe(
        '[Work Routine]: In my workflow with coding, I use AI when stuck to get concise answers.'
      );

      const goalItem: PersonalKnowledgeItem = {
        category: 'goals',
        subject: 'AI and Machine Learning',
        predicate: 'plans to learn',
        value: 'in the near future',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };
      expect(formatKnowledgeItemForGeneration(goalItem)).toBe(
        '[Future Goal]: My goal is to learn in the near future (AI and Machine Learning).'
      );
    });

    it('does not include source excerpts or unrelated fields and truncates to 500 chars', () => {
      const item: PersonalKnowledgeItem = {
        category: 'skills',
        subject: 'TypeScript',
        predicate: 'uses',
        value: 'X'.repeat(600),
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'secondary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      };

      const formatted = formatKnowledgeItemForGeneration(item);
      expect(formatted.length).toBeLessThanOrEqual(500);
      expect(formatted.startsWith('[Current Skill]: I learned and actively use TypeScript')).toBe(true);
    });
  });

  describe('retrieveRelevantKnowledge - Filtering and Constraints', () => {
    async function seedItem(
      overrides: Partial<PersonalKnowledgeItem> = {},
      hasValidEvidence: boolean = true
    ): Promise<number> {
      const item: PersonalKnowledgeItem = {
        category: 'projects',
        subject: 'Task Tracker',
        predicate: 'built',
        value: 'task management platform',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
        ...overrides,
      };

      const kId = await savePersonalKnowledge(item, testDb);

      if (hasValidEvidence) {
        await saveKnowledgeEvidence(
          {
            knowledgeId: kId,
            messageId: 101,
            sourceExcerpt: 'I built Task Tracker, a task management platform.',
            isValid: true,
            extractedAt: new Date(),
          },
          testDb
        );
      }

      return kId;
    }

    it('retrieves active, verified knowledge items that match the topic keywords', async () => {
      await seedItem({ subject: 'Task Tracker', value: 'task platform' });
      await seedItem({ subject: 'Docker', value: 'container workflow' });

      const res = await retrieveRelevantKnowledge('Tell me about Task Tracker platform', testDb);

      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.items.length).toBe(1);
      expect(res.items[0]!.subject).toBe('Task Tracker');
      expect(res.formattedStrings[0]).toBe(
        '[Completed Project]: I built Task Tracker (task platform).'
      );
    });

    it('excludes items with non-active statuses (archived, superseded, conflicted)', async () => {
      // Archived
      const idArchived = await seedItem({ subject: 'Task Tracker', status: 'archived' });
      // Superseded
      const idSuperseded = await seedItem({ subject: 'Task Tracker', status: 'superseded' });
      // Conflicted
      const idConflicted = await seedItem({ subject: 'Task Tracker', status: 'conflicted' });
      // Active
      const idActive = await seedItem({ subject: 'Task Tracker', status: 'active' });

      const res = await retrieveRelevantKnowledge('Task Tracker', testDb);

      expect(res.items.length).toBe(1);
      expect(res.items[0]!.id).toBe(idActive);
    });

    it('excludes items with low confidence or unrated confidence', async () => {
      await seedItem({ subject: 'React Native', value: 'cross-platform mobile', confidence: 'low' });
      const idMedium = await seedItem({ subject: 'React Native', value: 'mobile components', confidence: 'medium' });
      const idHigh = await seedItem({ subject: 'React Native', value: 'task management platform', confidence: 'high' });

      const res = await retrieveRelevantKnowledge('React Native', testDb);

      expect(res.items.length).toBe(2);
      const ids = res.items.map((i) => i.id);
      expect(ids).toContain(idMedium);
      expect(ids).toContain(idHigh);
    });

    it('excludes items with no evidence records or where all evidence is marked invalid', async () => {
      // 1. No evidence at all
      await seedItem({ subject: 'GraphQL' }, false);

      // 2. Evidence exists but isValid === false
      const kIdInvalid = await seedItem({ subject: 'GraphQL' }, false);
      await saveKnowledgeEvidence(
        {
          knowledgeId: kIdInvalid,
          messageId: 102,
          sourceExcerpt: 'GraphQL excerpt',
          isValid: false,
          invalidationReason: 'source_message_deleted',
          extractedAt: new Date(),
        },
        testDb
      );

      // 3. Valid evidence
      const kIdValid = await seedItem({ subject: 'GraphQL' }, true);

      const res = await retrieveRelevantKnowledge('GraphQL APIs', testDb);

      expect(res.items.length).toBe(1);
      expect(res.items[0]!.id).toBe(kIdValid);
    });

    it('caps retrieved results at maximum 10 items even when more match', async () => {
      // Seed 15 items with keyword "Cloud"
      for (let i = 1; i <= 15; i++) {
        await seedItem({
          subject: `Cloud Platform ${i}`,
          value: `Cloud deployment ${i}`,
          importance: i % 2 === 0 ? 'primary' : 'secondary',
        });
      }

      const res = await retrieveRelevantKnowledge('Cloud platforms', testDb, 10);

      expect(res.items.length).toBe(10);
      expect(res.formattedStrings.length).toBe(10);
    });

    it('returns empty result with hasRelevantKnowledge: false when no keyword matches occur', async () => {
      await seedItem({ subject: 'Vue.js', value: 'frontend UI library' });

      const res = await retrieveRelevantKnowledge('Rust systems programming', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.length).toBe(0);
      expect(res.formattedStrings.length).toBe(0);
    });

    it('ranks subject matches and primary importance higher than incidental matches', async () => {
      // Incidental match in value
      const idIncidental = await seedItem({
        subject: 'Database',
        predicate: 'uses',
        value: 'PostgreSQL for data storage',
        importance: 'incidental',
      });

      // Primary match in subject
      const idPrimary = await seedItem({
        subject: 'PostgreSQL',
        predicate: 'expert in',
        value: 'relational database design',
        importance: 'primary',
      });

      const res = await retrieveRelevantKnowledge('PostgreSQL', testDb);

      expect(res.items.length).toBe(2);
      expect(res.items[0]!.id).toBe(idPrimary);
      expect(res.items[1]!.id).toBe(idIncidental);
    });

    it('identifies autobiographical topics lacking personal facts and provides safety notice', async () => {
      const res = await retrieveRelevantKnowledge('My daily morning routine and coffee habits', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.isAutobiographicalWithoutFacts).toBe(true);
      expect(res.safetyNotice).toContain('No verified personal notes exist');
    });

    it('does not flag general conceptual topics as autobiographical without facts', async () => {
      const res = await retrieveRelevantKnowledge('Why I prefer learning by building projects', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.isAutobiographicalWithoutFacts).toBe(false);
      expect(res.safetyNotice).toBeUndefined();
    });
  });

  describe('Affirmative Constraint Formatting & Negation Handling', () => {
    it('expresses static vs live functionality affirmatively for Match Point', () => {
      const raw = 'could not get live match data, so built a static version where matches never end';
      const formatted = formatAffirmativeConstraint('Match Point', raw);

      expect(formatted).toBe(
        'Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.'
      );
      expect(formatted).not.toContain('could not');
      expect(formatted).toContain('currently uses static match data');
    });

    it('expresses offline sync vs websocket functionality affirmatively for Task Tracker', () => {
      const raw = 'could not get real-time websocket access, so built an offline sync queue';
      const formatted = formatAffirmativeConstraint('Task Tracker', raw);

      expect(formatted).toBe(
        'Task Tracker currently operates as an offline sync queue. It does not have real-time websocket access or live external sync.'
      );
    });

    it('handles general could-not-get X so-built Y patterns without inventing features', () => {
      const raw = 'unable to access cloud postgres, so implemented local indexeddb storage';
      const formatted = formatAffirmativeConstraint('LocalNotes', raw);

      expect(formatted).toBe(
        'LocalNotes currently uses local indexeddb storage. It does not have cloud postgres connectivity.'
      );
    });

    it('handles raw negation strings safely while preserving boundaries', () => {
      const raw = 'lacks automated cloud backup';
      const formatted = formatAffirmativeConstraint('UtilityApp', raw);

      expect(formatted).toContain('actual reality: operates as a local or static implementation without automated cloud backup.');
    });
  });

  describe('isAutobiographicalTopic', () => {
    it('detects daily routines and habits', () => {
      expect(isAutobiographicalTopic('My daily morning routine and coffee habits')).toBe(true);
      expect(isAutobiographicalTopic('My morning routine')).toBe(true);
      expect(isAutobiographicalTopic('My evening routine')).toBe(true);
      expect(isAutobiographicalTopic('How I spend my mornings')).toBe(true);
    });

    it('detects pets and personal life', () => {
      expect(isAutobiographicalTopic('My experience with owning pets')).toBe(true);
      expect(isAutobiographicalTopic('My experience with pets')).toBe(true);
      expect(isAutobiographicalTopic('My pet dog')).toBe(true);
      expect(isAutobiographicalTopic('My childhood memories')).toBe(true);
      expect(isAutobiographicalTopic('Where I grew up and my hometown')).toBe(true);
    });

    it('returns false for general conceptual questions about pets and routines', () => {
      expect(isAutobiographicalTopic('Why do people keep pets?')).toBe(false);
      expect(isAutobiographicalTopic('Why are morning routines useful?')).toBe(false);
      expect(isAutobiographicalTopic('Benefits of morning routines')).toBe(false);
      expect(isAutobiographicalTopic('The history of domestic pets')).toBe(false);
    });

    it('returns false for general, conceptual, and technical topics', () => {
      expect(isAutobiographicalTopic('Why I prefer learning by building projects rather than just reading documentation')).toBe(false);
      expect(isAutobiographicalTopic('My thoughts on remote work and collaborating across different time zones')).toBe(false);
      expect(isAutobiographicalTopic('Why writing automated tests for every feature saves time in the long run')).toBe(false);
      expect(isAutobiographicalTopic('IndexedDB')).toBe(false);
      expect(isAutobiographicalTopic('Why user chat data should stay stored locally in the browser')).toBe(false);
    });
  });

  describe('retrieveRelevantKnowledge - Provenance & Manual Curation Support', () => {
    it('1. retrieves a manual fact without evidence when relevant', async () => {
      const manualId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'TypeScript Compilers',
          predicate: 'uses',
          value: 'AST transformations for optimization',
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('TypeScript compiler AST optimization', testDb);

      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.items.some((i) => i.id === manualId)).toBe(true);
      expect(res.formattedStrings.length).toBeGreaterThan(0);
      expect(res.formattedStrings[0]).toContain('TypeScript Compilers');
    });

    it('2. excludes an extracted fact without valid evidence records', async () => {
      const extractedId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'OrphanedExtractedSkill',
          predicate: 'uses',
          value: 'unverified tool',
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

      const res = await retrieveRelevantKnowledge('OrphanedExtractedSkill', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.some((i) => i.id === extractedId)).toBe(false);
    });

    it('3. retrieves an extracted fact with valid evidence', async () => {
      const factId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'Rust Systems',
          predicate: 'uses',
          value: 'async-std for networking',
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
          messageId: 101,
          sourceExcerpt: 'I use Rust systems with async-std for networking daily.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('Rust systems networking', testDb);

      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.items.some((i) => i.id === factId)).toBe(true);
    });

    it('4. excludes a legacy fact without sourceType when it lacks valid evidence', async () => {
      const legacyId = await testDb.personalKnowledge.add({
        category: 'skills',
        subject: 'LegacyUnverifiedTool',
        predicate: 'uses',
        value: 'tooling without evidence',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      } as PersonalKnowledgeItem);

      const res = await retrieveRelevantKnowledge('LegacyUnverifiedTool', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.some((i) => i.id === legacyId)).toBe(false);
    });

    it('5. retrieves a legacy fact without sourceType when it has valid evidence', async () => {
      const legacyId = (await testDb.personalKnowledge.add({
        category: 'skills',
        subject: 'LegacyVerifiedTool',
        predicate: 'uses',
        value: 'reliable build scripts',
        stability: 'stable',
        status: 'active',
        freshnessState: 'fresh',
        confidence: 'high',
        importance: 'primary',
        createdAt: new Date(),
        updatedAt: new Date(),
        lastConfirmedAt: new Date(),
      } as PersonalKnowledgeItem)) as number;

      await saveKnowledgeEvidence(
        {
          knowledgeId: legacyId,
          messageId: 202,
          sourceExcerpt: 'I write reliable build scripts for our toolchain.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('LegacyVerifiedTool build scripts', testDb);

      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.items.some((i) => i.id === legacyId)).toBe(true);
    });

    it('6. excludes an archived manual fact even when matching topic keywords', async () => {
      const manualId = await addManualPersonalKnowledge(
        {
          category: 'goals',
          subject: 'OldArchivedGoal',
          predicate: 'aims to',
          value: 'finish obsolete migration',
        },
        testDb
      );

      await deletePersonalKnowledge(manualId, 'user_manually_rejected', testDb);

      const res = await retrieveRelevantKnowledge('OldArchivedGoal migration', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.some((i) => i.id === manualId)).toBe(false);
    });

    it('7. excludes an archived extracted fact even if it has valid evidence records', async () => {
      const factId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'ArchivedExtractedSkill',
          predicate: 'uses',
          value: 'deprecated library',
          stability: 'stable',
          status: 'archived',
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
          messageId: 303,
          sourceExcerpt: 'Evidence for archived skill.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('ArchivedExtractedSkill', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.some((i) => i.id === factId)).toBe(false);
    });

    it('8. excludes extracted facts whose linked evidence records are all invalidated', async () => {
      const factId = await savePersonalKnowledge(
        {
          category: 'preferences',
          subject: 'InvalidatedPreference',
          predicate: 'prefers',
          value: 'rejected workflow',
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
          messageId: 404,
          sourceExcerpt: 'Old excerpt that was revoked',
          isValid: false,
          invalidationReason: 'source_message_deleted',
          invalidatedAt: new Date(),
          extractedAt: new Date(),
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('InvalidatedPreference rejected workflow', testDb);

      expect(res.hasRelevantKnowledge).toBe(false);
      expect(res.items.some((i) => i.id === factId)).toBe(false);
    });

    it('9. preserves ranking by relevance score and limits to maxItems when combining manual and extracted facts', async () => {
      // 1. High-relevance manual item (matching subject and value)
      const highRelManualId = await addManualPersonalKnowledge(
        {
          category: 'skills',
          subject: 'Database Optimization',
          predicate: 'specializes in',
          value: 'query indexing and explain plans',
          importance: 'primary',
        },
        testDb
      );

      // 2. Low-relevance extracted item with valid evidence (matching only incidental value)
      const lowRelExtractedId = await savePersonalKnowledge(
        {
          category: 'skills',
          subject: 'General Backend',
          predicate: 'uses',
          value: 'basic database configurations',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'incidental',
          sourceType: 'extracted',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      );
      await saveKnowledgeEvidence(
        {
          knowledgeId: lowRelExtractedId,
          messageId: 501,
          sourceExcerpt: 'Evidence for basic database configurations',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      // Retrieve with topic matching both
      const res = await retrieveRelevantKnowledge('Database Optimization explain plans', testDb, 1);

      // Max items was 1 -> highest relevance manual item must be ranked first and capped at 1
      expect(res.items.length).toBe(1);
      expect(res.items[0]?.id).toBe(highRelManualId);
    });

    it('10. formats manual facts into the exact first-person bracketed statements required for generation', async () => {
      await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Echo Extension',
          predicate: 'built',
          value: 'a local-first personalized AI writing assistant',
          contextScope: 'browser extension',
        },
        testDb
      );

      await addManualPersonalKnowledge(
        {
          category: 'preferences',
          subject: 'Code Formatting',
          predicate: 'prefers',
          value: 'strict linters without noisy warnings',
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('Echo Extension and Code Formatting preferences', testDb);

      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.formattedStrings.length).toBe(2);

      // Verify exact bracketed generation format
      const projectStr = res.formattedStrings.find((s) => s.includes('Echo Extension'));
      expect(projectStr).toBeDefined();
      expect(projectStr).toBe('[Completed Project]: I built Echo Extension, a local-first personalized AI writing assistant.');

      const prefStr = res.formattedStrings.find((s) => s.includes('Code Formatting'));
      expect(prefStr).toBeDefined();
      expect(prefStr).toContain('[Personal Preference]: I prefer Code Formatting: strict linters without noisy warnings.');
    });

    it('11. excludes distant incidental hits when strong subject anchor matches exist (relative score cutoff)', async () => {
      // 1. Strong anchor hit for Match Point
      await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Match Point',
          predicate: 'built',
          value: 'tennis scorecard application',
        },
        testDb
      );

      // 2. Weak incidental hit that only matches the word 'scorecard' in value
      const incidentalId = await savePersonalKnowledge(
        {
          category: 'habits',
          subject: 'Office Routine',
          predicate: 'maintains',
          value: 'daily scorecard for sprint planning',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'incidental',
          sourceType: 'extracted',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        },
        testDb
      );
      await saveKnowledgeEvidence(
        {
          knowledgeId: incidentalId,
          messageId: 601,
          sourceExcerpt: 'I maintain a daily scorecard for sprint planning.',
          isValid: true,
          extractedAt: new Date(),
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('Match Point tennis', testDb);
      expect(res.hasRelevantKnowledge).toBe(true);
      expect(res.items.some((i) => i.subject === 'Match Point')).toBe(true);
      // Incidental hit should be filtered out by relative threshold
      expect(res.items.some((i) => i.subject === 'Office Routine')).toBe(false);
    });

    it('12. deduplicates near-duplicate facts with same subject and predicate or subsumed values', async () => {
      // Near-duplicate 1: same subject & predicate, subsumed value
      await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Match Point',
          predicate: 'is',
          value: 'an offline tennis scorecard app',
        },
        testDb
      );
      await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Match Point',
          predicate: 'is',
          value: 'tennis scorecard app',
        },
        testDb
      );

      // Complementary fact: same subject, distinct predicate & value
      await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Match Point',
          predicate: 'constraint',
          value: 'built 100% offline with zero server dependencies',
        },
        testDb
      );

      const res = await retrieveRelevantKnowledge('Match Point scorecard app', testDb);
      expect(res.hasRelevantKnowledge).toBe(true);
      // Deduplication should leave exactly 2 items instead of 3
      expect(res.items.length).toBe(2);
      expect(res.items.filter((i) => i.subject === 'Match Point' && i.predicate === 'is').length).toBe(1);
      expect(res.items.some((i) => i.predicate === 'constraint')).toBe(true);
    });

    it('13. limits retrieved knowledge items to maxItems (default 5)', async () => {
      for (let i = 1; i <= 8; i++) {
        await addManualPersonalKnowledge(
          {
            category: 'projects',
            subject: `Project ${i}`,
            predicate: 'built_with',
            value: `TypeScript framework number ${i}`,
          },
          testDb
        );
      }

      const res = await retrieveRelevantKnowledge('TypeScript framework', testDb);
      expect(res.items.length).toBe(5);
    });
  });

  describe('Evaluation Report Regression Suite - Real Failure Cases', () => {
    describe('1. Word variations matching', () => {
      it('correctly stems verb and noun morphological variants', () => {
        expect(stemWord('argue')).toBe('argu');
        expect(stemWord('arguments')).toBe('argu');
        expect(stemWord('arguing')).toBe('argu');
        expect(stemWord('argument')).toBe('argu');

        expect(stemWord('listen')).toBe('listen');
        expect(stemWord('listening')).toBe('listen');
        expect(stemWord('listened')).toBe('listen');

        expect(stemWord('learn')).toBe('learn');
        expect(stemWord('learning')).toBe('learn');
        expect(stemWord('learned')).toBe('learn');

        expect(stemWord('built')).toBe('build');
        expect(stemWord('building')).toBe('build');
        expect(stemWord('builds')).toBe('build');

        expect(stemWord('thought')).toBe('think');
        expect(stemWord('thinking')).toBe('think');
      });

      it('retrieves both argue and listen records on technical disagreements query', async () => {
        // Real user facts ID 1 and ID 2
        await addManualPersonalKnowledge(
          {
            category: 'goals',
            subject: 'the user',
            predicate: 'argue to prove how they are wrong',
            value: 'If I have enough knowledge about that topic',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        await addManualPersonalKnowledge(
          {
            category: 'goals',
            subject: 'the user',
            predicate: 'listen to theirs as well',
            value: 'if I am not good enough in that topic then I will put my point and listen to theirs as well',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const res = await retrieveRelevantKnowledge(
          'How I approach technical arguments and listening to other developers',
          testDb
        );

        expect(res.hasRelevantKnowledge).toBe(true);
        expect(res.items.length).toBe(2);

        const predicates = res.items.map((i) => i.predicate);
        expect(predicates).toContain('argue to prove how they are wrong');
        expect(predicates).toContain('listen to theirs as well');

        // Verify formatting for both records is clean first-person and never contains placeholder
        for (const str of res.formattedStrings) {
          expect(str).not.toContain('the user');
          expect(str).toContain('[Personal Approach]: In discussions,');
        }
      });
    });

    describe('2. Low-score leakage and irrelevant retrieval prevention', () => {
      it('avoids retrieving unrelated learning records when asking about AI problem-solving', async () => {
        // Real user facts ID 8, ID 3, ID 4, ID 14
        const id8 = await addManualPersonalKnowledge(
          {
            category: 'habits',
            subject: 'the user',
            predicate: 'uses AI',
            value: 'whenever stuck',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id3 = await addManualPersonalKnowledge(
          {
            category: 'states',
            subject: 'user',
            predicate: 'thought too much',
            value: 'about a product idea',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id4 = await addManualPersonalKnowledge(
          {
            category: 'states',
            subject: 'the user',
            predicate: 'hate',
            value: 'people who are learning having no value',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id14 = await addManualPersonalKnowledge(
          {
            category: 'goals',
            subject: 'the user',
            predicate: 'planning to try',
            value: 'AI and some ML',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const res = await retrieveRelevantKnowledge(
          'How I use AI when I get stuck on coding and product ideas',
          testDb
        );

        expect(res.hasRelevantKnowledge).toBe(true);
        const retrievedIds = res.items.map((i) => i.id);

        // Core relevant facts must be present
        expect(retrievedIds).toContain(id8);
        expect(retrievedIds).toContain(id3);

        // Irrelevant learning fact (ID 4) MUST NOT be retrieved
        expect(retrievedIds).not.toContain(id4);

        // Ensure "the user" placeholder did not trigger substring match on "use"
        expect(scoreKnowledgeItem({
          id: 999,
          category: 'states',
          subject: 'the user',
          predicate: 'hate',
          value: 'people who are learning having no value',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        }, ['use', 'ai', 'get', 'stuck', 'coding', 'product', 'ideas'])).toBe(0);
      });
    });

    describe('3. First-person formatting for author-subject facts', () => {
      it('formats habits with "the user" as natural first-person without awkward phrasing', () => {
        const item: PersonalKnowledgeItem = {
          category: 'habits',
          subject: 'the user',
          predicate: 'uses AI',
          value: 'whenever stuck',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };

        const formatted = formatKnowledgeItemForGeneration(item);
        expect(formatted).toBe(
          '[Personal Habit]: I use AI whenever stuck.'
        );
        expect(formatted).not.toContain('with the user');
        expect(formatted).not.toContain('I uses');
      });

      it('formats states and facts with "user" or "the user" without inserting placeholder into sentence', () => {
        const item1: PersonalKnowledgeItem = {
          category: 'states',
          subject: 'user',
          predicate: 'thought too much',
          value: 'about a product idea',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };
        expect(formatKnowledgeItemForGeneration(item1)).toBe(
          '[Personal Fact]: I thought too much about a product idea.'
        );

        const item2: PersonalKnowledgeItem = {
          category: 'states',
          subject: 'the user',
          predicate: 'hate',
          value: 'people who are learning having no value',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };
        expect(formatKnowledgeItemForGeneration(item2)).toBe(
          '[Personal Fact]: I hate people who are learning having no value.'
        );
      });

      it('formats goals with "the user" as direct authorial objective', () => {
        const item: PersonalKnowledgeItem = {
          category: 'goals',
          subject: 'the user',
          predicate: 'planning to try',
          value: 'AI and some ML',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };

        const formatted = formatKnowledgeItemForGeneration(item);
        expect(formatted).toBe(
          '[Future Goal]: My goal is to try AI and some ML.'
        );
        expect(formatted).not.toContain('for the user');
      });

      it('formats preferences and opinions with "the user" cleanly', () => {
        const prefItem: PersonalKnowledgeItem = {
          category: 'preferences',
          subject: 'the user',
          predicate: 'prefers',
          value: 'clean modular code',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };
        expect(formatKnowledgeItemForGeneration(prefItem)).toBe(
          '[Personal Preference]: I prefer clean modular code.'
        );

        const opItem: PersonalKnowledgeItem = {
          category: 'opinions',
          subject: 'user',
          predicate: 'believes',
          value: 'simplicity beats cleverness',
          stability: 'stable',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'high',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };
        expect(formatKnowledgeItemForGeneration(opItem)).toBe(
          '[Personal Perspective]: My perspective is that simplicity beats cleverness.'
        );
      });
    });

    describe('4. Match Point preservation', () => {
      it('preserves successful retrieval of Match Point on detailed and casual prompts', async () => {
        const mpId = await addManualPersonalKnowledge(
          {
            category: 'projects',
            subject: 'match point',
            predicate: 'built',
            value: 'a betting platform',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        // Detailed prompt
        const resDetailed = await retrieveRelevantKnowledge(
          'Why and how I built Match Point betting platform',
          testDb
        );
        expect(resDetailed.hasRelevantKnowledge).toBe(true);
        expect(resDetailed.items.length).toBe(1);
        expect(resDetailed.items[0]!.id).toBe(mpId);
        expect(resDetailed.formattedStrings[0]).toBe(
          '[Completed Project]: I built match point, a betting platform.'
        );

        // Casual prompt
        const resCasual = await retrieveRelevantKnowledge(
          'Casual social media post about Match Point',
          testDb
        );
        expect(resCasual.hasRelevantKnowledge).toBe(true);
        expect(resCasual.items.length).toBe(1);
        expect(resCasual.items[0]!.id).toBe(mpId);
      });
    });

    describe('5. Web development journey preservation', () => {
      it('preserves retrieval of web development, React, and JavaScript without regressions', async () => {
        const id13 = await addManualPersonalKnowledge(
          {
            category: 'goals',
            subject: 'web development',
            predicate: 'plan to learn',
            value: 'latest technologies',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id12 = await addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'React',
            predicate: 'learned',
            value: 'React',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id11 = await addManualPersonalKnowledge(
          {
            category: 'skills',
            subject: 'JavaScript',
            predicate: 'learned',
            value: 'JavaScript',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );
        const id4 = await addManualPersonalKnowledge(
          {
            category: 'states',
            subject: 'the user',
            predicate: 'hate',
            value: 'people who are learning having no value',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const res = await retrieveRelevantKnowledge(
          'My journey learning web development with React and JavaScript and what I plan to learn next',
          testDb
        );

        expect(res.hasRelevantKnowledge).toBe(true);
        const retrievedIds = res.items.map((i) => i.id);

        // Relevant skills & goals must all be preserved
        expect(retrievedIds).toContain(id13);
        expect(retrievedIds).toContain(id12);
        expect(retrievedIds).toContain(id11);

        // Negative unrelated opinion must NOT leak into web dev retrieval
        expect(retrievedIds).not.toContain(id4);
      });
    });

    describe('6. Semantic Formatting & Match Point Companion Verification', () => {
      it('formats debate records naturally as personal approaches without duplicate predicates or future goal labels', async () => {
        const item1: PersonalKnowledgeItem = {
          category: 'habits',
          subject: 'the user',
          predicate: 'argue to prove how they are wrong',
          value: 'If I have enough knowledge about that topic',
          sourceType: 'manual',
          editedByUser: true,
          stability: 'evolving',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'medium',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };

        const item2: PersonalKnowledgeItem = {
          category: 'habits',
          subject: 'the user',
          predicate: 'listen to theirs as well',
          value: 'if I am not good enough in that topic then I will put my point and listen to theirs as well',
          sourceType: 'manual',
          editedByUser: true,
          stability: 'evolving',
          status: 'active',
          freshnessState: 'fresh',
          confidence: 'medium',
          importance: 'primary',
          createdAt: new Date(),
          updatedAt: new Date(),
          lastConfirmedAt: new Date(),
        };

        const formatted1 = formatKnowledgeItemForGeneration(item1);
        expect(formatted1).toBe(
          '[Personal Approach]: In discussions, if I have enough knowledge about that topic, I argue to prove how they are wrong.'
        );
        expect(formatted1).not.toContain('Future Learning Goal');
        expect(formatted1).not.toContain('Work Routine');

        const formatted2 = formatKnowledgeItemForGeneration(item2);
        expect(formatted2).toBe(
          '[Personal Approach]: In discussions, if I am not good enough in that topic then I will put my point and listen to theirs as well.'
        );
        expect(formatted2).not.toContain('listen to theirs as well listen to theirs as well');
        expect(formatted2).not.toContain('Future Learning Goal');
      });

      it('retrieves Match Point motivation and constraint companion facts alongside the completed project', async () => {
        const mpBase = await addManualPersonalKnowledge(
          {
            category: 'projects',
            subject: 'match point',
            predicate: 'built',
            value: 'a betting platform',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const mpMotiv = await addManualPersonalKnowledge(
          {
            category: 'projects',
            subject: 'match point',
            predicate: 'built for',
            value: 'applying to an iGaming company as my first complete project',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const mpConstraint = await addManualPersonalKnowledge(
          {
            category: 'projects',
            subject: 'match point',
            predicate: 'constraint',
            value: 'I could not obtain live match data, so I built a static simulation where matches never end and users place bets.',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const res = await retrieveRelevantKnowledge(
          'Why and how I built Match Point betting platform and the challenges with match data',
          testDb
        );

        expect(res.hasRelevantKnowledge).toBe(true);
        const retrievedIds = res.items.map((i) => i.id);
        expect(retrievedIds).toContain(mpBase);
        expect(retrievedIds).toContain(mpMotiv);
        expect(retrievedIds).toContain(mpConstraint);

        const motivStr = res.formattedStrings.find((s) => s.includes('iGaming company'));
        expect(motivStr).toBeDefined();
        expect(motivStr).toBe(
          '[Project Context]: I built match point for applying to an iGaming company as my first complete project.'
        );

        const constraintStr = res.formattedStrings.find((s) => s.includes('static match data'));
        expect(constraintStr).toBeDefined();
        expect(constraintStr).toBe(
          '[Project Architecture & Reality]: match point currently uses static match data and simulated match progression. It does not have live match connectivity.'
        );
      });

      it('preserves manual curated status and editedByUser protection flags', async () => {
        const manualFactId = await addManualPersonalKnowledge(
          {
            category: 'habits',
            subject: 'the user',
            predicate: 'argue to prove how they are wrong',
            value: 'If I have enough knowledge about that topic',
            importance: 'primary',
            confidence: 'high',
          },
          testDb
        );

        const record = await testDb.personalKnowledge.get(manualFactId);
        expect(record).toBeDefined();
        expect(record?.sourceType).toBe('manual');
        expect(record?.status).toBe('active');
      });
    });
  });
});
