import 'fake-indexeddb/auto';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  createEchoDatabase,
  savePersonalKnowledge,
  saveKnowledgeEvidence,
  type EchoDatabase,
  type PersonalKnowledgeItem,
  type KnowledgeEvidence,
} from './database';
import {
  extractTopicKeywords,
  formatKnowledgeItemForGeneration,
  retrieveRelevantKnowledge,
  scoreKnowledgeItem,
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
        '[Completed Project]: I built Task Tracker, a task management platform for developers. (Status: Completed by me; not an ongoing attempt or another person\'s project.)'
      );
    });

    it('formats project limitations into explicit first-person constraints', () => {
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
        '[Project Constraint & Reality]: For my project Task Tracker, could not get real-time websocket access, so built an offline sync queue. (Strict reality: This is an offline/static simulation; never claim live, real-time, or external feed connectivity.)'
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
        '[Current Mastered Skill]: I already learned and actively use React and TypeScript (for web applications). (This is an existing skill, not a future goal.)'
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
        '[Work Routine]: In my workflow with coding, I use AI when stuck to get concise answers. (Only use the tools explicitly stated; do not extrapolate unmentioned software or platforms.)'
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
        '[Future Learning Goal (Not yet started)]: My goal is to learn in the near future (AI and Machine Learning). (Strictly a personal learning objective; not a business venture or commercial plan.)'
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
      expect(formatted.startsWith('[Current Mastered Skill]: I already learned and actively use TypeScript')).toBe(true);
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
        "[Completed Project]: I built Task Tracker (task platform). (Status: Completed by me; not an ongoing attempt or another person's project.)"
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
      await seedItem({ subject: 'React Native', confidence: 'low' });
      const idMedium = await seedItem({ subject: 'React Native', confidence: 'medium' });
      const idHigh = await seedItem({ subject: 'React Native', confidence: 'high' });

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
  });
});
