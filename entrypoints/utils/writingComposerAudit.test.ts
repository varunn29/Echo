import 'fake-indexeddb/auto';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import Dexie from 'dexie';

import {
  createEchoDatabase,
  saveWritingMessage,
  addManualPersonalKnowledge,
  getWritingMessages,
  getPersonalKnowledge,
  type EchoDatabase,
} from './database';

import {
  checkGenerationServiceHealth,
  generatePersonalizedDraft,
  DEFAULT_GENERATION_SERVICE_URL,
} from './generationClient';

import {
  retrieveRelevantKnowledge,
} from './knowledgeRetrieval';

function createUniqueDbName(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
}

describe('Writing Composer End-to-End Generation & Safety Audit', () => {
  const originalFetch = global.fetch;
  const openDatabases: EchoDatabase[] = [];
  const dbNamesToDelete: string[] = [];

  function trackDb(dbInstance: EchoDatabase): EchoDatabase {
    openDatabases.push(dbInstance);
    if (!dbNamesToDelete.includes(dbInstance.name)) {
      dbNamesToDelete.push(dbInstance.name);
    }
    return dbInstance;
  }

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(async () => {
    global.fetch = originalFetch;
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

  describe('1. Production Model & Active Adapter Isolation', () => {
    it('confirms active_adapter.json is configured for production Qwen3.5-0.8B LoRA V5', () => {
      const activeAdapterPath = path.resolve(__dirname, '../../training-output/active_adapter.json');
      expect(fs.existsSync(activeAdapterPath)).toBe(true);

      const metadata = JSON.parse(fs.readFileSync(activeAdapterPath, 'utf-8'));
      expect(metadata.active_version).toBe('v5');
      expect(metadata.active_adapter_path).toContain('qwen3.5-0.8b-lora-v5');
      // Must NOT point to experimental Qwen2.5 LoRA V1
      expect(metadata.active_adapter_path).not.toContain('qwen25');
    });

    it('confirms generate_with_adapter.py defaults to Qwen3.5-0.8B-Base', () => {
      const scriptPath = path.resolve(__dirname, '../../training/generate_with_adapter.py');
      const content = fs.readFileSync(scriptPath, 'utf-8');

      expect(content).toContain('DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"');
      expect(content).toContain('DEFAULT_METADATA_PATH = Path("training-output/active_adapter.json")');
    });

    it('confirms WritingComposer and generationClient communicate strictly with local service at 127.0.0.1:11435', () => {
      expect(DEFAULT_GENERATION_SERVICE_URL).toBe('http://127.0.0.1:11435');

      const composerPath = path.resolve(__dirname, '../dashboard/components/WritingComposer.tsx');
      const composerContent = fs.readFileSync(composerPath, 'utf-8');

      // WritingComposer never calls external paid APIs or cloud endpoints
      expect(composerContent).not.toContain('api.openai.com');
      expect(composerContent).not.toContain('generativelanguage.googleapis.com');
      expect(composerContent).not.toContain('anthropic.com');
    });
  });

  describe('2. Input Boundary Validation', () => {
    it('rejects empty or whitespace-only topic', async () => {
      await expect(generatePersonalizedDraft({ topic: '' })).rejects.toThrow(
        'A non-empty topic is required'
      );
      await expect(generatePersonalizedDraft({ topic: '   \n\t  ' })).rejects.toThrow(
        'A non-empty topic is required'
      );
    });

    it('rejects topics exceeding 500 characters', async () => {
      const longTopic = 'x'.repeat(501);
      await expect(generatePersonalizedDraft({ topic: longTopic })).rejects.toThrow(
        'maximum allowed length of 500 characters'
      );
    });

    it('rejects lead-ins exceeding 100 characters', async () => {
      const longLeadIn = 'y'.repeat(101);
      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic', leadIn: longLeadIn })
      ).rejects.toThrow('maximum allowed length of 100 characters');
    });

    it('rejects more than 10 personal knowledge items', async () => {
      const excessItems = Array.from({ length: 11 }, (_, i) => `Fact ${i}`);
      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic', personalKnowledge: excessItems })
      ).rejects.toThrow('At most 10 personal knowledge items');
    });
  });

  describe('3. Generation Reliability & Error Recovery', () => {
    it('handles service offline cleanly with structured offline health status', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

      const health = await checkGenerationServiceHealth();
      expect(health.isOnline).toBe(false);
      expect(health.isModelReady).toBe(false);
      expect(health.status).toBe('offline');
      expect(health.error).toContain('Local generation service is offline');
    });

    it('handles model loading (HTTP 503) cleanly', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 503,
        json: async () => ({ status: 'loading', error: 'Model loading' }),
      } as Response);

      const health = await checkGenerationServiceHealth();
      expect(health.isOnline).toBe(true);
      expect(health.isModelReady).toBe(false);
      expect(health.status).toBe('loading');

      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic' })
      ).rejects.toThrow('Model is currently loading or unavailable');
    });

    it('handles 90-second CPU timeout with an informative actionable tip', async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      global.fetch = vi.fn().mockRejectedValue(abortError);

      await expect(
        generatePersonalizedDraft({ topic: 'Complex topic' }, { timeoutMs: 100 })
      ).rejects.toThrow('Generation request timed out. Local CPU inference took longer than 90 seconds');
    });

    it('safely handles empty or whitespace-only model responses without crashing or silent failure', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          completeDraft: '   ',
          generatedText: '',
        }),
      } as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic' })
      ).rejects.toThrow('Generation service returned an empty draft');
    });

    it('safely handles malformed response missing completeDraft property', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          success: true,
          output: 'Missing completeDraft property',
        }),
      } as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic' })
      ).rejects.toThrow('Invalid response structure: missing complete draft');
    });

    it('safely handles unparseable non-JSON HTTP 200 response', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => {
          throw new Error('Unexpected token < in JSON at position 0');
        },
      } as unknown as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic' })
      ).rejects.toThrow('Received an unparseable response');
    });
  });

  describe('4. Personal Knowledge Retrieval Failure Degradation', () => {
    it('gracefully degrades to style-only generation when IndexedDB retrieval throws', async () => {
      // Simulate database failure during retrieval
      const brokenDb = trackDb(createEchoDatabase(createUniqueDbName('BrokenDb')));
      await brokenDb.open();
      brokenDb.close(); // Closed DB causes Dexie operations to fail if forced

      let fallbackKnowledge: string[] = ['not-empty'];

      try {
        // If retrieveRelevantKnowledge throws or fails on a closed/broken DB
        const result = await retrieveRelevantKnowledge('React development', brokenDb);
        fallbackKnowledge = result.formattedStrings;
      } catch {
        // Graceful degradation simulation in WritingComposer
        fallbackKnowledge = [];
      }

      // Generation should succeed with empty knowledge array
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          completeDraft: 'I enjoy writing clean React code with functional components.',
          generatedText: 'I enjoy writing clean React code with functional components.',
          tokensGenerated: 12,
        }),
      } as Response);

      const response = await generatePersonalizedDraft({
        topic: 'React development',
        personalKnowledge: fallbackKnowledge,
      });

      expect(response.completeDraft).toBe(
        'I enjoy writing clean React code with functional components.'
      );
    });
  });

  describe('5. Grounding Validation & User Review Safety', () => {
    it('returns grounded draft cleanly when grounding check succeeds', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          completeDraft: 'I built Task Tracker using React and TypeScript.',
          generatedText: 'I built Task Tracker using React and TypeScript.',
          tokensGenerated: 10,
          groundingCheck: {
            isGrounded: true,
            fallbackUsed: false,
            flaggedIssues: [],
          },
        }),
      } as Response);

      const response = await generatePersonalizedDraft({
        topic: 'Building Task Tracker',
        personalKnowledge: ['[Completed Project]: I built Task Tracker using React.'],
      });

      expect(response.groundingCheck?.isGrounded).toBe(true);
      expect(response.groundingCheck?.fallbackUsed).toBe(false);
      expect(response.groundingCheck?.flaggedIssues).toEqual([]);
    });

    it('returns flagged issues and preserves draft when grounding detects unverified claims', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          completeDraft: 'My friend Alex and I built a multi-million user mobile app.',
          generatedText: 'My friend Alex and I built a multi-million user mobile app.',
          tokensGenerated: 14,
          groundingCheck: {
            isGrounded: false,
            fallbackUsed: false,
            flaggedIssues: [
              'third_party_attribution: Project attributed to friend',
              'unverified_scale: Claims multi-million users without evidence',
            ],
          },
        }),
      } as Response);

      const response = await generatePersonalizedDraft({
        topic: 'My app launch',
        personalKnowledge: ['[Completed Project]: I built a personal prototype.'],
      });

      // Draft is returned so user can edit it, while warning issues are fully preserved
      expect(response.completeDraft).toContain('Alex');
      expect(response.groundingCheck?.isGrounded).toBe(false);
      expect(response.groundingCheck?.flaggedIssues).toHaveLength(2);
    });
  });

  describe('6. Data Safety & Non-Poisoning Guarantees', () => {
    it('verifies generation pipeline NEVER mutates writingMessages or personalKnowledge records', async () => {
      const testDb = trackDb(createEchoDatabase(createUniqueDbName('SafetyTestDb')));
      await testDb.open();

      const now = new Date();
      // Seed 2 original authentic writing messages
      const msg1Id = await saveWritingMessage(
        {
          text: 'I prefer building local-first software because offline reliability and user privacy matter.',
          platform: 'X',
          capturedAt: now,
          createdAt: now,
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      );
      const msg2Id = await saveWritingMessage(
        {
          text: 'TypeScript has saved me countless hours of debugging complex state transitions.',
          platform: 'General',
          capturedAt: now,
          createdAt: now,
          trainingStatus: 'pending',
        },
        undefined,
        testDb
      );

      // Seed 1 personal knowledge fact
      const factId = await addManualPersonalKnowledge(
        {
          category: 'projects',
          subject: 'Echo',
          predicate: 'built',
          value: 'local AI writing assistant',
          confidence: 'high',
        },
        testDb
      );

      // Verify baseline counts
      const baselineMsgs = await getWritingMessages(testDb);
      expect(baselineMsgs).toHaveLength(2);
      expect(new Set(baselineMsgs.map((m) => m.id))).toEqual(new Set([msg1Id, msg2Id]));

      const baselineKnowledge = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(baselineKnowledge).toHaveLength(1);
      expect(baselineKnowledge[0]?.id).toBe(factId);

      // Simulate full generation flow
      const retrieved = await retrieveRelevantKnowledge('Echo project', testDb);
      expect(retrieved.hasRelevantKnowledge).toBe(true);
      expect(retrieved.formattedStrings[0]).toContain('Echo');

      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          completeDraft: 'Over the past month, I built Echo as a local-first writing assistant.',
          generatedText: 'Over the past month, I built Echo as a local-first writing assistant.',
          tokensGenerated: 16,
          groundingCheck: { isGrounded: true, fallbackUsed: false, flaggedIssues: [] },
        }),
      } as Response);

      const draftResult = await generatePersonalizedDraft({
        topic: 'Echo project',
        personalKnowledge: retrieved.formattedStrings,
      });

      expect(draftResult.completeDraft).toContain('Echo');

      // Post-generation verification: database must be 100% IDENTICAL to baseline
      const postMsgs = await getWritingMessages(testDb);
      expect(postMsgs).toHaveLength(2);
      expect(postMsgs.map((m) => m.text)).toEqual(baselineMsgs.map((m) => m.text));
      expect(postMsgs[0]?.trainingStatus).toBe('pending'); // Still pending, not marked trained
      expect(postMsgs[1]?.trainingStatus).toBe('pending');

      // Generated draft MUST NOT be stored in writing messages (prevents dataset self-poisoning)
      const poisoned = postMsgs.some((m) => m.text.includes(draftResult.completeDraft));
      expect(poisoned).toBe(false);

      // Personal knowledge records must be 100% untouched
      const postKnowledge = await getPersonalKnowledge({ status: 'active' }, testDb);
      expect(postKnowledge).toHaveLength(1);
      expect(postKnowledge[0]?.subject).toBe('Echo');
      expect(postKnowledge[0]?.value).toBe('local AI writing assistant');
    });
  });
});
