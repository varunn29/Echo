import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  checkGenerationServiceHealth,
  generatePersonalizedDraft,
  DEFAULT_GENERATION_SERVICE_URL,
} from './generationClient';

describe('Echo Generation Service Client', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe('checkGenerationServiceHealth', () => {
    it('returns ready status when service returns HTTP 200 with status ready', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => ({
          status: 'ready',
          model: 'Qwen/Qwen3.5-0.8B-Base',
          adapter: 'training-output/qwen3.5-0.8b-lora-v2',
          loadTimeSec: 2.76,
          residentRamMb: 3600.0,
          uptimeSec: 45.2,
        }),
      } as Response);

      const health = await checkGenerationServiceHealth('http://127.0.0.1:11435');
      expect(health.isOnline).toBe(true);
      expect(health.isModelReady).toBe(true);
      expect(health.status).toBe('ready');
      expect(health.model).toBe('Qwen/Qwen3.5-0.8B-Base');
      expect(health.adapter).toBe('training-output/qwen3.5-0.8b-lora-v2');
      expect(health.loadTimeSec).toBe(2.76);
    });

    it('returns loading status when service returns HTTP 503', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 503,
        json: async () => ({
          status: 'error',
          error: 'Model is unavailable. Check server logs.',
        }),
      } as Response);

      const health = await checkGenerationServiceHealth('http://127.0.0.1:11435');
      expect(health.isOnline).toBe(true);
      expect(health.isModelReady).toBe(false);
      expect(health.status).toBe('loading');
      expect(health.error).toContain('loading or unavailable');
    });

    it('returns offline status when connection fails', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

      const health = await checkGenerationServiceHealth('http://127.0.0.1:11435');
      expect(health.isOnline).toBe(false);
      expect(health.isModelReady).toBe(false);
      expect(health.status).toBe('offline');
      expect(health.error).toContain('offline');
    });

    it('returns offline status when health check times out', async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      global.fetch = vi.fn().mockRejectedValue(abortError);

      const health = await checkGenerationServiceHealth('http://127.0.0.1:11435', 100);
      expect(health.isOnline).toBe(false);
      expect(health.status).toBe('offline');
      expect(health.error).toContain('timed out');
    });
  });

  describe('generatePersonalizedDraft - Input Boundaries', () => {
    it('throws error when topic is empty or whitespace', async () => {
      await expect(generatePersonalizedDraft({ topic: '' })).rejects.toThrow(
        'A non-empty topic is required'
      );
      await expect(generatePersonalizedDraft({ topic: '   ' })).rejects.toThrow(
        'A non-empty topic is required'
      );
    });

    it('throws error when topic exceeds 500 characters', async () => {
      const longTopic = 'a'.repeat(501);
      await expect(generatePersonalizedDraft({ topic: longTopic })).rejects.toThrow(
        'maximum allowed length of 500 characters'
      );
    });

    it('throws error when personalKnowledge exceeds 10 items', async () => {
      const items = Array.from({ length: 11 }, (_, i) => `Fact ${i}`);
      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic', personalKnowledge: items })
      ).rejects.toThrow('At most 10 personal knowledge items');
    });

    it('throws error when a personalKnowledge item exceeds 500 characters', async () => {
      const longItem = 'x'.repeat(501);
      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic', personalKnowledge: [longItem] })
      ).rejects.toThrow('at most 500 characters');
    });

    it('throws error when leadIn exceeds 100 characters', async () => {
      const longLeadIn = 'L'.repeat(101);
      await expect(
        generatePersonalizedDraft({ topic: 'Valid topic', leadIn: longLeadIn })
      ).rejects.toThrow('maximum allowed length of 100 characters');
    });
  });

  describe('generatePersonalizedDraft - Service Interactions', () => {
    it('successfully receives and returns generated draft from local service', async () => {
      const mockResponse = {
        completeDraft: 'Over the last few months, I built Task Tracker using React and TypeScript.',
        generatedText: 'I built Task Tracker using React and TypeScript.',
        rawGeneratedText: 'I built Task Tracker using React and TypeScript.',
        leadIn: 'Over the last few months, ',
        tokensGenerated: 45,
        tokenCount: 45,
        finalTokenCount: 45,
        rawTokenCount: 45,
        wordCount: 12,
        sentenceCount: 1,
        paragraphCount: 1,
        source: 'lora_sampled',
        elapsedSec: 1.15,
        tokensPerSec: 39.1,
        peakRamMb: 3750.2,
        metadata: {
          topic: 'Building Task Tracker',
          mode: 'sampled',
          model: 'Qwen/Qwen3.5-0.8B-Base',
          adapter: 'training-output/qwen3.5-0.8b-lora-v2',
          repetitionPenalty: 1.15,
          maxTokens: 85,
          source: 'lora_sampled',
          seed: 42,
          tokenCount: 45,
          finalTokenCount: 45,
          rawTokenCount: 45,
        },
      };

      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => mockResponse,
      } as Response);

      const res = await generatePersonalizedDraft({
        topic: 'Building Task Tracker',
        personalKnowledge: ['[projects] Task Tracker built task application'],
        leadIn: 'Over the last few months, ',
      });

      expect(res.completeDraft).toBe(mockResponse.completeDraft);
      expect(res.generatedText).toBe(mockResponse.generatedText);
      expect(res.rawGeneratedText).toBe(mockResponse.rawGeneratedText);
      expect(res.tokensGenerated).toBe(45);
      expect(res.tokenCount).toBe(45);
      expect(res.finalTokenCount).toBe(45);
      expect(res.rawTokenCount).toBe(45);
      expect(res.wordCount).toBe(12);
      expect(res.sentenceCount).toBe(1);
      expect(res.paragraphCount).toBe(1);
      expect(res.source).toBe('lora_sampled');
      expect(res.elapsedSec).toBe(1.15);
      expect(res.metadata?.adapter).toContain('lora-v2');
      expect(res.metadata?.tokenCount).toBe(45);
      expect(res.metadata?.finalTokenCount).toBe(45);
      expect(res.metadata?.rawTokenCount).toBe(45);
    });

    it('throws user-friendly error on HTTP 503 model loading', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 503,
        json: async () => ({ error: 'Model is unavailable. Check server logs.' }),
      } as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Test topic' })
      ).rejects.toThrow('Model is currently loading or unavailable');
    });

    it('throws user-friendly error on HTTP 403 origin rejection', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 403,
        json: async () => ({ error: 'Forbidden: Unauthorized origin' }),
      } as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Test topic' })
      ).rejects.toThrow('unauthorized origin');
    });

    it('throws sanitized error on HTTP 500 generation failure', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 500,
        json: async () => ({ error: 'Model generation failed. Check server logs.' }),
      } as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Test topic' })
      ).rejects.toThrow('Model generation failed');
    });

    it('throws error when server response is not valid JSON or lacks completeDraft', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        status: 200,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      } as unknown as Response);

      await expect(
        generatePersonalizedDraft({ topic: 'Test topic' })
      ).rejects.toThrow('unparseable response');
    });

    it('throws error when generation request times out', async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      global.fetch = vi.fn().mockRejectedValue(abortError);

      await expect(
        generatePersonalizedDraft({ topic: 'Test topic' }, { timeoutMs: 50 })
      ).rejects.toThrow('Generation request timed out');
    });

    it('correctly parses groundingCheck metadata for grounded, fallback, and ungrounded states', async () => {
      // 1. Clean grounded draft
      global.fetch = vi.fn().mockResolvedValueOnce({
        status: 200,
        json: async () => ({
          completeDraft: 'Valid grounded text',
          generatedText: 'Valid grounded text',
          groundingCheck: { isGrounded: true, flaggedIssues: [], fallbackUsed: false },
        }),
      } as Response);

      const cleanRes = await generatePersonalizedDraft({ topic: 'Topic' });
      expect(cleanRes.groundingCheck?.isGrounded).toBe(true);
      expect(cleanRes.groundingCheck?.fallbackUsed).toBe(false);
      expect(cleanRes.groundingCheck?.flaggedIssues).toEqual([]);

      // 2. Fallback-used draft
      global.fetch = vi.fn().mockResolvedValueOnce({
        status: 200,
        json: async () => ({
          completeDraft: 'Fallback rescued text',
          generatedText: 'Fallback rescued text',
          groundingCheck: { isGrounded: true, flaggedIssues: [], fallbackUsed: true },
        }),
      } as Response);

      const fallbackRes = await generatePersonalizedDraft({ topic: 'Topic' });
      expect(fallbackRes.groundingCheck?.isGrounded).toBe(true);
      expect(fallbackRes.groundingCheck?.fallbackUsed).toBe(true);

      // 3. Flagged ungrounded draft
      global.fetch = vi.fn().mockResolvedValueOnce({
        status: 200,
        json: async () => ({
          completeDraft: 'Flagged ungrounded text',
          generatedText: 'Flagged ungrounded text',
          groundingCheck: {
            isGrounded: false,
            flaggedIssues: ['third_party_attribution: Project attributed to friend'],
            fallbackUsed: true,
          },
        }),
      } as Response);

      const flaggedRes = await generatePersonalizedDraft({ topic: 'Topic' });
      expect(flaggedRes.groundingCheck?.isGrounded).toBe(false);
      expect(flaggedRes.groundingCheck?.fallbackUsed).toBe(true);
      expect(flaggedRes.groundingCheck?.flaggedIssues).toHaveLength(1);

      // 4. Missing/malformed groundingCheck metadata: does not crash
      global.fetch = vi.fn().mockResolvedValueOnce({
        status: 200,
        json: async () => ({
          completeDraft: 'Draft with no metadata',
          generatedText: 'Draft with no metadata',
        }),
      } as Response);

      const missingRes = await generatePersonalizedDraft({ topic: 'Topic' });
      expect(missingRes.completeDraft).toBe('Draft with no metadata');
      expect(missingRes.groundingCheck).toBeUndefined();
    });
  });
});
