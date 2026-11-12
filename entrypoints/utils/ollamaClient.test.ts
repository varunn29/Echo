import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkOllamaStatus,
  DEFAULT_HEALTH_CHECK_TIMEOUT_MS,
  DEFAULT_TIMEOUT_MS,
  isModelAvailable,
  isOllamaRunning,
  OllamaError,
  queryOllamaJson,
} from './ollamaClient';

describe('Ollama Client Suite', function () {
  afterEach(function () {
    vi.restoreAllMocks();
  });

  describe('isOllamaRunning', function () {
    it('should return true when Ollama returns 200 OK', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 200 }));
      const result = await isOllamaRunning();
      expect(result).toBe(true);
    });

    it('should return false when fetch rejects with connection error', async function () {
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('ECONNREFUSED'));
      const result = await isOllamaRunning();
      expect(result).toBe(false);
    });

    it('should return false on non-200 status', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 500 }));
      const result = await isOllamaRunning();
      expect(result).toBe(false);
    });
  });

  describe('isModelAvailable', function () {
    it('should return true when target model is present in tags list', async function () {
      const mockTags = {
        models: [
          { name: 'qwen3.5:2b', model: 'qwen3.5:2b', size: 2000000000 },
          { name: 'llama3:8b', model: 'llama3:8b', size: 4000000000 },
        ],
      };
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockTags), { status: 200 })
      );

      const result = await isModelAvailable('qwen3.5:2b');
      expect(result).toBe(true);
    });

    it('should return false when model is not installed', async function () {
      const mockTags = {
        models: [{ name: 'llama3:8b', model: 'llama3:8b', size: 4000000000 }],
      };
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockTags), { status: 200 })
      );

      const result = await isModelAvailable('qwen3.5:2b');
      expect(result).toBe(false);
    });
  });

  describe('checkOllamaStatus', function () {
    it('should have DEFAULT_HEALTH_CHECK_TIMEOUT_MS set to 10000ms', function () {
      expect(DEFAULT_HEALTH_CHECK_TIMEOUT_MS).toBe(10000);
    });

    it('should return healthy status when Ollama and model are available', async function () {
      const mockTags = {
        models: [{ name: 'qwen3.5:2b', model: 'qwen3.5:2b', size: 2000000000 }],
      };
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockTags), { status: 200 })
      );

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(true);
      expect(status.isModelReady).toBe(true);
      expect(status.availability).toBe('online');
      expect(status.error).toBeUndefined();
    });

    it('should identify origin rejection (403) and mark availability blocked', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 403 }));

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(true);
      expect(status.isModelReady).toBe(false);
      expect(status.availability).toBe('blocked');
      expect(status.error).toContain('Ollama rejected the extension origin (403 Forbidden)');
    });

    it('should report offline error when fetch fails with network connection error', async function () {
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Failed to connect'));

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(false);
      expect(status.isModelReady).toBe(false);
      expect(status.availability).toBe('offline');
      expect(status.error).toContain('Ollama is offline');
    });

    it('should classify health-check AbortError as unknown/transient, not offline', async function () {
      const abortError = new DOMException('The operation was aborted', 'AbortError');
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(abortError);

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(false);
      expect(status.isModelReady).toBe(false);
      expect(status.availability).toBe('unknown');
      expect(status.error).toContain('Ollama health check timed out');
      expect(status.error).not.toContain('Ollama is offline');
    });

    it('should report availability blocked when model is not installed', async function () {
      const mockTags = {
        models: [{ name: 'llama3:8b', model: 'llama3:8b', size: 4000000000 }],
      };
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockTags), { status: 200 })
      );

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(true);
      expect(status.isModelReady).toBe(false);
      expect(status.availability).toBe('blocked');
      expect(status.error).toContain('is not installed in Ollama');
    });
  });

  describe('queryOllamaJson', function () {
    it('should have DEFAULT_TIMEOUT_MS configured to 120000ms (120s for CPU inference)', function () {
      expect(DEFAULT_TIMEOUT_MS).toBe(120000);
    });

    it('should classify default inference timeout as kind timeout with 120s message', async function () {
      const abortError = new DOMException('The operation was aborted', 'AbortError');
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(abortError);

      let caughtErr!: OllamaError;
      try {
        await queryOllamaJson('Test');
      } catch (err: unknown) {
        caughtErr = err as OllamaError;
      }

      expect(caughtErr).toBeInstanceOf(OllamaError);
      expect(caughtErr.kind).toBe('timeout');
      expect(caughtErr.message).toBe('Ollama request timed out after 120s.');
    });

    it('should respect custom timeout override and report actual seconds on timeout', async function () {
      const abortError = new DOMException('The operation was aborted', 'AbortError');
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(abortError);

      let caughtErr!: OllamaError;
      try {
        await queryOllamaJson('Test', { timeoutMs: 5000 });
      } catch (err: unknown) {
        caughtErr = err as OllamaError;
      }

      expect(caughtErr).toBeInstanceOf(OllamaError);
      expect(caughtErr.kind).toBe('timeout');
      expect(caughtErr.message).toBe('Ollama request timed out after 5s.');
    });
    it('should return parsed response content on success', async function () {
      const mockResponse = {
        message: {
          role: 'assistant',
          content: '{"knowledge": []}',
        },
      };

      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockResponse), { status: 200 })
      );

      const result = await queryOllamaJson('Test prompt');
      expect(result).toBe('{"knowledge": []}');
    });

    it('should throw clear error on 404 model missing', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response('Not found', { status: 404, statusText: 'Not Found' })
      );

      await expect(queryOllamaJson('Test', { model: 'missing-model' })).rejects.toThrow(
        'Model "missing-model" not found in Ollama.'
      );
    });

    it('should throw clear error on empty response content', async function () {
      const mockResponse = {
        message: {
          role: 'assistant',
          content: '   ',
        },
      };

      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify(mockResponse), { status: 200 })
      );

      await expect(queryOllamaJson('Test')).rejects.toThrow(
        'Ollama returned an empty response.'
      );
    });

    it('should throw clear error on 403 Forbidden with exact origin recommendation', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response('Forbidden', { status: 403, statusText: 'Forbidden' })
      );

      await expect(queryOllamaJson('Test')).rejects.toThrow(
        'Ollama rejected the request origin (403 Forbidden). Set OLLAMA_ORIGINS to allow this extension'
      );
    });
  });
});

