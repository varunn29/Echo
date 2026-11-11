import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkOllamaStatus,
  isModelAvailable,
  isOllamaRunning,
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
      expect(status.error).toBeUndefined();
    });

    it('should identify origin rejection (403)', async function () {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 403 }));

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(true);
      expect(status.isModelReady).toBe(false);
      expect(status.error).toContain('Ollama rejected the extension origin (403 Forbidden)');
    });

    it('should report offline error when fetch fails', async function () {
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Failed to connect'));

      const status = await checkOllamaStatus('qwen3.5:2b');
      expect(status.isRunning).toBe(false);
      expect(status.isModelReady).toBe(false);
      expect(status.error).toContain('Ollama is offline');
    });
  });

  describe('queryOllamaJson', function () {
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

