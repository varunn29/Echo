import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { handleOffscreenMessage, resetWorkerForTesting, setWorkerInstanceForTesting } from './main';
import { handleWorkerMessage, type WorkerInferenceRequest } from '../utils/ollamaWorker';
import { OllamaError } from '../utils/ollamaClient';

describe('Offscreen Main and Web Worker Suite', function () {
  afterEach(function () {
    vi.restoreAllMocks();
    resetWorkerForTesting();
  });

  describe('handleWorkerMessage (Dedicated Worker)', function () {
    it('should ignore non-QUERY_OLLAMA messages', async function () {
      const postFn = vi.fn();
      await handleWorkerMessage({ id: '1', type: 'OTHER' as any, prompt: 'hi' }, postFn);
      expect(postFn).not.toHaveBeenCalled();
    });

    it('should post success response when direct inference succeeds', async function () {
      const postFn = vi.fn();
      const executeFn = vi.fn().mockResolvedValue('{"knowledge": [{"fact": "test"}]}');

      const req: WorkerInferenceRequest = {
        id: 'inf_123',
        type: 'QUERY_OLLAMA',
        prompt: 'User prompt',
        options: { timeoutMs: 60000 },
      };

      await handleWorkerMessage(req, postFn, executeFn);

      expect(executeFn).toHaveBeenCalledWith('User prompt', { timeoutMs: 60000 });
      expect(postFn).toHaveBeenCalledWith({
        id: 'inf_123',
        success: true,
        result: '{"knowledge": [{"fact": "test"}]}',
      });
    });

    it('should post typed error response preserving kind and statusCode when inference fails', async function () {
      const postFn = vi.fn();
      const executeFn = vi.fn().mockRejectedValue(
        new OllamaError('Ollama returned HTTP 500: internal server error', 'http_error', 500)
      );

      const req: WorkerInferenceRequest = {
        id: 'inf_456',
        type: 'QUERY_OLLAMA',
        prompt: 'Failing prompt',
      };

      await handleWorkerMessage(req, postFn, executeFn);

      expect(postFn).toHaveBeenCalledWith({
        id: 'inf_456',
        success: false,
        error: {
          message: 'Ollama returned HTTP 500: internal server error',
          kind: 'http_error',
          statusCode: 500,
        },
      });
    });
  });

  describe('handleOffscreenMessage (Offscreen Host)', function () {
    it('should ignore messages not targeted to echo-offscreen', function () {
      const sendResponse = vi.fn();
      const handled = handleOffscreenMessage({ target: 'other', type: 'EXECUTE_OLLAMA_INFERENCE' }, {}, sendResponse);
      expect(handled).toBe(false);
      expect(sendResponse).not.toHaveBeenCalled();
    });

    it('should reject invalid payload with message_failure', function () {
      const sendResponse = vi.fn();
      const handled = handleOffscreenMessage(
        { target: 'echo-offscreen', type: 'EXECUTE_OLLAMA_INFERENCE', id: '', prompt: 123 },
        {},
        sendResponse
      );
      expect(handled).toBe(false);
      expect(sendResponse).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({ kind: 'message_failure' }),
        })
      );
    });

    it('should forward request to worker and deliver worker response via sendResponse', function () {
      const sendResponse = vi.fn();
      const mockWorker = {
        postMessage: vi.fn(),
        terminate: vi.fn(),
        onmessage: null as any,
        onerror: null as any,
      };
      setWorkerInstanceForTesting(mockWorker as any);

      const handled = handleOffscreenMessage(
        {
          target: 'echo-offscreen',
          type: 'EXECUTE_OLLAMA_INFERENCE',
          id: 'msg_999',
          prompt: 'Hello offscreen',
          options: { timeoutMs: 5000 },
        },
        {},
        sendResponse
      );

      expect(handled).toBe(true);
      expect(mockWorker.postMessage).toHaveBeenCalledWith({
        id: 'msg_999',
        type: 'QUERY_OLLAMA',
        prompt: 'Hello offscreen',
        options: { timeoutMs: 5000 },
      });

      // Simulate worker message response
      mockWorker.onmessage({
        data: {
          id: 'msg_999',
          success: true,
          result: '{"knowledge": []}',
        },
      } as any);

      expect(sendResponse).toHaveBeenCalledWith({
        success: true,
        result: '{"knowledge": []}',
      });
    });

    it('should handle worker crash error and notify pending requests', function () {
      const sendResponse = vi.fn();
      const mockWorker = {
        postMessage: vi.fn(),
        terminate: vi.fn(),
        onmessage: null as any,
        onerror: null as any,
      };
      setWorkerInstanceForTesting(mockWorker as any);

      handleOffscreenMessage(
        {
          target: 'echo-offscreen',
          type: 'EXECUTE_OLLAMA_INFERENCE',
          id: 'msg_crash',
          prompt: 'Crash prompt',
        },
        {},
        sendResponse
      );

      // Simulate worker error crash event
      mockWorker.onerror(new Error('Out of memory thread termination'));

      expect(sendResponse).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            message: expect.stringContaining('Ollama Web Worker crashed'),
            kind: 'message_failure',
          }),
        })
      );
      expect(mockWorker.terminate).toHaveBeenCalled();
    });
  });
});
