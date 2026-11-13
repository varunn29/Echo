import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  closeOffscreenDocument,
  ensureOffscreenDocument,
  hasOffscreenDocument,
  isOffscreenSupportedAndRequired,
  OFFSCREEN_DOCUMENT_PATH,
  queryOllamaViaOffscreen,
  setOffscreenBridgeEnabledForTesting,
} from './offscreenManager';
import { OllamaError } from './ollamaClient';

describe('Offscreen Manager Suite', function () {
  let mockChrome: any;

  beforeEach(function () {
    setOffscreenBridgeEnabledForTesting(null);

    mockChrome = {
      runtime: {
        getURL: vi.fn((path: string) => `chrome-extension://test-id/${path}`),
        getContexts: vi.fn().mockResolvedValue([]),
        sendMessage: vi.fn(),
        lastError: null,
      },
      offscreen: {
        createDocument: vi.fn().mockResolvedValue(undefined),
        closeDocument: vi.fn().mockResolvedValue(undefined),
        hasDocument: vi.fn().mockResolvedValue(false),
      },
    };

    (globalThis as any).chrome = mockChrome;
  });

  afterEach(function () {
    vi.restoreAllMocks();
    setOffscreenBridgeEnabledForTesting(null);
    delete (globalThis as any).chrome;
  });

  describe('isOffscreenSupportedAndRequired', function () {
    it('should return true when chrome.offscreen is available and window is undefined', function () {
      expect(isOffscreenSupportedAndRequired()).toBe(true);
    });

    it('should return false when chrome.offscreen is absent', function () {
      delete mockChrome.offscreen;
      expect(isOffscreenSupportedAndRequired()).toBe(false);
    });

    it('should return false when window is defined (e.g., in UI or content script)', function () {
      (globalThis as any).window = {};
      expect(isOffscreenSupportedAndRequired()).toBe(false);
      delete (globalThis as any).window;
    });

    it('should respect manual test override when set', function () {
      setOffscreenBridgeEnabledForTesting(false);
      expect(isOffscreenSupportedAndRequired()).toBe(false);

      setOffscreenBridgeEnabledForTesting(true);
      expect(isOffscreenSupportedAndRequired()).toBe(true);
    });
  });

  describe('hasOffscreenDocument', function () {
    it('should return true when getContexts finds an active OFFSCREEN_DOCUMENT', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([
        { contextType: 'OFFSCREEN_DOCUMENT', documentUrl: 'chrome-extension://test-id/offscreen.html' },
      ]);

      const exists = await hasOffscreenDocument();
      expect(exists).toBe(true);
      expect(mockChrome.runtime.getContexts).toHaveBeenCalledWith({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: ['chrome-extension://test-id/offscreen.html'],
      });
    });

    it('should return false when getContexts returns empty array', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);
      const exists = await hasOffscreenDocument();
      expect(exists).toBe(false);
    });

    it('should fall back to chrome.offscreen.hasDocument if getContexts is unavailable', async function () {
      delete mockChrome.runtime.getContexts;
      mockChrome.offscreen.hasDocument.mockResolvedValue(true);

      const exists = await hasOffscreenDocument();
      expect(exists).toBe(true);
      expect(mockChrome.offscreen.hasDocument).toHaveBeenCalled();
    });

    it('should return false when chrome is not defined', async function () {
      delete (globalThis as any).chrome;
      const exists = await hasOffscreenDocument();
      expect(exists).toBe(false);
    });
  });

  describe('ensureOffscreenDocument', function () {
    it('should create offscreen document with genuine WORKERS reason and correct URL', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);

      await ensureOffscreenDocument();

      expect(mockChrome.offscreen.createDocument).toHaveBeenCalledWith({
        url: OFFSCREEN_DOCUMENT_PATH,
        reasons: ['WORKERS'],
        justification: expect.stringContaining('dedicated Web Worker'),
      });
    });

    it('should not create a second document if one already exists', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);

      await ensureOffscreenDocument();

      expect(mockChrome.offscreen.createDocument).not.toHaveBeenCalled();
    });

    it('should gracefully ignore race-condition errors if document was created concurrently', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);
      mockChrome.offscreen.createDocument.mockRejectedValue(
        new Error('Only a single offscreen document may be created.')
      );

      await expect(ensureOffscreenDocument()).resolves.toBeUndefined();
    });

    it('should rethrow unexpected document creation errors', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);
      mockChrome.offscreen.createDocument.mockRejectedValue(
        new Error('Permission denied to create offscreen document.')
      );

      await expect(ensureOffscreenDocument()).rejects.toThrow(
        'Permission denied to create offscreen document.'
      );
    });
  });

  describe('closeOffscreenDocument', function () {
    it('should close offscreen document when one exists', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);

      await closeOffscreenDocument();

      expect(mockChrome.offscreen.closeDocument).toHaveBeenCalled();
    });

    it('should do nothing if no offscreen document is active', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);

      await closeOffscreenDocument();

      expect(mockChrome.offscreen.closeDocument).not.toHaveBeenCalled();
    });

    it('should gracefully catch and ignore errors during document closure', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);
      mockChrome.offscreen.closeDocument.mockRejectedValue(new Error('Document already closed.'));

      await expect(closeOffscreenDocument()).resolves.toBeUndefined();
    });

    it('should rethrow unexpected errors during document closure', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);
      mockChrome.offscreen.closeDocument.mockRejectedValue(new Error('Permission denied to close offscreen document.'));

      await expect(closeOffscreenDocument()).rejects.toThrow('Permission denied to close offscreen document.');
    });
  });

  describe('queryOllamaViaOffscreen', function () {
    it('should successfully dispatch prompt to offscreen worker and return string response', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([]);
      mockChrome.runtime.sendMessage.mockImplementation(
        (_msg: any, callback: (res: any) => void) => {
          callback({
            success: true,
            result: '{"knowledge": []}',
          });
        }
      );

      const result = await queryOllamaViaOffscreen('Test prompt', { timeoutMs: 5000 });

      expect(result).toBe('{"knowledge": []}');
      expect(mockChrome.offscreen.createDocument).toHaveBeenCalled();
      expect(mockChrome.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          target: 'echo-offscreen',
          type: 'EXECUTE_OLLAMA_INFERENCE',
          prompt: 'Test prompt',
        }),
        expect.any(Function)
      );
    });

    it('should preserve and rethrow typed OllamaError with original errorKind and statusCode', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);
      mockChrome.runtime.sendMessage.mockImplementation(
        (_msg: any, callback: (res: any) => void) => {
          callback({
            success: false,
            error: {
              message: 'Ollama returned HTTP 500: internal server error',
              kind: 'http_error',
              statusCode: 500,
            },
          });
        }
      );

      await expect(queryOllamaViaOffscreen('Test prompt')).rejects.toSatisfy((err: any) => {
        expect(err).toBeInstanceOf(OllamaError);
        expect(err.kind).toBe('http_error');
        expect(err.statusCode).toBe(500);
        expect(err.message).toContain('HTTP 500');
        return true;
      });
    });

    it('should reject with offline OllamaError when runtime messaging fails', async function () {
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);
      mockChrome.runtime.sendMessage.mockImplementation(
        (_msg: any, callback: (res: any) => void) => {
          mockChrome.runtime.lastError = { message: 'Could not establish connection. Receiving end does not exist.' };
          callback(undefined);
        }
      );

      await expect(queryOllamaViaOffscreen('Test prompt')).rejects.toSatisfy((err: any) => {
        expect(err).toBeInstanceOf(OllamaError);
        expect(err.kind).toBe('offline');
        expect(err.message).toContain('Could not establish connection');
        return true;
      });
    });

    it('should reject with timeout OllamaError when offscreen bridge times out', async function () {
      vi.useFakeTimers();
      mockChrome.runtime.getContexts.mockResolvedValue([{ contextType: 'OFFSCREEN_DOCUMENT' }]);
      // sendMessage does not call callback (simulating hung/stalled worker)
      mockChrome.runtime.sendMessage.mockImplementation(() => {});

      const promise = queryOllamaViaOffscreen('Test prompt', { timeoutMs: 1000 });
      // Attach rejection handler to prevent unhandled rejection during timer advance
      let caughtError: any = null;
      promise.catch((e) => {
        caughtError = e;
      });

      // Timeout is timeoutMs + 10000ms = 11000ms
      await vi.advanceTimersByTimeAsync(11500);

      expect(caughtError).toBeInstanceOf(OllamaError);
      expect(caughtError.kind).toBe('timeout');
      expect(caughtError.message).toContain('Offscreen request timed out');

      vi.useRealTimers();
    });
  });
});
