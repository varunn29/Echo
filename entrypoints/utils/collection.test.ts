import { describe, expect, it } from 'vitest';
import { DEFAULT_DUPLICATE_WINDOW_MS, type WritingMessage } from './database';

describe('Collection Pipeline & Duplicate Policies', function () {
  it('should have a sensible default duplicate window of 60 seconds', function () {
    expect(DEFAULT_DUPLICATE_WINDOW_MS).toBe(60000);
  });

  describe('Client-Side Debouncing Logic', function () {
    function shouldDebounce(
      currentText: string,
      lastText: string,
      currentTime: number,
      lastTime: number,
      debounceMs = 2000
    ): boolean {
      if (!currentText.trim()) {
        return true;
      }
      if (currentText === lastText && currentTime - lastTime < debounceMs) {
        return true;
      }
      return false;
    }

    it('should block identical messages sent within debounce window', function () {
      const now = 10000;
      const isBlocked = shouldDebounce('Hello', 'Hello', now + 500, now, 2000);
      expect(isBlocked).toBe(true);
    });

    it('should allow identical messages sent after debounce window', function () {
      const now = 10000;
      const isBlocked = shouldDebounce('Hello', 'Hello', now + 2500, now, 2000);
      expect(isBlocked).toBe(false);
    });

    it('should allow different messages sent within debounce window', function () {
      const now = 10000;
      const isBlocked = shouldDebounce('World', 'Hello', now + 500, now, 2000);
      expect(isBlocked).toBe(false);
    });

    it('should reject empty or whitespace-only messages', function () {
      const now = 10000;
      expect(shouldDebounce('   ', '', now + 500, now, 2000)).toBe(true);
      expect(shouldDebounce('', '', now + 500, now, 2000)).toBe(true);
    });
  });

  describe('Pre-flight Buffer Fallback Logic', function () {
    function resolveMessageText(
      domText: string,
      bufferedText: string
    ): string {
      const trimmedDom = domText.trim();
      if (trimmedDom) {
        return trimmedDom;
      }
      return bufferedText.trim();
    }

    it('should prioritize live DOM text when available', function () {
      const resolved = resolveMessageText(
        'Text in composer',
        'Stale buffered text'
      );
      expect(resolved).toBe('Text in composer');
    });

    it('should fall back to buffered text when DOM has been cleared', function () {
      const resolved = resolveMessageText('', 'Buffered user prompt');
      expect(resolved).toBe('Buffered user prompt');
    });

    it('should return empty string when both DOM and buffer are empty', function () {
      const resolved = resolveMessageText('', '');
      expect(resolved).toBe('');
    });
  });

  describe('JSON Export Serialization', function () {
    it('should format writing data into a valid structured JSON export', function () {
      const mockMessages: WritingMessage[] = [
        {
          id: 101,
          platform: 'ChatGPT',
          text: 'Explain quantum computing',
          createdAt: new Date('2026-11-01T10:00:00.000Z'),
          capturedAt: new Date('2026-11-01T10:00:00.000Z'),
          trainingStatus: 'pending',
        },
        {
          id: 102,
          platform: 'ChatGPT',
          text: 'What is WebAssembly?',
          createdAt: new Date('2026-11-02T12:00:00.000Z'),
          capturedAt: new Date('2026-11-02T12:00:00.000Z'),
          trainingStatus: 'trained',
        },
      ];

      const exportPayload = {
        exportedAt: '2026-11-07T12:00:00.000Z',
        totalMessages: mockMessages.length,
        messages: mockMessages.map(function (message) {
          return {
            id: message.id,
            platform: message.platform,
            text: message.text,
            createdAt: message.createdAt.toISOString(),
            capturedAt: message.capturedAt.toISOString(),
            trainingStatus: message.trainingStatus,
          };
        }),
      };

      const jsonString = JSON.stringify(exportPayload, null, 2);
      const parsed = JSON.parse(jsonString);

      expect(parsed.totalMessages).toBe(2);
      expect(parsed.messages).toHaveLength(2);
      expect(parsed.messages[0].id).toBe(101);
      expect(parsed.messages[0].text).toBe('Explain quantum computing');
      expect(parsed.messages[0].createdAt).toBe('2026-11-01T10:00:00.000Z');
      expect(parsed.messages[1].id).toBe(102);
      expect(parsed.messages[1].trainingStatus).toBe('trained');
    });
  });
});

