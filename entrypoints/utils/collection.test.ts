import { describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_DUPLICATE_WINDOW_MS,
  type WritingMessage,
} from './database';
import {
  CLIENT_DEBOUNCE_MS,
  deliverMessageWithRetry,
  extractUserMessageText,
  findComposerElement,
  getElementText,
  isAssistantMessageElement,
  isSendButton,
  isUserMessageElement,
  resolveMessageText,
  shouldDebounce,
  type SaveMessageResponse,
} from './collectionHelper';

describe('Collection Pipeline & Duplicate Policies', function () {
  it('should have a sensible default duplicate window of 60 seconds', function () {
    expect(DEFAULT_DUPLICATE_WINDOW_MS).toBe(60000);
  });

  describe('Client-Side Debouncing Logic', function () {
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

  describe('Send Button Detection Across Dynamic Layouts', function () {
    function createMockElement(tag: string, attrs: Record<string, string> = {}, parent?: any) {
      const el: any = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        parentElement: parent || null,
        getAttribute(name: string) {
          return attrs[name] ?? null;
        },
        hasAttribute(name: string) {
          return name in attrs;
        },
        matches(selector: string) {
          if (selector.includes('data-testid="send-button"') && attrs['data-testid'] === 'send-button') return true;
          if (selector.includes('data-testid="fruitjuice-send-button"') && attrs['data-testid'] === 'fruitjuice-send-button') return true;
          if (selector.includes('data-testid="composer-send-button"') && attrs['data-testid'] === 'composer-send-button') return true;
          if (selector.includes('composer-submit-button') && attrs['id'] === 'composer-submit-button') return true;
          if (selector.includes('aria-label*="Send" i') && attrs['aria-label'] && /send/i.test(attrs['aria-label'])) return true;
          if (selector.includes('aria-label*="Submit" i') && attrs['aria-label'] && /submit/i.test(attrs['aria-label'])) return true;
          if (selector.includes('button[type="submit"]') && tag === 'button' && attrs['type'] === 'submit') return true;
          if (selector.includes('data-testid="stop-button"') && attrs['data-testid'] === 'stop-button') return true;
          if (selector.includes('aria-label*="Stop" i') && attrs['aria-label'] && /stop/i.test(attrs['aria-label'])) return true;
          if (selector.includes('data-message-author-role="assistant"') && attrs['data-message-author-role'] === 'assistant') return true;
          return false;
        },
        closest(selector: string) {
          let curr = el;
          while (curr) {
            if (curr.matches(selector)) return curr;
            curr = curr.parentElement;
          }
          return null;
        },
      };
      return el;
    }

    it('should recognize standard data-testid="send-button"', function () {
      const btn = createMockElement('button', { 'data-testid': 'send-button' });
      expect(isSendButton(btn)).toBe(true);
    });

    it('should recognize newer data-testid="fruitjuice-send-button"', function () {
      const btn = createMockElement('button', { 'data-testid': 'fruitjuice-send-button' });
      expect(isSendButton(btn)).toBe(true);
    });

    it('should recognize button with aria-label="Send prompt"', function () {
      const btn = createMockElement('button', { 'aria-label': 'Send prompt' });
      expect(isSendButton(btn)).toBe(true);
    });

    it('should recognize inner SVG and path elements inside Send button', function () {
      const btn = createMockElement('button', { 'data-testid': 'send-button' });
      const svg = createMockElement('svg', {}, btn);
      const path = createMockElement('path', {}, svg);

      expect(isSendButton(path)).toBe(true);
      expect(isSendButton(svg)).toBe(true);
    });

    it('should reject Stop Generation buttons', function () {
      const stopBtn = createMockElement('button', { 'data-testid': 'stop-button', 'aria-label': 'Stop streaming' });
      const path = createMockElement('path', {}, stopBtn);

      expect(isSendButton(stopBtn)).toBe(false);
      expect(isSendButton(path)).toBe(false);
    });

    it('should reject assistant message action buttons (thumbs up, copy)', function () {
      const assistantTurn = createMockElement('div', { 'data-message-author-role': 'assistant' });
      const actionBtn = createMockElement('button', { 'aria-label': 'Copy message' }, assistantTurn);

      expect(isSendButton(actionBtn)).toBe(false);
    });
  });

  describe('Composer Text Extraction & Resolution', function () {
    it('should extract trimmed value from mock textarea', function () {
      const mockTextarea = {
        value: '   What is machine learning?   ',
      };
      expect(getElementText(mockTextarea as any)).toBe('What is machine learning?');
    });

    it('should extract trimmed innerText from mock contenteditable ProseMirror', function () {
      const mockProseMirror = {
        innerText: '\nI love programming with TypeScript.\n\n',
      };
      expect(getElementText(mockProseMirror as any)).toBe('I love programming with TypeScript.');
    });

    it('should return empty string for null composer', function () {
      expect(getElementText(null)).toBe('');
    });
  });

  describe('Background Service Worker Communication with Retry', function () {
    it('should successfully deliver message when background responds on first try', async function () {
      const mockSend = vi.fn().mockResolvedValue({
        success: true,
        status: 'saved',
        id: 42,
      });

      const response = await deliverMessageWithRetry('Hello Echo', mockSend, 3, [1, 2, 3]);

      expect(response).not.toBeNull();
      expect(response?.success).toBe(true);
      expect(response?.status).toBe('saved');
      expect(response?.id).toBe(42);
      expect(mockSend).toHaveBeenCalledTimes(1);
    });

    it('should retry and succeed when service worker is waking up from dormancy', async function () {
      let attempts = 0;
      const mockSend = vi.fn().mockImplementation(async () => {
        attempts++;
        if (attempts === 1) {
          throw new Error('Could not establish connection. Receiving end does not exist.');
        }
        return {
          success: true,
          status: 'saved',
          id: 43,
        };
      });

      const response = await deliverMessageWithRetry('Recovered message', mockSend, 3, [5, 10, 15]);

      expect(response).not.toBeNull();
      expect(response?.status).toBe('saved');
      expect(response?.id).toBe(43);
      expect(attempts).toBe(2);
      expect(mockSend).toHaveBeenCalledTimes(2);
    });

    it('should handle permanent failure gracefully and return null after exceeding retries', async function () {
      const mockSend = vi.fn().mockRejectedValue(
        new Error('Could not establish connection. Receiving end does not exist.')
      );

      const response = await deliverMessageWithRetry('Doomed message', mockSend, 2, [5, 10]);

      expect(response).toBeNull();
      expect(mockSend).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });

    it('should immediately return null for empty text without calling background', async function () {
      const mockSend = vi.fn();
      const response = await deliverMessageWithRetry('   ', mockSend);

      expect(response).toBeNull();
      expect(mockSend).not.toHaveBeenCalled();
    });
  });

  describe('10 Consecutive Messages Collection Sequence', function () {
    it('should collect and persist 10 sequential distinct messages', async function () {
      const savedMessages: string[] = [];
      let lastSent = '';
      let lastTime = 0;

      const mockSend = vi.fn().mockImplementation(async (payload) => {
        savedMessages.push(payload.text);
        return {
          success: true,
          status: 'saved',
          id: savedMessages.length,
        };
      });

      for (let i = 1; i <= 10; i++) {
        const text = `Message number ${i} in sequence`;
        const now = 1000 + i * 3000; // 3 seconds apart (exceeds 2000ms debounce)

        const isDebounced = shouldDebounce(text, lastSent, now, lastTime, CLIENT_DEBOUNCE_MS);
        expect(isDebounced).toBe(false);

        lastSent = text;
        lastTime = now;

        const res = await deliverMessageWithRetry(text, mockSend);
        expect(res?.status).toBe('saved');
        expect(res?.id).toBe(i);
      }

      expect(savedMessages).toHaveLength(10);
      expect(savedMessages[0]).toBe('Message number 1 in sequence');
      expect(savedMessages[9]).toBe('Message number 10 in sequence');
    });
  });

  describe('Chat Feed Turn Extraction (Safety Net)', function () {
    it('should recognize user messages and ignore assistant responses', function () {
      const userElement = {
        getAttribute: (attr: string) => (attr === 'data-message-author-role' ? 'user' : null),
        closest: (selector: string) => (selector.includes('user') ? userElement : null),
      };

      const assistantElement = {
        getAttribute: (attr: string) => (attr === 'data-message-author-role' ? 'assistant' : null),
        closest: (selector: string) => (selector.includes('assistant') ? assistantElement : null),
      };

      expect(isUserMessageElement(userElement as any)).toBe(true);
      expect(isUserMessageElement(assistantElement as any)).toBe(false);

      expect(isAssistantMessageElement(assistantElement as any)).toBe(true);
      expect(isAssistantMessageElement(userElement as any)).toBe(false);
    });

    it('should extract user text while stripping action buttons', function () {
      const mockClone = {
        querySelectorAll: () => [
          { remove: vi.fn() },
          { remove: vi.fn() },
        ],
        textContent: '  Explain WebSockets in detail  ',
      };

      const mockUserTurn = {
        matches: () => true,
        cloneNode: () => mockClone,
        textContent: 'Explain WebSockets in detail',
      };

      const extracted = extractUserMessageText(mockUserTurn as any);
      expect(extracted).toBe('Explain WebSockets in detail');
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
