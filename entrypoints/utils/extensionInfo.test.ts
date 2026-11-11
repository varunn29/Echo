import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import {
  getExtensionId,
  buildServiceStartupCommand,
  getExtensionOrigin,
  copyToClipboard,
} from './extensionInfo';
import { DEFAULT_OLLAMA_MODEL } from './ollamaClient';

describe('Echo Extension Runtime Info & Service Startup Command Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    delete (globalThis as Record<string, unknown>).browser;
    delete (globalThis as Record<string, unknown>).chrome;
    delete (globalThis as Record<string, unknown>).document;
  });

  afterEach(() => {
    delete (globalThis as Record<string, unknown>).browser;
    delete (globalThis as Record<string, unknown>).chrome;
    delete (globalThis as Record<string, unknown>).document;
  });

  describe('getExtensionId', () => {
    it('returns empty string when neither browser nor chrome is defined', () => {
      expect(getExtensionId()).toBe('');
    });

    it('returns extension ID from browser.runtime.id when available', () => {
      (globalThis as Record<string, unknown>).browser = {
        runtime: { id: 'test_extension_id_browser_123' },
      };
      expect(getExtensionId()).toBe('test_extension_id_browser_123');
    });

    it('returns extension ID from chrome.runtime.id when browser is not available', () => {
      (globalThis as Record<string, unknown>).chrome = {
        runtime: { id: 'test_extension_id_chrome_456' },
      };
      expect(getExtensionId()).toBe('test_extension_id_chrome_456');
    });

    it('prefers browser.runtime.id over chrome.runtime.id when both are present', () => {
      (globalThis as Record<string, unknown>).browser = {
        runtime: { id: 'browser_primary_id' },
      };
      (globalThis as Record<string, unknown>).chrome = {
        runtime: { id: 'chrome_secondary_id' },
      };
      expect(getExtensionId()).toBe('browser_primary_id');
    });

    it('safely handles throwing getters on runtime object', () => {
      (globalThis as Record<string, unknown>).browser = {
        get runtime() {
          throw new Error('Access denied');
        },
      };
      expect(getExtensionId()).toBe('');
    });
  });

  describe('buildServiceStartupCommand', () => {
    it('formats command with placeholder when no extension ID is available', () => {
      const cmd = buildServiceStartupCommand();
      expect(cmd).toBe(
        'python training/generation_service.py --allowed-origin "chrome-extension://<your-extension-id>"'
      );
    });

    it('formats command with detected runtime ID when browser.runtime.id exists', () => {
      (globalThis as Record<string, unknown>).browser = {
        runtime: { id: 'mbghefdhlgoladjadphnagkiclmckohf' },
      };
      const cmd = buildServiceStartupCommand();
      expect(cmd).toBe(
        'python training/generation_service.py --allowed-origin "chrome-extension://mbghefdhlgoladjadphnagkiclmckohf"'
      );
    });

    it('uses explicitly passed extension ID over runtime ID', () => {
      (globalThis as Record<string, unknown>).browser = {
        runtime: { id: 'runtime_id' },
      };
      const cmd = buildServiceStartupCommand('custom_user_id_789');
      expect(cmd).toBe(
        'python training/generation_service.py --allowed-origin "chrome-extension://custom_user_id_789"'
      );
    });

    it('trims whitespace on explicit ID', () => {
      const cmd = buildServiceStartupCommand('  trimmed_id_123  ');
      expect(cmd).toBe(
        'python training/generation_service.py --allowed-origin "chrome-extension://trimmed_id_123"'
      );
    });

    it('uses placeholder when explicitly passed empty or whitespace string', () => {
      const cmd = buildServiceStartupCommand('   ');
      expect(cmd).toBe(
        'python training/generation_service.py --allowed-origin "chrome-extension://<your-extension-id>"'
      );
    });
  });

  describe('getExtensionOrigin', () => {
    it('returns placeholder origin when ID is missing', () => {
      expect(getExtensionOrigin()).toBe('chrome-extension://<your-extension-id>');
    });

    it('returns formatted origin with detected runtime ID', () => {
      (globalThis as Record<string, unknown>).browser = {
        runtime: { id: 'real_extension_id' },
      };
      expect(getExtensionOrigin()).toBe('chrome-extension://real_extension_id');
    });

    it('returns formatted origin with custom ID', () => {
      expect(getExtensionOrigin('custom_id')).toBe('chrome-extension://custom_id');
    });
  });

  describe('copyToClipboard', () => {
    it('returns false for empty text', async () => {
      const result = await copyToClipboard('');
      expect(result).toBe(false);
    });

    it('uses navigator.clipboard.writeText when available', async () => {
      const writeTextMock = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: writeTextMock },
        configurable: true,
        writable: true,
      });

      const result = await copyToClipboard('python training/generation_service.py');
      expect(result).toBe(true);
      expect(writeTextMock).toHaveBeenCalledWith('python training/generation_service.py');
    });

    it('falls back to document.execCommand when navigator.clipboard fails', async () => {
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          writeText: vi.fn().mockRejectedValue(new Error('Permission denied')),
        },
        configurable: true,
        writable: true,
      });

      const execCommandMock = vi.fn().mockReturnValue(true);
      const mockElement = {
        value: '',
        style: {},
        setAttribute: vi.fn(),
        select: vi.fn(),
      };
      (globalThis as Record<string, unknown>).document = {
        createElement: vi.fn().mockReturnValue(mockElement),
        body: {
          appendChild: vi.fn(),
          removeChild: vi.fn(),
        },
        execCommand: execCommandMock,
      };

      const result = await copyToClipboard('fallback copy text');
      expect(result).toBe(true);
      expect(execCommandMock).toHaveBeenCalledWith('copy');
    });
  });

  describe('WritingComposer Ollama Model Synchronization', () => {
    it('verifies DEFAULT_OLLAMA_MODEL is qwen3.5:2b', () => {
      expect(DEFAULT_OLLAMA_MODEL).toBe('qwen3.5:2b');
    });

    it('verifies WritingComposer imports and uses DEFAULT_OLLAMA_MODEL', () => {
      const composerPath = new URL('../dashboard/components/WritingComposer.tsx', import.meta.url);
      const composerSource = fs.readFileSync(composerPath, 'utf-8');

      // Verify import from ollamaClient
      expect(composerSource).toContain("import { DEFAULT_OLLAMA_MODEL } from '../../utils/ollamaClient';");

      // Verify template usage
      expect(composerSource).toContain('{DEFAULT_OLLAMA_MODEL}');

      // Verify neither gemma2:2b nor llama3.2:3b is hardcoded
      expect(composerSource).not.toContain('gemma2:2b');
      expect(composerSource).not.toContain('llama3.2:3b');
    });
  });
});
