declare const chrome: any;

import {
  DEFAULT_TIMEOUT_MS,
  OllamaError,
  type OllamaClientOptions,
} from './ollamaClient';

export const OFFSCREEN_DOCUMENT_PATH = 'offscreen.html';

let offscreenBridgeOverrideForTesting: boolean | null = null;

export function setOffscreenBridgeEnabledForTesting(enabled: boolean | null): void {
  offscreenBridgeOverrideForTesting = enabled;
}

/**
 * Checks whether the current runtime environment is a Manifest V3 background service worker
 * where chrome.offscreen is available and should be used to avoid the 30-second fetch watchdog.
 */
export function isOffscreenSupportedAndRequired(): boolean {
  if (offscreenBridgeOverrideForTesting !== null) {
    return offscreenBridgeOverrideForTesting;
  }

  // If chrome.offscreen is available and we are running in a service worker (window is undefined)
  if (
    typeof chrome !== 'undefined' &&
    Boolean(chrome.offscreen) &&
    typeof chrome.offscreen.createDocument === 'function' &&
    typeof window === 'undefined'
  ) {
    return true;
  }

  return false;
}

/**
 * Checks if an offscreen document is currently active for this extension.
 */
export async function hasOffscreenDocument(): Promise<boolean> {
  if (typeof chrome === 'undefined' || !chrome.runtime) {
    return false;
  }

  // 1. Chrome 116+ getContexts API
  if (typeof (chrome.runtime as any).getContexts === 'function') {
    try {
      const contexts = await (chrome.runtime as any).getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH)],
      });
      return Array.isArray(contexts) && contexts.length > 0;
    } catch {
      // Fall through to fallback
    }
  }

  // 2. Fallback: try checking chrome.offscreen.hasDocument if present
  if (typeof (chrome.offscreen as any)?.hasDocument === 'function') {
    try {
      return await (chrome.offscreen as any).hasDocument();
    } catch {
      // Fall through
    }
  }

  return false;
}

/**
 * Ensures an offscreen document exists, creating one with the genuine WORKERS reason if missing.
 */
export async function ensureOffscreenDocument(): Promise<void> {
  if (typeof chrome === 'undefined' || !chrome.offscreen) {
    return;
  }

  const exists = await hasOffscreenDocument();
  if (exists) {
    return;
  }

  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT_PATH,
      reasons: ['WORKERS' as any],
      justification: 'Spawn dedicated Web Worker to execute long-running local Ollama inference without service worker termination',
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    // Ignore error if document was created concurrently
    if (msg.includes('Only a single offscreen document may be created') || msg.includes('already exists')) {
      return;
    }
    throw err;
  }
}

/**
 * Safely closes the offscreen document when queue processing is completed.
 */
export async function closeOffscreenDocument(): Promise<void> {
  if (typeof chrome === 'undefined' || !chrome.offscreen) {
    return;
  }

  const exists = await hasOffscreenDocument();
  if (!exists) {
    return;
  }

  try {
    await chrome.offscreen.closeDocument();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    // Ignore benign errors if document was already closed or disposed concurrently
    if (
      msg.includes('No current offscreen document') ||
      msg.includes('already closed') ||
      msg.includes('not found')
    ) {
      return;
    }
    throw err;
  }
}

/**
 * Dispatches an inference prompt through the offscreen document to its dedicated Web Worker.
 * Preserves exact OllamaError kinds, status codes, and timeouts.
 */
export async function queryOllamaViaOffscreen(
  prompt: string,
  options: OllamaClientOptions = {}
): Promise<string> {
  await ensureOffscreenDocument();

  const id = `inf_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  const timeoutMs = (options.timeoutMs || DEFAULT_TIMEOUT_MS) + 10000;

  const response = await new Promise<{ success: boolean; result?: string; error?: any }>(
    (resolve, reject) => {
      const timer = setTimeout(() => {
        reject(
          new OllamaError(
            `Offscreen request timed out after ${Math.round(timeoutMs / 1000)}s.`,
            'timeout'
          )
        );
      }, timeoutMs);

      const targetChrome = typeof chrome !== 'undefined' ? chrome : (globalThis as any).chrome;
      if (!targetChrome?.runtime?.sendMessage) {
        clearTimeout(timer);
        reject(new OllamaError('Extension runtime messaging is not available.', 'offline'));
        return;
      }

      targetChrome.runtime.sendMessage(
        {
          target: 'echo-offscreen',
          type: 'EXECUTE_OLLAMA_INFERENCE',
          id,
          prompt,
          options,
        },
        (res: any) => {
          clearTimeout(timer);
          if (targetChrome.runtime?.lastError) {
            reject(
              new OllamaError(
                targetChrome.runtime.lastError.message || 'Failed to communicate with offscreen document.',
                'offline'
              )
            );
            return;
          }
          resolve(res);
        }
      );
    }
  );

  if (!response || !response.success) {
    const err = response?.error;
    const kind = err?.kind || 'http_error';
    const statusCode = err?.statusCode;
    throw new OllamaError(
      err?.message || 'Offscreen Ollama inference failed.',
      kind,
      statusCode
    );
  }

  return response.result!;
}
