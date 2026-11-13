declare const chrome: any;
declare const browser: any;

import type { WorkerInferenceRequest, WorkerInferenceResponse } from '../utils/ollamaWorker';

export interface PendingOffscreenRequest {
  resolve: (res: { success: boolean; result?: string; error?: any }) => void;
  reject: (err: Error) => void;
  timeoutId: ReturnType<typeof setTimeout>;
}

let workerInstance: Worker | null = null;
const activeRequests = new Map<string, PendingOffscreenRequest>();

export function attachWorkerListeners(worker: Worker): void {
  worker.onmessage = (event: MessageEvent<WorkerInferenceResponse>) => {
    const data = event.data;
    if (!data || !data.id) return;

    const pending = activeRequests.get(data.id);
    if (!pending) return;

    clearTimeout(pending.timeoutId);
    activeRequests.delete(data.id);

    if (data.success) {
      pending.resolve({ success: true, result: data.result });
    } else {
      pending.resolve({ success: false, error: data.error });
    }
  };

  worker.onerror = (err) => {
    console.error('Echo: Dedicated Ollama Web Worker error:', err);
    // Reject any pending requests with clear worker crash error
    for (const [, pending] of activeRequests.entries()) {
      clearTimeout(pending.timeoutId);
      pending.resolve({
        success: false,
        error: {
          message: `Ollama Web Worker crashed: ${err instanceof Error ? err.message : String(err)}`,
          kind: 'message_failure',
        },
      });
    }
    activeRequests.clear();
    if (workerInstance) {
      workerInstance.terminate();
      workerInstance = null;
    }
  };
}

export function getWorker(): Worker {
  if (!workerInstance) {
    // Instantiate genuine dedicated Web Worker outside service worker scope
    workerInstance = new Worker(new URL('../utils/ollamaWorker.ts', import.meta.url), {
      type: 'module',
    });
    attachWorkerListeners(workerInstance);
  }

  return workerInstance;
}

export function setWorkerInstanceForTesting(worker: Worker | null): void {
  resetWorkerForTesting();
  if (worker) {
    workerInstance = worker;
    attachWorkerListeners(worker);
  }
}

export function resetWorkerForTesting(): void {
  for (const [, pending] of activeRequests.entries()) {
    clearTimeout(pending.timeoutId);
  }
  activeRequests.clear();
  if (workerInstance) {
    workerInstance.terminate();
    workerInstance = null;
  }
}

/**
 * Handles incoming messages from background service worker to execute Ollama inference
 * inside the dedicated Web Worker.
 */
export function handleOffscreenMessage(
  message: any,
  _sender: any,
  sendResponse: (res: any) => void,
  workerFactory: () => Worker = getWorker
): boolean {
  if (
    !message ||
    typeof message !== 'object' ||
    message.target !== 'echo-offscreen' ||
    message.type !== 'EXECUTE_OLLAMA_INFERENCE'
  ) {
    return false;
  }

  const { id, prompt, options } = message;
  if (!id || typeof prompt !== 'string') {
    sendResponse({
      success: false,
      error: { message: 'Invalid inference payload sent to offscreen document.', kind: 'message_failure' },
    });
    return false;
  }

  let worker: Worker;
  try {
    worker = workerFactory();
  } catch (factoryErr: unknown) {
    sendResponse({
      success: false,
      error: {
        message: `Failed to spawn dedicated Web Worker in offscreen host: ${factoryErr instanceof Error ? factoryErr.message : String(factoryErr)}`,
        kind: 'message_failure',
      },
    });
    return false;
  }

  const timeoutMs = (options?.timeoutMs || 120000) + 5000;
  const timeoutId = setTimeout(() => {
    activeRequests.delete(id);
    sendResponse({
      success: false,
      error: {
        message: `Offscreen inference bridge timed out after ${Math.round(timeoutMs / 1000)}s.`,
        kind: 'timeout',
      },
    });
  }, timeoutMs);

  activeRequests.set(id, {
    resolve: (res) => sendResponse(res),
    reject: (err) =>
      sendResponse({
        success: false,
        error: { message: err.message, kind: 'message_failure' },
      }),
    timeoutId,
  });

  worker.postMessage({
    id,
    type: 'QUERY_OLLAMA',
    prompt,
    options,
  } satisfies WorkerInferenceRequest);

  // Return true to keep message channel open for asynchronous sendResponse
  return true;
}

// Register runtime message listener in offscreen document window context
if (typeof chrome !== 'undefined' && chrome.runtime?.onMessage) {
  chrome.runtime.onMessage.addListener(handleOffscreenMessage);
} else if (typeof browser !== 'undefined' && browser.runtime?.onMessage) {
  browser.runtime.onMessage.addListener(handleOffscreenMessage);
}
