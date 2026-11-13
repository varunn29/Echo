import { executeDirectOllamaJson, type OllamaClientOptions } from './ollamaClient';

export interface WorkerInferenceRequest {
  id: string;
  type: 'QUERY_OLLAMA';
  prompt: string;
  options?: OllamaClientOptions;
}

export interface WorkerInferenceSuccess {
  id: string;
  success: true;
  result: string;
}

export interface WorkerInferenceError {
  id: string;
  success: false;
  error: {
    message: string;
    kind?: string;
    statusCode?: number;
  };
}

export type WorkerInferenceResponse = WorkerInferenceSuccess | WorkerInferenceError;

/**
 * Genuine dedicated Web Worker execution handler for Ollama inference.
 * Runs in its own worker thread (WorkerGlobalScope), executing HTTP fetch outside
 * the Manifest V3 background service worker's 30-second fetch-response watchdog.
 */
export async function handleWorkerMessage(
  data: WorkerInferenceRequest,
  postFn: (msg: WorkerInferenceResponse) => void = (msg) => self.postMessage(msg),
  executeFn: (prompt: string, options?: OllamaClientOptions) => Promise<string> = executeDirectOllamaJson
): Promise<void> {
  if (!data || data.type !== 'QUERY_OLLAMA') {
    return;
  }

  try {
    const result = await executeFn(data.prompt, data.options);
    postFn({
      id: data.id,
      success: true,
      result,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    const kind = (err as any)?.kind;
    const statusCode = (err as any)?.statusCode;
    postFn({
      id: data.id,
      success: false,
      error: {
        message,
        kind,
        statusCode,
      },
    });
  }
}

// Register standard self.onmessage listener when running in genuine Worker environment
if (typeof self !== 'undefined' && typeof (self as any).postMessage === 'function') {
  self.onmessage = (event: MessageEvent<WorkerInferenceRequest>) => {
    handleWorkerMessage(event.data).catch((unhandledErr) => {
      console.error('Echo: Unhandled error in Ollama Web Worker message handler:', unhandledErr);
    });
  };
}
