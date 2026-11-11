/**
 * Echo — Local Training Service Client (Phase 1, Step 6)
 * Handles communication with the persistent local LoRA training endpoints on generation_service:
 * - GET /training/status
 * - POST /train
 */

export type TrainingState =
  | 'idle'
  | 'preparing'
  | 'training'
  | 'validating'
  | 'activating'
  | 'completed'
  | 'error'
  | 'offline';

export interface TrainingStatusResponse {
  status: TrainingState;
  jobId: string | null;
  stage: string;
  sampleCount: number;
  adapterVersion: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationSec: number;
  error: string | null;
  isOnline: boolean;
}

export interface TrainJobSubmissionResponse {
  status: 'accepted' | 'conflict' | 'rejected' | 'error' | 'offline';
  jobId?: string;
  message?: string;
  activeJobId?: string;
  sampleCount?: number;
  error?: string;
  httpStatus?: number;
}

export interface TrainingClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
}

export const DEFAULT_TRAINING_SERVICE_URL = 'http://127.0.0.1:11435';
export const DEFAULT_TRAINING_STATUS_TIMEOUT_MS = 4000;
export const DEFAULT_TRAINING_SUBMISSION_TIMEOUT_MS = 10000;

/**
 * Checks whether the local training service is online and retrieves its current lifecycle status.
 */
export async function getTrainingStatus(
  options: TrainingClientOptions = {}
): Promise<TrainingStatusResponse> {
  const {
    baseUrl = DEFAULT_TRAINING_SERVICE_URL,
    timeoutMs = DEFAULT_TRAINING_STATUS_TIMEOUT_MS,
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/training/status`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: controller.signal,
    });

    if (res.status === 200) {
      let data: any;
      try {
        data = await res.json();
      } catch {
        return {
          status: 'error',
          jobId: null,
          stage: 'error',
          sampleCount: 0,
          adapterVersion: null,
          startedAt: null,
          completedAt: null,
          durationSec: 0,
          error: 'Malformed JSON received from training status endpoint.',
          isOnline: true,
        };
      }

      const rawState = data.status || data.stage || 'idle';
      const validStates: Set<TrainingState> = new Set([
        'idle',
        'preparing',
        'training',
        'validating',
        'activating',
        'completed',
        'error',
      ]);

      const state: TrainingState = validStates.has(rawState) ? rawState : 'idle';

      return {
        status: state,
        jobId: data.jobId ?? null,
        stage: data.stage ?? state,
        sampleCount: typeof data.sampleCount === 'number' ? data.sampleCount : 0,
        adapterVersion: data.adapterVersion ?? null,
        startedAt: data.startedAt ?? null,
        completedAt: data.completedAt ?? null,
        durationSec: typeof data.durationSec === 'number' ? data.durationSec : 0,
        error: data.error ?? null,
        isOnline: true,
      };
    }

    return {
      status: 'error',
      jobId: null,
      stage: 'error',
      sampleCount: 0,
      adapterVersion: null,
      startedAt: null,
      completedAt: null,
      durationSec: 0,
      error: `Training status endpoint responded with HTTP ${res.status}.`,
      isOnline: true,
    };
  } catch (err: unknown) {
    const isTimeout = err instanceof Error && err.name === 'AbortError';
    return {
      status: 'offline',
      jobId: null,
      stage: 'offline',
      sampleCount: 0,
      adapterVersion: null,
      startedAt: null,
      completedAt: null,
      durationSec: 0,
      error: isTimeout
        ? 'Training status request timed out.'
        : 'Local training service is offline (127.0.0.1:11435).',
      isOnline: false,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}
/**
 * Submits eligible writing samples to start a background LoRA training job.
 * Sends either raw sample strings or object representations.
 */
export async function submitTrainingJob(
  samples: Array<{ id?: number; text: string } | string>,
  options: TrainingClientOptions = {}
): Promise<TrainJobSubmissionResponse> {
  const {
    baseUrl = DEFAULT_TRAINING_SERVICE_URL,
    timeoutMs = DEFAULT_TRAINING_SUBMISSION_TIMEOUT_MS,
  } = options;

  if (!Array.isArray(samples) || samples.length === 0) {
    return {
      status: 'rejected',
      error: 'Cannot submit training job: samples array is required and must not be empty.',
      httpStatus: 400,
    };
  }

  // Format samples for payload: string list or {text: string}
  const formattedSamples = samples.map(item => {
    if (typeof item === 'string') {
      return item.trim();
    }
    if (item && typeof item.text === 'string') {
      return item.text.trim();
    }
    return '';
  }).filter(text => text.length > 0);

  if (formattedSamples.length === 0) {
    return {
      status: 'rejected',
      error: 'Cannot submit training job: all provided samples were empty.',
      httpStatus: 400,
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        samples: formattedSamples,
      }),
      signal: controller.signal,
    });

    let data: any = {};
    try {
      data = await res.json();
    } catch {
      // response without JSON body
    }

    if (res.status === 202) {
      return {
        status: 'accepted',
        jobId: data.jobId,
        message: data.message ?? 'Training job accepted and running in background.',
        sampleCount: data.sampleCount ?? formattedSamples.length,
        httpStatus: 202,
      };
    }

    if (res.status === 409) {
      return {
        status: 'conflict',
        activeJobId: data.activeJobId,
        error: data.error ?? 'A training job is already in progress.',
        httpStatus: 409,
      };
    }

    if (res.status === 400) {
      return {
        status: 'rejected',
        error: data.error ?? 'Training request was rejected due to invalid samples or parameters.',
        httpStatus: 400,
      };
    }

    return {
      status: 'error',
      error: data.error ?? `Training request failed with HTTP ${res.status}.`,
      httpStatus: res.status,
    };
  } catch (err: unknown) {
    const isTimeout = err instanceof Error && err.name === 'AbortError';
    return {
      status: 'offline',
      error: isTimeout
        ? 'Training submission timed out.'
        : 'Local training service is unreachable (127.0.0.1:11435).',
      httpStatus: 0,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}
