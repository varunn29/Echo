/**
 * Echo — Automatic Training Trigger & Job Tracking (Phase 1, Step 6)
 *
 * Coordinates:
 * - Checking training readiness after messages are saved
 * - Enforcing 25-sample minimum threshold
 * - Submitting only eligible pending samples to POST /train
 * - Persisting active job ID and submitted message IDs across restarts
 * - Polling GET /training/status without blocking message collection
 * - Marking samples as 'trained' ONLY upon verified completion and activation
 * - Handling 409 conflicts, service outages, and failure retries cleanly
 */

import {
  db,
  getTrainingReadiness,
  markMessagesTrained,
  DEFAULT_TRAINING_READINESS_THRESHOLD,
  type EchoDatabase,
  type TrainingReadinessResult,
} from './database';
import {
  getTrainingStatus,
  submitTrainingJob,
  DEFAULT_TRAINING_SERVICE_URL,
  type TrainingStatusResponse,
  type TrainJobSubmissionResponse,
} from './trainingClient';

export const ACTIVE_TRAINING_JOB_STORAGE_KEY = 'echo_active_training_job';
export const TRAINING_COOLDOWN_MS = 60 * 1000; // 60s cooldown after failure
export const RETRY_ALARM_NAME = 'echo_training_service_retry';
export const RETRY_ALARM_PERIOD_MINUTES = 1;

export interface ActiveTrainingJobRecord {
  jobId: string;
  submittedMessageIds: number[];
  submittedAt: string;
  sampleCount: number;
  status: 'active' | 'completed' | 'failed';
  lastCheckedAt: string;
  adapterVersion?: string | null;
  completedAt?: string | null;
  failedAt?: string | null;
  error?: string | null;
}

export interface TrainingTriggerOptions {
  threshold?: number;
  targetDb?: EchoDatabase;
  baseUrl?: string;
  storage?: StorageLike;
  forceCooldownBypass?: boolean;
  autoTriggerFollowUp?: boolean;
}

export async function scheduleServiceRetryAlarm(): Promise<void> {
  if (typeof browser !== 'undefined' && browser.alarms) {
    try {
      const existing = await browser.alarms.get(RETRY_ALARM_NAME);
      if (!existing) {
        browser.alarms.create(RETRY_ALARM_NAME, {
          delayInMinutes: RETRY_ALARM_PERIOD_MINUTES,
          periodInMinutes: RETRY_ALARM_PERIOD_MINUTES,
        });
      }
    } catch {
      // Ignored in non-extension or testing environment
    }
  }
}

export async function clearServiceRetryAlarm(): Promise<void> {
  if (typeof browser !== 'undefined' && browser.alarms) {
    try {
      await browser.alarms.clear(RETRY_ALARM_NAME);
    } catch {
      // Ignored in non-extension or testing environment
    }
  }
}

export interface TrainingTriggerResult {
  action:
    | 'submitted'
    | 'active_job_pending'
    | 'completed'
    | 'insufficient_samples'
    | 'service_offline'
    | 'cooldown'
    | 'error';
  jobId?: string;
  eligibleCount: number;
  threshold: number;
  submittedCount?: number;
  message?: string;
  error?: string;
}

export interface StorageLike {
  get(key: string): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
  remove(key: string): Promise<void>;
}

let inMemoryStorage: Record<string, unknown> = {};

export function getStorage(): StorageLike {
  if (typeof browser !== 'undefined' && browser.storage?.local) {
    return browser.storage.local as unknown as StorageLike;
  }
  const globalChrome = (globalThis as unknown as { chrome?: { storage?: { local?: StorageLike } } }).chrome;
  if (globalChrome?.storage?.local) {
    return globalChrome.storage.local;
  }
  return {
    async get(key: string) {
      return { [key]: inMemoryStorage[key] };
    },
    async set(items: Record<string, unknown>) {
      Object.assign(inMemoryStorage, items);
    },
    async remove(key: string) {
      delete inMemoryStorage[key];
    },
  };
}

export function resetInMemoryStorageForTesting(): void {
  inMemoryStorage = {};
}

// In-memory execution mutex to avoid concurrent execution in the same context
let isTriggerRunning = false;

/**
 * Retrieves the currently persisted active training job from storage.
 */
export async function getActiveTrainingJob(
  storage: StorageLike = getStorage()
): Promise<ActiveTrainingJobRecord | null> {
  try {
    const data = await storage.get(ACTIVE_TRAINING_JOB_STORAGE_KEY);
    const record = data?.[ACTIVE_TRAINING_JOB_STORAGE_KEY];
    if (record && typeof record === 'object' && 'jobId' in record) {
      return record as ActiveTrainingJobRecord;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Persists active training job metadata to storage.
 */
export async function saveActiveTrainingJob(
  record: ActiveTrainingJobRecord,
  storage: StorageLike = getStorage()
): Promise<void> {
  await storage.set({ [ACTIVE_TRAINING_JOB_STORAGE_KEY]: record });
}

/**
 * Clears active training job from storage.
 */
export async function clearActiveTrainingJob(
  storage: StorageLike = getStorage()
): Promise<void> {
  await storage.remove(ACTIVE_TRAINING_JOB_STORAGE_KEY);
}

/**
 * Polls the local training service to check if an active job has completed or failed.
 * If completed successfully, marks the submitted messages as 'trained'.
 */
export async function pollActiveTrainingJob(
  options: TrainingTriggerOptions = {}
): Promise<ActiveTrainingJobRecord | null> {
  const {
    targetDb = db,
    baseUrl = DEFAULT_TRAINING_SERVICE_URL,
    storage = getStorage(),
    autoTriggerFollowUp = true,
  } = options;

  const currentJob = await getActiveTrainingJob(storage);
  if (!currentJob || currentJob.status !== 'active') {
    return currentJob;
  }

  const statusResponse = await getTrainingStatus({ baseUrl });

  if (!statusResponse.isOnline) {
    // Service offline; preserve job state, do not mark samples trained
    currentJob.lastCheckedAt = new Date().toISOString();
    await saveActiveTrainingJob(currentJob, storage);
    return currentJob;
  }

  // 1. Verified successful completion
  if (statusResponse.status === 'completed') {
    // Verify job match: if service reports jobId, it must match currentJob.jobId
    if (!statusResponse.jobId || statusResponse.jobId === currentJob.jobId) {
      console.log(
        `Echo: Training job ${currentJob.jobId} completed successfully. Marking ${currentJob.submittedMessageIds.length} samples as trained.`
      );

      if (currentJob.submittedMessageIds.length > 0) {
        await markMessagesTrained(currentJob.submittedMessageIds, targetDb);
      }

      currentJob.status = 'completed';
      currentJob.completedAt = statusResponse.completedAt || new Date().toISOString();
      currentJob.adapterVersion = statusResponse.adapterVersion;
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);

      // Automatic follow-up: Check if new messages collected during training meet threshold for next cycle
      if (autoTriggerFollowUp) {
        console.log('Echo: Training job completed. Automatically checking readiness for next training cycle...');
        try {
          await checkAndTriggerTraining({
            targetDb,
            baseUrl,
            storage,
            threshold: options.threshold,
            forceCooldownBypass: true,
            autoTriggerFollowUp: false,
          });
        } catch (err) {
          console.warn('Echo: Follow-up training trigger error:', err);
        }
      }

      return currentJob;
    } else {
      // Service completed a different job! Our job was displaced or lost.
      console.warn(
        `Echo: Service completed job ${statusResponse.jobId}, but expected ${currentJob.jobId}. Retaining samples as pending.`
      );
      currentJob.status = 'failed';
      currentJob.error = `Training job ${currentJob.jobId} was lost or displaced on service. Samples retained for retry.`;
      currentJob.failedAt = new Date().toISOString();
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);
      return currentJob;
    }
  }

  // 2. Verified error reported by service
  if (statusResponse.status === 'error') {
    if (!statusResponse.jobId || statusResponse.jobId === currentJob.jobId) {
      console.warn(
        `Echo: Training job ${currentJob.jobId} failed. Preserving ${currentJob.submittedMessageIds.length} pending samples for retry.`
      );

      currentJob.status = 'failed';
      currentJob.error = statusResponse.error || 'Training job failed.';
      currentJob.failedAt = new Date().toISOString();
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);
      return currentJob;
    } else {
      currentJob.status = 'failed';
      currentJob.error = `Training job was lost on service. Samples retained for retry.`;
      currentJob.failedAt = new Date().toISOString();
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);
      return currentJob;
    }
  }

  // 3. Still actively running on service
  if (
    statusResponse.status === 'preparing' ||
    statusResponse.status === 'training' ||
    statusResponse.status === 'validating' ||
    statusResponse.status === 'activating'
  ) {
    if (!statusResponse.jobId || statusResponse.jobId === currentJob.jobId) {
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);
      return currentJob;
    } else {
      // Running another job
      console.warn(`Echo: Service is running job ${statusResponse.jobId}, displaced ${currentJob.jobId}.`);
      currentJob.status = 'failed';
      currentJob.error = 'Training job was displaced by another job.';
      currentJob.failedAt = new Date().toISOString();
      currentJob.lastCheckedAt = new Date().toISOString();
      await saveActiveTrainingJob(currentJob, storage);
      return currentJob;
    }
  }

  // 4. Service is online but IDLE (Python service restarted/crashed mid-job)
  if (statusResponse.status === 'idle') {
    console.warn(
      `Echo: Active training job ${currentJob.jobId} disappeared from service (service is idle). Detecting lost job from service restart.`
    );

    // Retain all submitted samples as pending (do NOT mark trained!)
    currentJob.status = 'failed';
    currentJob.error = `Training job ${currentJob.jobId} was lost due to service restart. Samples retained for retry.`;
    currentJob.failedAt = new Date().toISOString();
    currentJob.lastCheckedAt = new Date().toISOString();
    await saveActiveTrainingJob(currentJob, storage);
    return currentJob;
  }

  // Fallback for any other state
  currentJob.lastCheckedAt = new Date().toISOString();
  await saveActiveTrainingJob(currentJob, storage);
  return currentJob;
}

/**
 * Checks training readiness and automatically triggers a local LoRA training job
 * if the threshold of 25 eligible samples has been met and the service is available.
 */
export async function checkAndTriggerTraining(
  options: TrainingTriggerOptions = {}
): Promise<TrainingTriggerResult> {
  const {
    threshold = DEFAULT_TRAINING_READINESS_THRESHOLD,
    targetDb = db,
    baseUrl = DEFAULT_TRAINING_SERVICE_URL,
    storage = getStorage(),
    forceCooldownBypass = false,
  } = options;

  if (isTriggerRunning) {
    const existing = await getActiveTrainingJob(storage);
    return {
      action: 'active_job_pending',
      jobId: existing?.jobId,
      eligibleCount: 0,
      threshold,
      message: 'A training trigger check is already in progress.',
    };
  }

  isTriggerRunning = true;
  try {
    // 1. First poll any existing active job (without recursive follow-up)
    const activeJob = await pollActiveTrainingJob({
      targetDb,
      baseUrl,
      storage,
      threshold,
      autoTriggerFollowUp: false,
    });
    if (activeJob && activeJob.status === 'active') {
      return {
        action: 'active_job_pending',
        jobId: activeJob.jobId,
        eligibleCount: 0,
        threshold,
        message: `Training job ${activeJob.jobId} is currently running in background.`,
      };
    }

    // 2. Cooldown check: if previous job failed recently, don't spam (unless lost job from restart)
    if (!forceCooldownBypass && activeJob && activeJob.status === 'failed' && activeJob.failedAt) {
      const isLostJob = activeJob.error?.includes('lost due to service restart');
      if (!isLostJob) {
        const elapsed = Date.now() - new Date(activeJob.failedAt).getTime();
        if (elapsed < TRAINING_COOLDOWN_MS) {
          return {
            action: 'cooldown',
            eligibleCount: 0,
            threshold,
            message: `Training retry is in cooldown (${Math.ceil((TRAINING_COOLDOWN_MS - elapsed) / 1000)}s remaining).`,
            error: activeJob.error ?? undefined,
          };
        }
      }
    }

    // 3. Query training readiness from database
    const readiness: TrainingReadinessResult = await getTrainingReadiness(threshold, targetDb);

    if (!readiness.isReady || readiness.eligibleSamples.length < threshold) {
      await clearServiceRetryAlarm();
      return {
        action: 'insufficient_samples',
        eligibleCount: readiness.eligibleCount,
        threshold,
        message: `Insufficient eligible samples: ${readiness.eligibleCount}/${threshold}`,
      };
    }

    // 4. Verify local training service availability
    const serviceStatus: TrainingStatusResponse = await getTrainingStatus({ baseUrl });
    if (!serviceStatus.isOnline) {
      // Schedule lightweight retry alarm so service is detected when user starts it later
      await scheduleServiceRetryAlarm();
      return {
        action: 'service_offline',
        eligibleCount: readiness.eligibleCount,
        threshold,
        message: 'Local training service is offline (127.0.0.1:11435). Collected samples remain safe and pending.',
      };
    }

    // If the service is already busy executing another job
    if (
      serviceStatus.status === 'preparing' ||
      serviceStatus.status === 'training' ||
      serviceStatus.status === 'validating' ||
      serviceStatus.status === 'activating'
    ) {
      const externalJobId = serviceStatus.jobId || 'active_service_job';
      await saveActiveTrainingJob(
        {
          jobId: externalJobId,
          submittedMessageIds: [],
          submittedAt: serviceStatus.startedAt || new Date().toISOString(),
          sampleCount: serviceStatus.sampleCount,
          status: 'active',
          lastCheckedAt: new Date().toISOString(),
          error: null,
        },
        storage
      );

      return {
        action: 'active_job_pending',
        jobId: externalJobId,
        eligibleCount: readiness.eligibleCount,
        threshold,
        message: `Local service is already executing training job ${externalJobId}.`,
      };
    }

    // 5. Filter and prepare only eligible samples with exact message IDs
    const submittedIds: number[] = [];
    const samplePayloads: Array<{ id: number; text: string }> = [];

    for (const sample of readiness.eligibleSamples) {
      if (typeof sample.id === 'number') {
        submittedIds.push(sample.id);
        samplePayloads.push({ id: sample.id, text: sample.text });
      }
    }

    if (submittedIds.length < threshold) {
      return {
        action: 'insufficient_samples',
        eligibleCount: submittedIds.length,
        threshold,
        message: `Insufficient eligible samples with valid IDs: ${submittedIds.length}/${threshold}`,
      };
    }

    console.log(
      `Echo: Submitting ${submittedIds.length} eligible samples to local LoRA training service...`
    );

    // 6. Submit training job to POST /train
    const submissionResult: TrainJobSubmissionResponse = await submitTrainingJob(
      samplePayloads,
      { baseUrl }
    );

    if (submissionResult.status === 'accepted' && submissionResult.jobId) {
      // Successfully accepted by server (HTTP 202)
      // Persist active job with submitted IDs; samples remain pending until completed!
      const jobRecord: ActiveTrainingJobRecord = {
        jobId: submissionResult.jobId,
        submittedMessageIds: submittedIds,
        submittedAt: new Date().toISOString(),
        sampleCount: submittedIds.length,
        status: 'active',
        lastCheckedAt: new Date().toISOString(),
        error: null,
      };

      await saveActiveTrainingJob(jobRecord, storage);
      await clearServiceRetryAlarm();

      console.log(`Echo: Training job ${submissionResult.jobId} accepted by local service.`);

      return {
        action: 'submitted',
        jobId: submissionResult.jobId,
        eligibleCount: readiness.eligibleCount,
        threshold,
        submittedCount: submittedIds.length,
        message: `Training job ${submissionResult.jobId} accepted and running in background.`,
      };
    }

    if (submissionResult.status === 'conflict') {
      // HTTP 409 Conflict: Another job is in progress or reserved
      const activeJobId = submissionResult.activeJobId || 'active_job';
      await saveActiveTrainingJob(
        {
          jobId: activeJobId,
          submittedMessageIds: [],
          submittedAt: new Date().toISOString(),
          sampleCount: 0,
          status: 'active',
          lastCheckedAt: new Date().toISOString(),
          error: null,
        },
        storage
      );

      return {
        action: 'active_job_pending',
        jobId: activeJobId,
        eligibleCount: readiness.eligibleCount,
        threshold,
        message: `Training job is already in progress (${activeJobId}).`,
      };
    }

    // Submission error or rejection (HTTP 400 / 500 / offline)
    const errorMsg = submissionResult.error || 'Training job submission failed.';
    console.warn(`Echo: Training submission failed: ${errorMsg}`);

    // Persist failure record to enforce cooldown and avoid hammering service
    await saveActiveTrainingJob(
      {
        jobId: 'failed_submission',
        submittedMessageIds: [],
        submittedAt: new Date().toISOString(),
        sampleCount: 0,
        status: 'failed',
        lastCheckedAt: new Date().toISOString(),
        failedAt: new Date().toISOString(),
        error: errorMsg,
      },
      storage
    );

    return {
      action: 'error',
      eligibleCount: readiness.eligibleCount,
      threshold,
      error: errorMsg,
      message: `Training submission failed: ${errorMsg}`,
    };
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`Echo: Unexpected error during training trigger: ${errorMsg}`);
    return {
      action: 'error',
      eligibleCount: 0,
      threshold,
      error: errorMsg,
    };
  } finally {
    isTriggerRunning = false;
  }
}
