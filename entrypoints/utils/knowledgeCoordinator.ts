/**
 * Echo — Persistent Background Knowledge Coordinator (Phase 3, Step 3B)
 *
 * Coordinates:
 * - Enqueueing eligible writing messages for personal knowledge extraction.
 * - Fast-path pre-classification (ineligible content skipped without LLM compute).
 * - Five-state message lifecycle: 'pending' | 'extracted' | 'skipped' | 'failed_temporary' | 'failed_permanent'.
 * - Versioned persistent state under 'echo_knowledge_coordinator_state'.
 * - Serialized read-modify-write state mutations preventing concurrency race conditions.
 * - Single execution mutex ensuring sequential processing (concurrency = 1).
 * - In-flight job persistence and crash/restart recovery.
 * - Distinct handling of service outages (alarm retry, no attempt increment) vs actionable config errors (blocked, no infinite alarm) vs message-level errors (retry up to 3 times).
 * - Bounded retries: 3 attempts before moving to 'failed_permanent' dead-letter state.
 * - Manual retry for permanently failed messages and manual reconsideration for skipped messages.
 */

import { classifyContent } from './contentClassifier';
import { db, type EchoDatabase, type WritingMessage } from './database';
import {
  extractKnowledgeFromMessage,
  type ExtractKnowledgeOptions,
  type ExtractSingleMessageResult,
} from './knowledgeExtractor';
import {
  checkOllamaStatus,
  DEFAULT_OLLAMA_BASE_URL,
  DEFAULT_OLLAMA_MODEL,
  OllamaError,
  type OllamaClientOptions,
  type OllamaStatus,
} from './ollamaClient';

export const COORDINATOR_STORAGE_KEY = 'echo_knowledge_coordinator_state';
export const KNOWLEDGE_RETRY_ALARM_NAME = 'echo_knowledge_service_retry';
export const KNOWLEDGE_RETRY_ALARM_PERIOD_MINUTES = 1;
export const MAX_EXTRACTION_ATTEMPTS = 3;

export type KnowledgeMessageStatus =
  | 'pending'
  | 'extracted'
  | 'skipped'
  | 'failed_temporary'
  | 'failed_permanent';

export type CoordinatorServiceStatus =
  | 'online'
  | 'offline'
  | 'blocked_configuration'
  | 'unknown';

export interface CoordinatorQueueItem {
  messageId: number;
  enqueuedAt: string;
  attempts: number;
}

export interface InFlightExtractionRecord {
  messageId: number;
  startedAt: string;
  attempt: number;
}

export interface MessageRegistryEntry {
  messageId: number;
  status: KnowledgeMessageStatus;
  attempts: number;
  lastError?: string;
  enqueuedAt: string;
  updatedAt: string;
  lastAttemptAt?: string;
  extractedCount?: number;
  duplicateCount?: number;
}

export interface KnowledgeCoordinatorState {
  version: 1;
  pendingQueue: CoordinatorQueueItem[];
  inFlight: InFlightExtractionRecord | null;
  registry: Record<number, MessageRegistryEntry>;
  serviceStatus: CoordinatorServiceStatus;
  serviceError?: string;
  lastCheckedAt?: string;
}

export interface KnowledgeCoordinatorStatus {
  pendingCount: number;
  extractedCount: number;
  skippedCount: number;
  failedTemporaryCount: number;
  failedPermanentCount: number;
  inFlightMessageId: number | null;
  serviceStatus: CoordinatorServiceStatus;
  serviceError?: string;
  isProcessing: boolean;
}

export interface StorageLike {
  get(key: string): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
  remove(key: string): Promise<void>;
}

export interface CoordinatorOptions {
  targetDb?: EchoDatabase;
  storage?: StorageLike;
  ollamaOptions?: OllamaClientOptions;
  cooldownMs?: number;
  sleepFn?: (ms: number) => Promise<void>;
}

export interface EnqueueResult {
  enqueued: boolean;
  status: KnowledgeMessageStatus;
  reason?: 'ineligible_content' | 'already_queued_or_extracted' | 'missing_message_id';
}

export interface SyncUntrackedResult {
  totalChecked: number;
  newlyQueuedCount: number;
  newlySkippedCount: number;
  alreadyTrackedCount: number;
  invalidCount: number;
}

export interface ProcessQueueResult {
  status:
    | 'completed'
    | 'already_running'
    | 'service_offline'
    | 'service_blocked'
    | 'service_unknown'
    | 'empty';
  processedCount: number;
  error?: string;
}

export interface RecoveryResult {
  recovered: boolean;
  action: 'idle' | 'requeued' | 'discarded_deleted';
  messageId?: number;
}

// In-memory fallback storage
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

// In-memory execution mutex preventing concurrent processing within the same service-worker instance
let isProcessingQueue = false;

// Serialized mutation chain preventing lost updates during concurrent read-modify-write calls
let mutationChain: Promise<unknown> = Promise.resolve();

export function resetCoordinatorStateForTesting(): void {
  inMemoryStorage = {};
  isProcessingQueue = false;
  mutationChain = Promise.resolve();
}

/**
 * Creates the initial default state for the coordinator.
 */
export function createDefaultCoordinatorState(): KnowledgeCoordinatorState {
  return {
    version: 1,
    pendingQueue: [],
    inFlight: null,
    registry: {},
    serviceStatus: 'unknown',
  };
}

/**
 * Retrieves the persisted coordinator state, initializing defaults if missing.
 */
export async function getCoordinatorState(
  storage: StorageLike = getStorage()
): Promise<KnowledgeCoordinatorState> {
  try {
    const data = await storage.get(COORDINATOR_STORAGE_KEY);
    const state = data?.[COORDINATOR_STORAGE_KEY];
    if (state && typeof state === 'object' && (state as { version?: number }).version === 1) {
      const validState = state as KnowledgeCoordinatorState;
      if (!Array.isArray(validState.pendingQueue)) validState.pendingQueue = [];
      if (!validState.registry || typeof validState.registry !== 'object') validState.registry = {};
      return validState;
    }
  } catch {
    // Ignore storage errors and fall back to default
  }
  return createDefaultCoordinatorState();
}

/**
 * Persists the coordinator state to storage.
 */
export async function saveCoordinatorState(
  state: KnowledgeCoordinatorState,
  storage: StorageLike = getStorage()
): Promise<void> {
  await storage.set({ [COORDINATOR_STORAGE_KEY]: state });
}

/**
 * Serializes all read-modify-write mutations on the coordinator state.
 * Prevents concurrent operations from overwriting each other's updates.
 */
export async function mutateCoordinatorState<T>(
  mutator: (state: KnowledgeCoordinatorState) => Promise<T> | T,
  storage: StorageLike = getStorage()
): Promise<T> {
  const execute = async (): Promise<T> => {
    const state = await getCoordinatorState(storage);
    const result = await mutator(state);
    await saveCoordinatorState(state, storage);
    return result;
  };

  const currentChain = mutationChain;
  let nextResolve!: (val: T) => void;
  let nextReject!: (err: unknown) => void;
  const nextPromise = new Promise<T>((resolve, reject) => {
    nextResolve = resolve;
    nextReject = reject;
  });

  mutationChain = currentChain.then(
    async () => {
      try {
        const res = await execute();
        nextResolve(res);
      } catch (err) {
        nextReject(err);
      }
    },
    async () => {
      try {
        const res = await execute();
        nextResolve(res);
      } catch (err) {
        nextReject(err);
      }
    }
  );

  return nextPromise;
}

/**
 * Schedules a Chrome alarm for periodic service recovery retries.
 * Adheres to Chrome MV3 minimum interval (1 minute).
 */
export async function scheduleKnowledgeRetryAlarm(): Promise<void> {
  if (typeof browser !== 'undefined' && browser.alarms) {
    try {
      const existing = await browser.alarms.get(KNOWLEDGE_RETRY_ALARM_NAME);
      if (!existing) {
        browser.alarms.create(KNOWLEDGE_RETRY_ALARM_NAME, {
          delayInMinutes: KNOWLEDGE_RETRY_ALARM_PERIOD_MINUTES,
          periodInMinutes: KNOWLEDGE_RETRY_ALARM_PERIOD_MINUTES,
        });
      }
    } catch {
      // Ignored in test/non-extension environments
    }
  }
}

/**
 * Clears the Chrome alarm when service is restored or queue is empty.
 */
export async function clearKnowledgeRetryAlarm(): Promise<void> {
  if (typeof browser !== 'undefined' && browser.alarms) {
    try {
      await browser.alarms.clear(KNOWLEDGE_RETRY_ALARM_NAME);
    } catch {
      // Ignored in test/non-extension environments
    }
  }
}

export type ErrorClassification = {
  category: 'service_offline' | 'service_blocked' | 'service_unknown' | 'message_failure';
  message: string;
};

/**
 * Determines whether an extraction failure warrants a brief cooldown period
 * (e.g. to allow Ollama to cleanly terminate a canceled runner subprocess before another health check).
 */
export function isCooldownEligible(err: unknown): boolean {
  if (err instanceof OllamaError) {
    return err.kind === 'timeout' || (err.kind === 'http_error' && err.statusCode === 500);
  }
  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    return msg.includes('timed out') || msg.includes('timeout') || msg.includes('500');
  }
  return false;
}

/**
 * Sanitizes error messages to prevent exposing personal message text or token payloads.
 */
export function sanitizeErrorMessage(err: unknown): string {
  if (!err) return 'Unknown extraction error';
  const raw = err instanceof Error ? err.message : String(err);
  const cleaned = raw.replace(/\r?\n/g, ' ').trim();
  if (cleaned.length > 200) {
    return cleaned.slice(0, 197) + '...';
  }
  return cleaned;
}

/**
 * Distinguishes service availability problems from actionable configuration blocks and message-level errors.
 */
export function classifyExtractionError(err: unknown, status?: OllamaStatus): ErrorClassification {
  if (status) {
    if (status.availability === 'unknown') {
      return {
        category: 'service_unknown',
        message: sanitizeErrorMessage(status.error || 'Ollama is responding slowly; health check timed out.'),
      };
    }
    if (!status.isRunning) {
      if (
        status.error &&
        (status.error.toLowerCase().includes('timed out') ||
          status.error.toLowerCase().includes('responding slowly'))
      ) {
        return {
          category: 'service_unknown',
          message: sanitizeErrorMessage(status.error),
        };
      }
      return {
        category: 'service_offline',
        message: sanitizeErrorMessage(status.error || 'Ollama is offline.'),
      };
    }
    if (!status.isModelReady) {
      return {
        category: 'service_blocked',
        message: sanitizeErrorMessage(status.error || 'Ollama model is missing or access is blocked.'),
      };
    }
  }

  if (err instanceof OllamaError) {
    if (err.kind === 'offline') {
      return { category: 'service_offline', message: sanitizeErrorMessage(err.message) };
    }
    if (err.kind === 'blocked_origin' || err.kind === 'model_missing') {
      return { category: 'service_blocked', message: sanitizeErrorMessage(err.message) };
    }
    return { category: 'message_failure', message: sanitizeErrorMessage(err.message) };
  }

  if (err instanceof Error) {
    const lower = err.message.toLowerCase();
    if (
      lower.includes('health check timed out') ||
      lower.includes('responding slowly')
    ) {
      return { category: 'service_unknown', message: sanitizeErrorMessage(err.message) };
    }
    if (
      lower.includes('offline') ||
      lower.includes('unreachable') ||
      lower.includes('econnrefused') ||
      lower.includes('fetch failed')
    ) {
      return { category: 'service_offline', message: sanitizeErrorMessage(err.message) };
    }
    if (
      lower.includes('403 forbidden') ||
      lower.includes('rejected the request origin') ||
      lower.includes('rejected the extension origin') ||
      lower.includes('not found in ollama')
    ) {
      return { category: 'service_blocked', message: sanitizeErrorMessage(err.message) };
    }
    return { category: 'message_failure', message: sanitizeErrorMessage(err.message) };
  }

  return { category: 'message_failure', message: sanitizeErrorMessage(err) };
}

/**
 * Enqueues a newly saved writing message into the knowledge extraction pipeline.
 * Performs fast-path pre-classification:
 * - Ineligible messages (code, terminal, structured data, short fragment) are recorded as 'skipped'.
 * - Eligible prose messages are queued under 'pending'.
 * Existing registry records (extracted, skipped, failed_permanent, pending, failed_temporary)
 * are left completely untouched and will never be overwritten by ordinary enqueue.
 */
export async function enqueueKnowledgeMessage(
  message: WritingMessage,
  options: CoordinatorOptions = {}
): Promise<EnqueueResult> {
  if (typeof message.id !== 'number' || isNaN(message.id) || message.id <= 0) {
    return { enqueued: false, status: 'skipped', reason: 'missing_message_id' };
  }

  const messageId = message.id;
  const storage = options.storage || getStorage();
  const now = new Date().toISOString();

  return mutateCoordinatorState((state) => {
    // 1. Existing registry entry check: must never be overwritten by ordinary enqueue
    const reg = state.registry[messageId];
    if (reg) {
      if (
        reg.status === 'extracted' ||
        reg.status === 'failed_permanent' ||
        reg.status === 'skipped'
      ) {
        return { enqueued: false, status: reg.status, reason: 'already_queued_or_extracted' };
      }

      const inFlightOrQueue =
        state.inFlight?.messageId === messageId ||
        state.pendingQueue.some((item) => item.messageId === messageId);

      if (inFlightOrQueue) {
        return { enqueued: false, status: reg.status, reason: 'already_queued_or_extracted' };
      }

      // If pending or failed_temporary in registry but missing from pendingQueue (and not inFlight)
      state.pendingQueue.push({
        messageId,
        enqueuedAt: reg.enqueuedAt,
        attempts: reg.attempts || 0,
      });
      return { enqueued: true, status: reg.status };
    }

    // 2. Untracked message: check duplicate in-flight / pending queue
    if (state.inFlight?.messageId === messageId) {
      return { enqueued: false, status: 'pending', reason: 'already_queued_or_extracted' };
    }

    const inQueue = state.pendingQueue.some((item) => item.messageId === messageId);
    if (inQueue) {
      return { enqueued: false, status: 'pending', reason: 'already_queued_or_extracted' };
    }

    // 3. Fast-path content classification for genuinely untracked message
    const classification = classifyContent(message.text || '');
    const isEligible =
      (classification.contentType === 'prose' || classification.contentType === 'mixed') &&
      classification.knowledgeSignal === 'possible';

    if (!isEligible) {
      state.registry[messageId] = {
        messageId,
        status: 'skipped',
        attempts: 0,
        enqueuedAt: now,
        updatedAt: now,
      };
      return { enqueued: false, status: 'skipped', reason: 'ineligible_content' };
    }

    // 4. Untracked eligible message: add to queue and registry
    state.pendingQueue.push({
      messageId,
      enqueuedAt: now,
      attempts: 0,
    });

    state.registry[messageId] = {
      messageId,
      status: 'pending',
      attempts: 0,
      enqueuedAt: now,
      updatedAt: now,
    };

    return { enqueued: true, status: 'pending' };
  }, storage);
}

/**
 * Efficiently synchronizes untracked database writing messages with the coordinator.
 * Performs a single serialized coordinator state mutation for the entire batch.
 * Genuinely untracked messages are classified and either queued as 'pending' or marked 'skipped'.
 * Existing registry records (extracted, skipped, failed_permanent, pending, failed_temporary)
 * are left completely untouched.
 */
export async function syncUntrackedKnowledgeMessages(
  messages: WritingMessage[],
  options: CoordinatorOptions = {}
): Promise<SyncUntrackedResult> {
  const storage = options.storage || getStorage();
  const now = new Date().toISOString();

  if (!Array.isArray(messages) || messages.length === 0) {
    return {
      totalChecked: 0,
      newlyQueuedCount: 0,
      newlySkippedCount: 0,
      alreadyTrackedCount: 0,
      invalidCount: 0,
    };
  }

  return mutateCoordinatorState((state) => {
    let newlyQueuedCount = 0;
    let newlySkippedCount = 0;
    let alreadyTrackedCount = 0;
    let invalidCount = 0;

    // Track existing message IDs in pendingQueue and inFlight to prevent duplicates within batch
    const pendingSet = new Set(state.pendingQueue.map((item) => item.messageId));
    if (state.inFlight?.messageId !== undefined && state.inFlight.messageId !== null) {
      pendingSet.add(state.inFlight.messageId);
    }

    for (const message of messages) {
      if (!message || typeof message.id !== 'number' || isNaN(message.id) || message.id <= 0) {
        invalidCount++;
        continue;
      }

      const messageId = message.id;

      // 1. Existing registry entries must remain completely untouched
      if (state.registry[messageId]) {
        alreadyTrackedCount++;
        continue;
      }

      // 2. Prevent duplicate entries if the input batch contains duplicate IDs
      if (pendingSet.has(messageId)) {
        alreadyTrackedCount++;
        continue;
      }

      // 3. Fast-path content classification for genuinely untracked message
      const classification = classifyContent(message.text || '');
      const isEligible =
        (classification.contentType === 'prose' || classification.contentType === 'mixed') &&
        classification.knowledgeSignal === 'possible';

      if (!isEligible) {
        state.registry[messageId] = {
          messageId,
          status: 'skipped',
          attempts: 0,
          enqueuedAt: now,
          updatedAt: now,
        };
        newlySkippedCount++;
      } else {
        state.pendingQueue.push({
          messageId,
          enqueuedAt: now,
          attempts: 0,
        });
        state.registry[messageId] = {
          messageId,
          status: 'pending',
          attempts: 0,
          enqueuedAt: now,
          updatedAt: now,
        };
        pendingSet.add(messageId);
        newlyQueuedCount++;
      }
    }

    return {
      totalChecked: messages.length,
      newlyQueuedCount,
      newlySkippedCount,
      alreadyTrackedCount,
      invalidCount,
    };
  }, storage);
}

/**
 * Processes queued messages sequentially with a concurrency of one.
 * Adheres to error classification:
 * - Temporary service outage -> halts queue, sets alarm, preserves attempts.
 * - Actionable blocked configuration -> halts queue, clears alarm, sets status.
 * - Message-level failure -> increments attempts, retries up to 3 times before moving to failed_permanent.
 */
export async function processKnowledgeQueue(
  options: CoordinatorOptions = {}
): Promise<ProcessQueueResult> {
  if (isProcessingQueue) {
    return { status: 'already_running', processedCount: 0 };
  }

  isProcessingQueue = true;
  const storage = options.storage || getStorage();
  const targetDb = options.targetDb || db;
  const ollamaOptions = options.ollamaOptions || {};
  const baseUrl = ollamaOptions.baseUrl || DEFAULT_OLLAMA_BASE_URL;
  const model = ollamaOptions.model || DEFAULT_OLLAMA_MODEL;
  const cooldownMs = options.cooldownMs ?? 1500;
  const sleepFn = options.sleepFn ?? ((ms: number) => new Promise((resolve) => setTimeout(resolve, ms)));

  let processedCount = 0;
  const seenInThisRun = new Set<number>();

  try {
    while (true) {
      // 1. Check service availability pre-flight before popping work
      let serviceCheck: OllamaStatus;
      try {
        serviceCheck = await checkOllamaStatus(model, baseUrl);
      } catch (err: unknown) {
        const isAbort = err instanceof Error && err.name === 'AbortError';
        serviceCheck = {
          isRunning: false,
          isModelReady: false,
          modelName: model,
          availability: isAbort ? 'unknown' : 'offline',
          error: sanitizeErrorMessage(err),
        };
      }

      if (!serviceCheck.isRunning) {
        const isUnknown =
          serviceCheck.availability === 'unknown' ||
          Boolean(
            serviceCheck.error &&
              (serviceCheck.error.toLowerCase().includes('timed out') ||
                serviceCheck.error.toLowerCase().includes('responding slowly'))
          );

        if (isUnknown) {
          await mutateCoordinatorState((state) => {
            state.serviceStatus = 'unknown';
            state.serviceError = serviceCheck.error || 'Ollama is responding slowly.';
            state.lastCheckedAt = new Date().toISOString();
          }, storage);

          const currentQueue = (await getCoordinatorState(storage)).pendingQueue;
          if (currentQueue.length > 0) {
            await scheduleKnowledgeRetryAlarm();
          }
          return { status: 'service_unknown', processedCount, error: serviceCheck.error };
        }

        await mutateCoordinatorState((state) => {
          state.serviceStatus = 'offline';
          state.serviceError = serviceCheck.error || 'Ollama is offline.';
          state.lastCheckedAt = new Date().toISOString();
        }, storage);

        const currentQueue = (await getCoordinatorState(storage)).pendingQueue;
        if (currentQueue.length > 0) {
          await scheduleKnowledgeRetryAlarm();
        }
        return { status: 'service_offline', processedCount, error: serviceCheck.error };
      }

      if (!serviceCheck.isModelReady) {
        await mutateCoordinatorState((state) => {
          state.serviceStatus = 'blocked_configuration';
          state.serviceError = serviceCheck.error || 'Model is not ready or origin rejected.';
          state.lastCheckedAt = new Date().toISOString();
        }, storage);

        // Do not create an endless alarm loop for actionable configuration blocks
        await clearKnowledgeRetryAlarm();
        return { status: 'service_blocked', processedCount, error: serviceCheck.error };
      }

      // Mark service online
      await mutateCoordinatorState((state) => {
        state.serviceStatus = 'online';
        state.serviceError = undefined;
        state.lastCheckedAt = new Date().toISOString();
      }, storage);

      // 2. Pop next item from pending queue and record in-flight
      const popped = await mutateCoordinatorState((state) => {
        if (state.pendingQueue.length === 0) {
          state.inFlight = null;
          return null;
        }

        // Avoid busy-looping over the same item within the same queue run
        const head = state.pendingQueue[0]!;
        if (seenInThisRun.has(head.messageId)) {
          state.inFlight = null;
          return null;
        }

        const item = state.pendingQueue.shift()!;
        seenInThisRun.add(item.messageId);
        const nextAttempt = item.attempts + 1;
        const now = new Date().toISOString();

        state.inFlight = {
          messageId: item.messageId,
          startedAt: now,
          attempt: nextAttempt,
        };

        const existingReg = state.registry[item.messageId];
        state.registry[item.messageId] = {
          messageId: item.messageId,
          status: 'pending',
          attempts: nextAttempt,
          enqueuedAt: existingReg?.enqueuedAt || item.enqueuedAt,
          updatedAt: now,
          lastAttemptAt: now,
        };

        return { item, nextAttempt };
      }, storage);

      if (!popped) {
        // If there are still pending items (e.g. ones that failed and were requeued for next run), schedule retry alarm
        const remainingQueue = (await getCoordinatorState(storage)).pendingQueue;
        if (remainingQueue.length > 0) {
          await scheduleKnowledgeRetryAlarm();
        } else {
          await clearKnowledgeRetryAlarm();
        }
        return { status: processedCount > 0 ? 'completed' : 'empty', processedCount };
      }

      const { item, nextAttempt } = popped;
      const messageId = item.messageId;

      // 3. Verify message still exists in database
      if (!targetDb.isOpen()) {
        await targetDb.open();
      }

      const sourceMsg = await targetDb.writingMessages.get(messageId);
      if (!sourceMsg) {
        // Source message was deleted before extraction started
        await mutateCoordinatorState((state) => {
          state.inFlight = null;
          delete state.registry[messageId];
        }, storage);
        continue;
      }

      // 4. Execute extraction using reusable extraction function
      try {
        const result: ExtractSingleMessageResult = await extractKnowledgeFromMessage(sourceMsg, {
          targetDb,
          ollamaOptions,
        });

        if (result.outcome === 'aborted_deleted') {
          // Message was deleted during extraction
          await mutateCoordinatorState((state) => {
            state.inFlight = null;
            delete state.registry[messageId];
          }, storage);
          continue;
        }

        // Successfully evaluated (either with grounded facts or verified empty)
        await mutateCoordinatorState((state) => {
          state.inFlight = null;
          const reg = state.registry[messageId];
          if (reg) {
            reg.status = 'extracted';
            reg.attempts = nextAttempt;
            reg.extractedCount = result.extractedCount;
            reg.duplicateCount = result.duplicateCount;
            reg.lastError = undefined;
            reg.updatedAt = new Date().toISOString();
          }
        }, storage);

        processedCount++;
      } catch (extractErr: unknown) {
        const classification = classifyExtractionError(extractErr);

        if (classification.category === 'service_unknown') {
          // Transient health check timeout / slow response: preserve attempts, pause queue, schedule alarm
          await mutateCoordinatorState((state) => {
            state.inFlight = null;
            state.pendingQueue.unshift({
              messageId,
              enqueuedAt: item.enqueuedAt,
              attempts: item.attempts, // preserved without increment
            });
            const reg = state.registry[messageId];
            if (reg) {
              reg.status = 'pending';
              reg.attempts = item.attempts;
              reg.lastError = classification.message;
              reg.updatedAt = new Date().toISOString();
            }
            state.serviceStatus = 'unknown';
            state.serviceError = classification.message;
            state.lastCheckedAt = new Date().toISOString();
          }, storage);

          await scheduleKnowledgeRetryAlarm();
          return { status: 'service_unknown', processedCount, error: classification.message };
        }

        if (classification.category === 'service_offline') {
          // Temporary service outage: do NOT increment attempt count, restore message to pending
          await mutateCoordinatorState((state) => {
            state.inFlight = null;
            state.pendingQueue.unshift({
              messageId,
              enqueuedAt: item.enqueuedAt,
              attempts: item.attempts, // preserved without increment
            });
            const reg = state.registry[messageId];
            if (reg) {
              reg.status = 'pending';
              reg.attempts = item.attempts;
              reg.lastError = classification.message;
              reg.updatedAt = new Date().toISOString();
            }
            state.serviceStatus = 'offline';
            state.serviceError = classification.message;
            state.lastCheckedAt = new Date().toISOString();
          }, storage);

          await scheduleKnowledgeRetryAlarm();
          return { status: 'service_offline', processedCount, error: classification.message };
        }

        if (classification.category === 'service_blocked') {
          // Actionable configuration block: do NOT increment attempts, halt queue, clear alarm
          await mutateCoordinatorState((state) => {
            state.inFlight = null;
            state.pendingQueue.unshift({
              messageId,
              enqueuedAt: item.enqueuedAt,
              attempts: item.attempts,
            });
            const reg = state.registry[messageId];
            if (reg) {
              reg.status = 'pending';
              reg.attempts = item.attempts;
              reg.lastError = classification.message;
              reg.updatedAt = new Date().toISOString();
            }
            state.serviceStatus = 'blocked_configuration';
            state.serviceError = classification.message;
            state.lastCheckedAt = new Date().toISOString();
          }, storage);

          await clearKnowledgeRetryAlarm();
          return { status: 'service_blocked', processedCount, error: classification.message };
        }

        // Message-level failure (JSON parsing, message timeout, empty body)
        await mutateCoordinatorState((state) => {
          state.inFlight = null;
          const now = new Date().toISOString();
          const reg = state.registry[messageId];

          if (nextAttempt < MAX_EXTRACTION_ATTEMPTS) {
            // Retries remain: mark failed_temporary and requeue at tail
            if (reg) {
              reg.status = 'failed_temporary';
              reg.attempts = nextAttempt;
              reg.lastError = classification.message;
              reg.updatedAt = now;
            }
            state.pendingQueue.push({
              messageId,
              enqueuedAt: item.enqueuedAt,
              attempts: nextAttempt,
            });
          } else {
            // Exhausted 3 retries: move to failed_permanent dead-letter state
            if (reg) {
              reg.status = 'failed_permanent';
              reg.attempts = nextAttempt;
              reg.lastError = classification.message;
              reg.updatedAt = now;
            }
          }
        }, storage);

        // Cooldown after an inference timeout or HTTP 500 before continuing
        if (isCooldownEligible(extractErr)) {
          await sleepFn(cooldownMs);
        }
      }
    }
  } finally {
    isProcessingQueue = false;
  }
}

/**
 * Recovers interrupted in-flight work after a service worker restart or browser restart.
 *
 * Recovery policy:
 * - If no in-flight message exists, do nothing ('idle').
 * - If the source writing message no longer exists in writingMessages, clear the in-flight
 *   record and discard its registry entry ('discarded_deleted').
 * - If the source message still exists, always requeue it at the front of pendingQueue ('requeued').
 *   Preserves previous attempt count and relevant timestamps.
 * - Never infers extraction completion from pre-existing evidence records.
 *   Accepts at-least-once execution; the extraction engine already handles duplicate knowledge safely.
 */
export async function recoverKnowledgeCoordinator(
  options: CoordinatorOptions = {}
): Promise<RecoveryResult> {
  const storage = options.storage || getStorage();
  const targetDb = options.targetDb || db;

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return mutateCoordinatorState(async (state) => {
    if (!state.inFlight) {
      return { recovered: false, action: 'idle' };
    }

    const inFlightId = state.inFlight.messageId;
    const inFlightAttempt = state.inFlight.attempt;

    // 1. Check if source message was deleted
    const exists = await targetDb.writingMessages.get(inFlightId);
    if (!exists) {
      state.inFlight = null;
      delete state.registry[inFlightId];
      return { recovered: true, action: 'discarded_deleted', messageId: inFlightId };
    }

    // 2. Always requeue at the front of pending queue (at-least-once guarantee)
    // Never infer completion from evidence: evidence could be pre-existing or empty results have no evidence
    state.inFlight = null;
    const preservedAttempts = Math.max(0, inFlightAttempt - 1);

    state.pendingQueue.unshift({
      messageId: inFlightId,
      enqueuedAt: state.registry[inFlightId]?.enqueuedAt || new Date().toISOString(),
      attempts: preservedAttempts,
    });

    const reg = state.registry[inFlightId];
    if (reg) {
      reg.status = 'pending';
      reg.attempts = preservedAttempts;
      reg.updatedAt = new Date().toISOString();
    }

    return { recovered: true, action: 'requeued', messageId: inFlightId };
  }, storage);
}

/**
 * Manually retries permanently failed messages by resetting their attempt count and requeueing them.
 * Does NOT delete the original writing message or alter existing knowledge.
 */
export async function retryFailedKnowledgeMessages(
  messageIds?: number[],
  options: CoordinatorOptions = {}
): Promise<number> {
  const storage = options.storage || getStorage();
  const targetSet = messageIds ? new Set(messageIds) : null;
  const now = new Date().toISOString();

  return mutateCoordinatorState((state) => {
    let requeuedCount = 0;

    for (const [idStr, entry] of Object.entries(state.registry)) {
      const msgId = Number(idStr);
      if (entry.status === 'failed_permanent' && (!targetSet || targetSet.has(msgId))) {
        entry.status = 'pending';
        entry.attempts = 0;
        entry.lastError = undefined;
        entry.updatedAt = now;

        // Add to queue if not already present
        if (!state.pendingQueue.some((q) => q.messageId === msgId)) {
          state.pendingQueue.push({
            messageId: msgId,
            enqueuedAt: now,
            attempts: 0,
          });
          requeuedCount++;
        }
      }
    }

    return requeuedCount;
  }, storage);
}

/**
 * Manually reconsiders previously skipped messages.
 * Re-evaluates eligibility against contentClassifier before enqueueing.
 * Does NOT blindly enqueue ineligible messages.
 */
export async function reconsiderSkippedKnowledgeMessages(
  messageIds?: number[],
  options: CoordinatorOptions = {}
): Promise<number> {
  const storage = options.storage || getStorage();
  const targetDb = options.targetDb || db;
  const targetSet = messageIds ? new Set(messageIds) : null;
  const now = new Date().toISOString();

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return mutateCoordinatorState(async (state) => {
    let enqueuedCount = 0;

    for (const [idStr, entry] of Object.entries(state.registry)) {
      const msgId = Number(idStr);
      if (entry.status === 'skipped' && (!targetSet || targetSet.has(msgId))) {
        const msg = await targetDb.writingMessages.get(msgId);
        if (!msg) {
          delete state.registry[msgId];
          continue;
        }

        const classification = classifyContent(msg.text);
        const isEligible =
          (classification.contentType === 'prose' || classification.contentType === 'mixed') &&
          classification.knowledgeSignal === 'possible';

        if (isEligible) {
          entry.status = 'pending';
          entry.attempts = 0;
          entry.updatedAt = now;

          if (!state.pendingQueue.some((q) => q.messageId === msgId)) {
            state.pendingQueue.push({
              messageId: msgId,
              enqueuedAt: now,
              attempts: 0,
            });
            enqueuedCount++;
          }
        } else {
          entry.updatedAt = now;
        }
      }
    }

    return enqueuedCount;
  }, storage);
}

/**
 * Returns a high-level summary of the coordinator's current state.
 */
export async function getKnowledgeCoordinatorStatus(
  storage: StorageLike = getStorage()
): Promise<KnowledgeCoordinatorStatus> {
  const state = await getCoordinatorState(storage);

  let pendingCount = 0;
  let extractedCount = 0;
  let skippedCount = 0;
  let failedTemporaryCount = 0;
  let failedPermanentCount = 0;

  for (const entry of Object.values(state.registry)) {
    switch (entry.status) {
      case 'pending':
        pendingCount++;
        break;
      case 'extracted':
        extractedCount++;
        break;
      case 'skipped':
        skippedCount++;
        break;
      case 'failed_temporary':
        failedTemporaryCount++;
        break;
      case 'failed_permanent':
        failedPermanentCount++;
        break;
    }
  }

  return {
    pendingCount: state.pendingQueue.length,
    extractedCount,
    skippedCount,
    failedTemporaryCount,
    failedPermanentCount,
    inFlightMessageId: state.inFlight?.messageId || null,
    serviceStatus: state.serviceStatus,
    serviceError: state.serviceError,
    isProcessing: isProcessingQueue,
  };
}
