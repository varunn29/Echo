import {
  saveWritingMessage,
  db,
  type EchoDatabase,
  type WritingMessage,
} from './utils/database';
import {
  checkAndTriggerTraining,
  pollActiveTrainingJob,
  clearServiceRetryAlarm,
  RETRY_ALARM_NAME,
} from './utils/trainingTrigger';
import {
  enqueueKnowledgeMessage,
  processKnowledgeQueue,
  recoverKnowledgeCoordinator,
  retryFailedKnowledgeMessages,
  reconsiderSkippedKnowledgeMessages,
  getKnowledgeCoordinatorStatus,
  syncUntrackedKnowledgeMessages,
  KNOWLEDGE_RETRY_ALARM_NAME,
  type CoordinatorOptions,
} from './utils/knowledgeCoordinator';

export interface SaveMessageResponse {
  success: boolean;
  status: 'saved' | 'duplicate' | 'invalid' | 'error';
  id?: number;
  message?: string;
  error?: string;
}

export type BackgroundOptions = CoordinatorOptions;

let startupPromise: Promise<void> | null = null;

/**
 * Initializes background services upon service worker startup.
 * Guaranteed to be idempotent: concurrent or repeated calls return the same in-flight promise.
 */
export function initializeBackgroundServices(
  options?: BackgroundOptions
): Promise<void> {
  if (startupPromise) {
    return startupPromise;
  }

  startupPromise = (async () => {
    // 1. Recover and poll active training job (independent)
    pollActiveTrainingJob({ targetDb: options?.targetDb }).catch(function (err) {
      console.warn('Echo: Startup active job poll error:', err);
    });

    // 2. Recover knowledge coordinator and resume pending queue (independent)
    try {
      const recovery = await recoverKnowledgeCoordinator(options);
      if (recovery.recovered) {
        console.log('Echo: Knowledge coordinator recovered in-flight work:', recovery);
      }
      await processKnowledgeQueue(options);
    } catch (err) {
      console.warn('Echo: Startup knowledge coordinator recovery error:', err);
    }
  })();

  return startupPromise;
}

export function resetBackgroundStateForTesting(): void {
  startupPromise = null;
}

/**
 * Handles Chrome alarms for service retry polling.
 * Dispatches independently to training or knowledge workflows based on alarm name.
 */
export function handleBackgroundAlarm(
  alarm: { name: string },
  options?: BackgroundOptions
): void {
  if (alarm.name === RETRY_ALARM_NAME) {
    console.log('Echo: Retry alarm fired. Checking training service availability...');
    checkAndTriggerTraining({ targetDb: options?.targetDb })
      .then(function (res) {
        if (res.action === 'submitted' || res.action === 'insufficient_samples') {
          clearServiceRetryAlarm().catch(function () {});
        }
      })
      .catch(function (err) {
        console.warn('Echo: Alarm trigger error:', err);
      });
  } else if (alarm.name === KNOWLEDGE_RETRY_ALARM_NAME) {
    console.log('Echo: Knowledge retry alarm fired. Processing knowledge queue...');
    processKnowledgeQueue(options).catch(function (err) {
      console.warn('Echo: Knowledge retry alarm processing error:', err);
    });
  }
}

/**
 * Handles incoming runtime messages from content scripts and dashboard.
 * Returns true if an asynchronous sendResponse will be invoked (required for Chrome MV3).
 */
export function handleRuntimeMessage(
  message: any,
  _sender: any,
  sendResponse: (response: any) => void,
  options?: BackgroundOptions
): boolean {
  if (!message || typeof message !== 'object') {
    return false;
  }

  // Handle training status queries from dashboard
  if (message.type === 'CHECK_TRAINING_STATUS') {
    pollActiveTrainingJob({ targetDb: options?.targetDb })
      .then(function (job) {
        sendResponse({ success: true, activeJob: job });
      })
      .catch(function (err) {
        sendResponse({ success: false, error: err instanceof Error ? err.message : String(err) });
      });
    return true;
  }

  if (message.type === 'TRIGGER_TRAINING_CHECK') {
    checkAndTriggerTraining({ targetDb: options?.targetDb })
      .then(function (result) {
        sendResponse({ success: true, result });
      })
      .catch(function (err) {
        sendResponse({ success: false, error: err instanceof Error ? err.message : String(err) });
      });
    return true;
  }

  // Handle knowledge coordinator status queries
  if (message.type === 'GET_KNOWLEDGE_COORDINATOR_STATUS') {
    getKnowledgeCoordinatorStatus(options?.storage)
      .then(function (status) {
        sendResponse({ success: true, status });
      })
      .catch(function (err) {
        sendResponse({
          success: false,
          error: err instanceof Error ? err.message : String(err),
        });
      });
    return true;
  }

  // Handle manual knowledge extraction trigger
  if (message.type === 'START_KNOWLEDGE_EXTRACTION') {
    (async () => {
      // Ensure any untracked writing messages in the database are enqueued in a single batch
      const targetDb = options?.targetDb || db;
      if (!targetDb.isOpen()) {
        await targetDb.open();
      }
      const existingMessages = await targetDb.writingMessages.toArray();
      await syncUntrackedKnowledgeMessages(existingMessages, options);
      return processKnowledgeQueue(options);
    })()
      .then(function (result) {
        sendResponse({ success: true, result });
      })
      .catch(function (err) {
        sendResponse({
          success: false,
          error: err instanceof Error ? err.message : String(err),
        });
      });
    return true;
  }

  // Handle manual retry of failed knowledge messages
  if (message.type === 'RETRY_FAILED_KNOWLEDGE_MESSAGES') {
    if (
      message.messageIds !== undefined &&
      (!Array.isArray(message.messageIds) ||
        message.messageIds.some(
          (id: unknown) => typeof id !== 'number' || !Number.isInteger(id) || id <= 0
        ))
    ) {
      sendResponse({
        success: false,
        error: 'Invalid messageIds: must be an array of positive integers.',
      });
      return false;
    }

    retryFailedKnowledgeMessages(message.messageIds, options)
      .then(function (requeuedCount) {
        sendResponse({ success: true, requeuedCount });
        processKnowledgeQueue(options).catch(function (procErr) {
          console.warn('Echo: Knowledge queue processing error after retry:', procErr);
        });
      })
      .catch(function (err) {
        sendResponse({
          success: false,
          error: err instanceof Error ? err.message : String(err),
        });
      });
    return true;
  }

  // Handle reconsideration of skipped knowledge messages
  if (message.type === 'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES') {
    if (
      message.messageIds !== undefined &&
      (!Array.isArray(message.messageIds) ||
        message.messageIds.some(
          (id: unknown) => typeof id !== 'number' || !Number.isInteger(id) || id <= 0
        ))
    ) {
      sendResponse({
        success: false,
        error: 'Invalid messageIds: must be an array of positive integers.',
      });
      return false;
    }

    reconsiderSkippedKnowledgeMessages(message.messageIds, options)
      .then(function (requeuedCount) {
        sendResponse({ success: true, requeuedCount });
        processKnowledgeQueue(options).catch(function (procErr) {
          console.warn('Echo: Knowledge queue processing error after reconsider:', procErr);
        });
      })
      .catch(function (err) {
        sendResponse({
          success: false,
          error: err instanceof Error ? err.message : String(err),
        });
      });
    return true;
  }

  // Handle ChatGPT message capture
  if (message.type !== 'NEW_USER_MESSAGE') {
    return false;
  }

  if (
    typeof message.text !== 'string' ||
    !message.text.trim()
  ) {
    console.warn('Echo: Invalid message rejected (empty text).');
    sendResponse({
      success: false,
      status: 'invalid',
      error: 'Message text must not be empty.',
    });
    return false;
  }

  const currentTime = new Date();
  const trimmedText = message.text.trim();

  saveWritingMessage(
    {
      platform: 'ChatGPT',
      text: trimmedText,
      createdAt: currentTime,
      capturedAt: currentTime,
      trainingStatus: 'pending',
    },
    undefined,
    options?.targetDb
  )
    .then(function (id) {
      if (id === undefined) {
        console.log('Echo: Duplicate message ignored within deduplication window.');
        sendResponse({
          success: true,
          status: 'duplicate',
          message: 'Duplicate message ignored within deduplication window.',
        });
        return;
      }

      if (typeof id !== 'number' || isNaN(id) || id <= 0) {
        console.warn('Echo: Message persisted but ID is invalid or missing:', id);
        sendResponse({
          success: true,
          status: 'saved',
          id: id,
        });
        return;
      }

      console.log('Echo: Message persisted successfully with ID:', id);
      sendResponse({
        success: true,
        status: 'saved',
        id: id,
      });

      // 1. Check training readiness and automatically trigger training if ready (independent)
      checkAndTriggerTraining({ targetDb: options?.targetDb }).catch(function (triggerErr) {
        console.warn('Echo: Background training trigger check error:', triggerErr);
      });

      // 2. Enqueue for automatic personal knowledge extraction (independent)
      const persistedMsg: WritingMessage = {
        id,
        platform: 'ChatGPT',
        text: trimmedText,
        createdAt: currentTime,
        capturedAt: currentTime,
        trainingStatus: 'pending',
      };

      enqueueKnowledgeMessage(persistedMsg, options)
        .then(function (enqueueRes) {
          if (enqueueRes.enqueued) {
            processKnowledgeQueue(options).catch(function (processErr) {
              console.warn('Echo: Background knowledge queue processing error:', processErr);
            });
          }
        })
        .catch(function (enqueueErr) {
          console.warn('Echo: Background knowledge enqueue error:', enqueueErr);
        });
    })
    .catch(function (error) {
      console.error('Echo: Failed to persist message to database:', error);
      sendResponse({
        success: false,
        status: 'error',
        error: error instanceof Error ? error.message : String(error),
      });
    });

  // Returning true is required in Chrome MV3 to keep the message channel open
  // until sendResponse is invoked asynchronously.
  return true;
}

// Polyfill for test environments where WXT global auto-imports are not present
if (typeof defineBackground === 'undefined') {
  (globalThis as any).defineBackground = (fn: () => void) => fn;
}

/**
 * Background entrypoint definition executed by WXT service worker bootstrap.
 */
export function runBackgroundEntrypoint(): void {
  console.log('Echo background service worker initialized.', {
    id: typeof browser !== 'undefined' && browser.runtime ? browser.runtime.id : undefined,
  });

  // On service worker startup, initialize training and knowledge services
  initializeBackgroundServices().catch(function (err) {
    console.warn('Echo: Background services initialization error:', err);
  });

  // Register alarm listener for automatic service recovery retries (Manifest V3)
  if (typeof browser !== 'undefined' && browser.alarms?.onAlarm) {
    browser.alarms.onAlarm.addListener(function (alarm) {
      handleBackgroundAlarm(alarm);
    });
  }

  // Register runtime message listener
  if (typeof browser !== 'undefined' && browser.runtime?.onMessage) {
    browser.runtime.onMessage.addListener(
      (message, sender, sendResponse: (response: any) => void) => {
        return handleRuntimeMessage(message, sender, sendResponse);
      }
    );
  }
}

export default defineBackground(runBackgroundEntrypoint);
