import { saveWritingMessage } from './utils/database';
import {
  checkAndTriggerTraining,
  pollActiveTrainingJob,
  clearServiceRetryAlarm,
  RETRY_ALARM_NAME,
} from './utils/trainingTrigger';

export interface SaveMessageResponse {
  success: boolean;
  status: 'saved' | 'duplicate' | 'invalid' | 'error';
  id?: number;
  message?: string;
  error?: string;
}

export default defineBackground(() => {
  console.log('Echo background service worker initialized.', { id: browser.runtime.id });

  // On service worker startup, recover and poll any active in-flight training job
  pollActiveTrainingJob().catch(function (err) {
    console.warn('Echo: Startup active job poll error:', err);
  });

  // Register alarm listener for automatic service recovery retries (Manifest V3)
  if (typeof browser !== 'undefined' && browser.alarms?.onAlarm) {
    browser.alarms.onAlarm.addListener(function (alarm) {
      if (alarm.name === RETRY_ALARM_NAME) {
        console.log('Echo: Retry alarm fired. Checking training service availability...');
        checkAndTriggerTraining()
          .then(function (res) {
            if (res.action === 'submitted' || res.action === 'insufficient_samples') {
              clearServiceRetryAlarm().catch(function () {});
            }
          })
          .catch(function (err) {
            console.warn('Echo: Alarm trigger error:', err);
          });
      }
    });
  }

  browser.runtime.onMessage.addListener(
    (message, _sender, sendResponse: (response: any) => void) => {
      // Handle training status queries from dashboard
      if (message?.type === 'CHECK_TRAINING_STATUS') {
        pollActiveTrainingJob()
          .then(function (job) {
            sendResponse({ success: true, activeJob: job });
          })
          .catch(function (err) {
            sendResponse({ success: false, error: String(err) });
          });
        return true;
      }

      if (message?.type === 'TRIGGER_TRAINING_CHECK') {
        checkAndTriggerTraining()
          .then(function (result) {
            sendResponse({ success: true, result });
          })
          .catch(function (err) {
            sendResponse({ success: false, error: String(err) });
          });
        return true;
      }

      if (message?.type !== 'NEW_USER_MESSAGE') {
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

      saveWritingMessage({
        platform: 'ChatGPT',
        text: message.text.trim(),
        createdAt: currentTime,
        capturedAt: currentTime,
        trainingStatus: 'pending',
      })
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

          console.log('Echo: Message persisted successfully with ID:', id);
          sendResponse({
            success: true,
            status: 'saved',
            id: id,
          });

          // Check training readiness and automatically trigger training if ready
          checkAndTriggerTraining().catch(function (triggerErr) {
            console.warn('Echo: Background training trigger check error:', triggerErr);
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
  );
});