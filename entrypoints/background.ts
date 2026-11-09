import { saveWritingMessage } from './utils/database';

export interface SaveMessageResponse {
  success: boolean;
  status: 'saved' | 'duplicate' | 'invalid' | 'error';
  id?: number;
  message?: string;
  error?: string;
}

export default defineBackground(() => {
  console.log('Echo background service worker initialized.', { id: browser.runtime.id });

  browser.runtime.onMessage.addListener(
    (message, _sender, sendResponse: (response: SaveMessageResponse) => void) => {
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