
import { saveWritingMessage } from './utils/database';

export default defineBackground(() => {
  console.log('Hello background!', { id: browser.runtime.id });

  browser.runtime.onMessage.addListener(function (message) {
    if (message.type !== 'NEW_USER_MESSAGE') {
      return;
    }

    if (
      typeof message.text !== 'string' ||
      !message.text.trim()
    ) {
      console.warn('Echo: Invalid message ignored.');
      return;
    }

    const currentTime = new Date();

    saveWritingMessage({
      platform: 'ChatGPT',
      text: message.text,
      createdAt: currentTime,
      capturedAt: currentTime,
      trainingStatus: 'pending',
    })
      .then(function (id) {
        if (id === undefined) {
          console.log('Echo: Duplicate message ignored.');
          return;
        }

        console.log('Echo: Message saved with ID:', id);
      })
      .catch(function (error) {
        console.error('Echo: Failed to save message:', error);
      });
  });
});
