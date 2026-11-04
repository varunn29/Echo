
import { db } from './utils/database';

export default defineBackground(() => {
  console.log('Hello background!', { id: browser.runtime.id });

  browser.runtime.onMessage.addListener(function (message) {
    if (message.type === 'NEW_USER_MESSAGE') {
      const currentTime = new Date();

      db.writingMessages.add({
        platform: 'ChatGPT',
        text: message.text,
        createdAt: currentTime,
        capturedAt: currentTime,
        trainingStatus: 'pending',
      })
        .then(function (id) {
          console.log('Echo: Message saved with ID:', id);
        })
        .catch(function (error) {
          console.error('Echo: Failed to save message:', error);
        });
    }
  });
});
