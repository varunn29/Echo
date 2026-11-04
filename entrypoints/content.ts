
export default defineContentScript({
  matches: ['https://chatgpt.com/*'],

  main() {
    console.log('Echo: ChatGPT content script loaded.');

    let lastCapturedMessage = '';
    let lastCapturedTime = 0;

    function captureMessage() {
      const form = document.querySelector(
        'form[data-chatgpt-composer]'
      );

      const editor = form?.querySelector(
        '[contenteditable="true"]'
      );

      const message = editor?.innerText.trim();

      if (!message) {
        return;
      }

      const currentTime = Date.now();

      if (
        message === lastCapturedMessage &&
        currentTime - lastCapturedTime < 1500
      ) {
        return;
      }

      lastCapturedMessage = message;
      lastCapturedTime = currentTime;

      console.log('Echo captured:', message);
    }

    document.addEventListener(
      'click',
      function (event) {
        const target = event.target;

        if (!(target instanceof Element)) {
          return;
        }

        const sendButton = target.closest(
          'form[data-chatgpt-composer] button[aria-label="Send"]'
        );

        if (sendButton) {
          captureMessage();
        }
      },
      true
    );

    document.addEventListener(
      'keydown',
      function (event) {
        if (
          event.key === 'Enter' &&
          !event.shiftKey &&
          !event.isComposing
        ) {
          const target = event.target;

          if (
            target instanceof Element &&
            target.closest(
              'form[data-chatgpt-composer] [contenteditable="true"]'
            )
          ) {
            captureMessage();
          }
        }
      },
      true
    );
  },
});
