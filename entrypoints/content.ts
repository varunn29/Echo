export default defineContentScript({
  matches: ['https://chatgpt.com/*'],

  main() {
    console.log('Echo: ChatGPT content script loaded.');

    let lastBufferedText = '';
    let lastSentMessage = '';
    let lastSentTime = 0;
    const CLIENT_DEBOUNCE_MS = 2000;

    function findComposerElement(): HTMLElement | null {
      return (
        document.getElementById('prompt-textarea') ||
        document.querySelector<HTMLElement>(
          'form[data-chatgpt-composer] [contenteditable="true"]'
        ) ||
        document.querySelector<HTMLElement>('form [contenteditable="true"]') ||
        document.querySelector<HTMLElement>('form textarea') ||
        document.querySelector<HTMLElement>(
          '[contenteditable="true"][data-placeholder]'
        ) ||
        document.querySelector<HTMLElement>('div[contenteditable="true"]')
      );
    }

    function getElementText(element: HTMLElement | null): string {
      if (!element) {
        return '';
      }

      if (
        element instanceof HTMLTextAreaElement ||
        element instanceof HTMLInputElement
      ) {
        return element.value.trim();
      }

      return (element.innerText || element.textContent || '').trim();
    }

    // Continually buffer text as the user types/pastes so we never lose it
    // if React clears the DOM before submit/click listeners execute.
    document.addEventListener(
      'input',
      function () {
        const composer = findComposerElement();
        const text = getElementText(composer);
        if (text) {
          lastBufferedText = text;
        }
      },
      true
    );

    function isSendButton(target: Element): boolean {
      return Boolean(
        target.closest(
          'button[data-testid="send-button"], button[aria-label*="Send" i], button[data-testid="composer-send-button"], form button[type="submit"]'
        )
      );
    }

    async function captureAndSendMessage() {
      // 1. Read directly from the DOM composer first.
      const composer = findComposerElement();
      let messageText = getElementText(composer);

      // 2. Fall back to buffered text if the DOM was already wiped by ChatGPT.
      if (!messageText && lastBufferedText) {
        messageText = lastBufferedText;
      }

      if (!messageText) {
        return;
      }

      const now = Date.now();

      // Prevent accidental duplicate sends across multiple listeners for the same user action.
      if (
        messageText === lastSentMessage &&
        now - lastSentTime < CLIENT_DEBOUNCE_MS
      ) {
        return;
      }

      lastSentMessage = messageText;
      lastSentTime = now;
      lastBufferedText = '';

      try {
        const response = await browser.runtime.sendMessage({
          type: 'NEW_USER_MESSAGE',
          text: messageText,
        });

        if (!response) {
          console.warn(
            'Echo: No acknowledgment received from background service worker.'
          );
          return;
        }

        if (!response.success) {
          console.error(
            'Echo: Background failed to persist message:',
            response.error
          );
          return;
        }

        if (response.status === 'duplicate') {
          console.log(
            'Echo: Duplicate message acknowledged and safely ignored by database.'
          );
          return;
        }

        if (response.status === 'saved') {
          console.log(
            'Echo: Message persisted to database with ID:',
            response.id
          );
        }
      } catch (error) {
        console.error(
          'Echo: Failed to deliver message to background service worker:',
          error
        );
      }
    }

    // Buffer on pointerdown before React starts unmounting/clearing the input on click.
    document.addEventListener(
      'pointerdown',
      function (event) {
        if (event.target instanceof Element && isSendButton(event.target)) {
          const composer = findComposerElement();
          const text = getElementText(composer);
          if (text) {
            lastBufferedText = text;
          }
        }
      },
      true
    );

    // Capture when Send button is clicked.
    document.addEventListener(
      'click',
      function (event) {
        if (event.target instanceof Element && isSendButton(event.target)) {
          captureAndSendMessage();
        }
      },
      true
    );

    // Capture when Enter is pressed without Shift or IME composition.
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
            (target.id === 'prompt-textarea' ||
              target.closest('#prompt-textarea') ||
              target.closest('[contenteditable="true"]') ||
              target instanceof HTMLTextAreaElement)
          ) {
            captureAndSendMessage();
          }
        }
      },
      true
    );

    // Capture when the ChatGPT form is submitted.
    document.addEventListener(
      'submit',
      function (event) {
        const target = event.target;

        if (
          target instanceof HTMLFormElement &&
          (target.matches('form[data-chatgpt-composer]') ||
            target.querySelector('#prompt-textarea') ||
            target.querySelector('[contenteditable="true"]') ||
            target.querySelector('textarea'))
        ) {
          captureAndSendMessage();
        }
      },
      true
    );
  },
});