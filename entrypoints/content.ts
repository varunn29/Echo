import {
  CLIENT_DEBOUNCE_MS,
  deliverMessageWithRetry,
  extractUserMessageText,
  findComposerElement,
  getElementText,
  isSendButton,
  resolveMessageText,
  shouldDebounce,
} from './utils/collectionHelper';

export default defineContentScript({
  matches: ['https://chatgpt.com/*'],

  main(ctx) {
    console.log('Echo: ChatGPT continuous collection script active.');

    let lastBufferedText = '';
    let lastSentMessage = '';
    let lastSentTime = 0;
    const seenUserMessageElements = new WeakSet<Element>();

    // Mark pre-existing conversation messages so they are never collected on initial load or navigation
    function markExistingMessages() {
      const existingUserTurns = document.querySelectorAll('[data-message-author-role="user"]');
      Array.from(existingUserTurns).forEach(function (turn) {
        seenUserMessageElements.add(turn);
      });
    }

    markExistingMessages();

    async function handleSendMessage(explicitText?: string) {
      const composer = findComposerElement();
      const domText = explicitText !== undefined && explicitText.trim()
        ? explicitText.trim()
        : getElementText(composer);

      const messageText = resolveMessageText(domText, lastBufferedText);
      if (!messageText) {
        return;
      }

      const now = Date.now();
      if (shouldDebounce(messageText, lastSentMessage, now, lastSentTime, CLIENT_DEBOUNCE_MS)) {
        return;
      }

      // Record dispatch immediately to prevent concurrent event loops
      lastSentMessage = messageText;
      lastSentTime = now;
      lastBufferedText = '';

      const response = await deliverMessageWithRetry(messageText, (msg) =>
        browser.runtime.sendMessage(msg)
      );

      if (!response) {
        // Reset on total delivery failure so future attempts/manual retries are not locked out
        lastSentMessage = '';
        return;
      }

      if (response.status === 'saved') {
        console.log('Echo: Message persisted successfully with ID:', response.id);
      } else if (response.status === 'duplicate') {
        console.log('Echo: Duplicate message acknowledged and safely ignored by database.');
      } else if (!response.success) {
        console.warn('Echo: Background could not save message:', response.error);
      }
    }

    // 1. Live Input Buffering
    // Continually buffers user typing so input is never lost if React resets the DOM before click completes
    document.addEventListener(
      'input',
      function () {
        const composer = findComposerElement();
        const text = getElementText(composer);
        if (text) {
          lastBufferedText = text;
        } else {
          lastBufferedText = '';
        }
      },
      true
    );

    document.addEventListener(
      'compositionend',
      function () {
        const composer = findComposerElement();
        const text = getElementText(composer);
        if (text) {
          lastBufferedText = text;
        }
      },
      true
    );

    // 2. Button Activation (pointerdown + mousedown)
    // Critical: React can unmount the send button or replace it with the stop button during mousedown.
    // Capturing on pointerdown/mousedown guarantees the message is read and dispatched BEFORE React tears down the DOM.
    function onSendButtonInteraction(event: Event) {
      const target = event.target as Node | null;
      if (isSendButton(target)) {
        const composer = findComposerElement();
        const text = getElementText(composer);
        if (text) {
          lastBufferedText = text;
        }
        handleSendMessage(text);
      }
    }

    document.addEventListener('pointerdown', onSendButtonInteraction, true);
    document.addEventListener('mousedown', onSendButtonInteraction, true);

    // 3. Button Click (Native click fallback for keyboard-triggered button activations)
    document.addEventListener(
      'click',
      function (event) {
        const target = event.target as Node | null;
        if (isSendButton(target)) {
          handleSendMessage();
        }
      },
      true
    );

    // 4. Keyboard Enter Key
    document.addEventListener(
      'keydown',
      function (event) {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          const target = event.target as Node | null;
          const element = target instanceof Element ? target : target?.parentElement;

          if (
            element &&
            (element.id === 'prompt-textarea' ||
              element.closest('#prompt-textarea') ||
              element.closest('div[contenteditable="true"]') ||
              element.closest('form[data-chatgpt-composer]') ||
              element instanceof HTMLTextAreaElement)
          ) {
            const composer = findComposerElement();
            const text = getElementText(composer);
            handleSendMessage(text);
          }
        }
      },
      true
    );

    // 5. Form Submission
    document.addEventListener(
      'submit',
      function (event) {
        const target = event.target as Node | null;
        if (target instanceof HTMLFormElement) {
          handleSendMessage();
        }
      },
      true
    );

    // 6. Chat Feed Mutation Observer (Safety Net)
    // Catches messages sent via suggestion prompts, voice dictation, or when event listeners were bypassed.
    // Crucially ONLY checks elements with data-message-author-role="user", completely ignoring assistant turns.
    let mutationDebounceTimer: number | null = null;

    function processAddedChatTurns() {
      const userTurns = document.querySelectorAll<HTMLElement>('[data-message-author-role="user"]');
      Array.from(userTurns).forEach(function (turn) {
        if (!seenUserMessageElements.has(turn)) {
          seenUserMessageElements.add(turn);
          const messageText = extractUserMessageText(turn);
          if (messageText) {
            handleSendMessage(messageText);
          }
        }
      });
    }

    const chatObserver = new MutationObserver(function () {
      if (mutationDebounceTimer !== null) {
        clearTimeout(mutationDebounceTimer);
      }
      mutationDebounceTimer = window.setTimeout(function () {
        mutationDebounceTimer = null;
        processAddedChatTurns();
      }, 100);
    });

    chatObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // 7. SPA Navigation and History Transitions
    function handleNavigation() {
      lastBufferedText = '';
      // Allow new route DOM to stabilize, then mark historical messages
      window.setTimeout(function () {
        markExistingMessages();
      }, 300);
    }

    window.addEventListener('popstate', handleNavigation);
    window.addEventListener('hashchange', handleNavigation);

    if (ctx && typeof ctx.addEventListener === 'function') {
      try {
        ctx.addEventListener(window, 'wxt:locationchange' as any, handleNavigation);
        ctx.onInvalidated(function () {
          chatObserver.disconnect();
          if (mutationDebounceTimer !== null) {
            clearTimeout(mutationDebounceTimer);
          }
        });
      } catch (err) {
        console.debug('Echo: WXT context listener registered with fallback:', err);
      }
    }
  },
});
