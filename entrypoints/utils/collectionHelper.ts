export interface SaveMessageResponse {
  success: boolean;
  status: 'saved' | 'duplicate' | 'invalid' | 'error';
  id?: number;
  message?: string;
  error?: string;
}

export const CLIENT_DEBOUNCE_MS = 2000;

/**
 * Safe cross-environment check for DOM Element instance.
 */
export function isElement(node: any): node is Element {
  if (!node || typeof node !== 'object') {
    return false;
  }
  if (typeof Element !== 'undefined') {
    return node instanceof Element;
  }
  return node.nodeType === 1;
}

/**
 * Determines whether a DOM node is or is contained within a ChatGPT Send/Submit button.
 * Matches known ChatGPT button attributes across desktop and mobile layouts.
 */
export function isSendButton(target: any): boolean {
  if (!target) {
    return false;
  }

  const element: any = isElement(target) ? target : target.parentElement;
  if (!element || typeof element.closest !== 'function') {
    return false;
  }

  // Explicitly ignore Stop Generation buttons and Assistant response action buttons
  if (
    element.closest(
      'button[data-testid="stop-button"], button[aria-label*="Stop" i], [data-message-author-role="assistant"]'
    )
  ) {
    return false;
  }

  const sendButton = element.closest(
    [
      'button[data-testid="send-button"]',
      'button[data-testid="fruitjuice-send-button"]',
      'button[data-testid="composer-send-button"]',
      'button#composer-submit-button',
      'button[aria-label*="Send" i]',
      'button[aria-label*="Submit" i]',
      'form[data-chatgpt-composer] button[type="submit"]',
      'form button[type="submit"]',
    ].join(', ')
  );

  return Boolean(sendButton);
}

/**
 * Finds the currently active ChatGPT composer element.
 * Prioritizes connected and visible elements to avoid stale React DOM fragments.
 */
export function findComposerElement(root?: any): HTMLElement | null {
  const docRoot = root || (typeof document !== 'undefined' ? document : null);
  if (!docRoot || typeof docRoot.querySelectorAll !== 'function') {
    return null;
  }

  const candidateSelectors = [
    '#prompt-textarea',
    'div[contenteditable="true"].ProseMirror',
    'div[role="textbox"][contenteditable="true"]',
    'form[data-chatgpt-composer] [contenteditable="true"]',
    'form [contenteditable="true"]',
    'form textarea',
    'textarea[data-id="root"]',
    'div[contenteditable="true"][data-placeholder]',
    'div[contenteditable="true"]',
  ];

  for (const selector of candidateSelectors) {
    const matches: any[] = Array.from(docRoot.querySelectorAll(selector));
    for (const el of matches) {
      if (el.isConnected !== false) {
        // If element has layout and is not hidden
        if (el.offsetParent !== null || (el.getClientRects && el.getClientRects().length > 0)) {
          return el;
        }
      }
    }
    // Fall back to first matching connected element if layout info isn't available (e.g. in test env)
    if (matches.length > 0 && matches[0].isConnected !== false) {
      return matches[0];
    }
  }

  if (typeof docRoot.getElementById === 'function') {
    return docRoot.getElementById('prompt-textarea');
  }

  return null;
}

/**
 * Extracts and cleans the text value from a composer element.
 * Works with native HTML textareas, inputs, and ProseMirror contenteditable elements.
 */
export function getElementText(element: any): string {
  if (!element) {
    return '';
  }

  if (typeof HTMLTextAreaElement !== 'undefined' && element instanceof HTMLTextAreaElement) {
    return element.value.trim();
  }
  if (typeof HTMLInputElement !== 'undefined' && element instanceof HTMLInputElement) {
    return element.value.trim();
  }

  // If mock or value is directly present
  if (typeof element.value === 'string') {
    return element.value.trim();
  }

  // For contenteditable/ProseMirror: innerText preserves paragraph breaks; fallback to textContent
  const rawText = (element.innerText || element.textContent || '').trim();
  return rawText;
}

/**
 * Pure debouncing logic for client-side message submissions.
 * Prevents duplicate submissions caused by multiple event triggers for the same user action.
 */
export function shouldDebounce(
  currentText: string,
  lastText: string,
  currentTime: number,
  lastTime: number,
  debounceMs: number = CLIENT_DEBOUNCE_MS
): boolean {
  const trimmed = currentText.trim();
  if (!trimmed) {
    return true;
  }

  if (trimmed === lastText.trim() && currentTime - lastTime < debounceMs) {
    return true;
  }

  return false;
}

/**
 * Resolves the message text, prioritizing live DOM text and falling back to buffered input.
 */
export function resolveMessageText(
  domText: string,
  bufferedText: string
): string {
  const trimmedDom = domText.trim();
  if (trimmedDom) {
    return trimmedDom;
  }
  return bufferedText.trim();
}

/**
 * Checks if a DOM element represents a user message in ChatGPT's chat stream.
 */
export function isUserMessageElement(element: any): boolean {
  if (!element) {
    return false;
  }
  if (typeof element.getAttribute === 'function' && element.getAttribute('data-message-author-role') === 'user') {
    return true;
  }
  if (typeof element.closest === 'function') {
    return Boolean(element.closest('[data-message-author-role="user"]'));
  }
  return false;
}

/**
 * Checks if a DOM element represents an assistant message in ChatGPT's chat stream.
 */
export function isAssistantMessageElement(element: any): boolean {
  if (!element) {
    return false;
  }
  if (typeof element.getAttribute === 'function' && element.getAttribute('data-message-author-role') === 'assistant') {
    return true;
  }
  if (typeof element.closest === 'function') {
    return Boolean(element.closest('[data-message-author-role="assistant"]'));
  }
  return false;
}

/**
 * Extracts text from a user message turn container in the chat feed.
 * Strips out extraneous UI elements like edit buttons, copy buttons, etc.
 */
export function extractUserMessageText(container: any): string {
  if (!container) {
    return '';
  }

  const userTurn =
    typeof container.matches === 'function' && container.matches('[data-message-author-role="user"]')
      ? container
      : typeof container.closest === 'function'
      ? container.closest('[data-message-author-role="user"]')
      : container;

  if (!userTurn) {
    return '';
  }

  if (typeof userTurn.cloneNode === 'function') {
    const clone = userTurn.cloneNode(true);
    if (typeof clone.querySelectorAll === 'function') {
      const buttons = clone.querySelectorAll('button, svg, [role="button"], [aria-label]');
      buttons.forEach((b: any) => {
        if (typeof b.remove === 'function') {
          b.remove();
        }
      });
    }
    return (clone.textContent || '').trim();
  }

  return (userTurn.textContent || '').trim();
}

/**
 * Delivers a user message to the background service worker with retry and exponential backoff.
 * Gracefully handles Manifest V3 service worker dormancy/wake-up latency.
 */
export async function deliverMessageWithRetry(
  text: string,
  sendFn: (payload: { type: string; text: string }) => Promise<SaveMessageResponse | undefined>,
  maxRetries: number = 3,
  delays: number[] = [150, 400, 1000]
): Promise<SaveMessageResponse | null> {
  const trimmed = text.trim();
  if (!trimmed) {
    return null;
  }

  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const response = await sendFn({
        type: 'NEW_USER_MESSAGE',
        text: trimmed,
      });

      if (response) {
        return response;
      }

      // If response is undefined, background channel closed without response
      throw new Error('Could not establish connection. Receiving end did not respond.');
    } catch (error) {
      attempt++;
      const errorMsg = error instanceof Error ? error.message : String(error);
      const isTransient =
        errorMsg.includes('Could not establish connection') ||
        errorMsg.includes('Receiving end does not exist') ||
        errorMsg.includes('did not respond') ||
        errorMsg.includes('service worker');

      if (attempt <= maxRetries && isTransient) {
        const delay = delays[attempt - 1] ?? 1000;
        console.warn(
          `Echo: Background service worker dormant or reconnecting (attempt ${attempt}/${maxRetries}). Retrying in ${delay}ms...`
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
      } else {
        console.error('Echo: Failed to deliver message to background service worker:', error);
        return null;
      }
    }
  }

  return null;
}
