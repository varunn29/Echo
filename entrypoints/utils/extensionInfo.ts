/**
 * Echo — Extension Runtime Information & Service Setup Utilities
 *
 * Resolves the dynamic extension ID at runtime and formats the exact
 * startup command needed to authorize the extension with the local
 * Python LoRA generation service.
 */

/**
 * Returns the current runtime extension ID across Chrome and Firefox MV3 contexts.
 * Falls back to an empty string in non-extension (e.g. test or server) environments.
 */
export function getExtensionId(): string {
  try {
    // 1. WXT / WebExtension browser API
    if (typeof browser !== 'undefined' && browser.runtime?.id) {
      return browser.runtime.id;
    }
  } catch {
    // Ignore cross-context / environment errors
  }

  try {
    // 2. Standard Chromium chrome.runtime API
    const globalChrome = (globalThis as unknown as { chrome?: { runtime?: { id?: string } } }).chrome;
    if (globalChrome?.runtime?.id) {
      return globalChrome.runtime.id;
    }
  } catch {
    // Ignore cross-context / environment errors
  }

  return '';
}

/**
 * Formats the full terminal command required to start the local Python generation service
 * with CORS authorization for the user's specific extension origin.
 *
 * @param extensionId Optional explicit extension ID. If omitted, resolved dynamically.
 * @returns The command line string.
 */
export function buildServiceStartupCommand(extensionId?: string): string {
  const resolvedId = (extensionId !== undefined ? extensionId : getExtensionId()).trim();
  const idPlaceholder = resolvedId || '<your-extension-id>';
  return `python training/generation_service.py --allowed-origin "chrome-extension://${idPlaceholder}"`;
}

/**
 * Returns the fully qualified extension origin URL string.
 *
 * @param extensionId Optional explicit extension ID. If omitted, resolved dynamically.
 */
export function getExtensionOrigin(extensionId?: string): string {
  const resolvedId = (extensionId !== undefined ? extensionId : getExtensionId()).trim();
  const idPlaceholder = resolvedId || '<your-extension-id>';
  return `chrome-extension://${idPlaceholder}`;
}

/**
 * Utility to copy text to the system clipboard with browser fallback.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (!text) return false;

  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall back to DOM execCommand
  }

  try {
    if (typeof document !== 'undefined') {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-9999px';
      textArea.style.top = '-9999px';
      textArea.setAttribute('aria-hidden', 'true');
      document.body.appendChild(textArea);
      textArea.select();
      const success = document.execCommand('copy');
      document.body.removeChild(textArea);
      return success;
    }
  } catch {
    return false;
  }

  return false;
}
