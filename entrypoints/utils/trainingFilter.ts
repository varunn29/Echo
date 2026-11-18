import { classifyContent } from './contentClassifier';

function isObviousNonContent(text: string): boolean {
  const normalizedText = text.trim();

  if (!normalizedText) {
    return true;
  }

  // Reject messages consisting entirely of a Markdown code block.
  if (/^```[\s\S]*```$/.test(normalizedText)) {
    return true;
  }

  // Reject shell prompts (PowerShell, Windows, POSIX, Virtualenv) across lines
  if (
    /(^|\n)\s*(\([a-zA-Z0-9_.-]+\)\s+)?(P?S\s+[A-Za-z]:\\|[A-Za-z]:\\[^>]*>|PS>|[$#>]\s+|[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+:)/im.test(
      normalizedText
    ) ||
    /\bPS\s+[A-Za-z]:\\/i.test(normalizedText) ||
    /\bGet-Process\b/i.test(normalizedText) ||
    /Windows PowerShell.*Copyright \(C\) Microsoft Corporation/i.test(normalizedText)
  ) {
    return true;
  }

  // Reject obvious terminal commands.
  const terminalCommandPattern =
    /^(npm|npx|pnpm|yarn|bun|node|python|python3|pip|pip3|git|docker|cd|mkdir|rmdir|del|dir|ls|pwd|clear|cls|ollama|wsl|cargo|go)\b/i;

  if (terminalCommandPattern.test(normalizedText)) {
    return true;
  }

  return false;
}

function isPotentialStyleSample(text: string): boolean {
  return !isObviousNonContent(text);
}

/**
 * Canonical eligibility predicate for Echo personal writing style and model training.
 *
 * A message is eligible if and only if:
 * 1. It is a non-empty string.
 * 2. It is not obvious non-content (code blocks, shell prompts, CLI commands).
 * 3. It is classified as Tier A (Authentic Personal Prose) or Tier B (Authentic Technical Prose).
 * 4. Its content type is 'prose' (specifically excluding code, terminal, structured data, short fragments, and mixed).
 * 5. Its style signal is 'strong'.
 */
function isEligibleStyleSample(text: string): boolean {
  if (typeof text !== 'string' || !text.trim()) {
    return false;
  }

  if (!isPotentialStyleSample(text)) {
    return false;
  }

  const classification = classifyContent(text);

  const isEligibleTier =
    classification.tier === 'tier_a' || classification.tier === 'tier_b';
  const isStrongProse =
    classification.contentType === 'prose' && classification.styleSignal === 'strong';

  return isEligibleTier && isStrongProse;
}

/**
 * Canonical alias for training sample eligibility — identical to isEligibleStyleSample.
 */
function isEligibleTrainingSample(text: string): boolean {
  return isEligibleStyleSample(text);
}

export {
  isObviousNonContent,
  isPotentialStyleSample,
  isEligibleStyleSample,
  isEligibleTrainingSample,
};
