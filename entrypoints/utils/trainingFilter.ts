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

  // Reject shell prompts (PowerShell, Windows, POSIX)
  if (
    /^(PS\s+[A-Za-z]:\\|[A-Za-z]:\\[^>]*>|PS>|[$#>]\s+|[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+:)/i.test(
      normalizedText
    )
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

function isEligibleTrainingSample(text: string): boolean {
  if (typeof text !== 'string' || !text.trim()) {
    return false;
  }

  if (!isPotentialStyleSample(text)) {
    return false;
  }

  const classification = classifyContent(text);

  // V2 Six-Tier Architecture:
  // ONLY Tier A (Authentic personal prose) and Tier B (Authentic technical prose) are eligible for style SFT.
  // Tier C (Agent directives), Tier D (Terminal/code), Tier E (External), and Tier F (Mixed/ambiguous) are excluded.
  return classification.tier === 'tier_a' || classification.tier === 'tier_b';
}

export {
  isObviousNonContent,
  isPotentialStyleSample,
  isEligibleTrainingSample,
};
