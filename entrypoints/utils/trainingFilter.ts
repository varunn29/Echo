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

  // Reject obvious terminal commands.
  const terminalCommandPattern =
    /^(npm|npx|node|python|python3|pip|pip3|git|docker|cd|mkdir|rmdir|del|dir|ls|pwd|clear|cls|ollama|wsl)\b/i;

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

  // Exclude code, terminal output, structured data, and short fragments
  if (
    classification.contentType === 'code' ||
    classification.contentType === 'terminal' ||
    classification.contentType === 'structured_data' ||
    classification.contentType === 'short_fragment'
  ) {
    return false;
  }

  // Exclude weak or no-style samples according to existing rules
  if (classification.styleSignal !== 'strong') {
    return false;
  }

  return true;
}

export {
  isObviousNonContent,
  isPotentialStyleSample,
  isEligibleTrainingSample,
};
