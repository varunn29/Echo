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

export {
  isObviousNonContent,
  isPotentialStyleSample,
};