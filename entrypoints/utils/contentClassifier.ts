export interface ContentClassification {
  contentType:
    | 'prose'
    | 'code'
    | 'terminal'
    | 'structured_data'
    | 'short_fragment'
    | 'mixed';

  styleSignal: 'strong' | 'weak' | 'none';

  knowledgeSignal: 'possible' | 'none';

  wordCount: number;

  reasons: string[];
}

// Known command line binaries and tools
const CLI_COMMANDS = new Set([
  'npm',
  'npx',
  'pnpm',
  'yarn',
  'bun',
  'git',
  'docker',
  'docker-compose',
  'podman',
  'kubectl',
  'pip',
  'pip3',
  'python',
  'python3',
  'node',
  'deno',
  'cargo',
  'rustc',
  'go',
  'dotnet',
  'mvn',
  'gradle',
  'cd',
  'mkdir',
  'rmdir',
  'rm',
  'del',
  'dir',
  'ls',
  'pwd',
  'cp',
  'mv',
  'cat',
  'grep',
  'curl',
  'wget',
  'ssh',
  'ollama',
  'wsl',
  'brew',
  'apt',
  'apt-get',
  'yum',
  'sudo',
  'clear',
  'cls',
  'chmod',
  'chown',
  'tar',
  'unzip',
  'ps',
  'kill',
  'find',
  'sed',
  'awk',
]);

// Natural language verbs, auxiliary verbs, and prepositions that distinguish prose starting with a tool name
// from an actual terminal command invocation (e.g. "Python was..." vs "python main.py").
const NATURAL_FOLLOWING_WORDS = new Set([
  'is',
  'was',
  'are',
  'were',
  'will',
  'would',
  'can',
  'could',
  'should',
  'has',
  'have',
  'had',
  'makes',
  'made',
  'helps',
  'helped',
  'allows',
  'allowed',
  'provides',
  'provided',
  'seems',
  'seemed',
  'feels',
  'felt',
  'gives',
  'gave',
  'creates',
  'created',
  'stands',
  'becomes',
  'became',
  'offers',
  'offered',
  'works',
  'worked',
  'changed',
  'started',
  'my',
  'our',
  'a',
  'an',
  'the',
  'in',
  'for',
  'with',
  'on',
  'at',
  'by',
  'to',
  'as',
  'and',
  'or',
  'so',
  'because',
  'communication', // e.g. "Clear communication..."
  'thinking',
  'code',
  'explanation',
]);

// Short conversational phrases with minimal stylistic or informational content
const CONVERSATIONAL_FRAGMENTS = new Set([
  'ok',
  'okay',
  'k',
  'yes',
  'yeah',
  'yep',
  'no',
  'nope',
  'nah',
  'sure',
  'thanks',
  'thank you',
  'thx',
  'got it',
  'understood',
  'continue',
  'go on',
  'more',
  'try again',
  'retry',
  'regenerate',
  'explain',
  'why',
  'why?',
  'how?',
  'what?',
  'hello',
  'hi',
  'hey',
  'good morning',
  'good evening',
  'good afternoon',
  'help',
  'fix it',
  'fix this',
  'please continue',
]);

// Personal knowledge triggers organized across 6 distinct categories:
// 1. Projects and experiences
// 2. Learning interests and goals
// 3. Personal habits and workflows
// 4. Learning and communication preferences
// 5. Personal opinions and behavioral tendencies
// 6. Current goals, difficulties, and self-evaluations
const PERSONAL_EXPERIENCE_PATTERNS = [
  // 1. Projects and experiences
  /\b(I|we)\b[\w\s]{0,20}\b(started by|started with|started learning|been learning|learned|learning|studying|studied|picked up|working on|working with)\b/i,
  /\b(I|we)\b[\w\s]{0,20}\b(built|building|created|creating|developed|developing|designed|designing|implemented|implementing|deployed|deploying)\b/i,
  /\b(my|our)\s+(first\s+)?(project|app|application|platform|startup|product|website|repo|codebase|company|team|client|job|role|interview|career)\b/i,
  /\b(I|we)\s+(worked\s+at|worked\s+for|interned\s+at|interviewed\s+with|applied\s+to|applying\s+to)\b/i,

  // 2. Learning interests, ambitions, and forward goals
  /\b(my\s+plan\s+was|planning\s+to|I\s+have\s+not\s+yet\s+decided|I\s+decided\s+to|I\s+want\s+to\s+learn|I\s+wanted\s+to\s+build|I\s+like\s+programming)\b/i,
  /\b(I\s+am|I'm|planning\s+to|hope\s+to|interested\s+in|curious\s+about|looking\s+to|aiming\s+to)\s+(learn|explore|try|study|pick\s+up|master|transition\s+to|work\s+with)\b/i,
  /\b(my\s+goal\s+is|my\s+aim\s+is|my\s+target\s+is|my\s+plan\s+is|I\s+plan\s+to|there's\s+a\s+long\s+way\s+to\s+go)\b/i,

  // 3. Personal habits and workflows
  /\b(whenever\s+i(\s+am|'m)?|when\s+i(\s+am|'m)?)\s+(stuck|debugging|coding|building|working|writing|learning)\b/i,
  /\b(I\s+usually|I\s+always|I\s+typically|I\s+often|I\s+tend\s+to|my\s+workflow|my\s+habit|my\s+routine)\b/i,
  /\bI\s+(use|rely\s+on|prefer\s+using|reach\s+for|turn\s+to)\s+[\w\s]{1,30}\s+because\b/i,

  // 4. Learning and communication preferences
  /\bI\s+(will|prefer\s+to|like\s+to|tend\s+to)\s+(take\s+the\s+help\s+of|use|provide|give)\s+(examples|analogies|diagrams|visuals|code\s+snippets)\b/i,
  /\b(my\s+way\s+of\s+explaining|how\s+I\s+explain|how\s+I\s+teach|how\s+I\s+learn|how\s+they\s+can\s+visualize|that's\s+how\s+they\s+can\s+visualize)\b/i,
  /\b(I\s+learn\s+best\s+by|I\s+prefer\s+learning|the\s+way\s+I\s+understand|helps\s+me\s+understand|helps\s+me\s+visualize)\b/i,

  // 5. Personal opinions and behavioral tendencies
  /\b(frustrates\s+me|I\s+just\s+hate|I\s+hate|I\s+really\s+like|I\s+love|I\s+prefer|I\s+value|in\s+my\s+opinion|from\s+my\s+experience|I\s+strongly\s+believe|my\s+take\s+on\s+this\s+is)\b/i,
  /\b(if\s+I\s+have\s+enough\s+knowledge|if\s+I\s+know\s+enough|when\s+I\s+disagree|in\s+disagreements|I\s+will\s+argue|I\s+will\s+put\s+my\s+point|listen\s+to\s+theirs)\b/i,
  /\b(my\s+approach\s+is|my\s+tendency\s+is|my\s+philosophy\s+is|how\s+I\s+approach)\b/i,

  // 6. Current goals, difficulties, and self-evaluations
  /\b(main\s+difficulty|biggest\s+difficulty|biggest\s+challenge|problem\s+I\s+faced|struggled\s+with|difficulty\s+which\s+I\s+faced)\b/i,
  /\b(it's\s+not\s+(really\s+)?simple\s+to\s+come\s+up\s+with|hard\s+for\s+me\s+to|I'm\s+struggling\s+to|I\s+am\s+struggling\s+to)\b/i,
  /\b(I\s+have\s+thought\s+too\s+much\s+about|I\s+don't\s+know\s+about\s+any\s+such|I\s+am\s+not\s+satisfied|I'm\s+not\s+satisfied|I\s+feel\s+I\s+am\s+slow|I\s+feel\s+slow|people\s+my\s+age)\b/i,
  /\b(come\s+up\s+with\s+a\s+product\s+idea|looking\s+for\s+a\s+project\s+idea|thinking\s+about\s+what\s+to\s+build|trying\s+to\s+figure\s+out\s+what)\b/i,
];

function isPureJson(text: string): boolean {
  const trimmed = text.trim();
  if (
    !(trimmed.startsWith('{') && trimmed.endsWith('}')) &&
    !(trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    return false;
  }

  try {
    const parsed = JSON.parse(trimmed);
    return typeof parsed === 'object' && parsed !== null;
  } catch {
    return false;
  }
}

function isXmlOrHtml(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.startsWith('<?xml') || trimmed.startsWith('<!DOCTYPE')) {
    return true;
  }

  // Self-contained tag block: <tag>...</tag> or <tag/>
  const htmlTagMatch = /^<([a-zA-Z0-9_-]+)(\s+[^>]*)?>[\s\S]*<\/\1>$/i.test(trimmed);
  const singleTagMatch = /^<([a-zA-Z0-9_-]+)(\s+[^>]*)?\s*\/>$/i.test(trimmed);
  return htmlTagMatch || singleTagMatch;
}

function isYamlOrConfig(text: string): boolean {
  const lines = text.trim().split('\n').map((line) => line.trim()).filter(Boolean);
  if (lines.length < 3) {
    return false;
  }

  const yamlKeyCount = lines.filter((line) =>
    /^[a-zA-Z0-9_.-]+:\s*(\S.*)?$/.test(line)
  ).length;

  return yamlKeyCount / lines.length >= 0.7;
}

function isStackTraceOrCompilerLog(text: string): boolean {
  const stackTraceSignatures = [
    /Traceback \(most recent call last\):/i,
    /^\s*at\s+[\w$.<>]+\s+\([^)]+:\d+:\d+\)/m,
    /^\s*at\s+async\s+[\w$.<>]+\s+\([^)]+:\d+:\d+\)/m,
    /error TS\d+:\s+/i,
    /npm ERR!\s+/i,
    /SyntaxError:\s+/i,
    /TypeError:\s+/i,
    /ReferenceError:\s+/i,
    /UnhandledPromiseRejection/i,
    /At line:\d+\s+char:\d+/i,
    /\+\s+CategoryInfo\s+:/i,
    /\+\s+FullyQualifiedErrorId\s+:/i,
    /fatal:\s+unable to access/i,
  ];

  return stackTraceSignatures.some((regex) => regex.test(text));
}

function isTerminalCommand(text: string): boolean {
  const trimmed = text.trim();
  const firstLine = trimmed.split('\n')[0]?.trim() || '';

  // PowerShell Cmdlets: Get-Process, Set-ExecutionPolicy, etc.
  if (/^(Get|Set|New|Remove|Start|Stop|Invoke|Test|Enable|Disable)-[a-zA-Z]+\b/.test(firstLine)) {
    return true;
  }

  // Shell prompts: $ git status, > npm install
  if (/^[$>]\s+[a-z0-9_-]+/i.test(firstLine)) {
    return true;
  }

  const tokens = firstLine.split(/\s+/);
  const firstToken = tokens[0]?.toLowerCase().replace(/[\\/].*[\\/]/, '') || '';

  if (CLI_COMMANDS.has(firstToken)) {
    // If followed by natural language verb/preposition, it is prose (e.g. "Python was...", "Git is...")
    const secondToken = tokens[1]?.toLowerCase() || '';
    if (secondToken && NATURAL_FOLLOWING_WORDS.has(secondToken)) {
      return false;
    }

    // Common terminal subcommand or flags (e.g., "npm install", "git status", "docker run -d")
    if (
      tokens.length === 1 ||
      tokens.some((token) => token.startsWith('-')) ||
      /^(install|i|add|run|build|test|commit|push|pull|status|checkout|branch|clone|init|diff|merge|rebase|config|exec|start|stop|restart|logs|ps|version)\b/i.test(
        secondToken
      )
    ) {
      return true;
    }

    // Single line without terminal sentence punctuation
    if (!firstLine.endsWith('.') && !firstLine.endsWith('?') && tokens.length <= 6) {
      return true;
    }
  }

  return false;
}

function isUnfencedCode(text: string): boolean {
  const trimmed = text.trim();
  const lines = trimmed.split('\n');

  // SQL Statements
  if (
    /^\s*(SELECT|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b/i.test(
      trimmed
    )
  ) {
    return true;
  }

  // Function / Class / Interface definitions
  const codeDefinitionSignatures = [
    /\bfunction\s*\w*\s*\([^)]*\)\s*\{/,
    /\bconst\s+\w+\s*=\s*(async\s*)?\([^)]*\)\s*=>/,
    /\bdef\s+\w+\s*\([^)]*\)\s*:/,
    /\b(public|private|protected)\s+(static\s+)?(async\s+)?(\w+|<.*>)\s+\w+\s*\([^)]*\)/,
    /\bclass\s+\w+(\s+extends\s+\w+)?\s*\{/,
    /\b(interface|type)\s+[A-Z]\w*\s*(=|\{)/,
    /\bimport\s+.*\s+from\s+['"][^'"]+['"]/,
    /\bconst\s+.*\s*=\s*require\(['"][^'"]+['"]\)/,
    /\bexport\s+(default\s+)?(function|const|class|interface|type)\b/,
  ];

  const hasDefinition = codeDefinitionSignatures.some((sig) => sig.test(trimmed));
  if (hasDefinition) {
    return true;
  }

  // Multiple lines ending in code syntax (semicolons, braces)
  let codeEndingLines = 0;
  for (const line of lines) {
    const l = line.trim();
    if (
      l.endsWith(';') ||
      l.endsWith('{') ||
      l.endsWith('}') ||
      l.endsWith(');') ||
      l.endsWith('},')
    ) {
      codeEndingLines++;
    }
  }

  if (lines.length >= 2 && codeEndingLines / lines.length >= 0.5) {
    return true;
  }

  return false;
}

function hasMixedProseAndCode(text: string): {
  isMixed: boolean;
  proseWords: number;
} {
  const trimmed = text.trim();

  // Check for fenced code block with surrounding text
  const fencedRegex = /```[\s\S]*?```/g;
  const matches = trimmed.match(fencedRegex);

  if (matches) {
    const textWithoutFences = trimmed.replace(fencedRegex, ' ').trim();
    const words = textWithoutFences.split(/\s+/).filter(Boolean);
    if (words.length >= 5) {
      return { isMixed: true, proseWords: words.length };
    }
  }

  // Check for unfenced code embedded in explanatory prose
  const lines = trimmed.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length >= 3) {
    let proseWordsCount = 0;
    let hasCodeLine = false;

    for (const line of lines) {
      if (
        isUnfencedCode(line) ||
        isTerminalCommand(line) ||
        isStackTraceOrCompilerLog(line)
      ) {
        hasCodeLine = true;
      } else {
        const words = line.split(/\s+/).filter(Boolean);
        proseWordsCount += words.length;
      }
    }

    if (hasCodeLine && proseWordsCount >= 10) {
      return { isMixed: true, proseWords: proseWordsCount };
    }
  }

  return { isMixed: false, proseWords: 0 };
}

const PROCEDURAL_QUESTION_PATTERNS = [
  /\bwhen\s+(are|do|will|should|can|could|shall)\s+(we|you|i)\s+[\w\s]{0,25}\b(start|building|build|do|begin|proceed)\b/i,
  /\b(so\s+)?will\s+(we|you|i)\s+(have\s+to|need\s+to|be\s+able\s+to)\b/i,
  /\b(should|can|could|do|did|would|shall|is|are)\s+(we|you)\s+(have\s+to|need\s+to|start|do|use|build|proceed)\b/i,
  /\b(what|where|how)\s+(are|do|will|should|can|could)\s+(we|you)\s+(do|go|start|build|implement|approach)\b/i,
  /\b(can\s+you|could\s+you|please\s+(explain|show|help|tell|clarify)|tell\s+me|show\s+me|help\s+me\s+with)\b/i,
  /\bok\s+then\s+we\s+can\s+go\s+with\b/i,
];

export const DECLARATIVE_PERSONAL_STATEMENTS = [
  /\b(recently\s+|so\s+)?I\s+(built|created|developed|designed|implemented|deployed|worked|interned|applied)\b/i,
  /\bI\s+(started\s+by\s+learning|learned|studied|picked\s+up|been\s+learning)\b/i,
  /\bI\s+(usually|always|typically|often|prefer\s+using|rely\s+on)\b/i,
  /\b(frustrates\s+me|I\s+hate|I\s+love|I\s+value|in\s+my\s+opinion|my\s+plan\s+was)\b/i,
  /\bI\s+(am\s+not\s+satisfied|feel\s+I\s+am\s+slow|have\s+thought\s+too\s+much)\b/i,
  /\bif\s+I\s+have\s+enough\s+knowledge\b/i,
  /\bwhenever\s+i(\s+am|'m)?\s+stuck\b/i,
  /\b(the\s+)?(main|biggest|primary)\s+(difficulty|challenge|problem)\s+(which\s+)?I\s+(faced|encountered|had)\b/i,
  /\b(problem|difficulty|challenge)\s+(which\s+)?I\s+(faced|encountered|had)\b/i,
  /\bI\s+(struggled\s+with|faced\s+difficulty|had\s+to\s+work\s+around|worked\s+around|experienced)\b/i,
  /\bI\s+(couldn't|could\s+not|was\s+not\s+able\s+to|cannot|can't)\s+(get|find|use|access)\b/i,
  /\b(so\s+)?I\s+(had\s+to\s+build|built|have\s+to\s+build)\b/i,
];

function isProceduralQuestionOnly(text: string): boolean {
  const matchesProcedural = PROCEDURAL_QUESTION_PATTERNS.some((pattern) => pattern.test(text));
  if (!matchesProcedural) {
    return false;
  }
  const hasDeclarativeFact = DECLARATIVE_PERSONAL_STATEMENTS.some((pattern) => pattern.test(text));
  return !hasDeclarativeFact;
}

function hasPersonalKnowledgeSignal(text: string): boolean {
  if (isProceduralQuestionOnly(text)) {
    return false;
  }
  return PERSONAL_EXPERIENCE_PATTERNS.some((pattern) => pattern.test(text));
}

/**
 * Classifies a captured writing message to evaluate content type,
 * writing-style signal, and personal knowledge potential without modifying data.
 */
export function classifyContent(text: string): ContentClassification {
  const normalizedText = text.trim();
  const wordCount = normalizedText ? normalizedText.split(/\s+/).length : 0;
  const reasons: string[] = [];

  // 1. Empty / Whitespace-only
  if (!normalizedText || wordCount === 0) {
    return {
      contentType: 'short_fragment',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount: 0,
      reasons: ['empty_message'],
    };
  }

  // 2. Structured data (JSON, XML/HTML, YAML)
  if (isPureJson(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['json_payload'],
    };
  }

  if (isXmlOrHtml(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['xml_or_html_markup'],
    };
  }

  if (isYamlOrConfig(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['yaml_or_configuration_data'],
    };
  }

  // 3. Stack traces & compiler error logs
  if (isStackTraceOrCompilerLog(normalizedText)) {
    return {
      contentType: 'terminal',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['stack_trace_or_compiler_output'],
    };
  }

  // 4. Entirely fenced code block (``` ... ```)
  if (/^```[\s\S]*```$/.test(normalizedText)) {
    return {
      contentType: 'code',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['fenced_code_block'],
    };
  }

  // 5. Mixed prose and code / commands
  const mixedCheck = hasMixedProseAndCode(normalizedText);
  if (mixedCheck.isMixed) {
    reasons.push('mixed_prose_and_code');
    const isKnowledge = hasPersonalKnowledgeSignal(normalizedText);
    if (isKnowledge) {
      reasons.push('personal_context_detected');
    }

    const styleSignal =
      mixedCheck.proseWords >= 14 ? 'strong' : mixedCheck.proseWords >= 6 ? 'weak' : 'none';

    return {
      contentType: 'mixed',
      styleSignal,
      knowledgeSignal: isKnowledge ? 'possible' : 'none',
      wordCount,
      reasons,
    };
  }

  // 6. Unfenced source code
  if (isUnfencedCode(normalizedText)) {
    return {
      contentType: 'code',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['unfenced_source_code'],
    };
  }

  // 7. Terminal command
  if (isTerminalCommand(normalizedText)) {
    return {
      contentType: 'terminal',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['terminal_command'],
    };
  }

  // 8. Short conversational fragments (evaluated after code, terminal, and structured data)
  const lowerText = normalizedText.toLowerCase().replace(/[.!?,]$/, '');
  if (
    CONVERSATIONAL_FRAGMENTS.has(lowerText) ||
    (wordCount <= 3 && !/[.!?]$/.test(normalizedText) && !hasPersonalKnowledgeSignal(normalizedText))
  ) {
    if (!hasPersonalKnowledgeSignal(normalizedText)) {
      return {
        contentType: 'short_fragment',
        styleSignal: 'none',
        knowledgeSignal: 'none',
        wordCount,
        reasons: ['conversational_fragment'],
      };
    }
  }

  // 9. Natural language prose
  reasons.push('natural_language_prose');
  const isPersonalKnowledge = hasPersonalKnowledgeSignal(normalizedText);
  if (isPersonalKnowledge) {
    reasons.push('personal_context_detected');
  }

  // Style signal evaluation:
  // 14+ words of prose provides solid sentence flow/cadence; 6-13 words with personal/emotional context also provides style signal.
  let styleSignal: 'strong' | 'weak' | 'none';
  if (wordCount >= 14) {
    styleSignal = 'strong';
  } else if (wordCount >= 6) {
    styleSignal = isPersonalKnowledge ? 'strong' : 'weak';
  } else {
    styleSignal = 'weak';
  }

  return {
    contentType: 'prose',
    styleSignal,
    knowledgeSignal: isPersonalKnowledge ? 'possible' : 'none',
    wordCount,
    reasons,
  };
}
