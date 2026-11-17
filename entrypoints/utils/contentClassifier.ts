export type ClassificationTier =
  | 'tier_a' // Authentic personal prose — eligible
  | 'tier_b' // Authentic technical prose — eligible
  | 'tier_c' // AI-agent instructions and prompt templates — segregated
  | 'tier_d' // Terminal sessions, build logs and raw code — excluded
  | 'tier_e' // External and third-party content — excluded
  | 'tier_f'; // Mixed or ambiguous content — manual review

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

  // Classifier V2 Architecture enhancements
  tier: ClassificationTier;
  tierReason: string;
  reasonCodes: string[];
}

export const CLASSIFICATION_REASON_CODES = {
  // Tier A: Authentic Personal Prose
  PERSONAL_EXPERIENCE_REFLECTION: 'personal_experience_reflection',
  PERSONAL_WORKFLOW_HABIT: 'personal_workflow_habit',
  PERSONAL_OPINION_OR_VALUE: 'personal_opinion_or_value',
  PERSONAL_LEARNING_JOURNEY: 'personal_learning_journey',

  // Tier B: Authentic Technical Prose
  AUTHENTIC_TECHNICAL_EXPOSITION: 'authentic_technical_exposition',
  NATURAL_CONVERSATIONAL_PROSE: 'natural_conversational_prose',

  // Tier C: AI-Agent Directives & Prompt Templates
  AGENT_DIRECTIVE_HEADING: 'agent_directive_heading',
  AGENT_DIRECTIVE_COMMAND: 'agent_directive_command',
  AGENT_STRUCTURED_TEMPLATE: 'agent_structured_template',

  // Tier D: Terminal, Logs, Code, Data, Fragments
  TERMINAL_POWERSHELL_PROMPT: 'terminal_powershell_prompt',
  TERMINAL_WINDOWS_CMD_PROMPT: 'terminal_windows_cmd_prompt',
  TERMINAL_POSIX_PROMPT: 'terminal_posix_prompt',
  TERMINAL_BUILD_LOG: 'terminal_build_log',
  TERMINAL_TEST_RUNNER_OUTPUT: 'terminal_test_runner_output',
  TERMINAL_PACKAGE_MANAGER_OUTPUT: 'terminal_package_manager_output',
  TERMINAL_STACK_TRACE: 'terminal_stack_trace',
  TERMINAL_COMMAND_INVOCATION: 'terminal_command_invocation',
  CODE_FENCED_BLOCK: 'code_fenced_block',
  CODE_UNFENCED_DEFINITION: 'code_unfenced_definition',
  STRUCTURED_DATA_JSON: 'structured_data_json',
  STRUCTURED_DATA_MARKUP: 'structured_data_markup',
  STRUCTURED_DATA_YAML: 'structured_data_yaml',
  EMPTY_OR_WHITESPACE: 'empty_or_whitespace',
  SHORT_FRAGMENT: 'short_fragment',

  // Tier E: External & Third-Party Content
  EXTERNAL_VIDEO_SCRIPT: 'external_video_script',
  EXTERNAL_SOCIAL_TAGS: 'external_social_tags',
  EXTERNAL_SCRAPED_DASHBOARD: 'external_scraped_dashboard',

  // Tier F: Mixed or Ambiguous Content
  MIXED_PROSE_AND_CODE: 'mixed_prose_and_code',
  MIXED_PROSE_AND_TERMINAL: 'mixed_prose_and_terminal',
  CONVERSATIONAL_QUERY_NOT_DRAFTING_TARGET: 'conversational_query_not_drafting_target',
  AMBIGUOUS_STYLE_SIGNAL: 'ambiguous_style_signal',
} as const;

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
  'branching',
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

export const PERSONAL_EXPERIENCE_PATTERNS = [
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

  // 5. Personal opinions, values and behavioral tendencies
  /\b(frustrates\s+me|I\s+just\s+hate|I\s+hate|I\s+really\s+like|I\s+love|I\s+prefer|I\s+value|in\s+my\s+opinion|from\s+my\s+experience|I\s+strongly\s+believe|my\s+take\s+on\s+this\s+is|I\s+believe\s+good\s+software|I\s+enjoy\s+building)\b/i,
  /\b(if\s+I\s+have\s+enough\s+knowledge|if\s+I\s+know\s+enough|when\s+I\s+disagree|in\s+disagreements|I\s+will\s+argue|I\s+will\s+put\s+my\s+point|listen\s+to\s+theirs)\b/i,
  /\b(my\s+approach\s+is|my\s+tendency\s+is|my\s+philosophy\s+is|how\s+I\s+approach)\b/i,

  // 6. Current goals, difficulties, and self-evaluations
  /\b(main\s+difficulty|biggest\s+difficulty|biggest\s+challenge|problem\s+I\s+faced|struggled\s+with|difficulty\s+which\s+I\s+faced)\b/i,
  /\b(it's\s+not\s+(really\s+)?simple\s+to\s+come\s+up\s+with|hard\s+for\s+me\s+to|I'm\s+struggling\s+to|I\s+am\s+struggling\s+to)\b/i,
  /\b(I\s+have\s+thought\s+too\s+much\s+about|I\s+don't\s+know\s+about\s+any\s+such|I\s+am\s+not\s+satisfied|I'm\s+not\s+satisfied|I\s+feel\s+I\s+am\s+slow|I\s+feel\s+slow|people\s+my\s+age)\b/i,
  /\b(come\s+up\s+with\s+a\s+product\s+idea|looking\s+for\s+a\s+project\s+idea|thinking\s+about\s+what\s+to\s+build|trying\s+to\s+figure\s+out\s+what)\b/i,
];

export const DECLARATIVE_PERSONAL_STATEMENTS = [
  /\b(recently\s+|so\s+)?I\s+(built|created|developed|designed|implemented|deployed|worked|interned|applied)\b/i,
  /\bI\s+(started\s+by\s+learning|learned|studied|picked\s+up|been\s+learning)\b/i,
  /\bI\s+(usually|always|typically|often|prefer\s+using|rely\s+on)\b/i,
  /\b(frustrates\s+me|I\s+hate|I\s+love|I\s+value|in\s+my\s+opinion|my\s+plan\s+was|I\s+enjoy\s+building|I\s+prefer\s+software)\b/i,
  /\bI\s+(am\s+not\s+satisfied|feel\s+I\s+am\s+slow|have\s+thought\s+too\s+much)\b/i,
  /\bif\s+I\s+have\s+enough\s+knowledge\b/i,
  /\bwhenever\s+i(\s+am|'m)?\s+stuck\b/i,
  /\b(the\s+)?(main|biggest|primary)\s+(difficulty|challenge|problem)\s+(which\s+)?I\s+(faced|encountered|had)\b/i,
  /\b(problem|difficulty|challenge)\s+(which\s+)?I\s+(faced|encountered|had)\b/i,
  /\bI\s+(struggled\s+with|faced\s+difficulty|had\s+to\s+work\s+around|worked\s+around|experienced)\b/i,
  /\bI\s+(couldn't|could\s+not|was\s+not\s+able\s+to|cannot|can't)\s+(get|find|use|access)\b/i,
  /\b(so\s+)?I\s+(had\s+to\s+build|built|have\s+to\s+build)\b/i,
];

// Patterns indicative of authentic technical discussion in user voice
const TECHNICAL_PROSE_PATTERNS = [
  /\b(architecture|database|adapter|model|service|token|worker|offscreen|IndexedDB|LevelDB|process|pipeline|transaction|component)\b/i,
  /\b(we\s+will\s+write\s+tests|run\s+the\s+test\s+command|code\s+will\s+run\s+against|check\s+whether\s+it\s+works|breaks\s+the\s+functionality)\b/i,
  /\b(asynchronous\s+event-driven|connection\s+pooling|dependency\s+injection|type\s+alias|state\s+management|client-side|server-side)\b/i,
  /\b(git\s+branching\s+strategies|deployment\s+pipeline|re-renders|microtask\s+queue|concurrency|local-first)\b/i,
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

/**
 * Detects cross-platform shell prompts, PowerShell cmdlets, and terminal command invocations across lines.
 */
function detectTerminalPromptOrCommand(text: string): { isTerminal: boolean; reasonCode?: string } {
  const trimmed = text.trim();
  const lines = trimmed.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return { isTerminal: false };

  // 1. Multi-platform Shell Prompts (Windows PowerShell, CMD, POSIX, Virtualenv)
  // Checked on first line or any prompt-like line
  const psPromptRegex = /^(PS\s+[A-Za-z]:\\|PS>)/i;
  const cmdPromptRegex = /^[A-Za-z]:\\[^>]*>/i;
  const posixPromptRegex = /^([$>]\s+|[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+:.*[$#]\s+|\([a-zA-Z0-9_.-]+\)\s+[a-zA-Z0-9_.-]+@|#\s+(apt|yum|systemctl|docker|npm|pnpm|yarn|git|cd|ls|rm|chmod|chown)\b)/i;

  for (const line of lines) {
    if (psPromptRegex.test(line)) {
      return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_POWERSHELL_PROMPT };
    }
    if (cmdPromptRegex.test(line)) {
      return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_WINDOWS_CMD_PROMPT };
    }
    if (posixPromptRegex.test(line)) {
      return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_POSIX_PROMPT };
    }
  }

  // 2. PowerShell Cmdlets (Get-Process, Select-Object, Where-Object, etc.)
  const firstLine = lines[0] || '';
  if (/^(Get|Set|New|Remove|Start|Stop|Invoke|Test|Enable|Disable|Select|Where|ForEach|Measure|Sort|Group)-[a-zA-Z]+\b/i.test(firstLine)) {
    return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_POWERSHELL_PROMPT };
  }

  // 3. PowerShell continuation prompt (>>)
  if (/^\s*>>\s+/m.test(trimmed)) {
    return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_POWERSHELL_PROMPT };
  }

  // 4. CLI tool commands on first line (e.g. npm install, git status)
  const tokens = firstLine.split(/\s+/);
  const firstToken = tokens[0]?.toLowerCase().replace(/[\\/].*[\\/]/, '') || '';

  if (CLI_COMMANDS.has(firstToken)) {
    const secondToken = tokens[1]?.toLowerCase() || '';
    if (secondToken && NATURAL_FOLLOWING_WORDS.has(secondToken)) {
      return { isTerminal: false };
    }

    if (
      tokens.length === 1 ||
      tokens.some((token) => token.startsWith('-')) ||
      /^(install|i|add|run|build|test|commit|push|pull|status|checkout|branch|clone|init|diff|merge|rebase|config|exec|start|stop|restart|logs|ps|version)\b/i.test(
        secondToken
      )
    ) {
      return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_COMMAND_INVOCATION };
    }

    if (!firstLine.endsWith('.') && !firstLine.endsWith('?') && tokens.length <= 6) {
      return { isTerminal: true, reasonCode: CLASSIFICATION_REASON_CODES.TERMINAL_COMMAND_INVOCATION };
    }
  }

  return { isTerminal: false };
}

/**
 * Detects successful/informational build logs, package manager output, and test runner summaries.
 */
function detectBuildOrPackageLog(text: string): { isBuildLog: boolean; reasonCode?: string } {
  const buildAndPackageSignatures: Array<{ pattern: RegExp; code: string }> = [
    // Package managers (npm, pnpm, yarn)
    { pattern: /\badded\s+\d+\s+packages(\s+in\s+\d+s)?/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },
    { pattern: /\bpackages\s+are\s+looking\s+for\s+funding\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },
    { pattern: /run\s+`npm\s+fund`\s+for\s+details/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },
    { pattern: /\bup\s+to\s+date\s+in\s+\d+/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },
    { pattern: /\bfound\s+\d+\s+vulnerabilities\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },
    { pattern: /\baudited\s+\d+\s+packages\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_PACKAGE_MANAGER_OUTPUT },

    // Build tools (WXT, Vite, Webpack)
    { pattern: /\bBuilt\s+extension\s+in\s+\d+/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /\bBuilding\s+chrome-mv3\s+for\s+production\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /\b(WXT|vite|webpack)\s+v?\d+\.\d+/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /\bBuild(ing)?\s+completed\s+in\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /✓\s+\.output\//i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /Σ\s+Total\s+size:\s+[\d.]+\s+[kKmMgG]?B/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /[├└│─]\s+\.output[\\\/]/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },
    { pattern: /[├└│─]\s+[\w.-]+\.(json|html|js|css|png|svg)\s+[\d.]+\s+[kKmMgG]?B/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG },

    // Test runners (Vitest, Jest)
    { pattern: /✓\s+entrypoints\/.*tests?\s+\(\d+\)/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
    { pattern: /✓\s+[\w/.-]+\.test\.ts\s+\(\d+\s+tests\)/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
    { pattern: /PASS\s+Waiting\s+for\s+file\s+changes/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
    { pattern: /\bTest\s+Files\s+\d+\s+passed\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
    { pattern: /\bTests\s+\d+\s+passed\b/i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
    { pattern: /DEV\s+v\d+\.\d+\.\d+\s+[A-Za-z]:\//i, code: CLASSIFICATION_REASON_CODES.TERMINAL_TEST_RUNNER_OUTPUT },
  ];

  for (const sig of buildAndPackageSignatures) {
    if (sig.pattern.test(text)) {
      return { isBuildLog: true, reasonCode: sig.code };
    }
  }

  return { isBuildLog: false };
}

/**
 * Detects AI-agent directives, prompt engineering scaffolding, and audit instructions (Tier C).
 */
function detectAgentDirectiveOrTemplate(text: string): { isAgentPrompt: boolean; reasonCode?: string } {
  const trimmed = text.trim();

  // 1. Prominent Project / Agent Headings (e.g. "# Echo — Final Collection Resilience Review")
  if (/^#+\s*(Echo\s*[\u2014\-]|(Task|Objective|Audit|Review|Implementation|Instructions|Requirements|Preflight|Read-Only)\b)/im.test(trimmed)) {
    return { isAgentPrompt: true, reasonCode: CLASSIFICATION_REASON_CODES.AGENT_DIRECTIVE_HEADING };
  }

  // 2. Directives commanding an AI assistant
  const commandPatterns = [
    /^(Perform\s+a\s+(strictly\s+)?(read-only|READ-ONLY|FINAL)\s+(pre-commit\s+)?(review|audit)|Prepare\s+the\s+approved\s+Echo\s+changes|Push\s+the\s+existing\s+commit|Phase\s+\d+\s+Step\s+\d+|Read-Only\s+Audit:|Review\s+the\s+following|Audit\s+the\s+implementation|Inspect\s+the\s+codebase|Verify\s+the\s+following\s+requirements|Please\s+verify\s+all|Check\s+whether\s+the\s+following)/im,
    /\b(Strict\s+safety\s+restrictions|Required\s+output|Zero\s+training\s+steps\s+must\s+be\s+executed|Wait\s+for\s+explicit\s+approval|READ-ONLY\s+investigation\s+only|Do\s+not\s+modify\s+any\s+files|Do\s+not\s+execute\s+training|Target\s+configuration|Pre-Implementation\s+Safety\s+Audit)\b/i,
    /^(Target|Task|Objective|Instructions|Requirements|Deliverables|Context|Verification|Pre-Implementation\s+Safety\s+Audit):\s*$/im,
    /^I\s+approve\s+Path\s+\d+:\s+/im,
  ];

  for (const cp of commandPatterns) {
    if (cp.test(trimmed)) {
      return { isAgentPrompt: true, reasonCode: CLASSIFICATION_REASON_CODES.AGENT_DIRECTIVE_COMMAND };
    }
  }

  // 3. Multi-heading task scaffolding (## 1. ... ## 2. ...)
  if (/(^|\n)##\s+1\.\s+[\s\S]*(^|\n)##\s+2\.\s+/im.test(trimmed)) {
    return { isAgentPrompt: true, reasonCode: CLASSIFICATION_REASON_CODES.AGENT_STRUCTURED_TEMPLATE };
  }

  return { isAgentPrompt: false };
}

/**
 * Detects external, third-party, or scraped content (Tier E).
 */
function detectExternalContent(text: string): { isExternal: boolean; reasonCode?: string } {
  // Video generation scripts
  if (/\b(Video\s+\d+\s*—|Target\s+duration:\s*\d+–\d+\s+seconds|Narration:\s+Energetic|A\s+cinematic\s+Roblox-style\s+3D|A\s+high-quality\s+3D\s+animated\s+Roblox)/i.test(text)) {
    return { isExternal: true, reasonCode: CLASSIFICATION_REASON_CODES.EXTERNAL_VIDEO_SCRIPT };
  }

  // Social gaming captions and channel descriptions
  if (/\b(Subway\s+Surfers|Surf\s+N\s+Play|Subway\s+Surfers\s+gameplay|mobile\s+gaming|Subscribe\s+to\s+Surf\s+N\s+Play|hitting\s+300K\s+in\s+Subway\s+Surfers)\b/i.test(text)) {
    return { isExternal: true, reasonCode: CLASSIFICATION_REASON_CODES.EXTERNAL_SOCIAL_TAGS };
  }

  // Scraped third-party interface text
  if (/\b(Earn\s+on\s+YouTube\s+YouTube\s+Studio\s+dashboard|Channel\s+customisation\s+YouTube\s+Studio\s+dashboard|Skip\s+navigation\s+Search)\b/i.test(text)) {
    return { isExternal: true, reasonCode: CLASSIFICATION_REASON_CODES.EXTERNAL_SCRAPED_DASHBOARD };
  }

  // Repeated hashtags
  if (/#\w+\s+#\w+\s+#\w+/i.test(text)) {
    return { isExternal: true, reasonCode: CLASSIFICATION_REASON_CODES.EXTERNAL_SOCIAL_TAGS };
  }

  return { isExternal: false };
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

  // Check for unfenced code or terminal commands embedded in explanatory prose
  const lines = trimmed.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length >= 3) {
    let proseWordsCount = 0;
    let hasCodeOrTerminalLine = false;

    for (const line of lines) {
      if (
        isUnfencedCode(line) ||
        detectTerminalPromptOrCommand(line).isTerminal ||
        detectBuildOrPackageLog(line).isBuildLog ||
        isStackTraceOrCompilerLog(line)
      ) {
        hasCodeOrTerminalLine = true;
      } else {
        const words = line.split(/\s+/).filter(Boolean);
        proseWordsCount += words.length;
      }
    }

    if (hasCodeOrTerminalLine && proseWordsCount >= 10) {
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

function isProceduralQuestionOnly(text: string): boolean {
  const matchesProcedural = PROCEDURAL_QUESTION_PATTERNS.some((pattern) => pattern.test(text));
  if (!matchesProcedural) {
    return false;
  }
  const hasDeclarativeFact = DECLARATIVE_PERSONAL_STATEMENTS.some((pattern) => pattern.test(text));
  return !hasDeclarativeFact;
}

function isConversationalQuestion(text: string): boolean {
  const trimmed = text.trim();
  if (isProceduralQuestionOnly(trimmed)) {
    return true;
  }
  if (trimmed.endsWith('?')) {
    // If it ends with ? and lacks personal declarative facts, it is a conversational query turn
    const hasDeclarativeFact = DECLARATIVE_PERSONAL_STATEMENTS.some((pattern) => pattern.test(trimmed));
    return !hasDeclarativeFact;
  }
  return false;
}

function hasPersonalKnowledgeSignal(text: string): boolean {
  if (isProceduralQuestionOnly(text)) {
    return false;
  }
  return PERSONAL_EXPERIENCE_PATTERNS.some((pattern) => pattern.test(text));
}

/**
 * Classifies a captured writing message into the six-tier architecture:
 * Tier A: Authentic personal prose (eligible)
 * Tier B: Authentic technical prose (eligible)
 * Tier C: AI-agent instructions & prompt templates (segregated)
 * Tier D: Terminal sessions, build logs, raw code & structured data (excluded)
 * Tier E: External & third-party content (excluded)
 * Tier F: Mixed or ambiguous content (manual review)
 */
export function classifyContent(text: string): ContentClassification {
  const normalizedText = text.trim();
  const wordCount = normalizedText ? normalizedText.split(/\s+/).length : 0;
  const reasons: string[] = [];
  const reasonCodes: string[] = [];

  // 1. Empty / Whitespace-only -> Tier D (Excluded)
  if (!normalizedText || wordCount === 0) {
    return {
      contentType: 'short_fragment',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount: 0,
      reasons: ['empty_message'],
      tier: 'tier_d',
      tierReason: 'Empty or whitespace message',
      reasonCodes: [CLASSIFICATION_REASON_CODES.EMPTY_OR_WHITESPACE],
    };
  }

  // 2. Structured data (JSON, XML/HTML, YAML) -> Tier D (Excluded)
  if (isPureJson(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['json_payload'],
      tier: 'tier_d',
      tierReason: 'Structured JSON payload',
      reasonCodes: [CLASSIFICATION_REASON_CODES.STRUCTURED_DATA_JSON],
    };
  }

  if (isXmlOrHtml(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['xml_or_html_markup'],
      tier: 'tier_d',
      tierReason: 'XML or HTML markup',
      reasonCodes: [CLASSIFICATION_REASON_CODES.STRUCTURED_DATA_MARKUP],
    };
  }

  if (isYamlOrConfig(normalizedText)) {
    return {
      contentType: 'structured_data',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['yaml_or_configuration_data'],
      tier: 'tier_d',
      tierReason: 'YAML or configuration data',
      reasonCodes: [CLASSIFICATION_REASON_CODES.STRUCTURED_DATA_YAML],
    };
  }

  // 3. Stack traces & compiler error logs -> Tier D (Excluded)
  if (isStackTraceOrCompilerLog(normalizedText)) {
    return {
      contentType: 'terminal',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['stack_trace_or_compiler_output'],
      tier: 'tier_d',
      tierReason: 'Stack trace or compiler error log',
      reasonCodes: [CLASSIFICATION_REASON_CODES.TERMINAL_STACK_TRACE],
    };
  }

  // 4. Build, package manager, and test runner outputs -> Tier D (Excluded)
  const buildLogCheck = detectBuildOrPackageLog(normalizedText);
  if (buildLogCheck.isBuildLog) {
    const code = buildLogCheck.reasonCode || CLASSIFICATION_REASON_CODES.TERMINAL_BUILD_LOG;
    return {
      contentType: 'terminal',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['build_or_package_manager_log'],
      tier: 'tier_d',
      tierReason: 'Build log, package manager output, or test runner summary',
      reasonCodes: [code],
    };
  }

  // 5. AI-Agent Directives & Prompt Templates -> Tier C (Segregated)
  const agentCheck = detectAgentDirectiveOrTemplate(normalizedText);
  if (agentCheck.isAgentPrompt) {
    const code = agentCheck.reasonCode || CLASSIFICATION_REASON_CODES.AGENT_DIRECTIVE_COMMAND;
    return {
      contentType: 'prose',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['agent_directive_or_prompt_template'],
      tier: 'tier_c',
      tierReason: 'AI-agent instruction, audit directive, or prompt template',
      reasonCodes: [code],
    };
  }

  // 6. Cross-platform terminal shell prompts & commands -> Tier D (Excluded)
  const terminalCheck = detectTerminalPromptOrCommand(normalizedText);
  if (terminalCheck.isTerminal) {
    const code = terminalCheck.reasonCode || CLASSIFICATION_REASON_CODES.TERMINAL_COMMAND_INVOCATION;
    return {
      contentType: 'terminal',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['terminal_command', 'terminal_command_or_prompt'],
      tier: 'tier_d',
      tierReason: 'Cross-platform terminal prompt or command invocation',
      reasonCodes: [code],
    };
  }

  // 7. Entirely fenced code block (``` ... ```) -> Tier D (Excluded)
  if (/^```[\s\S]*```$/.test(normalizedText)) {
    return {
      contentType: 'code',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['fenced_code_block'],
      tier: 'tier_d',
      tierReason: 'Markdown fenced code block',
      reasonCodes: [CLASSIFICATION_REASON_CODES.CODE_FENCED_BLOCK],
    };
  }

  // 8. External / Third-Party Content -> Tier E (Excluded)
  const externalCheck = detectExternalContent(normalizedText);
  if (externalCheck.isExternal) {
    const code = externalCheck.reasonCode || CLASSIFICATION_REASON_CODES.EXTERNAL_SOCIAL_TAGS;
    return {
      contentType: 'prose',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['external_or_third_party_content'],
      tier: 'tier_e',
      tierReason: 'External video generation prompt, social tags, or scraped interface content',
      reasonCodes: [code],
    };
  }

  // 9. Mixed prose and code / terminal -> Tier F (Manual Review)
  const mixedCheck = hasMixedProseAndCode(normalizedText);
  if (mixedCheck.isMixed) {
    reasons.push('mixed_prose_and_code');
    reasonCodes.push(CLASSIFICATION_REASON_CODES.MIXED_PROSE_AND_CODE);
    const isKnowledge = hasPersonalKnowledgeSignal(normalizedText);
    if (isKnowledge) {
      reasons.push('personal_context_detected');
    }

    return {
      contentType: 'mixed',
      styleSignal: mixedCheck.proseWords >= 14 ? 'strong' : 'weak',
      knowledgeSignal: isKnowledge ? 'possible' : 'none',
      wordCount,
      reasons,
      tier: 'tier_f',
      tierReason: 'Mixed prose and code/terminal requiring manual review',
      reasonCodes,
    };
  }

  // 10. Raw Unfenced source code -> Tier D (Excluded)
  if (isUnfencedCode(normalizedText)) {
    return {
      contentType: 'code',
      styleSignal: 'none',
      knowledgeSignal: 'none',
      wordCount,
      reasons: ['unfenced_source_code'],
      tier: 'tier_d',
      tierReason: 'Raw unfenced source code',
      reasonCodes: [CLASSIFICATION_REASON_CODES.CODE_UNFENCED_DEFINITION],
    };
  }

  // 11. Short conversational fragments (< 4 words or static phrases) -> Tier D (Excluded)
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
        tier: 'tier_d',
        tierReason: 'Short conversational fragment without style signal',
        reasonCodes: [CLASSIFICATION_REASON_CODES.SHORT_FRAGMENT],
      };
    }
  }

  // 12. Conversational Questions (e.g. asking an AI questions) -> Tier F (Manual Review / Non-drafting)
  if (isConversationalQuestion(normalizedText)) {
    reasons.push('conversational_query');
    reasonCodes.push(CLASSIFICATION_REASON_CODES.CONVERSATIONAL_QUERY_NOT_DRAFTING_TARGET);
    return {
      contentType: 'prose',
      styleSignal: 'weak',
      knowledgeSignal: 'none',
      wordCount,
      reasons,
      tier: 'tier_f',
      tierReason: 'Conversational question directed at assistant; not self-contained drafting prose',
      reasonCodes,
    };
  }

  // 13. Natural Language Prose Evaluation (Tier A vs Tier B)
  const isPersonalKnowledge = hasPersonalKnowledgeSignal(normalizedText);
  if (isPersonalKnowledge) {
    reasons.push('personal_context_detected');
  }

  // Check for Tier A (Authentic Personal Prose)
  const hasPersonalStatement = DECLARATIVE_PERSONAL_STATEMENTS.some((p) => p.test(normalizedText));

  if (!isProceduralQuestionOnly(normalizedText) && (isPersonalKnowledge || hasPersonalStatement)) {
    reasons.push('natural_language_prose');
    reasonCodes.push(CLASSIFICATION_REASON_CODES.PERSONAL_EXPERIENCE_REFLECTION);
    return {
      contentType: 'prose',
      styleSignal: 'strong',
      knowledgeSignal: 'possible',
      wordCount,
      reasons,
      tier: 'tier_a',
      tierReason: 'Authentic personal prose with genuine style and experiential signal',
      reasonCodes,
    };
  }

  // Check for Tier B (Authentic Technical Prose)
  const isTechnicalProse = TECHNICAL_PROSE_PATTERNS.some((p) => p.test(normalizedText));
  if (isTechnicalProse && wordCount >= 14) {
    reasons.push('natural_language_prose');
    reasons.push('technical_exposition');
    reasonCodes.push(CLASSIFICATION_REASON_CODES.AUTHENTIC_TECHNICAL_EXPOSITION);
    return {
      contentType: 'prose',
      styleSignal: 'strong',
      knowledgeSignal: 'none',
      wordCount,
      reasons,
      tier: 'tier_b',
      tierReason: 'Authentic technical prose explaining concepts, tests, or architecture',
      reasonCodes,
    };
  }

  // General prose check
  if (wordCount >= 16 && /[.!?]$/.test(normalizedText)) {
    // Complete grammatical sentence structure
    reasons.push('natural_language_prose');
    reasonCodes.push(CLASSIFICATION_REASON_CODES.NATURAL_CONVERSATIONAL_PROSE);
    return {
      contentType: 'prose',
      styleSignal: 'strong',
      knowledgeSignal: 'none',
      wordCount,
      reasons,
      tier: 'tier_b',
      tierReason: 'Authentic grammatical conversational prose',
      reasonCodes,
    };
  }

  // Ambiguous or weak style prose -> Tier F (Manual Review)
  reasons.push('natural_language_prose');
  reasonCodes.push(CLASSIFICATION_REASON_CODES.AMBIGUOUS_STYLE_SIGNAL);
  return {
    contentType: 'prose',
    styleSignal: 'weak',
    knowledgeSignal: 'none',
    wordCount,
    reasons,
    tier: 'tier_f',
    tierReason: 'Ambiguous or weak style signal requiring manual review',
    reasonCodes,
  };
}
