import {
  classifyContent,
  DECLARATIVE_PERSONAL_STATEMENTS,
} from './contentClassifier';
import {
  db,
  type EchoDatabase,
  type KnowledgeCategory,
  type KnowledgeConfidence,
  type KnowledgeImportance,
  type KnowledgeStability,
  type PersonalKnowledgeItem,
  type WritingMessage,
} from './database';
import {
  checkOllamaStatus,
  DEFAULT_OLLAMA_BASE_URL,
  DEFAULT_OLLAMA_MODEL,
  queryOllamaJson,
  type OllamaClientOptions,
} from './ollamaClient';

export interface RawExtractedItem {
  category?: string;
  subject?: string;
  predicate?: string;
  value?: string;
  contextScope?: string;
  sourceExcerpt?: string;
  confidence?: string;
  importance?: string;
}

export interface RawExtractionPayload {
  knowledge?: RawExtractedItem[];
}

export interface ValidatedKnowledgeDraft {
  category: KnowledgeCategory;
  subject: string;
  predicate: string;
  value: string;
  contextScope?: string;
  sourceExcerpt: string;
  confidence: KnowledgeConfidence;
  importance: KnowledgeImportance;
  stability: KnowledgeStability;
}

export interface ExtractionProgress {
  totalEligible: number;
  processedCount: number;
  currentMessageText?: string;
  extractedKnowledgeCount: number;
  status: 'idle' | 'checking' | 'running' | 'completed' | 'error';
  errorMessage?: string;
}

export interface ExtractionSummary {
  totalAnalyzed: number;
  candidateCount: number;
  extractedCount: number;
  skippedCount: number;
  duplicateCount: number;
  emptyResultCount: number;
  errors: string[];
}

export type ExtractionProgressCallback = (progress: ExtractionProgress) => void;

const VALID_CATEGORIES = new Set<KnowledgeCategory>([
  'projects',
  'skills',
  'goals',
  'habits',
  'preferences',
  'opinions',
  'states',
]);

const PROCESSED_STORAGE_KEY = 'echo_processed_message_ids';

interface StorageLike {
  get(key: string): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
  remove(key: string): Promise<void>;
}

let memoryStorage: Record<string, unknown> = {};

function getStorage(): StorageLike {
  if (typeof browser !== 'undefined' && browser.storage?.local) {
    return browser.storage.local as unknown as StorageLike;
  }
  const globalChrome = (globalThis as unknown as { chrome?: { storage?: { local?: StorageLike } } }).chrome;
  if (globalChrome?.storage?.local) {
    return globalChrome.storage.local;
  }
  // In-memory fallback for test and non-extension environments
  return {
    async get(key: string) {
      return { [key]: memoryStorage[key] };
    },
    async set(items: Record<string, unknown>) {
      Object.assign(memoryStorage, items);
    },
    async remove(key: string) {
      delete memoryStorage[key];
    },
  };
}

/**
 * Retrieves the set of message IDs previously processed (either produced knowledge or pre-filtered/no knowledge).
 */
export async function getProcessedMessageIds(): Promise<Set<number>> {
  try {
    const storage = getStorage();
    const data = await storage.get(PROCESSED_STORAGE_KEY);
    const list = data[PROCESSED_STORAGE_KEY];
    if (Array.isArray(list)) {
      return new Set<number>(list);
    }
  } catch {
    // In test or non-extension environments, fall back to empty set
  }
  return new Set<number>();
}

/**
 * Marks message IDs as processed in local extension storage.
 */
export async function markMessagesAsProcessed(ids: number[]): Promise<void> {
  if (ids.length === 0) return;
  try {
    const current = await getProcessedMessageIds();
    for (const id of ids) {
      current.add(id);
    }
    const arr = Array.from(current);
    const storage = getStorage();
    await storage.set({ [PROCESSED_STORAGE_KEY]: arr });
  } catch {
    // Fall back gracefully
  }
}

/**
 * Resets the processed messages cache.
 */
export async function clearProcessedMessagesCache(): Promise<void> {
  memoryStorage = {};
  try {
    const storage = getStorage();
    await storage.remove(PROCESSED_STORAGE_KEY);
  } catch {
    // Fall back gracefully
  }
}

/**
 * Infers appropriate stability based on category.
 */
function inferStability(category: KnowledgeCategory): KnowledgeStability {
  switch (category) {
    case 'habits':
    case 'skills':
    case 'preferences':
    case 'opinions':
      return 'stable';
    case 'projects':
    case 'goals':
      return 'evolving';
    case 'states':
      return 'temporary';
    default:
      return 'evolving';
  }
}

export const PROCEDURAL_QUESTION_PATTERNS = [
  /\bwhen\s+(are|do|will|should|can|could|shall)\s+(we|you|i)\s+[\w\s]{0,25}\b(start|building|build|do|begin|proceed|go)\b/i,
  /\b(so\s+)?will\s+(we|you|i)\s+(have\s+to|need\s+to|be\s+able\s+to)\b/i,
  /\b(should|can|could|do|did|would|shall|is|are)\s+(we|you)\s+(have\s+to|need\s+to|start|do|use|build|proceed)\b/i,
  /\b(what|where|how)\s+(are|do|will|should|can|could)\s+(we|you)\s+(do|go|start|build|implement|approach)\b/i,
  /\b(can\s+you|could\s+you|please\s+(explain|show|help|tell|clarify)|tell\s+me|show\s+me|help\s+me\s+with)\b/i,
  /\b(how\s+do\s+i|how\s+can\s+i|how\s+to|what\s+is|why\s+does|explain)\b/i,
  /\bok\s+then\s+we\s+can\s+go\s+with\b/i,
];

export function isProceduralConversationQuestion(
  excerpt: string,
  sourceText?: string
): boolean {
  const norm = excerpt.toLowerCase().trim();

  // Explicit question mark check
  if (norm.endsWith('?') || norm.includes('?')) {
    if (
      /\b(when|why|where|how|what|which)\s+(are|do|will|should|can|could|shall|is)\s+(we|i|you|it)\b/i.test(norm) ||
      /\b(will|shall|should|can|could|do|did|would|is|are)\s+(we|you)\s+(have\s+to|need\s+to|start|be\s+able\s+to|do|go|use|try|build)\b/i.test(norm) ||
      /\b(can\s+you|could\s+you|please|tell\s+me|show\s+me)\b/i.test(norm)
    ) {
      return true;
    }
  }

  // Interrogative / workflow coordination without question mark
  if (PROCEDURAL_QUESTION_PATTERNS.some((p) => p.test(norm))) {
    return true;
  }

  // Check sourceText if provided
  if (sourceText) {
    const normSrc = sourceText.toLowerCase().trim();
    if (PROCEDURAL_QUESTION_PATTERNS.some((p) => p.test(normSrc))) {
      // Check if sourceText contains any authentic declarative personal statement
      const hasDeclarativeFact = DECLARATIVE_PERSONAL_STATEMENTS.some((p) =>
        p.test(normSrc)
      );

      if (!hasDeclarativeFact) {
        return true;
      }
    }
  }

  return false;
}

const VAGUE_VALUES = new Set([
  'many things',
  'something',
  'anything',
  'everything',
  'this and that',
  'every time',
  'everytime',
  "today's time",
  'todays time',
  'other technologies in the coming future',
  'other things',
  'some things',
  'start building',
  'start building now',
  'about a product idea',
  'now',
  'a lot',
  'stuff',
  'things',
  'slow',
]);

const PROCEDURAL_PREDICATES = new Set([
  'will be done',
  'start building',
  'have to do',
  'need to do',
  'going to start',
  'planning to start',
]);

export function isTautologicalOrVague(
  subject: string,
  predicate: string,
  value: string
): boolean {
  const normSubj = subject.toLowerCase().trim();
  const normPred = predicate.toLowerCase().trim();
  const normVal = value.toLowerCase().trim();

  // 1. Tautological assertions where subject matches value
  if (normSubj === normVal) {
    return true;
  }

  // 2. Tautological assertions where predicate equals value or subject
  if (normSubj === normPred || normPred === normVal) {
    return true;
  }

  // 3. Vague filler value check
  if (VAGUE_VALUES.has(normVal)) {
    return true;
  }

  if (
    /\b(many\s+things|this\s+and\s+that|every\s*time|other\s+technologies\s+in\s+the\s+coming\s+future|about\s+a\s+product\s+idea|todays?\s+time)\b/i.test(
      normVal
    )
  ) {
    return true;
  }

  // 4. Procedural predicates
  if (PROCEDURAL_PREDICATES.has(normPred)) {
    return true;
  }

  // 5. Generic subject with empty or vague assertion
  if (normSubj === 'user' || normSubj === 'the user' || normSubj === 'i' || normSubj === 'we') {
    if (
      normPred === 'thought too much' ||
      (normPred === 'want to learn' && normVal.split(/\s+/).length <= 2)
    ) {
      return true;
    }
  }

  return false;
}

const GENERIC_SUBJECTS = new Set([
  'it',
  'this',
  'that',
  'thing',
  'stuff',
  'they',
  'them',
  'these',
  'those',
  'something',
  'anything',
]);

const DISALLOWED_SKILL_PREDICATES = new Set([
  'built',
  'build',
  'created',
  'create',
  'developed',
  'develop',
  'made',
  'make',
  'deployed',
  'deploy',
  'faced',
  'struggled',
  'failed',
]);

const LIMITATION_INDICATORS = [
  'difficulty',
  'not able to',
  'unable to',
  'could not',
  "couldn't",
  "can't",
  'never end',
  'static one where',
  'limitation',
  'workaround',
  'problem was',
  'stuck because',
];

const DISALLOWED_GOAL_PREDICATES = new Set([
  'built',
  'created',
  'developed',
  'made',
  'finished',
  'completed',
]);

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
  'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'some', 'that', 'this', 'it',
  'not', 'too', 'very', 'can', 'will', 'just', 'more', 'much', 'also',
]);

function extractContentTokens(str: string): string[] {
  return str
    .toLowerCase()
    .split(/[^a-z0-9_#+.-]+/)
    .filter((t) => t.length >= 3 && !STOP_WORDS.has(t));
}

/**
 * Validates and grounds an extracted item against the source message text.
 * Strictly enforces that sourceExcerpt is a verbatim substring of message.text.
 * Rejects unsupported interpretations, project limitations incorrectly framed as skills,
 * and ambiguous or hallucinated values.
 */
export function validateAndGroundItem(
  item: RawExtractedItem,
  sourceText: string
): ValidatedKnowledgeDraft | null {
  if (!item || typeof item !== 'object') {
    return null;
  }

  // 1. Category validation
  let category = (item.category || '').toLowerCase().trim() as KnowledgeCategory;
  if (!VALID_CATEGORIES.has(category)) {
    return null;
  }

  // 2. Core content fields validation
  const subject = (item.subject || '').trim();
  let predicate = (item.predicate || '').trim();
  const value = (item.value || '').trim();
  const sourceExcerpt = (item.sourceExcerpt || '').trim();

  if (!subject || !predicate || !value || !sourceExcerpt) {
    return null;
  }

  // Minimum length check for subject and predicate
  if (subject.length < 2 || predicate.length < 2 || value.length < 2) {
    return null;
  }

  // Reject generic pronouns or vague placeholders as subjects
  if (GENERIC_SUBJECTS.has(subject.toLowerCase())) {
    return null;
  }

  // 3. Provenance Grounding Check: sourceExcerpt MUST be a verbatim substring of sourceText
  const normalizedSource = sourceText.toLowerCase();
  const normalizedExcerpt = sourceExcerpt.toLowerCase();

  if (!normalizedSource.includes(normalizedExcerpt)) {
    // Reject hallucinated or altered excerpts
    return null;
  }

  // 4. Excerpt quality check (must be at least 4 characters to prevent trivial matches)
  if (sourceExcerpt.length < 4) {
    return null;
  }

  // 5. Reject procedural conversation questions mistakenly framed as personal facts
  if (isProceduralConversationQuestion(sourceExcerpt, sourceText)) {
    return null;
  }

  // 6. Reject tautological triples and vague conversational filler
  if (isTautologicalOrVague(subject, predicate, value)) {
    return null;
  }

  // 7. Category semantic calibration & normalization
  const combinedContext = `${normalizedExcerpt} ${predicate.toLowerCase()} ${value.toLowerCase()}`;

  // A. Debate and communication approaches -> 'habits' (NOT 'goals')
  // Only recalibrate if strong context shows it describes an ongoing discussion/communication habit,
  // and NOT an explicit goal (e.g. "My goal is to participate in a debate tournament").
  if (category === 'goals') {
    const isExplicitGoal =
      /\b(goal|aim|target|plan\s+to|planning\s+to|participate(\s+in)?|compete(\s+in)?|tournament|competition|contest|join)\b/i.test(
        combinedContext
      );
    const isDiscussionHabit =
      /\b(when\s+in\s+a\s+debate|in\s+a\s+discussion|when\s+debating|in\s+discussions|listen\s+to\s+theirs|put\s+my\s+point|argue\s+to\s+prove|explain\s+with\s+examples|use\s+analogies)\b/i.test(
        combinedContext
      );

    if (isDiscussionHabit && !isExplicitGoal) {
      category = 'habits';
    }
  }

  // B. Durable technical convictions & philosophies -> 'opinions' (NOT 'states')
  // Only recalibrate if context describes durable philosophies / educational critique,
  // and NOT an ephemeral error or immediate bug state (e.g. "This compiler error frustrates me").
  if (category === 'states') {
    const isEphemeralState =
      /\b(compiler\s+error|build\s+error|type\s+error|syntax\s+error|runtime\s+error|this\s+error|this\s+bug|this\s+issue|debugging\s+this|at\s+line)\b/i.test(
        combinedContext
      );
    const isDurableTechnicalConviction =
      /\b(tutorials?\s+(that\s+)?skip|programming\s+tutorials?|skip\s+the\s+why|premature\s+microservice|unnecessary\s+microservice|monolithic\s+architecture|strongly\s+believe|in\s+my\s+opinion)\b/i.test(
        combinedContext
      );

    if (isDurableTechnicalConviction && !isEphemeralState) {
      category = 'opinions';
    }
  }

  // 8. Semantic Category Boundary Checks
  // A. Skills boundaries
  if (category === 'skills') {
    const normPred = predicate.toLowerCase();
    // Building or creating an app/product is a project, not a technical skill
    if (DISALLOWED_SKILL_PREDICATES.has(normPred)) {
      return null;
    }

    // Technical difficulties, missing APIs, and workarounds are limitations, NOT skills
    const normVal = value.toLowerCase();
    if (
      LIMITATION_INDICATORS.some(
        (indicator) =>
          normalizedExcerpt.includes(indicator) || normVal.includes(indicator)
      )
    ) {
      return null;
    }

    // Prevent unwarranted competency inflation: never upgrade "learned" to "proficient_in" or "expert" unless explicitly stated
    if (normPred.includes('proficient') || normPred.includes('expert')) {
      const mentionsExpertise = /\b(proficient|proficiency|expert|expertise|advanced|master|mastered)\b/i.test(
        normalizedExcerpt
      );
      if (!mentionsExpertise) {
        predicate = 'learned';
      }
    }
  }

  // B. Goals boundaries
  if (category === 'goals') {
    const normPred = predicate.toLowerCase();
    // Goals describe future ambitions or intentions, not past completed actions
    if (DISALLOWED_GOAL_PREDICATES.has(normPred)) {
      return null;
    }
  }

  // 7. Claim-to-Evidence Grounding Check:
  // The sourceExcerpt is the direct textual justification for this assertion.
  // It MUST directly support the claimed value and assertion, not merely be an arbitrary substring of the document.
  const valueTokens = extractContentTokens(value);
  const predicateTokens = extractContentTokens(predicate);
  const subjectTokens = extractContentTokens(subject);

  // A. Value grounding: If the value has meaningful content words, at least one MUST appear in the sourceExcerpt itself.
  if (valueTokens.length > 0) {
    const hasGroundedValueInExcerpt = valueTokens.some((token) =>
      normalizedExcerpt.includes(token)
    );
    if (!hasGroundedValueInExcerpt) {
      return null; // The excerpt does not contain or substantiate the claimed value!
    }
  }

  // B. Claim-to-evidence coherence: The sourceExcerpt must share semantic content with what is being asserted.
  // If the excerpt contains zero tokens from value AND zero tokens from predicate, the excerpt does not support the claim.
  if (valueTokens.length > 0 || predicateTokens.length > 0) {
    const hasAssertionToken =
      valueTokens.some((t) => normalizedExcerpt.includes(t)) ||
      predicateTokens.some((t) => normalizedExcerpt.includes(t));
    if (!hasAssertionToken) {
      return null; // Excerpt is completely disconnected from the asserted action and value!
    }
  }

  // 8. Confidence validation & calibration for tentative / developing claims:
  // Confidence represents how explicitly the source supports the extracted claim.
  let confidence: KnowledgeConfidence = 'medium';
  const rawConf = (item.confidence || '').toLowerCase().trim();
  if (rawConf === 'high' || rawConf === 'medium' || rawConf === 'low') {
    confidence = rawConf;
  }

  // Tentative intention calibration:
  // A forward-looking goal, planned learning, or tentative wish (e.g. "planning to learn X", "hope to", "want to")
  // represents a developing intention, NOT an accomplished or definitive established fact.
  // Tentative claims must not automatically receive 'high' confidence.
  const TENTATIVE_MARKERS = [
    'planning to',
    'plan to',
    'plan was',
    'planning on',
    'hope to',
    'hoping to',
    'might',
    'maybe',
    'thinking about',
    'thinking of',
    'trying to',
    'would like to',
    'want to',
    'considering',
    'aiming to',
    'interested in',
    'curious about',
  ];

  const isTentative =
    category === 'goals' ||
    TENTATIVE_MARKERS.some(
      (m) =>
        normalizedExcerpt.includes(m) ||
        predicate.toLowerCase().includes(m) ||
        value.toLowerCase().includes(m)
    );

  if (isTentative && confidence === 'high') {
    confidence = 'medium';
  }

  // 9. Importance validation: represents semantic prominence (independent of confidence)
  let importance: KnowledgeImportance = 'secondary';
  const rawImp = (item.importance || '').toLowerCase().trim();
  if (rawImp === 'primary' || rawImp === 'secondary' || rawImp === 'incidental') {
    importance = rawImp;
  }

  const stability = inferStability(category);
  const contextScope = item.contextScope && item.contextScope.trim()
    ? item.contextScope.trim()
    : undefined;

  return {
    category,
    subject,
    predicate,
    value,
    contextScope,
    sourceExcerpt,
    confidence,
    importance,
    stability,
  };
}

/**
 * Parses and validates raw LLM output into a list of grounded knowledge drafts.
 */
export function parseExtractionResponse(
  rawContent: string,
  sourceText: string
): ValidatedKnowledgeDraft[] {
  let parsed: RawExtractionPayload;
  try {
    parsed = JSON.parse(rawContent) as RawExtractionPayload;
  } catch {
    throw new Error('Failed to parse Ollama response as valid JSON.');
  }

  if (!parsed || !Array.isArray(parsed.knowledge)) {
    return [];
  }

  const validDrafts: ValidatedKnowledgeDraft[] = [];

  for (const item of parsed.knowledge) {
    const grounded = validateAndGroundItem(item, sourceText);
    if (grounded) {
      validDrafts.push(grounded);
    }
  }

  return validDrafts;
}

/**
 * Persists an extracted knowledge item and its evidence into the database atomically.
 * Handles deduplication: if an active item with identical (category, subject, predicate) exists,
 * it avoids creating a duplicate and links the new evidence to the existing knowledge item.
 * Guarantees zero orphaned knowledge items.
 */
export async function persistGroundedKnowledge(
  draft: ValidatedKnowledgeDraft,
  messageId: number,
  targetDb: EchoDatabase = db
): Promise<{ knowledgeId: number; isNewKnowledge: boolean }> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  const normalizedSubject = draft.subject.toLowerCase().trim();
  const normalizedPredicate = draft.predicate.toLowerCase().trim();
  const now = new Date();

  return targetDb.transaction(
    'rw',
    [targetDb.personalKnowledge, targetDb.knowledgeEvidence],
    async function () {
      // 1. Check for existing active knowledge item with the same category, subject, and predicate
      const existingItem = await targetDb.personalKnowledge
        .where('category')
        .equals(draft.category)
        .filter(function (k) {
          return (
            k.status === 'active' &&
            k.subject.toLowerCase().trim() === normalizedSubject &&
            k.predicate.toLowerCase().trim() === normalizedPredicate
          );
        })
        .first();

      let targetKnowledgeId: number;
      let isNewKnowledge = false;

      if (existingItem && existingItem.id !== undefined) {
        // Corroborate existing knowledge
        targetKnowledgeId = existingItem.id;

        // Check if this message is already linked to avoid duplicate evidence
        const duplicateEvidence = await targetDb.knowledgeEvidence
          .where('knowledgeId')
          .equals(targetKnowledgeId)
          .filter(function (e) {
            return e.messageId === messageId;
          })
          .first();

        if (duplicateEvidence) {
          // Already linked; do not insert duplicate evidence
          return { knowledgeId: targetKnowledgeId, isNewKnowledge: false };
        }

        // Update lastConfirmedAt on the existing knowledge item
        // CRITICAL: Confidence and importance are NEVER increased simply due to repeated mentions
        await targetDb.personalKnowledge.update(targetKnowledgeId, {
          lastConfirmedAt: now,
          updatedAt: now,
        });
      } else {
        // Create new knowledge item
        isNewKnowledge = true;
        const newItem: PersonalKnowledgeItem = {
          category: draft.category,
          subject: draft.subject,
          predicate: draft.predicate,
          value: draft.value,
          contextScope: draft.contextScope,
          stability: draft.stability,
          status: 'active',
          freshnessState: 'fresh',
          confidence: draft.confidence,
          importance: draft.importance,
          createdAt: now,
          updatedAt: now,
          lastConfirmedAt: now,
        };

        const assignedId = await targetDb.personalKnowledge.add(newItem);
        targetKnowledgeId = assignedId as number;
      }

      // Check evidence inside the Dexie transaction before inserting it to guarantee idempotency
      const existingEvidence = await targetDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(targetKnowledgeId)
        .filter(function (e) {
          return e.messageId === messageId;
        })
        .first();

      if (existingEvidence) {
        return { knowledgeId: targetKnowledgeId, isNewKnowledge };
      }

      // 2. Insert linked knowledge evidence row atomically
      await targetDb.knowledgeEvidence.add({
        knowledgeId: targetKnowledgeId,
        messageId,
        sourceExcerpt: draft.sourceExcerpt,
        relevanceNote: `Extracted from user message #${messageId}`,
        isValid: true,
        extractedAt: now,
      });

      return { knowledgeId: targetKnowledgeId, isNewKnowledge };
    }
  );
}

/**
 * Main Personal Knowledge Extraction pipeline execution function.
 * Runs sequentially with concurrency of one.
 * Preserves message collection, honors privacy deletion rules, and never hallucinates facts.
 */
export async function runKnowledgeExtraction(
  options: {
    targetDb?: EchoDatabase;
    ollamaOptions?: OllamaClientOptions;
    onProgress?: ExtractionProgressCallback;
    forceAll?: boolean;
    messages?: WritingMessage[];
  } = {}
): Promise<ExtractionSummary> {
  const targetDb = options.targetDb || db;
  const ollamaOptions = options.ollamaOptions || {};
  const baseUrl = ollamaOptions.baseUrl || DEFAULT_OLLAMA_BASE_URL;
  const model = ollamaOptions.model || DEFAULT_OLLAMA_MODEL;
  const onProgress = options.onProgress;

  const summary: ExtractionSummary = {
    totalAnalyzed: 0,
    candidateCount: 0,
    extractedCount: 0,
    skippedCount: 0,
    duplicateCount: 0,
    emptyResultCount: 0,
    errors: [],
  };

  // 1. Verify Ollama & Model availability
  if (onProgress) {
    onProgress({
      totalEligible: 0,
      processedCount: 0,
      extractedKnowledgeCount: 0,
      status: 'checking',
    });
  }

  const status = await checkOllamaStatus(model, baseUrl);
  if (!status.isRunning || !status.isModelReady) {
    const errorMsg = status.error || 'Ollama is offline or model is not ready.';
    if (onProgress) {
      onProgress({
        totalEligible: 0,
        processedCount: 0,
        extractedKnowledgeCount: 0,
        status: 'error',
        errorMessage: errorMsg,
      });
    }
    throw new Error(errorMsg);
  }

  // 2. Fetch messages from database or options
  let allMessages = options.messages;
  if (!allMessages) {
    if (!targetDb.isOpen()) {
      await targetDb.open();
    }
    allMessages = await targetDb.writingMessages.orderBy('id').toArray();
  }

  summary.totalAnalyzed = allMessages.length;

  // 3. Identify previously processed message IDs
  const processedIds = options.forceAll ? new Set<number>() : await getProcessedMessageIds();

  // Also query existing knowledgeEvidence messageIds from Dexie to be doubly safe
  if (!options.forceAll && targetDb.isOpen()) {
    const evidenceMessageIds = await targetDb.knowledgeEvidence.orderBy('messageId').uniqueKeys();
    for (const id of evidenceMessageIds) {
      if (typeof id === 'number') {
        processedIds.add(id);
      }
    }
  }

  // 4. Filter candidates using Content Classifier
  const eligibleCandidates: WritingMessage[] = [];
  const newlySkippedIds: number[] = [];

  for (const msg of allMessages) {
    if (!msg.id) continue;

    if (processedIds.has(msg.id)) {
      continue;
    }

    const classification = classifyContent(msg.text);

    // Only process prose or mixed content with a possible knowledge signal
    const isEligible =
      (classification.contentType === 'prose' || classification.contentType === 'mixed') &&
      classification.knowledgeSignal === 'possible';

    if (isEligible) {
      eligibleCandidates.push(msg);
    } else {
      newlySkippedIds.push(msg.id);
      summary.skippedCount++;
    }
  }

  // Mark non-eligible messages as processed so we don't re-classify them repeatedly
  await markMessagesAsProcessed(newlySkippedIds);

  summary.candidateCount = eligibleCandidates.length;

  if (onProgress) {
    onProgress({
      totalEligible: eligibleCandidates.length,
      processedCount: 0,
      extractedKnowledgeCount: 0,
      status: 'running',
    });
  }

  if (eligibleCandidates.length === 0) {
    if (onProgress) {
      onProgress({
        totalEligible: 0,
        processedCount: 0,
        extractedKnowledgeCount: 0,
        status: 'completed',
      });
    }
    return summary;
  }

  // 5. Sequential processing (concurrency = 1)
  const newlyProcessedCandidateIds: number[] = [];

  for (let i = 0; i < eligibleCandidates.length; i++) {
    const candidate = eligibleCandidates[i]!;

    if (onProgress) {
      onProgress({
        totalEligible: eligibleCandidates.length,
        processedCount: i,
        currentMessageText: candidate.text,
        extractedKnowledgeCount: summary.extractedCount,
        status: 'running',
      });
    }

    try {
      // Query Ollama in JSON mode with thinking disabled for fast CPU inference
      const rawJson = await queryOllamaJson(candidate.text, ollamaOptions);

      // Parse and ground against original source message text
      const drafts = parseExtractionResponse(rawJson, candidate.text);

      if (drafts.length === 0) {
        summary.emptyResultCount++;
      } else {
        for (const draft of drafts) {
          const result = await persistGroundedKnowledge(draft, candidate.id!, targetDb);
          if (result.isNewKnowledge) {
            summary.extractedCount++;
          } else {
            summary.duplicateCount++;
          }
        }
      }

      // Mark this message ID as processed immediately in storage so progress is saved incrementally
      await markMessagesAsProcessed([candidate.id!]);
      newlyProcessedCandidateIds.push(candidate.id!);
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      summary.errors.push(`Message #${candidate.id}: ${errorMsg}`);
      // Do not mark failed messages as processed so they can safely be retried
    }
  }

  // Mark successfully processed candidate messages in storage
  await markMessagesAsProcessed(newlyProcessedCandidateIds);

  if (onProgress) {
    onProgress({
      totalEligible: eligibleCandidates.length,
      processedCount: eligibleCandidates.length,
      extractedKnowledgeCount: summary.extractedCount,
      status: 'completed',
    });
  }

  return summary;
}
