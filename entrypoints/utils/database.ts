import Dexie, { type EntityTable } from 'dexie';
import { classifyContent } from './contentClassifier';
import {
  isPotentialStyleSample,
  isEligibleStyleSample,
  isEligibleTrainingSample,
} from './trainingFilter';

type KnowledgeCategory =
  | 'projects'
  | 'skills'
  | 'goals'
  | 'habits'
  | 'preferences'
  | 'opinions'
  | 'states';

type KnowledgeStability = 'stable' | 'evolving' | 'temporary';

type KnowledgeStatus = 'active' | 'superseded' | 'conflicted' | 'archived';

type FreshnessState = 'fresh' | 'stale' | 'evergreen';

type KnowledgeConfidence = 'high' | 'medium' | 'low';

type KnowledgeImportance = 'primary' | 'secondary' | 'incidental';

type InvalidationReason =
  | 'source_message_deleted'
  | 'superseded_by_newer_fact'
  | 'user_manually_rejected';

interface WritingMessage {
  id?: number;
  platform: string;
  text: string;
  createdAt: Date;
  capturedAt: Date;
  trainingStatus: 'pending' | 'trained';
}

interface PersonalKnowledgeItem {
  id?: number;
  category: KnowledgeCategory;
  subject: string;
  predicate: string;
  value: string;
  contextScope?: string;
  stability: KnowledgeStability;
  status: KnowledgeStatus;
  freshnessState: FreshnessState;
  confidence: KnowledgeConfidence;
  importance: KnowledgeImportance;
  sourceType?: 'manual' | 'extracted';
  editedByUser?: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastConfirmedAt: Date;
}

interface KnowledgeEvidence {
  id?: number;
  knowledgeId: number;
  messageId: number;
  sourceExcerpt: string;
  relevanceNote?: string;
  isValid: boolean;
  invalidatedAt?: Date;
  invalidationReason?: InvalidationReason | string;
  extractedAt: Date;
}

interface PersonalKnowledgeFilter {
  category?: KnowledgeCategory;
  status?: KnowledgeStatus;
  sourceType?: 'manual' | 'extracted';
}

type EchoDatabase = Dexie & {
  writingMessages: EntityTable<WritingMessage, 'id'>;
  personalKnowledge: EntityTable<PersonalKnowledgeItem, 'id'>;
  knowledgeEvidence: EntityTable<KnowledgeEvidence, 'id'>;
};

function createEchoDatabase(name: string = 'EchoDatabase'): EchoDatabase {
  const instance = new Dexie(name) as EchoDatabase;

  instance.version(1).stores({
    writingMessages: '++id, platform, capturedAt, trainingStatus',
  });

  instance.version(2).stores({
    writingMessages: '++id, platform, capturedAt, trainingStatus',
    personalKnowledge:
      '++id, category, subject, stability, status, freshnessState, importance, createdAt, updatedAt',
    knowledgeEvidence: '++id, knowledgeId, messageId, isValid, extractedAt',
  });

  // Close connection gracefully when another tab or context requests a schema upgrade
  instance.on('versionchange', function () {
    instance.close();
  });

  instance.on('blocked', function () {
    console.warn(`EchoDatabase: Database upgrade blocked for "${name}". Please close other open Echo tabs.`);
  });

  return instance;
}

const db = createEchoDatabase('EchoDatabase');

const DEFAULT_DUPLICATE_WINDOW_MS = 60 * 1000; // 60 seconds

async function saveWritingMessage(
  message: WritingMessage,
  duplicateWindowMs: number = DEFAULT_DUPLICATE_WINDOW_MS,
  targetDb: EchoDatabase = db
): Promise<number | undefined> {
  if (
    typeof message.text !== 'string' ||
    !message.text.trim()
  ) {
    throw new Error('Invalid writing message: text must not be empty.');
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.transaction('rw', targetDb.writingMessages, async function () {
    const normalizedText = message.text.trim();
    const messageCapturedTime = new Date(message.capturedAt).getTime();

    // Check if an identical message on the same platform was captured within the deduplication window.
    // Messages captured outside this window are preserved as intentional repetitions.
    const recentDuplicate = await targetDb.writingMessages
      .where('platform')
      .equals(message.platform)
      .filter(function (savedMessage) {
        if (savedMessage.text.trim() !== normalizedText) {
          return false;
        }

        const savedCapturedTime = new Date(savedMessage.capturedAt).getTime();
        return Math.abs(messageCapturedTime - savedCapturedTime) <= duplicateWindowMs;
      })
      .first();

    if (recentDuplicate) {
      return undefined;
    }

    return targetDb.writingMessages.add(message);
  });
}

async function getWritingMessages(
  targetDb: EchoDatabase = db
): Promise<WritingMessage[]> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.writingMessages
    .orderBy('capturedAt')
    .reverse()
    .toArray();
}

async function deleteWritingMessage(
  id: number,
  targetDb: EchoDatabase = db
): Promise<void> {
  if (typeof id !== 'number' || isNaN(id)) {
    throw new Error('Invalid writing message ID: ID must be a valid number.');
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.transaction(
    'rw',
    [targetDb.writingMessages, targetDb.knowledgeEvidence, targetDb.personalKnowledge],
    async function () {
      // 1. Find all evidence records linked to this message ID
      const linkedEvidence = await targetDb.knowledgeEvidence
        .where('messageId')
        .equals(id)
        .toArray();

      const affectedKnowledgeIds = new Set<number>();

      // 2, 3, 4. Invalidate and scrub each evidence record linked to the deleted message
      const now = new Date();
      for (const evidence of linkedEvidence) {
        if (evidence.id !== undefined) {
          await targetDb.knowledgeEvidence.update(evidence.id, {
            isValid: false,
            sourceExcerpt: '',
            relevanceNote: '',
            invalidatedAt: now,
            invalidationReason: 'source_message_deleted',
          });
          affectedKnowledgeIds.add(evidence.knowledgeId);
        }
      }

      // 5, 6, 7. Scrub derived content and archive all affected knowledge items
      for (const knowledgeId of affectedKnowledgeIds) {
        await targetDb.personalKnowledge.update(knowledgeId, {
          status: 'archived',
          subject: '',
          predicate: '',
          value: '',
          contextScope: '',
          updatedAt: now,
        });
      }

      // 8. Delete the original writing message
      await targetDb.writingMessages.delete(id);
    }
  );
}

async function markMessagesTrained(
  ids: number[],
  targetDb: EchoDatabase = db
): Promise<number> {
  const validIds = ids.filter(id => typeof id === 'number' && !isNaN(id));
  if (validIds.length === 0) {
    return 0;
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.transaction('rw', targetDb.writingMessages, async function () {
    let updatedCount = 0;
    for (const id of validIds) {
      const updated = await targetDb.writingMessages.update(id, {
        trainingStatus: 'trained',
      });
      if (updated) {
        updatedCount++;
      }
    }
    return updatedCount;
  });
}

async function savePersonalKnowledge(
  item: PersonalKnowledgeItem,
  targetDb: EchoDatabase = db
): Promise<number> {
  if (
    !item ||
    typeof item.category !== 'string' ||
    typeof item.subject !== 'string' ||
    !item.subject.trim() ||
    typeof item.predicate !== 'string' ||
    !item.predicate.trim() ||
    typeof item.value !== 'string' ||
    !item.value.trim() ||
    typeof item.stability !== 'string' ||
    typeof item.status !== 'string' ||
    typeof item.freshnessState !== 'string' ||
    typeof item.confidence !== 'string' ||
    typeof item.importance !== 'string'
  ) {
    throw new Error(
      'Invalid personal knowledge item: category, subject, predicate, value, stability, status, freshnessState, confidence, and importance are required.'
    );
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  const normalizedItem: PersonalKnowledgeItem = {
    ...item,
    subject: item.subject.trim(),
    predicate: item.predicate.trim(),
    value: item.value.trim(),
    sourceType: item.sourceType ?? 'extracted',
    editedByUser: item.editedByUser ?? false,
    createdAt: item.createdAt ? new Date(item.createdAt) : new Date(),
    updatedAt: item.updatedAt ? new Date(item.updatedAt) : new Date(),
    lastConfirmedAt: item.lastConfirmedAt
      ? new Date(item.lastConfirmedAt)
      : new Date(),
  };

  const id = await targetDb.personalKnowledge.add(normalizedItem);
  return id as number;
}

async function getPersonalKnowledge(
  filter?: PersonalKnowledgeFilter,
  targetDb: EchoDatabase = db
): Promise<PersonalKnowledgeItem[]> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  let items: PersonalKnowledgeItem[];

  if (filter?.category && filter?.status) {
    items = await targetDb.personalKnowledge
      .where('category')
      .equals(filter.category)
      .filter(function (item) {
        return item.status === filter.status;
      })
      .toArray();
  } else if (filter?.category) {
    items = await targetDb.personalKnowledge
      .where('category')
      .equals(filter.category)
      .toArray();
  } else if (filter?.status) {
    items = await targetDb.personalKnowledge
      .where('status')
      .equals(filter.status)
      .toArray();
  } else {
    items = await targetDb.personalKnowledge.toArray();
  }

  if (filter?.sourceType) {
    items = items.filter(function (item) {
      return (item.sourceType ?? 'extracted') === filter.sourceType;
    });
  }

  return items;
}

async function getPersonalKnowledgeById(
  id: number,
  targetDb: EchoDatabase = db
): Promise<PersonalKnowledgeItem | undefined> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.personalKnowledge.get(id);
}

const VALID_KNOWLEDGE_CATEGORIES: ReadonlySet<string> = new Set([
  'projects',
  'skills',
  'goals',
  'habits',
  'preferences',
  'opinions',
  'states',
]);

const VALID_KNOWLEDGE_STABILITIES: ReadonlySet<string> = new Set([
  'stable',
  'evolving',
  'temporary',
]);

const VALID_KNOWLEDGE_CONFIDENCES: ReadonlySet<string> = new Set([
  'high',
  'medium',
  'low',
]);

const VALID_KNOWLEDGE_IMPORTANCES: ReadonlySet<string> = new Set([
  'primary',
  'secondary',
  'incidental',
]);

const VALID_FRESHNESS_STATES: ReadonlySet<string> = new Set([
  'fresh',
  'stale',
  'evergreen',
]);

const VALID_INVALIDATION_REASONS: ReadonlySet<InvalidationReason> = new Set<InvalidationReason>([
  'source_message_deleted',
  'superseded_by_newer_fact',
  'user_manually_rejected',
]);

interface ManualPersonalKnowledgeInput {
  category: KnowledgeCategory;
  subject: string;
  predicate: string;
  value: string;
  contextScope?: string;
  stability?: KnowledgeStability;
  freshnessState?: FreshnessState;
  confidence?: KnowledgeConfidence;
  importance?: KnowledgeImportance;
}

async function addManualPersonalKnowledge(
  input: ManualPersonalKnowledgeInput,
  targetDb: EchoDatabase = db
): Promise<number> {
  if (!input || typeof input !== 'object') {
    throw new Error('Invalid manual personal knowledge: input must be an object.');
  }

  if (typeof input.category !== 'string' || !VALID_KNOWLEDGE_CATEGORIES.has(input.category)) {
    throw new Error(`Invalid personal knowledge category: "${input.category}".`);
  }

  if (typeof input.subject !== 'string' || !input.subject.trim()) {
    throw new Error('Invalid personal knowledge subject: subject must not be empty.');
  }

  if (typeof input.predicate !== 'string' || !input.predicate.trim()) {
    throw new Error('Invalid personal knowledge predicate: predicate must not be empty.');
  }

  if (typeof input.value !== 'string' || !input.value.trim()) {
    throw new Error('Invalid personal knowledge value: value must not be empty.');
  }

  if (input.contextScope !== undefined && typeof input.contextScope !== 'string') {
    throw new Error('Invalid personal knowledge contextScope: contextScope must be a string if provided.');
  }

  if (input.stability !== undefined && !VALID_KNOWLEDGE_STABILITIES.has(input.stability)) {
    throw new Error(`Invalid personal knowledge stability: "${input.stability}".`);
  }

  if (input.confidence !== undefined && !VALID_KNOWLEDGE_CONFIDENCES.has(input.confidence)) {
    throw new Error(`Invalid personal knowledge confidence: "${input.confidence}".`);
  }

  if (input.importance !== undefined && !VALID_KNOWLEDGE_IMPORTANCES.has(input.importance)) {
    throw new Error(`Invalid personal knowledge importance: "${input.importance}".`);
  }

  if (input.freshnessState !== undefined && !VALID_FRESHNESS_STATES.has(input.freshnessState)) {
    throw new Error(`Invalid personal knowledge freshnessState: "${input.freshnessState}".`);
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  const now = new Date();
  const normalizedItem: PersonalKnowledgeItem = {
    category: input.category,
    subject: input.subject.trim(),
    predicate: input.predicate.trim(),
    value: input.value.trim(),
    contextScope: input.contextScope ? input.contextScope.trim() : undefined,
    stability: input.stability ?? 'stable',
    status: 'active',
    freshnessState: input.freshnessState ?? 'fresh',
    confidence: input.confidence ?? 'high',
    importance: input.importance ?? 'primary',
    sourceType: 'manual',
    editedByUser: false,
    createdAt: now,
    updatedAt: now,
    lastConfirmedAt: now,
  };

  const id = await targetDb.personalKnowledge.add(normalizedItem);
  return id as number;
}

interface UpdatePersonalKnowledgeInput {
  category?: KnowledgeCategory;
  subject?: string;
  predicate?: string;
  value?: string;
  contextScope?: string;
  stability?: KnowledgeStability;
  freshnessState?: FreshnessState;
  confidence?: KnowledgeConfidence;
  importance?: KnowledgeImportance;
}

const IMMUTABLE_KNOWLEDGE_FIELDS: ReadonlySet<string> = new Set([
  'id',
  'createdAt',
  'sourceType',
  'status',
]);

const INTERNAL_KNOWLEDGE_FIELDS: ReadonlySet<string> = new Set([
  'editedByUser',
  'updatedAt',
  'lastConfirmedAt',
]);

const ALLOWED_MUTABLE_KNOWLEDGE_FIELDS: ReadonlySet<keyof UpdatePersonalKnowledgeInput> = new Set([
  'category',
  'subject',
  'predicate',
  'value',
  'contextScope',
  'stability',
  'freshnessState',
  'confidence',
  'importance',
]);

async function updatePersonalKnowledge(
  id: number,
  updates: UpdatePersonalKnowledgeInput,
  targetDb: EchoDatabase = db
): Promise<void> {
  if (typeof id !== 'number' || isNaN(id) || id <= 0) {
    throw new Error('Invalid personal knowledge ID: ID must be a positive number.');
  }

  if (!updates || typeof updates !== 'object') {
    throw new Error('Invalid update payload: updates must be an object.');
  }

  const rawUpdates = updates as Record<string, unknown>;
  const suppliedKeys = Object.keys(rawUpdates);

  if (suppliedKeys.length === 0) {
    throw new Error('Invalid update payload: at least one mutable field must be provided.');
  }

  // Strictly inspect every key supplied in the update payload
  for (const key of suppliedKeys) {
    if (IMMUTABLE_KNOWLEDGE_FIELDS.has(key)) {
      throw new Error(`Cannot modify immutable field: "${key}".`);
    }
    if (INTERNAL_KNOWLEDGE_FIELDS.has(key)) {
      throw new Error(`Cannot modify internal field: "${key}".`);
    }
    if (!ALLOWED_MUTABLE_KNOWLEDGE_FIELDS.has(key as keyof UpdatePersonalKnowledgeInput)) {
      throw new Error(`Cannot update unsupported field: "${key}".`);
    }
  }

  const hasDefinedMutableField = Array.from(ALLOWED_MUTABLE_KNOWLEDGE_FIELDS).some(function (key) {
    return updates[key] !== undefined;
  });

  if (!hasDefinedMutableField) {
    throw new Error('Invalid update payload: at least one mutable field must be provided.');
  }

  if (updates.category !== undefined && !VALID_KNOWLEDGE_CATEGORIES.has(updates.category)) {
    throw new Error(`Invalid personal knowledge category: "${updates.category}".`);
  }

  if (updates.subject !== undefined && (typeof updates.subject !== 'string' || !updates.subject.trim())) {
    throw new Error('Invalid personal knowledge subject: subject must not be empty.');
  }

  if (updates.predicate !== undefined && (typeof updates.predicate !== 'string' || !updates.predicate.trim())) {
    throw new Error('Invalid personal knowledge predicate: predicate must not be empty.');
  }

  if (updates.value !== undefined && (typeof updates.value !== 'string' || !updates.value.trim())) {
    throw new Error('Invalid personal knowledge value: value must not be empty.');
  }

  if (updates.contextScope !== undefined && typeof updates.contextScope !== 'string') {
    throw new Error('Invalid personal knowledge contextScope: contextScope must be a string if provided.');
  }

  if (updates.stability !== undefined && !VALID_KNOWLEDGE_STABILITIES.has(updates.stability)) {
    throw new Error(`Invalid personal knowledge stability: "${updates.stability}".`);
  }

  if (updates.confidence !== undefined && !VALID_KNOWLEDGE_CONFIDENCES.has(updates.confidence)) {
    throw new Error(`Invalid personal knowledge confidence: "${updates.confidence}".`);
  }

  if (updates.importance !== undefined && !VALID_KNOWLEDGE_IMPORTANCES.has(updates.importance)) {
    throw new Error(`Invalid personal knowledge importance: "${updates.importance}".`);
  }

  if (updates.freshnessState !== undefined && !VALID_FRESHNESS_STATES.has(updates.freshnessState)) {
    throw new Error(`Invalid personal knowledge freshnessState: "${updates.freshnessState}".`);
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.transaction('rw', targetDb.personalKnowledge, async function () {
    const existing = await targetDb.personalKnowledge.get(id);
    if (!existing) {
      throw new Error(`Personal knowledge item #${id} not found.`);
    }

    if (existing.status !== 'active') {
      throw new Error(`Cannot update personal knowledge item #${id} because it is not active (status: "${existing.status}").`);
    }

    const now = new Date();
    const mutation: Partial<PersonalKnowledgeItem> = {
      updatedAt: now,
      editedByUser: true,
    };

    if (updates.category !== undefined) mutation.category = updates.category;
    if (updates.subject !== undefined) mutation.subject = updates.subject.trim();
    if (updates.predicate !== undefined) mutation.predicate = updates.predicate.trim();
    if (updates.value !== undefined) mutation.value = updates.value.trim();
    if (updates.contextScope !== undefined) mutation.contextScope = updates.contextScope.trim();
    if (updates.stability !== undefined) mutation.stability = updates.stability;
    if (updates.confidence !== undefined) mutation.confidence = updates.confidence;
    if (updates.importance !== undefined) mutation.importance = updates.importance;
    if (updates.freshnessState !== undefined) mutation.freshnessState = updates.freshnessState;

    await targetDb.personalKnowledge.update(id, mutation);
  });
}

async function deletePersonalKnowledge(
  id: number,
  reason: InvalidationReason = 'user_manually_rejected',
  targetDb: EchoDatabase = db
): Promise<void> {
  if (typeof id !== 'number' || isNaN(id) || id <= 0) {
    throw new Error('Invalid personal knowledge ID: ID must be a positive number.');
  }

  if (typeof reason !== 'string' || !VALID_INVALIDATION_REASONS.has(reason as InvalidationReason)) {
    throw new Error(`Invalid invalidation reason: "${String(reason)}".`);
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.transaction(
    'rw',
    [targetDb.personalKnowledge, targetDb.knowledgeEvidence],
    async function () {
      const existing = await targetDb.personalKnowledge.get(id);
      if (!existing) {
        throw new Error(`Personal knowledge item #${id} not found.`);
      }

      const now = new Date();

      // 1. Invalidate all linked evidence records atomically
      const linkedEvidence = await targetDb.knowledgeEvidence
        .where('knowledgeId')
        .equals(id)
        .toArray();

      for (const evidence of linkedEvidence) {
        if (evidence.id !== undefined) {
          await targetDb.knowledgeEvidence.update(evidence.id, {
            isValid: false,
            invalidatedAt: now,
            invalidationReason: reason,
          });
        }
      }

      // 2. Soft-delete the fact by setting status to 'archived'
      await targetDb.personalKnowledge.update(id, {
        status: 'archived',
        updatedAt: now,
      });
    }
  );
}

async function saveKnowledgeEvidence(
  evidence: KnowledgeEvidence,
  targetDb: EchoDatabase = db
): Promise<number> {
  if (
    !evidence ||
    typeof evidence.knowledgeId !== 'number' ||
    isNaN(evidence.knowledgeId) ||
    evidence.knowledgeId <= 0 ||
    typeof evidence.messageId !== 'number' ||
    isNaN(evidence.messageId) ||
    evidence.messageId <= 0 ||
    typeof evidence.sourceExcerpt !== 'string' ||
    typeof evidence.isValid !== 'boolean'
  ) {
    throw new Error(
      'Invalid knowledge evidence: knowledgeId and messageId must be positive numbers, and sourceExcerpt and isValid are required.'
    );
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  const normalizedEvidence: KnowledgeEvidence = {
    ...evidence,
    extractedAt: evidence.extractedAt
      ? new Date(evidence.extractedAt)
      : new Date(),
  };

  const id = await targetDb.knowledgeEvidence.add(normalizedEvidence);
  return id as number;
}

async function getKnowledgeEvidence(
  knowledgeId: number,
  targetDb: EchoDatabase = db
): Promise<KnowledgeEvidence[]> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.knowledgeEvidence
    .where('knowledgeId')
    .equals(knowledgeId)
    .toArray();
}

async function getEvidenceByMessageId(
  messageId: number,
  targetDb: EchoDatabase = db
): Promise<KnowledgeEvidence[]> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  return targetDb.knowledgeEvidence
    .where('messageId')
    .equals(messageId)
    .toArray();
}

const DEFAULT_TRAINING_READINESS_THRESHOLD = 25;

interface TrainingReadinessBreakdown {
  nonContent: number;
  code: number;
  terminal: number;
  structuredData: number;
  shortFragment: number;
  weakOrNoStyle: number;
  totalExcluded: number;
}

interface TrainingReadinessResult {
  totalPending: number;
  eligibleCount: number;
  threshold: number;
  isReady: boolean;
  eligibleSamples: WritingMessage[];
  breakdown: TrainingReadinessBreakdown;
}

interface TrainingReadinessOptions {
  threshold?: number;
}

async function getTrainingReadiness(
  thresholdOrOptionsOrDb?: number | TrainingReadinessOptions | EchoDatabase,
  maybeDb?: EchoDatabase
): Promise<TrainingReadinessResult> {
  let threshold = DEFAULT_TRAINING_READINESS_THRESHOLD;
  let targetDb = db;

  if (thresholdOrOptionsOrDb && typeof (thresholdOrOptionsOrDb as EchoDatabase).isOpen === 'function') {
    targetDb = thresholdOrOptionsOrDb as EchoDatabase;
  } else if (typeof thresholdOrOptionsOrDb === 'number') {
    threshold = thresholdOrOptionsOrDb;
    if (maybeDb) {
      targetDb = maybeDb;
    }
  } else if (thresholdOrOptionsOrDb && typeof thresholdOrOptionsOrDb === 'object') {
    if (typeof (thresholdOrOptionsOrDb as TrainingReadinessOptions).threshold === 'number') {
      threshold = (thresholdOrOptionsOrDb as TrainingReadinessOptions).threshold!;
    }
    if (maybeDb) {
      targetDb = maybeDb;
    }
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  // Retrieve writing messages with trainingStatus === 'pending' strictly read-only
  const pendingMessages = await targetDb.writingMessages
    .where('trainingStatus')
    .equals('pending')
    .toArray();

  const breakdown: TrainingReadinessBreakdown = {
    nonContent: 0,
    code: 0,
    terminal: 0,
    structuredData: 0,
    shortFragment: 0,
    weakOrNoStyle: 0,
    totalExcluded: 0,
  };

  const eligibleSamples: WritingMessage[] = [];

  for (const message of pendingMessages) {
    const text = message.text;

    // Canonical single source of truth for style-training eligibility
    if (isEligibleStyleSample(text)) {
      eligibleSamples.push(message);
      continue;
    }

    // Message is excluded — record in diagnostic breakdown counters
    breakdown.totalExcluded++;

    if (!isPotentialStyleSample(text)) {
      breakdown.nonContent++;
      continue;
    }

    const classification = classifyContent(text);

    if (classification.contentType === 'code') {
      breakdown.code++;
    } else if (classification.contentType === 'terminal') {
      breakdown.terminal++;
    } else if (classification.contentType === 'structured_data') {
      breakdown.structuredData++;
    } else if (classification.contentType === 'short_fragment') {
      breakdown.shortFragment++;
    } else {
      breakdown.weakOrNoStyle++;
    }
  }

  return {
    totalPending: pendingMessages.length,
    eligibleCount: eligibleSamples.length,
    threshold,
    isReady: eligibleSamples.length >= threshold,
    eligibleSamples,
    breakdown,
  };
}

export {
  db,
  createEchoDatabase,
  saveWritingMessage,
  getWritingMessages,
  deleteWritingMessage,
  markMessagesTrained,
  savePersonalKnowledge,
  getPersonalKnowledge,
  getPersonalKnowledgeById,
  addManualPersonalKnowledge,
  updatePersonalKnowledge,
  deletePersonalKnowledge,
  saveKnowledgeEvidence,
  getKnowledgeEvidence,
  getEvidenceByMessageId,
  getTrainingReadiness,
  isEligibleStyleSample,
  isEligibleTrainingSample,
  DEFAULT_DUPLICATE_WINDOW_MS,
  DEFAULT_TRAINING_READINESS_THRESHOLD,
  VALID_KNOWLEDGE_CATEGORIES,
  VALID_KNOWLEDGE_STABILITIES,
  VALID_KNOWLEDGE_CONFIDENCES,
  VALID_KNOWLEDGE_IMPORTANCES,
  VALID_FRESHNESS_STATES,
  VALID_INVALIDATION_REASONS,
  ALLOWED_MUTABLE_KNOWLEDGE_FIELDS,
};

export type {
  EchoDatabase,
  WritingMessage,
  PersonalKnowledgeItem,
  KnowledgeEvidence,
  KnowledgeCategory,
  KnowledgeStability,
  KnowledgeStatus,
  FreshnessState,
  KnowledgeConfidence,
  KnowledgeImportance,
  InvalidationReason,
  PersonalKnowledgeFilter,
  ManualPersonalKnowledgeInput,
  UpdatePersonalKnowledgeInput,
  TrainingReadinessResult,
  TrainingReadinessBreakdown,
  TrainingReadinessOptions,
};
