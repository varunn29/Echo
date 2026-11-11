import Dexie, { type EntityTable } from 'dexie';

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

  if (filter?.category && filter?.status) {
    return targetDb.personalKnowledge
      .where('category')
      .equals(filter.category)
      .filter(function (item) {
        return item.status === filter.status;
      })
      .toArray();
  }

  if (filter?.category) {
    return targetDb.personalKnowledge
      .where('category')
      .equals(filter.category)
      .toArray();
  }

  if (filter?.status) {
    return targetDb.personalKnowledge
      .where('status')
      .equals(filter.status)
      .toArray();
  }

  return targetDb.personalKnowledge.toArray();
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

export {
  db,
  createEchoDatabase,
  saveWritingMessage,
  getWritingMessages,
  deleteWritingMessage,
  savePersonalKnowledge,
  getPersonalKnowledge,
  getPersonalKnowledgeById,
  saveKnowledgeEvidence,
  getKnowledgeEvidence,
  getEvidenceByMessageId,
  DEFAULT_DUPLICATE_WINDOW_MS,
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
};
