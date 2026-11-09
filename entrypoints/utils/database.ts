
import Dexie, { type EntityTable } from 'dexie';

interface WritingMessage {
  id?: number;
  platform: string;
  text: string;
  createdAt: Date;
  capturedAt: Date;
  trainingStatus: 'pending' | 'trained';
}

const db = new Dexie('EchoDatabase') as Dexie & {
  writingMessages: EntityTable<WritingMessage, 'id'>;
};

db.version(1).stores({
  writingMessages: '++id, platform, capturedAt, trainingStatus',
});

const DEFAULT_DUPLICATE_WINDOW_MS = 60 * 1000; // 60 seconds

async function saveWritingMessage(
  message: WritingMessage,
  duplicateWindowMs: number = DEFAULT_DUPLICATE_WINDOW_MS
): Promise<number | undefined> {
  if (
    typeof message.text !== 'string' ||
    !message.text.trim()
  ) {
    throw new Error('Invalid writing message: text must not be empty.');
  }

  if (!db.isOpen()) {
    await db.open();
  }

  return db.transaction('rw', db.writingMessages, async function () {
    const normalizedText = message.text.trim();
    const messageCapturedTime = new Date(message.capturedAt).getTime();

    // Check if an identical message on the same platform was captured within the deduplication window.
    // Messages captured outside this window are preserved as intentional repetitions.
    const recentDuplicate = await db.writingMessages
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

    return db.writingMessages.add(message);
  });
}

async function getWritingMessages(): Promise<WritingMessage[]> {
  if (!db.isOpen()) {
    await db.open();
  }

  return db.writingMessages
    .orderBy('capturedAt')
    .reverse()
    .toArray();
}

async function deleteWritingMessage(id: number): Promise<void> {
  if (!db.isOpen()) {
    await db.open();
  }

  await db.writingMessages.delete(id);
}

export {
  db,
  saveWritingMessage,
  getWritingMessages,
  deleteWritingMessage,
  DEFAULT_DUPLICATE_WINDOW_MS,
};

export type { WritingMessage };
