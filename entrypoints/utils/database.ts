
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

export { db };
export type { WritingMessage };
