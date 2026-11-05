
import 'fake-indexeddb/auto';

import { beforeEach, describe, expect, it } from 'vitest';
import {
  db,
  saveWritingMessage,
  getWritingMessages,
  deleteWritingMessage,
  type WritingMessage,
} from './database';

function createMessage(
  text: string,
  platform = 'ChatGPT',
  capturedAt = new Date()
): WritingMessage {
  const currentTime = new Date();

  return {
    platform,
    text,
    createdAt: currentTime,
    capturedAt,
    trainingStatus: 'pending',
  };
}

describe('saveWritingMessage', function () {
  beforeEach(async function () {
    await db.writingMessages.clear();
  });

  it('should save a valid writing message', async function () {
    const message = createMessage('Hello, Echo!');

    const id = await saveWritingMessage(message);

    expect(id).toBeTypeOf('number');

    const savedMessage = await db.writingMessages.get(id!);

    expect(savedMessage?.text).toBe('Hello, Echo!');
  });

  it('should not save duplicate messages', async function () {
    const message = createMessage('Hello, Echo!');

    const firstId = await saveWritingMessage(message);
    const secondId = await saveWritingMessage(message);

    expect(firstId).toBeTypeOf('number');
    expect(secondId).toBeUndefined();

    const count = await db.writingMessages.count();

    expect(count).toBe(1);
  });

  it('should ignore leading and trailing spaces when checking duplicates', async function () {
    await saveWritingMessage(createMessage('Hello, Echo!'));

    const duplicateId = await saveWritingMessage(
      createMessage('  Hello, Echo!  ')
    );

    expect(duplicateId).toBeUndefined();
  });

  it('should reject messages containing only whitespace', async function () {
    await expect(
      saveWritingMessage(createMessage('   '))
    ).rejects.toThrow(
      'Invalid writing message: text must not be empty.'
    );
  });

  it('should allow identical messages from different platforms', async function () {
    await saveWritingMessage(
      createMessage('Hello, Echo!', 'ChatGPT')
    );

    const secondId = await saveWritingMessage(
      createMessage('Hello, Echo!', 'OtherPlatform')
    );

    expect(secondId).toBeTypeOf('number');

    const count = await db.writingMessages.count();

    expect(count).toBe(2);
  });
});

describe('getWritingMessages', function () {
  beforeEach(async function () {
    await db.writingMessages.clear();
  });

  it('should return messages with the newest message first', async function () {
    const oldest = createMessage(
      'Oldest message',
      'ChatGPT',
      new Date('2026-11-01T10:00:00')
    );

    const middle = createMessage(
      'Middle message',
      'ChatGPT',
      new Date('2026-11-02T10:00:00')
    );

    const newest = createMessage(
      'Newest message',
      'ChatGPT',
      new Date('2026-11-03T10:00:00')
    );

    await saveWritingMessage(oldest);
    await saveWritingMessage(middle);
    await saveWritingMessage(newest);

    const messages = await getWritingMessages();

    expect(messages).toHaveLength(3);
    expect(messages[0]?.text).toBe('Newest message');
    expect(messages[1]?.text).toBe('Middle message');
    expect(messages[2]?.text).toBe('Oldest message');
  });

  it('should return an empty array when there are no messages', async function () {
    const messages = await getWritingMessages();

    expect(messages).toEqual([]);
  });
});

describe('deleteWritingMessage', function () {
  beforeEach(async function () {
    await db.writingMessages.clear();
  });

  it('should delete a message using its ID', async function () {
    const id = await saveWritingMessage(
      createMessage('Message to delete')
    );

    await deleteWritingMessage(id!);

    const deletedMessage = await db.writingMessages.get(id!);

    expect(deletedMessage).toBeUndefined();
  });

  it('should not affect other messages when deleting one message', async function () {
    const firstId = await saveWritingMessage(
      createMessage('First message')
    );

    const secondId = await saveWritingMessage(
      createMessage('Second message')
    );

    await deleteWritingMessage(firstId!);

    const remainingMessages = await getWritingMessages();

    expect(remainingMessages).toHaveLength(1);
    expect(remainingMessages[0]?.id).toBe(secondId);
  });
});
