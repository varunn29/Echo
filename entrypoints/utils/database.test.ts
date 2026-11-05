import 'fake-indexeddb/auto';

import { beforeEach, describe, expect, it } from 'vitest';
import {
  db,
  saveWritingMessage,
  type WritingMessage,
} from './database';

function createMessage(
  text: string,
  platform = 'ChatGPT'
): WritingMessage {
  const currentTime = new Date();

  return {
    platform,
    text,
    createdAt: currentTime,
    capturedAt: currentTime,
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