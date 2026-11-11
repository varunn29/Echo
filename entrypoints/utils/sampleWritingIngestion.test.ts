import 'fake-indexeddb/auto';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Dexie from 'dexie';
import {
  createEchoDatabase,
  type EchoDatabase,
} from './database';
import {
  importSampleWritingSamples,
  SAMPLE_WRITING_SAMPLES,
} from './sampleWritingIngestion';

describe('Sample Writing Ingestion Suite', () => {
  let testDb: EchoDatabase;
  const dbName = `SampleIngestionTest_${Date.now()}`;

  beforeEach(async () => {
    testDb = createEchoDatabase(dbName);
    await testDb.open();
  });

  afterEach(async () => {
    if (testDb.isOpen()) {
      testDb.close();
    }
    await Dexie.delete(dbName);
  });

  it('contains valid, non-empty sample definitions', () => {
    expect(SAMPLE_WRITING_SAMPLES.length).toBeGreaterThan(0);
    for (const sample of SAMPLE_WRITING_SAMPLES) {
      expect(sample.id).toBeTruthy();
      expect(sample.text.trim().length).toBeGreaterThan(20);
      expect(sample.platform).toBe('Sample Data');
    }
  });

  it('imports sample writing messages into an empty database', async () => {
    const result = await importSampleWritingSamples(testDb);
    expect(result.importedCount).toBe(SAMPLE_WRITING_SAMPLES.length);
    expect(result.existingCount).toBe(0);

    const saved = await testDb.writingMessages.toArray();
    expect(saved.length).toBe(SAMPLE_WRITING_SAMPLES.length);
  });

  it('is completely idempotent when executed repeatedly', async () => {
    const firstRun = await importSampleWritingSamples(testDb);
    expect(firstRun.importedCount).toBe(SAMPLE_WRITING_SAMPLES.length);

    const secondRun = await importSampleWritingSamples(testDb);
    expect(secondRun.importedCount).toBe(0);
    expect(secondRun.existingCount).toBe(SAMPLE_WRITING_SAMPLES.length);

    const count = await testDb.writingMessages.count();
    expect(count).toBe(SAMPLE_WRITING_SAMPLES.length);
  });
});

