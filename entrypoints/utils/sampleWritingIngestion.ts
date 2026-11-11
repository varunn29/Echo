/**
 * Echo — Sample Writing Ingestion
 * Provides generic, public writing samples that allow users to test
 * message storage and Ollama personal knowledge extraction out of the box.
 */

import {
  db,
  type EchoDatabase,
  type WritingMessage,
} from './database';
import {
  runKnowledgeExtraction,
  type ExtractionProgressCallback,
  type ExtractionSummary,
} from './knowledgeExtractor';
import type { OllamaClientOptions } from './ollamaClient';

export interface SampleDefinition {
  id: string;
  text: string;
  platform: string;
  category: string;
  capturedAt: string;
}

export interface IngestionResult {
  importedCount: number;
  existingCount: number;
  totalSamples: number;
  messages: WritingMessage[];
}

export interface IngestionAndExtractionSummary {
  ingestion: IngestionResult;
  extraction: ExtractionSummary;
}

/**
 * Curated public developer writing samples for testing and evaluation.
 * Completely generic and free of private personal identifiers.
 */
export const SAMPLE_WRITING_SAMPLES: readonly SampleDefinition[] = [
  {
    id: 'sample_web_dev_1',
    platform: 'Sample Data',
    category: 'technical_skills',
    capturedAt: '2026-11-01T10:00:00Z',
    text: 'I started out learning HTML, CSS, and modern JavaScript, and then moved into frontend frameworks like React and TypeScript. Along the way, I also explored backend technologies like Node.js, Express, and PostgreSQL to understand full-stack application development.',
  },
  {
    id: 'sample_project_1',
    platform: 'Sample Data',
    category: 'project_experience',
    capturedAt: '2026-11-02T11:00:00Z',
    text: 'Recently I built an open-source task management application to help developers track their daily workflow. The most challenging part was designing offline synchronization when network connectivity drops, so I implemented a local-first queue that syncs whenever the connection is restored.',
  },
  {
    id: 'sample_workflow_1',
    platform: 'Sample Data',
    category: 'workflow_habits',
    capturedAt: '2026-11-03T12:00:00Z',
    text: 'Whenever I get stuck on a difficult bug or build error, I usually read the official library documentation and use debugging tools or AI for concise explanations rather than spending hours sifting through random forum posts.',
  },
  {
    id: 'sample_goals_1',
    platform: 'Sample Data',
    category: 'learning_goals',
    capturedAt: '2026-11-04T13:00:00Z',
    text: 'Lately I have been planning to expand my skills into machine learning and cloud architecture. I want to build more intelligent tools that can process natural language and automate repetitive development tasks.',
  },
  {
    id: 'sample_philosophy_1',
    platform: 'Sample Data',
    category: 'developer_philosophy',
    capturedAt: '2026-11-05T14:00:00Z',
    text: 'I believe good software is built by focusing on the fundamentals and understanding how tools work under the hood, rather than just copying templates without understanding the architectural trade-offs.',
  },
] as const;

/**
 * Idempotently imports sample writing messages into writingMessages.
 * Guarantees no duplicates if called repeatedly.
 */
export async function importSampleWritingSamples(
  targetDb: EchoDatabase = db
): Promise<IngestionResult> {
  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  const resultMessages: WritingMessage[] = [];
  let importedCount = 0;
  let existingCount = 0;

  for (const sample of SAMPLE_WRITING_SAMPLES) {
    const trimmedText = sample.text.trim();

    // Check if message already exists with identical text
    const existing = await targetDb.writingMessages
      .filter((m) => m.text.trim() === trimmedText)
      .first();

    if (existing && existing.id !== undefined) {
      existingCount++;
      resultMessages.push(existing);
    } else {
      const capturedDate = new Date(sample.capturedAt);
      const newMsg: WritingMessage = {
        platform: sample.platform,
        text: sample.text,
        createdAt: capturedDate,
        capturedAt: capturedDate,
        trainingStatus: 'pending',
      };

      const newId = await targetDb.writingMessages.add(newMsg);
      newMsg.id = newId as number;
      importedCount++;
      resultMessages.push(newMsg);
    }
  }

  return {
    importedCount,
    existingCount,
    totalSamples: SAMPLE_WRITING_SAMPLES.length,
    messages: resultMessages,
  };
}

/**
 * End-to-end pipeline: imports sample writing messages (idempotently)
 * and processes them through the Ollama knowledge extraction pipeline.
 */
export async function importAndExtractSampleKnowledge(
  options: {
    targetDb?: EchoDatabase;
    ollamaOptions?: OllamaClientOptions;
    onProgress?: ExtractionProgressCallback;
    forceAll?: boolean;
  } = {}
): Promise<IngestionAndExtractionSummary> {
  const targetDb = options.targetDb || db;

  // 1. Ingest sample messages
  const ingestion = await importSampleWritingSamples(targetDb);

  // 2. Run extraction on the imported writing messages
  const extraction = await runKnowledgeExtraction({
    targetDb,
    messages: ingestion.messages,
    ollamaOptions: options.ollamaOptions,
    onProgress: options.onProgress,
    forceAll: options.forceAll,
  });

  return {
    ingestion,
    extraction,
  };
}

