import 'fake-indexeddb/auto';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  createEchoDatabase,
  saveWritingMessage,
  getWritingMessages,
  type EchoDatabase,
  type WritingMessage,
} from './database';
import {
  checkAndTriggerTraining,
  pollActiveTrainingJob,
  getActiveTrainingJob,
  saveActiveTrainingJob,
  clearActiveTrainingJob,
  resetInMemoryStorageForTesting,
  ACTIVE_TRAINING_JOB_STORAGE_KEY,
  type ActiveTrainingJobRecord,
} from './trainingTrigger';
import { DEFAULT_TRAINING_SERVICE_URL } from './trainingClient';

describe('Automatic Training Trigger & Tracking Suite (Phase 1, Step 6)', () => {
  let testDb: EchoDatabase;
  const dbName = `TrainingTriggerTest_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  // Sample natural writing generator (guaranteed to pass contentClassifier styleSignal: 'strong')
  const createEligibleSample = (index: number): string =>
    `Personal reflection sample ${String(index).padStart(2, '0')}: I have noticed that when we document our architectural boundaries clearly, our velocity increases and code reviews become significantly more productive. We should maintain this engineering standard across our core repositories.`;

  beforeEach(async () => {
    testDb = createEchoDatabase(dbName);
    await testDb.open();
    resetInMemoryStorageForTesting();
    vi.restoreAllMocks();
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    if (testDb.isOpen()) {
      await testDb.delete();
    }
  });

  // Helper to populate database with N eligible messages
  async function populateEligibleMessages(count: number): Promise<number[]> {
    const ids: number[] = [];
    const baseTime = new Date('2026-03-01T10:00:00Z');

    for (let i = 1; i <= count; i++) {
      const capturedAt = new Date(baseTime.getTime() + i * 120000);
      const id = await saveWritingMessage(
        {
          platform: 'ChatGPT',
          text: createEligibleSample(i),
          createdAt: capturedAt,
          capturedAt: capturedAt,
          trainingStatus: 'pending',
        },
        0,
        testDb
      );
      if (id !== undefined) {
        ids.push(id);
      }
    }
    return ids;
  }

  // ---------------------------------------------------------------------------
  // 1. Fewer than 25 eligible samples: no training request
  // ---------------------------------------------------------------------------
  it('1. should not submit a training request when fewer than 25 eligible samples exist', async () => {
    await populateEligibleMessages(24); // 24 samples

    const fetchSpy = vi.spyOn(globalThis, 'fetch');

    const result = await checkAndTriggerTraining({
      targetDb: testDb,
      threshold: 25,
    });

    expect(result.action).toBe('insufficient_samples');
    expect(result.eligibleCount).toBe(24);
    expect(result.threshold).toBe(25);

    // Verify fetch was never invoked for /train
    expect(fetchSpy).not.toHaveBeenCalled();

    // Verify all messages remain pending
    const messages = await getWritingMessages(testDb);
    expect(messages.length).toBe(24);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 2. Exactly 25 eligible samples: one training request
  // ---------------------------------------------------------------------------
  it('2. should submit exactly one training request when 25 eligible samples exist', async () => {
    await populateEligibleMessages(25);

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      const urlStr = String(url);
      if (urlStr.endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle', jobId: null }), { status: 200 });
      }
      if (urlStr.endsWith('/train') && init?.method === 'POST') {
        return new Response(
          JSON.stringify({
            status: 'accepted',
            jobId: 'train_test_job_001',
            sampleCount: 25,
            message: 'Training job accepted.',
          }),
          { status: 202 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    const result = await checkAndTriggerTraining({
      targetDb: testDb,
      threshold: 25,
    });

    expect(result.action).toBe('submitted');
    expect(result.jobId).toBe('train_test_job_001');
    expect(result.submittedCount).toBe(25);

    // Verify POST /train was called exactly once
    const trainCalls = fetchSpy.mock.calls.filter(([url, init]) =>
      String(url).endsWith('/train') && init?.method === 'POST'
    );
    expect(trainCalls.length).toBe(1);

    const requestBody = JSON.parse(String(trainCalls[0]?.[1]?.body ?? '{}'));
    expect(requestBody.samples.length).toBe(25);
  });

  // ---------------------------------------------------------------------------
  // 3. Ineligible samples are never submitted
  // ---------------------------------------------------------------------------
  it('3. should exclude code, terminal, structured data, short fragments, and weak style samples from submission', async () => {
    await populateEligibleMessages(25);

    // Add ineligible noise samples
    const now = new Date();
    await saveWritingMessage(
      {
        platform: 'ChatGPT',
        text: 'const express = require("express");\nconst app = express();\napp.listen(3000);',
        createdAt: now,
        capturedAt: now,
        trainingStatus: 'pending',
      },
      0,
      testDb
    );
    await saveWritingMessage(
      {
        platform: 'ChatGPT',
        text: '$ npm test -- --run\n$ git commit -m "update"',
        createdAt: now,
        capturedAt: now,
        trainingStatus: 'pending',
      },
      0,
      testDb
    );
    await saveWritingMessage(
      {
        platform: 'ChatGPT',
        text: '{"name": "test", "status": "active", "version": "1.0.0"}',
        createdAt: now,
        capturedAt: now,
        trainingStatus: 'pending',
      },
      0,
      testDb
    );
    await saveWritingMessage(
      {
        platform: 'ChatGPT',
        text: 'ok thanks',
        createdAt: now,
        capturedAt: now,
        trainingStatus: 'pending',
      },
      0,
      testDb
    );

    let submittedPayload: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      const urlStr = String(url);
      if (urlStr.endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      if (urlStr.endsWith('/train')) {
        submittedPayload = JSON.parse(String(init?.body));
        return new Response(
          JSON.stringify({ status: 'accepted', jobId: 'job_clean_samples', sampleCount: 25 }),
          { status: 202 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    const result = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(result.action).toBe('submitted');

    expect(submittedPayload.samples.length).toBe(25);
    for (const sample of submittedPayload.samples) {
      expect(sample).not.toContain('require("express")');
      expect(sample).not.toContain('git commit');
      expect(sample).not.toContain('"version":');
      expect(sample).not.toBe('ok thanks');
    }
  });

  // ---------------------------------------------------------------------------
  // 4. HTTP 202 does not mark samples trained
  // ---------------------------------------------------------------------------
  it('4. should leave all samples as pending when HTTP 202 Accepted is returned', async () => {
    await populateEligibleMessages(25);

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      return new Response(
        JSON.stringify({ status: 'accepted', jobId: 'job_accepted_202' }),
        { status: 202 }
      );
    });

    const result = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(result.action).toBe('submitted');

    // Crucial safety invariant: samples MUST remain pending!
    const messages = await getWritingMessages(testDb);
    expect(messages.length).toBe(25);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);

    // Active job must be saved with status 'active'
    const activeJob = await getActiveTrainingJob();
    expect(activeJob).not.toBeNull();
    expect(activeJob?.status).toBe('active');
    expect(activeJob?.submittedMessageIds.length).toBe(25);
  });

  // ---------------------------------------------------------------------------
  // 5. Successful completion marks only submitted message IDs as trained
  // ---------------------------------------------------------------------------
  it('5. should mark only submitted message IDs as trained after service reports completion', async () => {
    const ids = await populateEligibleMessages(25);

    // Mock initial submission
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      return new Response(
        JSON.stringify({ status: 'accepted', jobId: 'job_to_complete' }),
        { status: 202 }
      );
    });

    await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });

    // Verify still pending before completion
    let messages = await getWritingMessages(testDb);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);

    // Now mock training service completing successfully
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(
          JSON.stringify({
            status: 'completed',
            jobId: 'job_to_complete',
            adapterVersion: 'v3',
            completedAt: '2026-03-01 12:00:00',
          }),
          { status: 200 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    // Poll active job
    const completedJob = await pollActiveTrainingJob({ targetDb: testDb });
    expect(completedJob?.status).toBe('completed');
    expect(completedJob?.adapterVersion).toBe('v3');

    // All 25 submitted messages must now be marked trained
    messages = await getWritingMessages(testDb);
    expect(messages.length).toBe(25);
    expect(messages.every(m => m.trainingStatus === 'trained')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 6. Messages collected during training remain pending
  // ---------------------------------------------------------------------------
  it('6. should keep messages collected during active training as pending when the job completes', async () => {
    await populateEligibleMessages(25);

    // Submit job (IDs 1..25)
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      return new Response(
        JSON.stringify({ status: 'accepted', jobId: 'job_with_midflight_msg' }),
        { status: 202 }
      );
    });

    await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });

    // A 26th message is saved by user during active training!
    const midflightTime = new Date('2026-03-01T11:00:00Z');
    const newMsgId = await saveWritingMessage(
      {
        platform: 'ChatGPT',
        text: 'New personal thought captured while background training was actively running on the GPU.',
        createdAt: midflightTime,
        capturedAt: midflightTime,
        trainingStatus: 'pending',
      },
      0,
      testDb
    );
    expect(newMsgId).toBeDefined();

    // Now training job completes
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(
          JSON.stringify({
            status: 'completed',
            jobId: 'job_with_midflight_msg',
            adapterVersion: 'v3',
          }),
          { status: 200 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    await pollActiveTrainingJob({ targetDb: testDb });

    // Query messages: exactly 25 trained, and exactly 1 pending (the midflight message)!
    const messages = await getWritingMessages(testDb);
    expect(messages.length).toBe(26);

    const trained = messages.filter(m => m.trainingStatus === 'trained');
    const pending = messages.filter(m => m.trainingStatus === 'pending');

    expect(trained.length).toBe(25);
    expect(pending.length).toBe(1);
    expect(pending[0]?.id).toBe(newMsgId);
  });

  // ---------------------------------------------------------------------------
  // 7. Training failure leaves submitted samples pending
  // ---------------------------------------------------------------------------
  it('7. should preserve all samples as pending when training fails', async () => {
    await populateEligibleMessages(25);

    // Initial submission
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      return new Response(
        JSON.stringify({ status: 'accepted', jobId: 'job_that_will_fail' }),
        { status: 202 }
      );
    });

    await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });

    // Mock failure status from service
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(
          JSON.stringify({
            status: 'error',
            jobId: 'job_that_will_fail',
            error: 'CUDA Out of Memory during backprop',
          }),
          { status: 200 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    const failedJob = await pollActiveTrainingJob({ targetDb: testDb });
    expect(failedJob?.status).toBe('failed');
    expect(failedJob?.error).toContain('CUDA Out of Memory');

    // Samples must remain pending for future retry
    const messages = await getWritingMessages(testDb);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 8. HTTP 409 is handled as an existing active job
  // ---------------------------------------------------------------------------
  it('8. should handle HTTP 409 Conflict by tracking the existing active job instead of failing', async () => {
    await populateEligibleMessages(25);

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
      }
      return new Response(
        JSON.stringify({
          status: 'conflict',
          error: 'A training job is already in progress.',
          activeJobId: 'train_external_job_555',
        }),
        { status: 409 }
      );
    });

    const result = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(result.action).toBe('active_job_pending');
    expect(result.jobId).toBe('train_external_job_555');

    // Storage must track this active job
    const activeJob = await getActiveTrainingJob();
    expect(activeJob?.jobId).toBe('train_external_job_555');
    expect(activeJob?.status).toBe('active');

    // Samples remain pending
    const messages = await getWritingMessages(testDb);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 9. Service unavailable does not lose or mark samples
  // ---------------------------------------------------------------------------
  it('9. should gracefully handle service offline without losing or marking samples', async () => {
    await populateEligibleMessages(25);

    // Mock service offline (fetch rejection)
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Failed to fetch (Connection refused)'));

    const result = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(result.action).toBe('service_offline');
    expect(result.eligibleCount).toBe(25);

    // Verify all 25 samples remain untouched in database
    const messages = await getWritingMessages(testDb);
    expect(messages.length).toBe(25);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 10. Service-worker/dashboard restart recovers the active job
  // ---------------------------------------------------------------------------
  it('10. should recover in-flight active job across simulated service-worker restarts', async () => {
    const ids = await populateEligibleMessages(25);

    // Persist active job in storage (simulating state before dormancy/restart)
    const storedJob: ActiveTrainingJobRecord = {
      jobId: 'train_job_across_restart',
      submittedMessageIds: ids,
      submittedAt: '2026-03-01T10:00:00Z',
      sampleCount: 25,
      status: 'active',
      lastCheckedAt: '2026-03-01T10:00:00Z',
      error: null,
    };
    await saveActiveTrainingJob(storedJob);

    // Service is still actively training
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(
          JSON.stringify({
            status: 'training',
            jobId: 'train_job_across_restart',
            stage: 'training',
          }),
          { status: 200 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    // Recover job
    const recovered = await pollActiveTrainingJob({ targetDb: testDb });
    expect(recovered?.jobId).toBe('train_job_across_restart');
    expect(recovered?.status).toBe('active');

    // Samples remain pending while still training
    let messages = await getWritingMessages(testDb);
    expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);

    // Later: job completes
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).endsWith('/training/status')) {
        return new Response(
          JSON.stringify({
            status: 'completed',
            jobId: 'train_job_across_restart',
            adapterVersion: 'v4',
          }),
          { status: 200 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    const completed = await pollActiveTrainingJob({ targetDb: testDb });
    expect(completed?.status).toBe('completed');

    messages = await getWritingMessages(testDb);
    expect(messages.every(m => m.trainingStatus === 'trained')).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 11. Repeated triggers do not submit duplicate jobs
  // ---------------------------------------------------------------------------
  it('11. should prevent duplicate submissions on repeated triggers while a job is active', async () => {
    await populateEligibleMessages(25);

    let submitCount = 0;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const urlStr = String(url);
      if (urlStr.endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: submitCount > 0 ? 'training' : 'idle' }), {
          status: 200,
        });
      }
      if (urlStr.endsWith('/train')) {
        submitCount++;
        return new Response(
          JSON.stringify({ status: 'accepted', jobId: 'job_single_submission' }),
          { status: 202 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    // First trigger
    const firstResult = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(firstResult.action).toBe('submitted');
    expect(submitCount).toBe(1);

    // Second trigger immediately after
    const secondResult = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
    expect(secondResult.action).toBe('active_job_pending');
    expect(submitCount).toBe(1); // Crucial: submitCount did NOT increment!
  });

  // ---------------------------------------------------------------------------
  // 12. Concurrent triggers submit only one job
  // ---------------------------------------------------------------------------
  it('12. should submit only one job when concurrent triggers execute simultaneously', async () => {
    await populateEligibleMessages(25);

    let submitCount = 0;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const urlStr = String(url);
      if (urlStr.endsWith('/training/status')) {
        return new Response(JSON.stringify({ status: submitCount > 0 ? 'training' : 'idle' }), {
          status: 200,
        });
      }
      if (urlStr.endsWith('/train')) {
        submitCount++;
        return new Response(
          JSON.stringify({ status: 'accepted', jobId: 'job_concurrent_safety' }),
          { status: 202 }
        );
      }
      return new Response('Not Found', { status: 404 });
    });

    // Fire 5 concurrent triggers simultaneously
    const results = await Promise.all([
      checkAndTriggerTraining({ targetDb: testDb, threshold: 25 }),
      checkAndTriggerTraining({ targetDb: testDb, threshold: 25 }),
      checkAndTriggerTraining({ targetDb: testDb, threshold: 25 }),
      checkAndTriggerTraining({ targetDb: testDb, threshold: 25 }),
      checkAndTriggerTraining({ targetDb: testDb, threshold: 25 }),
    ]);

    const submitted = results.filter(r => r.action === 'submitted');
    const pending = results.filter(r => r.action === 'active_job_pending');

    expect(submitted.length).toBe(1);
    expect(pending.length).toBe(4);
    expect(submitCount).toBe(1); // Exactly one HTTP submission!
  });

  // ===========================================================================
  // Step 6 Audit Suite: Recovery and Trigger Scenarios
  // ===========================================================================
  describe('Phase 1, Step 6 Focused Audit: Service Recovery, Restarts, and Follow-up', () => {
    it('Audit 1: should automatically submit training when service becomes available without new writing messages', async () => {
      await populateEligibleMessages(25);

      // 1. Service is offline
      vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Connection refused'));

      const offlineResult = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(offlineResult.action).toBe('service_offline');

      // Verify all 25 samples remain pending
      let messages = await getWritingMessages(testDb);
      expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);

      // 2. User starts the Python service (no new ChatGPT messages added)
      let trainCallMade = false;
      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
        const urlStr = String(url);
        if (urlStr.endsWith('/training/status')) {
          return new Response(JSON.stringify({ status: 'idle', jobId: null }), { status: 200 });
        }
        if (urlStr.endsWith('/train') && init?.method === 'POST') {
          trainCallMade = true;
          return new Response(
            JSON.stringify({ status: 'accepted', jobId: 'job_after_recovery', sampleCount: 25 }),
            { status: 202 }
          );
        }
        return new Response('Not Found', { status: 404 });
      });

      // 3. Retry trigger fires (e.g., via alarm or periodic check)
      const recoveryResult = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(recoveryResult.action).toBe('submitted');
      expect(recoveryResult.jobId).toBe('job_after_recovery');
      expect(trainCallMade).toBe(true);

      // Verify active job stored
      const activeJob = await getActiveTrainingJob();
      expect(activeJob?.jobId).toBe('job_after_recovery');
      expect(activeJob?.status).toBe('active');
    });

    it('Audit 2: should detect lost in-memory job after Python service restarts, leaving all samples pending', async () => {
      const ids = await populateEligibleMessages(25);

      // Simulate active job submitted before service crash
      const preCrashJob: ActiveTrainingJobRecord = {
        jobId: 'job_lost_on_crash',
        submittedMessageIds: ids,
        submittedAt: new Date().toISOString(),
        sampleCount: 25,
        status: 'active',
        lastCheckedAt: new Date().toISOString(),
      };
      await saveActiveTrainingJob(preCrashJob);

      // Service restarts and is now idle with NO knowledge of job_lost_on_crash
      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
        if (String(url).endsWith('/training/status')) {
          return new Response(
            JSON.stringify({
              status: 'idle',
              jobId: null,
              stage: 'idle',
              sampleCount: 0,
              adapterVersion: null,
            }),
            { status: 200 }
          );
        }
        return new Response('Not Found', { status: 404 });
      });

      // Poll active job
      const polled = await pollActiveTrainingJob({ targetDb: testDb });

      // Must detect that the job was lost due to restart
      expect(polled?.status).toBe('failed');
      expect(polled?.error).toContain('lost due to service restart');

      // Crucial Safety Invariant: Samples must NOT be marked trained!
      const messages = await getWritingMessages(testDb);
      expect(messages.length).toBe(25);
      expect(messages.every(m => m.trainingStatus === 'pending')).toBe(true);
    });

    it('Audit 3: should allow a subsequent retry to successfully submit those pending samples after a service restart', async () => {
      const ids = await populateEligibleMessages(25);

      // Record was marked failed due to service restart
      const lostJob: ActiveTrainingJobRecord = {
        jobId: 'job_lost_before_restart',
        submittedMessageIds: ids,
        submittedAt: new Date().toISOString(),
        sampleCount: 25,
        status: 'failed',
        failedAt: new Date().toISOString(),
        error: 'Training job job_lost_before_restart was lost due to service restart. Samples retained for retry.',
        lastCheckedAt: new Date().toISOString(),
      };
      await saveActiveTrainingJob(lostJob);

      // Service is online and idle, ready for new training jobs
      let submittedNewJob = false;
      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
        const urlStr = String(url);
        if (urlStr.endsWith('/training/status')) {
          return new Response(JSON.stringify({ status: 'idle', jobId: null }), { status: 200 });
        }
        if (urlStr.endsWith('/train') && init?.method === 'POST') {
          submittedNewJob = true;
          return new Response(
            JSON.stringify({ status: 'accepted', jobId: 'job_restart_retry_success', sampleCount: 25 }),
            { status: 202 }
          );
        }
        return new Response('Not Found', { status: 404 });
      });

      // Subsequent retry should NOT be blocked by cooldown and should submit cleanly
      const retryResult = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(retryResult.action).toBe('submitted');
      expect(retryResult.jobId).toBe('job_restart_retry_success');
      expect(submittedNewJob).toBe(true);

      const activeJob = await getActiveTrainingJob();
      expect(activeJob?.jobId).toBe('job_restart_retry_success');
      expect(activeJob?.status).toBe('active');
    });

    it('Audit 4: should automatically start Job B when 25 new messages were collected during Job A', async () => {
      // 1. Batch 1 (IDs 1..25)
      const batch1Ids = await populateEligibleMessages(25);

      // Start Job 1
      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
        if (String(url).endsWith('/training/status')) {
          return new Response(JSON.stringify({ status: 'idle' }), { status: 200 });
        }
        return new Response(
          JSON.stringify({ status: 'accepted', jobId: 'job_1_first_cycle', sampleCount: 25 }),
          { status: 202 }
        );
      });

      const job1Result = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(job1Result.action).toBe('submitted');
      expect(job1Result.jobId).toBe('job_1_first_cycle');

      // 2. User writes 25 more messages during Job 1 (IDs 26..50)
      const baseTime = new Date('2026-03-01T15:00:00Z');
      const batch2Ids: number[] = [];
      for (let i = 26; i <= 50; i++) {
        const capturedAt = new Date(baseTime.getTime() + i * 60000);
        const id = await saveWritingMessage(
          {
            platform: 'ChatGPT',
            text: createEligibleSample(i),
            createdAt: capturedAt,
            capturedAt: capturedAt,
            trainingStatus: 'pending',
          },
          0,
          testDb
        );
        if (id) batch2Ids.push(id);
      }
      expect(batch2Ids.length).toBe(25);

      // 3. Mock Job 1 completing, and service accepting Job 2
      let job2Submitted = false;
      let job2Payload: any = null;

      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
        const urlStr = String(url);
        if (urlStr.endsWith('/training/status')) {
          // Reports Job 1 completed
          return new Response(
            JSON.stringify({
              status: 'completed',
              jobId: 'job_1_first_cycle',
              adapterVersion: 'v3',
              completedAt: '2026-03-01 16:00:00',
            }),
            { status: 200 }
          );
        }
        if (urlStr.endsWith('/train') && init?.method === 'POST') {
          job2Submitted = true;
          job2Payload = JSON.parse(String(init?.body));
          return new Response(
            JSON.stringify({ status: 'accepted', jobId: 'job_2_second_cycle', sampleCount: 25 }),
            { status: 202 }
          );
        }
        return new Response('Not Found', { status: 404 });
      });

      // 4. Poll Job 1 with autoTriggerFollowUp enabled
      const completedJob1 = await pollActiveTrainingJob({
        targetDb: testDb,
        autoTriggerFollowUp: true,
      });
      expect(completedJob1?.status).toBe('completed');
      expect(completedJob1?.jobId).toBe('job_1_first_cycle');

      // 5. Verify Batch 1 messages are now marked 'trained'
      const allMessages = await getWritingMessages(testDb);
      const batch1Messages = allMessages.filter(m => m.id && batch1Ids.includes(m.id));
      expect(batch1Messages.every(m => m.trainingStatus === 'trained')).toBe(true);

      // 6. Verify Job 2 was automatically submitted for Batch 2!
      expect(job2Submitted).toBe(true);
      expect(job2Payload.samples.length).toBe(25);

      // Batch 2 messages must still be 'pending' because Job 2 is in-flight
      const batch2Messages = allMessages.filter(m => m.id && batch2Ids.includes(m.id));
      expect(batch2Messages.every(m => m.trainingStatus === 'pending')).toBe(true);

      // Storage now reflects active Job 2
      const activeJob = await getActiveTrainingJob();
      expect(activeJob?.jobId).toBe('job_2_second_cycle');
      expect(activeJob?.status).toBe('active');
    });

    it('Audit 5: should prevent duplicate submission or overlapping training jobs during active execution', async () => {
      await populateEligibleMessages(25);

      let submitCallCount = 0;
      vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
        const urlStr = String(url);
        if (urlStr.endsWith('/training/status')) {
          return new Response(
            JSON.stringify({
              status: submitCallCount > 0 ? 'training' : 'idle',
              jobId: submitCallCount > 0 ? 'job_in_progress' : null,
            }),
            { status: 200 }
          );
        }
        if (urlStr.endsWith('/train') && init?.method === 'POST') {
          submitCallCount++;
          return new Response(
            JSON.stringify({ status: 'accepted', jobId: 'job_in_progress', sampleCount: 25 }),
            { status: 202 }
          );
        }
        return new Response('Not Found', { status: 404 });
      });

      // First trigger starts the job
      const res1 = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(res1.action).toBe('submitted');
      expect(submitCallCount).toBe(1);

      // Multiple triggers while job is in progress
      const res2 = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(res2.action).toBe('active_job_pending');

      const res3 = await checkAndTriggerTraining({ targetDb: testDb, threshold: 25 });
      expect(res3.action).toBe('active_job_pending');

      // Verify no duplicate submission occurred
      expect(submitCallCount).toBe(1);
    });
  });
});
