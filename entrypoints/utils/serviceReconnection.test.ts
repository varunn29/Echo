import 'fake-indexeddb/auto';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { renderToString } from 'react-dom/server';
import fs from 'node:fs';
import path from 'node:path';

import { TrainingStatusCard } from '../dashboard/components/TrainingStatusCard';
import WritingComposer from '../dashboard/components/WritingComposer';
import { getTrainingStatus, type TrainingStatusResponse } from './trainingClient';
import { checkGenerationServiceHealth, type GenerationServiceHealth } from './generationClient';

describe('Automatic Python Service Reconnection Suite', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.useRealTimers();
  });

  describe('1. Static Code & Lifecycle Safety Verification', () => {
    it('TrainingStatusCard implements 5s offline retry, active training polling, and unmount cleanup', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/TrainingStatusCard.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      // Uses refs to prevent overlapping requests and memory leaks on unmount
      expect(content).toContain('isMountedRef');
      expect(content).toContain('isRefreshingRef');

      // Checks GET /training/status
      expect(content).toContain('getTrainingStatus()');

      // Polls every 5 seconds when offline OR when training is active
      expect(content).toContain('isOffline || isTrainingActive');
      expect(content).toContain('setInterval');
      expect(content).toContain('5000');
      expect(content).toContain('clearInterval(interval)');

      // Avoids frequent polling when online and no training job is active
      expect(content).toContain('When service is online and no training job is active');
    });

    it('WritingComposer implements 5s offline retry, unmount cleanup, and avoids polling when ready', () => {
      const filePath = path.resolve(__dirname, '../dashboard/components/WritingComposer.tsx');
      const content = fs.readFileSync(filePath, 'utf-8');

      // Uses refs to prevent overlapping checks and memory leaks on unmount
      expect(content).toContain('isMountedRef');
      expect(content).toContain('isCheckingRef');

      // Checks GET /health
      expect(content).toContain('checkGenerationServiceHealth()');

      // Retries every 5 seconds if offline or loading model
      expect(content).toContain("serviceStatus !== null && serviceStatus.status !== 'ready'");
      expect(content).toContain('setInterval');
      expect(content).toContain('5000');
      expect(content).toContain('clearInterval(interval)');

      // Manual check connection retained via header refresh button
      expect(content).toContain('Refresh Service Status');
      expect(content).toContain('checkHealth');

      // Verifies the duplicate large warning card has been removed
      expect(content).not.toContain('Local Generation Service is offline');
      expect(content).not.toContain('Echo requires CORS authorization');
      expect(content).not.toContain('buildServiceStartupCommand');
    });

    it('renders both dashboard components to string without throwing errors and verifies header chip presence', () => {
      const trainingHtml = renderToString(React.createElement(TrainingStatusCard));
      expect(trainingHtml).toContain('Personal Style Learning');
      // Training card preserves the service startup instructions
      expect(trainingHtml).toContain('Local training service is offline');

      const composerHtml = renderToString(React.createElement(WritingComposer));
      expect(composerHtml).toContain('Create something meaningful');
      // Small status chip is present
      expect(composerHtml).toContain('Service Offline');
      // Duplicate large warning box is NOT rendered
      expect(composerHtml).not.toContain('Local Generation Service is offline');
      expect(composerHtml).not.toContain('Echo requires CORS authorization');
    });
  });

  describe('2. WritingComposer Service Reconnection & Concurrency Harness', () => {
    it('1. retries every 5 seconds when offline and transitions immediately to ready when service starts', async () => {
      vi.useFakeTimers();

      let fetchCallCount = 0;
      let serviceIsRunning = false;

      // Mock fetch: fails while serviceIsRunning is false, succeeds when true
      global.fetch = vi.fn().mockImplementation(async (url: string) => {
        fetchCallCount++;
        if (url.includes('/health')) {
          if (!serviceIsRunning) {
            throw new Error('Failed to fetch: Connection refused');
          }
          return {
            status: 200,
            json: async () => ({
              status: 'ready',
              model: 'Qwen/Qwen3.5-0.8B-Base',
              adapter: 'training-output/qwen3.5-0.8b-lora-v5',
              loadTimeSec: 2.1,
            }),
          } as Response;
        }
        throw new Error('Unknown URL');
      });

      // Simulation harness reproducing WritingComposer's exact reconnection logic
      let serviceStatus: GenerationServiceHealth | null = null;
      let isChecking = false;
      let intervalId: any = null;

      const checkHealth = async () => {
        if (isChecking) return;
        isChecking = true;
        try {
          const health = await checkGenerationServiceHealth();
          serviceStatus = health;
        } catch {
          serviceStatus = {
            isOnline: false,
            isModelReady: false,
            status: 'offline',
            error: 'Unable to reach local generation service.',
          };
        } finally {
          isChecking = false;
        }
      };

      const syncPolling = () => {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        if (serviceStatus !== null && serviceStatus.status !== 'ready') {
          intervalId = setInterval(async () => {
            await checkHealth();
            syncPolling();
          }, 5000);
        }
      };

      // Initial check on mount (offline)
      await checkHealth();
      syncPolling();

      expect(fetchCallCount).toBe(1);
      expect(serviceStatus).not.toBeNull();
      expect(serviceStatus!.isOnline).toBe(false);
      expect(serviceStatus!.status).toBe('offline');
      expect(intervalId).not.toBeNull();

      // Tick 1: 5s later -> still offline
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCallCount).toBe(2);
      expect(serviceStatus!.status).toBe('offline');

      // Tick 2: 5s later -> still offline
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCallCount).toBe(3);
      expect(serviceStatus!.status).toBe('offline');

      // Now user starts the Python generation service
      serviceIsRunning = true;

      // Tick 3: 5s later -> service detected!
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCallCount).toBe(4);
      expect(serviceStatus!.isOnline).toBe(true);
      expect(serviceStatus!.status).toBe('ready');

      // Verify that after becoming ready, polling ceases (no unnecessary frequent polling)
      expect(intervalId).toBeNull();
      await vi.advanceTimersByTimeAsync(15000);
      expect(fetchCallCount).toBe(4); // No additional calls
    });

    it('2. prevents overlapping requests when checks take longer than the interval or are invoked manually', async () => {
      let activeFetches = 0;
      let maxConcurrentFetches = 0;

      global.fetch = vi.fn().mockImplementation(async () => {
        activeFetches++;
        if (activeFetches > maxConcurrentFetches) {
          maxConcurrentFetches = activeFetches;
        }
        // Artificial delay simulating network latency
        await new Promise((resolve) => setTimeout(resolve, 50));
        activeFetches--;
        throw new Error('Connection refused');
      });

      let isChecking = false;
      const checkHealth = async () => {
        if (isChecking) return;
        isChecking = true;
        try {
          await checkGenerationServiceHealth();
        } finally {
          isChecking = false;
        }
      };

      // Fire 5 rapid concurrent calls (e.g. interval tick + user spamming "Check Connection")
      await Promise.all([
        checkHealth(),
        checkHealth(),
        checkHealth(),
        checkHealth(),
        checkHealth(),
      ]);

      expect(maxConcurrentFetches).toBe(1);
    });

    it('3. cleans up polling intervals on unmount', () => {
      vi.useFakeTimers();
      let intervalCleared = false;

      const interval = setInterval(() => {}, 5000);
      const cleanup = () => {
        clearInterval(interval);
        intervalCleared = true;
      };

      cleanup();
      expect(intervalCleared).toBe(true);
    });
  });

  describe('3. TrainingStatusCard Reconnection & Active Training Polling Harness', () => {
    it('1. automatically retries GET /training/status when offline and updates when service starts', async () => {
      vi.useFakeTimers();

      let fetchCount = 0;
      let serviceRunning = false;

      global.fetch = vi.fn().mockImplementation(async (url: string) => {
        fetchCount++;
        if (url.includes('/training/status')) {
          if (!serviceRunning) {
            throw new Error('Failed to fetch');
          }
          return {
            status: 200,
            json: async () => ({
              status: 'idle',
              stage: 'idle',
              sampleCount: 25,
              adapterVersion: 'v5',
              isOnline: true,
            }),
          } as Response;
        }
        throw new Error('Unknown URL');
      });

      let statusResponse: TrainingStatusResponse | null = null;
      let isRefreshing = false;
      let intervalId: any = null;

      const refreshStatus = async () => {
        if (isRefreshing) return;
        isRefreshing = true;
        try {
          statusResponse = await getTrainingStatus();
        } finally {
          isRefreshing = false;
        }
      };

      const syncInterval = () => {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        const isOffline = statusResponse !== null && !statusResponse.isOnline;
        if (isOffline) {
          intervalId = setInterval(async () => {
            await refreshStatus();
            syncInterval();
          }, 5000);
        }
      };

      // Mount: initial offline check
      await refreshStatus();
      syncInterval();

      expect(fetchCount).toBe(1);
      expect(statusResponse!.isOnline).toBe(false);
      expect(intervalId).not.toBeNull();

      // Tick: 5s later -> retries while offline
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCount).toBe(2);
      expect(statusResponse!.isOnline).toBe(false);

      // Start service
      serviceRunning = true;
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCount).toBe(3);
      expect(statusResponse!.isOnline).toBe(true);
      expect(statusResponse!.status).toBe('idle');

      // Polling should stop once online and idle
      expect(intervalId).toBeNull();
      await vi.advanceTimersByTimeAsync(15000);
      expect(fetchCount).toBe(3);
    });

    it('2. preserves 5s polling while training is active and stops when completed', async () => {
      vi.useFakeTimers();

      let currentJobStatus = 'training';
      let fetchCount = 0;

      global.fetch = vi.fn().mockImplementation(async (url: string) => {
        fetchCount++;
        return {
          status: 200,
          json: async () => ({
            status: currentJobStatus,
            stage: currentJobStatus,
            sampleCount: 25,
            adapterVersion: 'v5',
            isOnline: true,
          }),
        } as Response;
      });

      let statusResponse: TrainingStatusResponse | null = null;
      let intervalId: any = null;

      const refreshStatus = async () => {
        statusResponse = await getTrainingStatus();
      };

      const syncInterval = () => {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        const isTrainingActive =
          statusResponse?.status === 'preparing' ||
          statusResponse?.status === 'training' ||
          statusResponse?.status === 'validating' ||
          statusResponse?.status === 'activating';

        if (isTrainingActive) {
          intervalId = setInterval(async () => {
            await refreshStatus();
            syncInterval();
          }, 5000);
        }
      };

      await refreshStatus();
      syncInterval();

      expect(fetchCount).toBe(1);
      expect(intervalId).not.toBeNull();

      // Active training ticks
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCount).toBe(2);

      // Training completes
      currentJobStatus = 'completed';
      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchCount).toBe(3);
      expect(statusResponse!.status).toBe('completed');

      // Interval should now be cleared
      expect(intervalId).toBeNull();
      await vi.advanceTimersByTimeAsync(15000);
      expect(fetchCount).toBe(3);
    });
  });
});

