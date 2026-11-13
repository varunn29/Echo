/**
 * Echo — Local Generation Service Client
 * Handles communication with the persistent local LoRA generation service running at 127.0.0.1:11435.
 */

export interface GenerationServiceHealth {
  isOnline: boolean;
  isModelReady: boolean;
  status: 'ready' | 'loading' | 'offline' | 'error';
  model?: string;
  adapter?: string;
  loadTimeSec?: number;
  residentRamMb?: number;
  uptimeSec?: number;
  error?: string;
}

export type GenerationLengthMode = 'short' | 'medium' | 'long';
export type GenerationPlatform = 'X' | 'LinkedIn' | 'General';

export interface GenerationRequest {
  topic: string;
  personalKnowledge?: string[];
  leadIn?: string;
  mode?: 'sampled' | 'greedy';
  lengthMode?: GenerationLengthMode;
  platform?: GenerationPlatform | string;
  maxTokens?: number;
  temperature?: number;
  repetitionPenalty?: number;
  seed?: number;
}

export interface GenerationMetadata {
  topic: string;
  mode: string;
  lengthMode?: string;
  platform?: string;
  model: string;
  adapter: string;
  repetitionPenalty: number;
  maxTokens: number;
  source?: string;
  seed?: number;
  tokenCount?: number;
  finalTokenCount?: number;
  rawTokenCount?: number;
}

export interface GroundingCheckResult {
  isGrounded: boolean;
  flaggedIssues?: string[];
  fallbackUsed?: boolean;
}

export interface GenerationResponse {
  completeDraft: string;
  generatedText: string;
  rawGeneratedText?: string;
  leadIn?: string;
  platform?: string;
  tokensGenerated?: number;
  tokenCount?: number;
  finalTokenCount?: number;
  rawTokenCount?: number;
  wordCount?: number;
  sentenceCount?: number;
  paragraphCount?: number;
  source?: 'lora_sampled' | 'lora_greedy' | 'base_model' | 'zero_fact_safe_response' | 'grounded_fallback';
  elapsedSec?: number;
  tokensPerSec?: number;
  peakRamMb?: number;
  metadata?: GenerationMetadata;
  groundingCheck?: GroundingCheckResult;
}

export interface GenerationClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
}

export const DEFAULT_GENERATION_SERVICE_URL = 'http://127.0.0.1:11435';
export const DEFAULT_GENERATION_TIMEOUT_MS = 60000; // 60s for CPU inference
export const DEFAULT_HEALTH_TIMEOUT_MS = 3000; // 3s for health checks

/**
 * Checks whether the local generation service is reachable and the model is resident in memory.
 */
export async function checkGenerationServiceHealth(
  baseUrl: string = DEFAULT_GENERATION_SERVICE_URL,
  timeoutMs: number = DEFAULT_HEALTH_TIMEOUT_MS
): Promise<GenerationServiceHealth> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: controller.signal,
    });

    if (res.status === 200) {
      const data = await res.json();
      return {
        isOnline: true,
        isModelReady: data.status === 'ready',
        status: data.status === 'ready' ? 'ready' : 'loading',
        model: data.model,
        adapter: data.adapter,
        loadTimeSec: data.loadTimeSec,
        residentRamMb: data.residentRamMb,
        uptimeSec: data.uptimeSec,
      };
    }

    if (res.status === 503) {
      return {
        isOnline: true,
        isModelReady: false,
        status: 'loading',
        error: 'Model is currently loading or unavailable. Please wait a moment.',
      };
    }

    return {
      isOnline: true,
      isModelReady: false,
      status: 'error',
      error: `Service responded with status ${res.status}.`,
    };
  } catch (err: unknown) {
    const isTimeout = err instanceof Error && err.name === 'AbortError';
    return {
      isOnline: false,
      isModelReady: false,
      status: 'offline',
      error: isTimeout
        ? 'Health check timed out. Service might be busy or unreachable.'
        : 'Local generation service is offline. Please start it at 127.0.0.1:11435.',
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Sends a generation request to the local LoRA service.
 */
export async function generatePersonalizedDraft(
  request: GenerationRequest,
  options: GenerationClientOptions = {}
): Promise<GenerationResponse> {
  const {
    baseUrl = DEFAULT_GENERATION_SERVICE_URL,
    timeoutMs = DEFAULT_GENERATION_TIMEOUT_MS,
  } = options;

  // Client-side validation matching service boundaries
  if (!request || typeof request.topic !== 'string' || !request.topic.trim()) {
    throw new Error('A non-empty topic is required to generate a draft.');
  }

  const trimmedTopic = request.topic.trim();
  if (trimmedTopic.length > 500) {
    throw new Error('Topic exceeds the maximum allowed length of 500 characters.');
  }

  const knowledge = request.personalKnowledge ?? [];
  if (!Array.isArray(knowledge)) {
    throw new Error('Personal knowledge must be an array of strings.');
  }
  if (knowledge.length > 10) {
    throw new Error('At most 10 personal knowledge items can be supplied.');
  }
  for (const item of knowledge) {
    if (typeof item !== 'string' || item.length > 500) {
      throw new Error('Each personal knowledge item must be a string of at most 500 characters.');
    }
  }

  const leadIn = (request.leadIn ?? '').trim();
  if (leadIn.length > 100) {
    throw new Error('Lead-in exceeds the maximum allowed length of 100 characters.');
  }

  const platform = request.platform ?? 'General';
  if (typeof platform !== 'string') {
    throw new Error('Platform must be a string.');
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const payload = {
      topic: trimmedTopic,
      personalKnowledge: knowledge,
      leadIn: leadIn,
      mode: request.mode ?? 'sampled',
      lengthMode: request.lengthMode ?? 'medium',
      platform,
      maxTokens: request.maxTokens,
      temperature: request.temperature ?? 0.65,
      repetitionPenalty: request.repetitionPenalty ?? 1.15,
      seed: request.seed ?? 42,
    };

    const res = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (res.status === 200) {
      let data: any;
      try {
        data = await res.json();
      } catch {
        throw new Error('Received an unparseable response from the generation service.');
      }

      if (!data || typeof data.completeDraft !== 'string') {
        throw new Error('Invalid response structure: missing complete draft.');
      }

      return {
        completeDraft: data.completeDraft,
        generatedText: data.generatedText ?? '',
        rawGeneratedText: data.rawGeneratedText,
        leadIn: data.leadIn ?? '',
        platform: data.platform ?? data.metadata?.platform ?? platform,
        tokensGenerated: data.tokensGenerated,
        tokenCount: data.tokenCount,
        finalTokenCount: data.finalTokenCount,
        rawTokenCount: data.rawTokenCount,
        wordCount: data.wordCount,
        sentenceCount: data.sentenceCount,
        paragraphCount: data.paragraphCount,
        source: data.source,
        elapsedSec: data.elapsedSec,
        tokensPerSec: data.tokensPerSec,
        peakRamMb: data.peakRamMb,
        metadata: data.metadata,
        groundingCheck: data.groundingCheck,
      };
    }

    // Handle sanitized error responses
    let errorMessage = `Generation service responded with HTTP ${res.status}`;
    try {
      const errorJson = await res.json();
      if (errorJson && typeof errorJson.error === 'string') {
        errorMessage = errorJson.error;
      }
    } catch {
      // Keep default errorMessage
    }

    if (res.status === 503) {
      throw new Error('Model is currently loading or unavailable. Please wait a moment and try again.');
    }
    if (res.status === 403) {
      throw new Error('Generation request was rejected by service security (unauthorized origin).');
    }
    if (res.status === 413) {
      throw new Error('Generation request payload is too large.');
    }

    throw new Error(errorMessage);
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.name === 'AbortError') {
        throw new Error('Generation request timed out. The local model took too long to respond.');
      }
      throw err;
    }
    throw new Error('An unexpected network error occurred while connecting to the generation service.');
  } finally {
    clearTimeout(timeoutId);
  }
}

