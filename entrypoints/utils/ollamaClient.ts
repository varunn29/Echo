export interface OllamaModelTag {
  name: string;
  model: string;
  size: number;
}

export interface OllamaTagsResponse {
  models?: OllamaModelTag[];
}

export interface OllamaChatResponse {
  message?: {
    role: string;
    content: string;
  };
  done?: boolean;
}

export interface OllamaStatus {
  isRunning: boolean;
  isModelReady: boolean;
  modelName: string;
  error?: string;
}

export interface OllamaClientOptions {
  baseUrl?: string;
  model?: string;
  timeoutMs?: number;
  temperature?: number;
  numPredict?: number;
}

export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434';
export const DEFAULT_OLLAMA_MODEL = 'qwen3.5:2b';
export const DEFAULT_TIMEOUT_MS = 60000; // 60s timeout for CPU inference

export const KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = `You are a personal knowledge extraction engine for a local-first writing assistant.
Your task is to extract verifiable personal facts explicitly stated by the user about themselves from their message.
Extract ONLY claims where the user explicitly declares something about their own projects, technical skills, goals, personal habits, learning/workflow preferences, opinions, or current states.

CATEGORIES & SEMANTIC BOUNDARIES:
- "projects": Concrete software, apps, sites, or tools built or actively being built by the user (e.g. "Task Tracker", "a finance platform").
  * CONSTRAINTS & WORKAROUNDS: When the user explains technical difficulties, unavailable data, or workarounds for a project (e.g. "could not get live banking API data, so had to build a static simulation where mock transactions loop"), PRESERVE the constraint as part of the project's description/value. NEVER ignore constraints or convert a limitation into a standard feature or a skill.
- "skills": Specific programming languages, frameworks, libraries, tools, or technical methodologies the user knows, uses, or has learned (e.g. HTML, CSS, JavaScript, React, TypeScript, Tailwind, Node.js, Express.js, PostgreSQL).
  * COMPLETE TECHNOLOGY LISTS: When the user describes a learning path or lists multiple technologies (e.g. 'HTML, CSS, JavaScript... React, TypeScript, Tailwind, Node.js, Express.js, PostgreSQL'), extract the COMPLETE list of technologies into the value or subject. NEVER drop technologies or fragment the list into single-item tautologies.
  * ACCURATE COMPETENCY LEVEL: Use the exact verb stated (e.g. "learned", "uses", "learning", "studying"). NEVER automatically upgrade "learned" or "uses" to "proficient_in" or "expert" unless the user explicitly claimed high proficiency or expertise.
  * Project limitations, missing APIs, or workarounds are NOT skills.
- "goals": Explicit future objectives, ambitions, learning plans, or career aspirations (e.g. "planning to learn AI and machine learning", "applying to fintech companies").
  * Predicates for goals must describe a future ambition or objective ("planning to", "aims to", "wants to learn"), NOT past completed actions ("built").
  * DO NOT extract vague filler (e.g. "want to learn many things", "will think about other technologies in the coming future") as goals.
- "habits": Established recurring personal workflows, routines, or communication/debate approaches (e.g. "uses AI when stuck for concise explanations", "in technical discussions, argues points if knowledgeable, otherwise listens").
- "preferences": Explicit personal tool choices, likes, dislikes, or learning styles (e.g. "prefers examples to visualize concepts").
- "opinions": Durable personal convictions, philosophies, or strong viewpoints (e.g. "frustrated that tools generate boilerplate without developers understanding core principles", "tutorials skip the why and just show the how"). Do NOT classify durable opinions as temporary states.
- "states": Current temporary emotional feelings, blockers, or self-evaluations (e.g. "feels progress is slow this month", "struggling to come up with a project idea").

STRICT EXTRACTION RULES:
1. PROCEDURAL CHAT QUESTIONS ARE NOT PERSONAL FACTS:
   * DO NOT extract facts from procedural chat questions or conversation instructions directed at ChatGPT/assistant (e.g. "When are we going to start building now?", "Will we have to do it every time even when our project is complete?", "Can you explain this?", "What should we do next?").
   * First-person pronouns ("we", "I") in a conversational question do NOT constitute personal knowledge about the user.
   * If a message is asking for instructions, clarification, troubleshooting, or planning next steps, return {"knowledge": []}.
2. NO TAUTOLOGICAL TRIPLES:
   * DO NOT produce triples where the subject and value are identical (e.g. subject: "HTML", predicate: "learned", value: "HTML" or subject: "latest technologies", predicate: "planning to learn", value: "latest technologies").
   * Subject should be the domain or user, predicate the action, and value the meaningful informative content.
3. NO VAGUE CONVERSATIONAL FILLER:
   * DO NOT extract non-informative generic statements (e.g. "user | want to learn | many things", "project | will be done | every time", "user | thought too much | about an idea", "user | will think about | other things in the future").
4. PRESERVE PROJECT CONSTRAINTS & LIMITATIONS:
   * When a project involved a workaround or technical limitation (e.g. static simulation because live external API was unavailable), record the workaround accurately under "projects"; never invert a limitation into a feature or misclassify it as a skill.
5. PROVENANCE GROUNDING:
   * Every extracted item MUST have an exact, verbatim "sourceExcerpt" that is a substring of the user's message. The "sourceExcerpt" must directly contain and support the specific claim (its subject, predicate, and value) — NEVER pair a claim from one part of the text with an excerpt from an unrelated clause or sentence.
6. CONFIDENCE & IMPORTANCE:
   * "confidence": Use "high" ONLY for direct, unambiguous, definitive completed facts ("Recently I built X", "I use Y daily", "I learned React"). Statements of future intent, plans, aspirations, or tentative goals ("planning to learn X", "want to try Y", "hope to move to Z") MUST be rated "medium", NEVER "high". Confidence represents source certainty, NOT repeated mentions or importance.
   * "importance": Use "secondary" by default. Reserve "primary" only for major standalone projects or primary career goals.
7. Return a valid JSON object matching:
{
  "knowledge": [
    {
      "category": "projects" | "skills" | "goals" | "habits" | "preferences" | "opinions" | "states",
      "subject": string,
      "predicate": string,
      "value": string,
      "contextScope": string,
      "sourceExcerpt": string,
      "confidence": "high" | "medium" | "low",
      "importance": "primary" | "secondary" | "incidental"
    }
  ]
}
If no genuine personal facts exist about the user, return {"knowledge": []}.`;

/**
 * Checks whether Ollama is online and responsive.
 */
export async function isOllamaRunning(
  baseUrl: string = DEFAULT_OLLAMA_BASE_URL,
  timeoutMs: number = 3000
): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: controller.signal,
    });
    return res.status === 200;
  } catch {
    return false;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Checks whether the target model is installed and listed in Ollama.
 */
export async function isModelAvailable(
  modelName: string = DEFAULT_OLLAMA_MODEL,
  baseUrl: string = DEFAULT_OLLAMA_BASE_URL,
  timeoutMs: number = 4000
): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: controller.signal,
    });

    if (res.status !== 200) {
      return false;
    }

    const data = (await res.json()) as OllamaTagsResponse;
    const target = modelName.toLowerCase();

    return Boolean(
      data.models?.some((m) => {
        const name = m.name?.toLowerCase() || '';
        return name === target || name.startsWith(`${target}:`);
      })
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Performs a comprehensive status check of Ollama and model availability.
 */
export async function checkOllamaStatus(
  modelName: string = DEFAULT_OLLAMA_MODEL,
  baseUrl: string = DEFAULT_OLLAMA_BASE_URL
): Promise<OllamaStatus> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: controller.signal,
    });

    if (res.status === 403) {
      return {
        isRunning: true,
        isModelReady: false,
        modelName,
        error:
          'Ollama rejected the extension origin (403 Forbidden). Set OLLAMA_ORIGINS to allow this extension (e.g. chrome-extension://<id>) and restart Ollama.',
      };
    }

    if (res.status !== 200) {
      return {
        isRunning: false,
        isModelReady: false,
        modelName,
        error: `Ollama returned HTTP status ${res.status}.`,
      };
    }

    const data = (await res.json()) as OllamaTagsResponse;
    const target = modelName.toLowerCase();

    const isModelReady = Boolean(
      data.models?.some((m) => {
        const name = m.name?.toLowerCase() || '';
        return name === target || name.startsWith(`${target}:`);
      })
    );

    return {
      isRunning: true,
      isModelReady,
      modelName,
      error: isModelReady
        ? undefined
        : `Model "${modelName}" is not installed in Ollama. Run: ollama pull ${modelName}`,
    };
  } catch (err: unknown) {
    const message =
      err instanceof Error && err.name === 'AbortError'
        ? 'Ollama health check timed out.'
        : 'Ollama is offline. Start Ollama to enable personal knowledge extraction.';

    return {
      isRunning: false,
      isModelReady: false,
      modelName,
      error: message,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Sends a structured prompt to Ollama in JSON mode and returns the raw JSON string response.
 */
export async function queryOllamaJson(
  userPrompt: string,
  options: OllamaClientOptions = {}
): Promise<string> {
  const baseUrl = options.baseUrl || DEFAULT_OLLAMA_BASE_URL;
  const model = options.model || DEFAULT_OLLAMA_MODEL;
  const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
  const temperature = options.temperature ?? 0.1;
  const numPredict = options.numPredict ?? 600;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const payload = {
    model,
    messages: [
      {
        role: 'system',
        content: KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT,
      },
      {
        role: 'user',
        content: userPrompt,
      },
    ],
    format: 'json',
    stream: false,
    options: {
      temperature,
      num_predict: numPredict,
    },
    // think: false disables chain-of-thought tokens on thinking models, speeding CPU inference up by 5x-10x
    think: false,
  };

  try {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (res.status === 403) {
      throw new Error(
        'Ollama rejected the request origin (403 Forbidden). Set OLLAMA_ORIGINS to allow this extension (e.g. chrome-extension://<id>) and restart Ollama.'
      );
    }

    if (res.status === 404) {
      throw new Error(
        `Model "${model}" not found in Ollama. Pull it via: ollama pull ${model}`
      );
    }

    if (!res.ok) {
      throw new Error(`Ollama returned HTTP error ${res.status}: ${res.statusText}`);
    }

    const data = (await res.json()) as OllamaChatResponse;
    const content = data.message?.content?.trim();

    if (!content) {
      throw new Error('Ollama returned an empty response.');
    }

    return content;
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.name === 'AbortError') {
        throw new Error(`Ollama request timed out after ${Math.round(timeoutMs / 1000)}s.`);
      }
      if (err.message.includes('ECONNREFUSED') || err.message.includes('fetch failed')) {
        throw new Error(
          'Ollama is offline or unreachable at http://localhost:11434. Please start Ollama.'
        );
      }
      throw err;
    }
    throw new Error(String(err));
  } finally {
    clearTimeout(timeoutId);
  }
}
