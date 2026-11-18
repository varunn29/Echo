import { describe, expect, it } from 'vitest';
import {
  isObviousNonContent,
  isPotentialStyleSample,
  isEligibleStyleSample,
  isEligibleTrainingSample,
} from './trainingFilter';

describe('isObviousNonContent', function () {
  it('should reject empty messages', function () {
    expect(isObviousNonContent('   ')).toBe(true);
  });

  it('should reject terminal commands', function () {
    expect(isObviousNonContent('npm install express')).toBe(true);
  });

  it('should reject Git commands', function () {
    expect(isObviousNonContent('git status')).toBe(true);
  });

  it('should reject messages consisting entirely of code blocks', function () {
    const text = '```javascript\nconsole.log("Hello");\n```';

    expect(isObviousNonContent(text)).toBe(true);
  });

  it('should accept natural language', function () {
    expect(
      isObviousNonContent('I enjoy building projects with React.')
    ).toBe(false);
  });
});

describe('isPotentialStyleSample', function () {
  it('should accept natural personal writing', function () {
    expect(
      isPotentialStyleSample(
        'I started learning React recently and I enjoy building projects.'
      )
    ).toBe(true);
  });

  it('should reject terminal commands', function () {
    expect(isPotentialStyleSample('npm install express')).toBe(false);
  });

  it('should reject empty messages', function () {
    expect(isPotentialStyleSample('   ')).toBe(false);
  });
});

describe('isEligibleTrainingSample', function () {
  it('should accept natural personal prose with strong style signal', function () {
    const text =
      'I started learning web development by building simple UI widgets and now I prefer TypeScript with React for larger fullstack projects.';

    expect(isEligibleTrainingSample(text)).toBe(true);
  });

  it('should quarantine mixed prose and code into Tier F (not eligible for training)', function () {
    const text = `I am trying to implement a debounce hook in React, but the timer keeps resetting on each render:

\`\`\`tsx
function useDebounce(value, delay) {
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
}
\`\`\`

How can I persist the timer reference across re-renders without triggering extra updates?`;

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject empty or whitespace text', function () {
    expect(isEligibleTrainingSample('')).toBe(false);
    expect(isEligibleTrainingSample('   \n  ')).toBe(false);
  });

  it('should reject pure fenced code blocks', function () {
    const text = '```python\ndef solve():\n    return 42\n```';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject unfenced source code', function () {
    const text = 'function calculateSum(a: number, b: number): number { return a + b; }';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject terminal commands', function () {
    expect(isEligibleTrainingSample('npm run build')).toBe(false);
    expect(isEligibleTrainingSample('git checkout -b feature/auth')).toBe(false);
  });

  it('should reject structured JSON data', function () {
    const text = '{"service": "echo", "status": "active", "version": "1.0.0"}';

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  it('should reject short conversational fragments', function () {
    expect(isEligibleTrainingSample('ok')).toBe(false);
    expect(isEligibleTrainingSample('sounds good')).toBe(false);
    expect(isEligibleTrainingSample('thanks!')).toBe(false);
  });

  it('should reject weak style samples with insufficient prose length or signal', function () {
    // 5 words without personal context
    expect(isEligibleTrainingSample('How does CSS flexbox work?')).toBe(false);

    // 9 words generic prose without personal context
    expect(isEligibleTrainingSample('The quick brown fox jumps over the lazy dog.')).toBe(false);
  });

  it('should accept authentic technical prose with strong style signal (Tier B)', function () {
    const text =
      'When designing an event-driven architecture, separating the command bus from query handlers prevents race conditions and makes testing individual state mutations predictable.';

    expect(isEligibleTrainingSample(text)).toBe(true);
  });

  it('should reject Windows PowerShell prompts and drive prefixes', function () {
    expect(isEligibleTrainingSample('PS D:\\Forge\\Echo> npm run build')).toBe(false);
    expect(isEligibleTrainingSample('D:\\Forge\\Echo> git status')).toBe(false);
  });

  it('should reject AI-agent directives and prompt scaffolding (Tier C)', function () {
    expect(
      isEligibleTrainingSample(
        '# Echo — Final Training Readiness Audit\n\nPerform a strictly read-only audit of the codebase and verify the following.'
      )
    ).toBe(false);
  });

  it('should reject build logs and package manager outputs (Tier D)', function () {
    const buildLog = `wxt v0.19.0 (vite v5.1.0)
Building for chrome (manifest v3)...
✓ .output/chrome-mv3/manifest.json
✓ .output/chrome-mv3/background.js
Build completed in 2.3s`;

    expect(isEligibleTrainingSample(buildLog)).toBe(false);
  });

  it('should reject external video generation and scraped social content (Tier E)', function () {
    const script =
      'Video 001 — Target duration: 30–45 seconds. A cinematic Roblox-style 3D avatar running through a futuristic neon city.';

    expect(isEligibleTrainingSample(script)).toBe(false);
  });

  it('should reject mixed prose and code when prose signal is weak', function () {
    const text = `Here is my code:

\`\`\`ts
const x = 1;
\`\`\`

Fix this.`;

    expect(isEligibleTrainingSample(text)).toBe(false);
  });

  describe('Hardened False-Positive Rejections', function () {
    it('should reject Antigravity assistant reports starting with "I have initiated" or "We have inspected"', function () {
      const report1 =
        'I have initiated the test of the existing fine-tuned LoRA adapter using training/test_adapter.py. I will wait for inference to finish. The service is active and listening on port 11435.';
      const report2 =
        'Echo — Personal Writing Dataset Preparation V1 Report\nWe have inspected the complete set of collected messages from Echo\'s IndexedDB export and verified all candidate samples.';
      const report3 =
        'I have launched the LoRA V2 training process (training/train_lora_v2.py) using the 38 training samples (1,054 tokens) on the CPU. I will wait for training to finish.';

      expect(isEligibleStyleSample(report1)).toBe(false);
      expect(isEligibleTrainingSample(report1)).toBe(false);
      expect(isEligibleStyleSample(report2)).toBe(false);
      expect(isEligibleTrainingSample(report2)).toBe(false);
      expect(isEligibleStyleSample(report3)).toBe(false);
      expect(isEligibleTrainingSample(report3)).toBe(false);
    });

    it('should reject Git commit, staging, and push instructions', function () {
      const commit1 =
        'Push the already verified Commit 1 to GitHub.\n\nCommit: b640bb4cc7c7bc62df7817bb389a346cd49de620\n\nSteps:\n1. Verify the current branch is main.\n2. Push using a normal push.\n3. Verify remote contains Commit 1.';
      const commit2 =
        'We are beginning the Echo commit process, starting with Commit 1 only.\n\nCommit 1: Data Layer & Content Classification\nStage exactly these 6 files: entrypoints/utils/database.ts, entrypoints/utils/contentClassifier.ts';
      const commit3 =
        'Commit 1 has been reviewed and approved. Create the commit using this exact message: feat(data): extend personal knowledge curation and harden content classifier';

      expect(isEligibleStyleSample(commit1)).toBe(false);
      expect(isEligibleTrainingSample(commit1)).toBe(false);
      expect(isEligibleStyleSample(commit2)).toBe(false);
      expect(isEligibleTrainingSample(commit2)).toBe(false);
      expect(isEligibleStyleSample(commit3)).toBe(false);
      expect(isEligibleTrainingSample(commit3)).toBe(false);
    });

    it('should reject multi-section task prompts containing Objective, Restrictions, and Deliverables', function () {
      const prompt = `Objective: Perform a strictly read-only baseline generation quality audit.
Restrictions: Do not modify or risk working code. Wait for explicit approval before proceeding.
Deliverables: Return a structured inspection report with metrics and evidence.`;

      expect(isEligibleStyleSample(prompt)).toBe(false);
      expect(isEligibleTrainingSample(prompt)).toBe(false);
    });

    it('should reject server logs with slot launch, mirostat, and sampler params', function () {
      const log =
        'slot launch_slot_: id 0 | task -1 | sampler params: repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000 mirostat = 0, mirostat_lr = 0.100, prompt eval time = 120.45 ms';

      expect(isEligibleStyleSample(log)).toBe(false);
      expect(isEligibleTrainingSample(log)).toBe(false);
    });

    it('should reject external application forms and scraped UI text', function () {
      const form =
        'YARC markARCBUILDER FELLOWSHIP Founding Cohort · 2027 · Stage 01 Application Step 4 of 7 — Where is your work 57% When applying please describe your project.';
      const scrapedUi =
        'Echo. Writing studio Workspace DashboardMy Writing DataMy Writing Style Preferences Settings Local first Your writing and personal data are stored on device.';

      expect(isEligibleStyleSample(form)).toBe(false);
      expect(isEligibleTrainingSample(form)).toBe(false);
      expect(isEligibleStyleSample(scrapedUi)).toBe(false);
      expect(isEligibleTrainingSample(scrapedUi)).toBe(false);
    });

    it('should reject synthetic few-shot writing templates', function () {
      const template =
        'You are a personal writing assistant. First, study the writing samples below.\n\nSTYLE A: "Finally deployed my first project. Took a few days to figure out authentication."';

      expect(isEligibleStyleSample(template)).toBe(false);
      expect(isEligibleTrainingSample(template)).toBe(false);
    });
  });

  describe('Genuine Natural Writing Preservation', function () {
    it('should preserve ID 8 (explanation of writing tests)', function () {
      const text =
        'so we will write tests for every feature or for whatever feature we want to write and then when we make some changes next time, we can just run that and it will tell if any previous feature is broken or not, and what is its relation with mock and all that';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });

    it('should preserve ID 38 (question about local training vs cloud GPU)', function () {
      const text =
        'I am confused what do I use now, do I download a model and train it on my system or do I use the cloud GPU';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });

    it('should preserve ID 43 (statement about first-version requirements)', function () {
      const text =
        'see we are just building the first version so it\'s fine if not every laptop can run it';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });

    it('should preserve ID 97 (reflection on learning web development)', function () {
      const text =
        'So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind and next js, and then did some small project then after that I built match point, but now what I am thinking is everyone can build this with AI';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });

    it('should preserve ID 159 (why user likes programming)', function () {
      const text =
        'I like programming because I wanted to build cool things which come to my mind and along the way solve problems and build software which many people can use and find helpful in their daily lives';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });

    it('should preserve ID 160 (reflection on AI and programming value)', function () {
      const text =
        'One thing that really frustrates me is the fact that now anyone with no coding or programming skill can build a software and make money from it, because it kind of makes me feel that my skills are becoming less valuable';

      expect(isEligibleStyleSample(text)).toBe(true);
      expect(isEligibleTrainingSample(text)).toBe(true);
    });
  });
});
