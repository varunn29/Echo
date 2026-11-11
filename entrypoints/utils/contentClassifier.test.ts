import { describe, expect, it } from 'vitest';
import { classifyContent } from './contentClassifier';

describe('ContentClassifier V1 Test Suite', function () {
  describe('1. Natural Prose Beginning with Technology Names (Bug Fixes)', function () {
    it('should classify "Python was..." as natural prose with personal knowledge signal', function () {
      const result = classifyContent('Python was the first language I learned.');
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
      expect(result.reasons).toContain('natural_language_prose');
      expect(result.reasons).toContain('personal_context_detected');
    });

    it('should classify "Docker makes..." as natural prose', function () {
      const result = classifyContent(
        'Docker makes our deployment pipeline predictable and isolated across environments.'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('weak');
      expect(result.knowledgeSignal).toBe('none');
      expect(result.reasons).toContain('natural_language_prose');
    });

    it('should classify "Git branching..." as natural prose', function () {
      const result = classifyContent(
        'Git branching strategies are often overcomplicated by teams when a simple trunk workflow would suffice.'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('should classify "Node has..." as natural prose', function () {
      const result = classifyContent(
        'Node has an asynchronous event-driven architecture that handles concurrent I/O operations smoothly.'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('weak');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('should classify "Clear communication..." as natural prose', function () {
      const result = classifyContent(
        'Clear communication prevents unnecessary bugs and saves development time across the team.'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('weak');
      expect(result.knowledgeSignal).toBe('none');
    });
  });

  describe('2. Terminal Commands & CLI Invocations', function () {
    it('should detect npm command', function () {
      const result = classifyContent('npm install express');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
      expect(result.reasons).toContain('terminal_command');
    });

    it('should detect git command with flags', function () {
      const result = classifyContent('git commit -m "feat: initial commit"');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('should detect docker run command', function () {
      const result = classifyContent('docker run -d -p 8080:80 nginx');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect python command invocation', function () {
      const result = classifyContent('python -m venv .venv');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect PowerShell cmdlets', function () {
      const result = classifyContent('Get-Process | Where-Object { $_.CPU -gt 10 }');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect shell prompts', function () {
      const result = classifyContent('$ ls -la /var/log');
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });
  });

  describe('3. Unfenced Source Code', function () {
    it('should detect JavaScript/TypeScript function', function () {
      const code = `function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}`;
      const result = classifyContent(code);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
      expect(result.reasons).toContain('unfenced_source_code');
    });

    it('should detect React hook with imports and syntax', function () {
      const code = `const [count, setCount] = useState(0);
useEffect(() => {
  document.title = String(count);
}, [count]);`;
      const result = classifyContent(code);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect Python function definition', function () {
      const code = `def filter_active_users(users):
    return [u for u in users if u.get("is_active")]`;
      const result = classifyContent(code);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect SQL query', function () {
      const query = `SELECT id, email, created_at FROM users WHERE status = 'active' ORDER BY created_at DESC;`;
      const result = classifyContent(query);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect Java method signature', function () {
      const code = `public static void main(String[] args) {
    System.out.println("Echo initialized");
}`;
      const result = classifyContent(code);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
    });
  });

  describe('4. Fenced Code Blocks', function () {
    it('should detect markdown-fenced code block', function () {
      const code = '```typescript\nconst greeting: string = "Hello, Echo";\nconsole.log(greeting);\n```';
      const result = classifyContent(code);
      expect(result.contentType).toBe('code');
      expect(result.styleSignal).toBe('none');
      expect(result.reasons).toContain('fenced_code_block');
    });
  });

  describe('5. Structured Data (JSON, XML/HTML, YAML)', function () {
    it('should detect JSON object', function () {
      const json = JSON.stringify(
        {
          name: 'Echo',
          version: '1.0.0',
          dependencies: { dexie: '^4.4.7' },
        },
        null,
        2
      );
      const result = classifyContent(json);
      expect(result.contentType).toBe('structured_data');
      expect(result.styleSignal).toBe('none');
      expect(result.reasons).toContain('json_payload');
    });

    it('should detect JSON array', function () {
      const jsonArray = '[{"id": 1, "text": "sample"}, {"id": 2, "text": "sample 2"}]';
      const result = classifyContent(jsonArray);
      expect(result.contentType).toBe('structured_data');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect XML / HTML markup', function () {
      const markup = '<div class="card">\n  <h2>Header</h2>\n  <p>Description text</p>\n</div>';
      const result = classifyContent(markup);
      expect(result.contentType).toBe('structured_data');
      expect(result.styleSignal).toBe('none');
      expect(result.reasons).toContain('xml_or_html_markup');
    });

    it('should detect YAML configuration', function () {
      const yaml = `server:
  port: 8080
  host: localhost
  max_connections: 100
  timeout_seconds: 30`;
      const result = classifyContent(yaml);
      expect(result.contentType).toBe('structured_data');
      expect(result.styleSignal).toBe('none');
      expect(result.reasons).toContain('yaml_or_configuration_data');
    });
  });

  describe('6. Stack Traces & Compiler Logs', function () {
    it('should detect Python traceback', function () {
      const trace = `Traceback (most recent call last):
  File "train_lora.py", line 87, in load_dataset
    raise ValueError("The training dataset is empty.")
ValueError: The training dataset is empty.`;
      const result = classifyContent(trace);
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
      expect(result.reasons).toContain('stack_trace_or_compiler_output');
    });

    it('should detect TypeScript compiler errors', function () {
      const tsError = `entrypoints/utils/integration.test.ts(30,11): error TS2322: Type 'string | undefined' is not assignable to type 'string'.
  Type 'undefined' is not assignable to type 'string'.`;
      const result = classifyContent(tsError);
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect Node runtime stack trace', function () {
      const nodeTrace = `Error: Cannot find module 'express'
    at Function.Module._resolveFilename (node:internal/modules/cjs/loader:1225:15)
    at Function.Module._load (node:internal/modules/cjs/loader:1051:27)`;
      const result = classifyContent(nodeTrace);
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });

    it('should detect PowerShell execution error output', function () {
      const psError = `npm : The term 'npm' is not recognized as the name of a cmdlet.
At line:1 char:1
+ npm test
+ ~~~
    + CategoryInfo          : ObjectNotFound: (npm:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException`;
      const result = classifyContent(psError);
      expect(result.contentType).toBe('terminal');
      expect(result.styleSignal).toBe('none');
    });
  });

  describe('7. Short Conversational Fragments', function () {
    it('should classify "ok" as short fragment', function () {
      const result = classifyContent('ok');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('should classify "continue" as short fragment', function () {
      const result = classifyContent('continue');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
    });

    it('should classify "thanks!" as short fragment', function () {
      const result = classifyContent('thanks!');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
    });

    it('should classify single question word "why?" as short fragment', function () {
      const result = classifyContent('why?');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
    });
  });

  describe('8. Technical Questions in Natural Language', function () {
    it('should classify technical question as prose with weak style and no personal knowledge', function () {
      const result = classifyContent(
        'How do I configure PostgreSQL connection pooling in Node.js using pg-pool?'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('weak');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('should classify concept inquiry as prose', function () {
      const result = classifyContent(
        'What is the difference between an interface and a type alias in TypeScript?'
      );
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('weak');
      expect(result.knowledgeSignal).toBe('none');
    });
  });

  describe('9. Mixed Content (Prose + Code)', function () {
    it('should classify question with embedded code block as mixed with style signal', function () {
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

      const result = classifyContent(text);
      expect(result.contentType).toBe('mixed');
      expect(result.styleSignal).toBe('strong');
      expect(result.reasons).toContain('mixed_prose_and_code');
    });
  });

  describe('10. Authentic User Writing Samples (Style + Knowledge)', function () {
    it('Sample 1: should recognize learning journey description as strong style and possible knowledge', function () {
      const text =
        "So I started by learning HTML, CSS and JavaScript and my plan was to learn web development and for that I also learned React, typescript, tailwind, node js, express js, postgres and now I feel that there is not much left for web development in today's time, so planning to move ahead and learn some latest technologies";
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
      expect(result.reasons).toContain('personal_context_detected');
    });

    it('Sample 2: should recognize future learning interest as strong style and possible knowledge', function () {
      const text =
        'I have not yet decided but planning to try with AI and some ML, also with cloud computing, will think about other technologies in the coming future';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 3: should recognize project building experience as strong style and possible knowledge', function () {
      const text =
        'Recently I built an analytics dashboard, a data tool and I built it because I was applying to a software company and it was my first complete project, the main difficulty which I faced while building it is that I was not able to get live event data and so I had to build a static simulation where events loop continuously';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 4: should recognize personal workflow habit as strong style and possible knowledge', function () {
      const text =
        'Whenever I am stuck on a problem I prefer using documentation and search tools because it gives a direct answer and saves debugging time';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 5: should recognize strong personal frustration/opinion as strong style and possible knowledge', function () {
      const text =
        'One thing that really frustrates me is when automated tools generate boilerplate without developers learning fundamental principles';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 6: should recognize coding motivation as strong style and possible knowledge', function () {
      const text =
        'I like programming because I want to build useful software that solves real problems and helps people work more effectively';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 7: should recognize goal exploration and difficulty as strong style and possible knowledge', function () {
      const text =
        'The main difficulty which I faced when building distributed systems is designing scalable caching and data synchronization across microservices';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 8: should recognize self-reflection as strong style and possible knowledge', function () {
      const text =
        'I feel I should speed up my learning pace because other engineers are mastering so many modern tools and there is still a long way to go';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 9: should recognize communication preference as strong style and possible knowledge', function () {
      const text =
        'I like to use examples and diagrams when explaining system design because that is how they can visualize complex distributed workflows';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });

    it('Sample 10: should recognize debate and behavioral tendency as strong style and possible knowledge', function () {
      const text =
        'In technical discussions, if I have deep experience with an architecture then I will argue my position, but if I am unfamiliar then I will listen to their perspective as well';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.styleSignal).toBe('strong');
      expect(result.knowledgeSignal).toBe('possible');
    });
  });

  describe('11. Refined Knowledge Signal Categories & Negative Controls', function () {
    // 1. Projects and experiences
    it('Category 1: should recognize past work and role experiences', function () {
      const text = 'Last summer I interned at a fintech startup where I developed internal dashboards.';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // 2. Learning interests and goals
    it('Category 2: should recognize forward learning goals', function () {
      const text = 'I am looking to explore distributed systems and master Kubernetes this year.';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // 3. Personal habits and workflows
    it('Category 3: should recognize workflow tool preferences', function () {
      const text = 'I prefer using Neovim because modal editing feels significantly faster for my workflow.';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // 4. Learning and communication preferences
    it('Category 4: should recognize learning style preference', function () {
      const text = 'I learn best by building a small prototype rather than just reading documentation.';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // 5. Personal opinions and behavioral tendencies
    it('Category 5: should recognize architectural convictions', function () {
      const text = 'In my opinion, premature microservice architectures cause far more harm than monolithic design.';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // 6. Current goals, difficulties, and states
    it('Category 6: should recognize current struggles and exploration', function () {
      const text = "I'm struggling to figure out what data structure works best for this graph traversal problem.";
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('possible');
    });

    // Negative controls: sentences with "I" or "my" that should NOT be personal knowledge
    it('Negative Control 1: generic technical question with "I"', function () {
      const text = 'How can I sort an array of objects by date in JavaScript?';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 2: generic troubleshooting question with "I"', function () {
      const text = 'What should I do if my Docker container exits unexpectedly with code 137?';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 3: generic debugging condition with "I"', function () {
      const text = 'If I run npm test, it fails with an unexpected token error.';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 4: generic framework query with "I"', function () {
      const text = 'Where should I store database connection credentials in a Next.js application?';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 5: prompt asking for an example with "I"', function () {
      const text = 'Can you give me an example of binary search implemented in Python?';
      const result = classifyContent(text);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 6: procedural question asking when to start building (no knowledge signal)', function () {
      const text = 'ok then we can go with this proposed plan then, and when are we going to start building now';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 7: procedural workflow question asking about repeated step (no knowledge signal)', function () {
      const text = 'so will we have to do it everytime even when our project is  complete';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 8: interrogative planning question directed at assistant', function () {
      const text = 'When are we going to start building now?';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Negative Control 9: interrogative repetition question directed at assistant', function () {
      const text = 'Will we have to do it every time even when our project is complete?';
      const result = classifyContent(text);
      expect(result.knowledgeSignal).toBe('none');
    });

    it('Positive Control: Mixed personal difficulty statement alongside procedural question preserves knowledge signal', function () {
      const mixedText =
        'The main difficulty which I faced while building the analytics platform was not being able to get live feed data, so I built a static simulation. Will we have to do it every time even when our project is complete?';
      const result = classifyContent(mixedText);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('possible');
      expect(result.reasons).toContain('personal_context_detected');
    });

    it('Positive Control: Mixed problem faced statement with interrogative closing', function () {
      const mixedText =
        'The problem I faced was that I could not get live API access so I worked around it. When are we going to deploy?';
      const result = classifyContent(mixedText);
      expect(result.contentType).toBe('prose');
      expect(result.knowledgeSignal).toBe('possible');
    });
  });

  describe('11. Edge Cases', function () {
    it('should handle empty string', function () {
      const result = classifyContent('');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
      expect(result.wordCount).toBe(0);
      expect(result.reasons).toContain('empty_message');
    });

    it('should handle whitespace-only string', function () {
      const result = classifyContent('     \n\t   ');
      expect(result.contentType).toBe('short_fragment');
      expect(result.styleSignal).toBe('none');
      expect(result.knowledgeSignal).toBe('none');
      expect(result.wordCount).toBe(0);
    });
  });
});

