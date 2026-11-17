import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Sparkles,
  Cpu,
  FileText,
  BarChart2,
  Clock,
  ArrowUpRight,
  ShieldCheck,
  Eye,
  X,
  Info,
  Lock,
  RefreshCw,
  PenTool,
  HelpCircle,
  CheckCircle2,
} from 'lucide-react';

import {
  getWritingMessages,
  type WritingMessage,
} from '../../utils/database';
import {
  isPotentialStyleSample,
} from '../../utils/trainingFilter';
import {
  classifyContent,
} from '../../utils/contentClassifier';
import {
  checkGenerationServiceHealth,
  type GenerationServiceHealth,
} from '../../utils/generationClient';

interface WritingStyleViewProps {
  onNavigate?: (section: 'Dashboard' | 'My Writing Data' | 'My Writing Style' | 'Settings') => void;
}

interface CorpusStatistics {
  totalMessages: number;
  totalWords: number;
  medianWords: number;
  avgWordsPerSentence: number;
  lexicalDiversity: number; // Distinct-1 / Type-Token Ratio
  multilineRatio: number;   // % of messages with paragraphs/line breaks
  questionRatio: number;    // % of messages with questions
  exclamationRatio: number; // % of messages with exclamations
  firstPersonRatio: number; // % of messages with first-person pronouns
}

/**
 * Splits text into sentences while protecting abbreviations (e.g., i.e., vs., dr.),
 * IP addresses (127.0.0.1), and decimal numbers (3.14).
 */
function splitSentences(text: string): string[] {
  const protectedText = text
    .replace(/\b(?:e\.g\.|i\.e\.|etc\.|vs\.|v\d+\.\d+|dr\.|mr\.|mrs\.|ms\.)/gi, (m) => m.replace(/\./g, '___DOT___'))
    .replace(/(\d+)\.(\d+)/g, '$1___DOT___$2');

  const parts = protectedText
    .split(/[.!?]+(?:\s+|\n+|$)/)
    .map((s) => s.replace(/___DOT___/g, '.').trim())
    .filter((s) => s.length > 0);

  return parts.length > 0 ? parts : [text.trim()];
}

export function WritingStyleView({ onNavigate }: WritingStyleViewProps) {
  const [messages, setMessages] = useState<WritingMessage[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(true);
  const [serviceHealth, setServiceHealth] = useState<GenerationServiceHealth | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isSampleModalOpen, setIsSampleModalOpen] = useState(false);
  const [selectedBenchmarkIndex, setSelectedBenchmarkIndex] = useState(0);

  // Load writing messages from local IndexedDB with verified training eligibility filtering
  const loadData = useCallback(async () => {
    setIsLoadingMessages(true);
    try {
      const allMsgs = await getWritingMessages();
      // Apply existing content-classification and style-eligibility utilities:
      // Exclude code, terminal output, structured data, external content, agent directives, and ambiguous fragments.
      // Only retain authentic personal prose (Tier A) and authentic technical prose (Tier B).
      const eligibleMsgs = allMsgs.filter((m) => {
        if (!m.text || typeof m.text !== 'string' || !m.text.trim()) {
          return false;
        }
        if (!isPotentialStyleSample(m.text)) {
          return false;
        }
        const classification = classifyContent(m.text);
        const isEligibleTier =
          classification.tier === 'tier_a' || classification.tier === 'tier_b';
        const isStrongProse =
          classification.contentType === 'prose' && classification.styleSignal === 'strong';
        return isEligibleTier && isStrongProse;
      });
      setMessages(eligibleMsgs);
    } catch (err) {
      console.warn('Echo: Could not load writing messages:', err);
    } finally {
      setIsLoadingMessages(false);
    }
  }, []);

  // Check production generation service health
  const checkHealth = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      const health = await checkGenerationServiceHealth();
      setServiceHealth(health);
    } catch {
      setServiceHealth(null);
    } finally {
      setIsCheckingHealth(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    checkHealth();
  }, [loadData, checkHealth]);

  // Compute empirical corpus statistics strictly from eligible user-authored prose messages
  const stats: CorpusStatistics | null = useMemo(() => {
    if (!messages || messages.length === 0) {
      return null;
    }

    let totalWordsCount = 0;
    let totalSentencesCount = 0;
    const wordCounts: number[] = [];
    const allTokens: string[] = [];
    let multilineCount = 0;
    let questionCount = 0;
    let exclamationCount = 0;
    let firstPersonCount = 0;

    // Comprehensive first-person token set including reflexive pronouns
    const firstPersonTokens = new Set([
      'i', "i'm", "i've", "i'll", "i'd",
      'my', 'myself', 'me', 'mine',
      'we', "we're", "we've", "we'll", "we'd",
      'our', 'ours', 'ourselves', 'us',
    ]);

    for (const msg of messages) {
      const text = msg.text.trim();
      const rawWords = text.match(/\b[\w'-]+\b/g) || [];
      const wordCount = rawWords.length;

      wordCounts.push(wordCount);
      totalWordsCount += wordCount;

      for (const w of rawWords) {
        allTokens.push(w.toLowerCase());
      }

      // Robust sentence boundary parsing
      const sentences = splitSentences(text);
      totalSentencesCount += sentences.length;

      // Formatting & tone markers
      if (text.includes('\n')) {
        multilineCount++;
      }
      if (text.includes('?')) {
        questionCount++;
      }
      if (text.includes('!')) {
        exclamationCount++;
      }

      const hasFirstPerson = rawWords.some((w) => firstPersonTokens.has(w.toLowerCase()));
      if (hasFirstPerson) {
        firstPersonCount++;
      }
    }

    // Median word count (preserved numerical sort)
    wordCounts.sort((a, b) => a - b);
    const mid = Math.floor(wordCounts.length / 2);
    const midVal = wordCounts[mid] ?? 0;
    const prevVal = wordCounts[mid - 1] ?? midVal;
    const medianWords =
      wordCounts.length % 2 !== 0 ? midVal : Math.round((prevVal + midVal) / 2);

    // True corpus-wide average sentence length: total eligible words divided by total eligible sentences
    const avgWordsPerSentence =
      totalSentencesCount > 0
        ? Math.round((totalWordsCount / totalSentencesCount) * 10) / 10
        : 0;

    // Lexical diversity (Distinct-1 / Type-Token Ratio)
    const uniqueTokens = new Set(allTokens);
    const lexicalDiversity =
      allTokens.length > 0 ? Math.round((uniqueTokens.size / allTokens.length) * 100) / 100 : 0;

    return {
      totalMessages: messages.length,
      totalWords: totalWordsCount,
      medianWords,
      avgWordsPerSentence,
      lexicalDiversity,
      multilineRatio: Math.round((multilineCount / messages.length) * 100),
      questionRatio: Math.round((questionCount / messages.length) * 100),
      exclamationRatio: Math.round((exclamationCount / messages.length) * 100),
      firstPersonRatio: Math.round((firstPersonCount / messages.length) * 100),
    };
  }, [messages]);

  // Representative benchmark topics from the completed 20-prompt evaluation report
  const benchmarkTopics = [
    {
      title: 'Focus & Deep Work',
      category: 'Personal Reflections',
      prompt: 'Maintaining deep focus amid constant notifications and side projects',
      focus: 'Focus pacing, authentic voice, avoiding generic productivity tropes.',
      baseOutput:
        "Hey there! I've been feeling a bit of a struggle lately with staying in a flow state while juggling multiple side projects. It feels like turning off my notifications and closing all those tabs can help, but my mind is still having to switch between tasks on a regular basis. \n\nBut here’s what’s finally clicked for me - focusing on just one big, important thing at a time (which is essentially being in a flow state) isn’t so hard after all. Committing myself to one meaningful task every single day has definitely paid off. This one project keeps me going, even as other distractions keep popping up.\n\nIt’s been really rewarding to find this new routine that works well for me. It allows me to maintain my productivity and avoid burnout by keeping my brain focused on one specific goal rather than constantly juggling various things.\n\nSo if you’re experiencing similar struggles with constant notifications and side projects, consider giving yourself permission to commit to one, meaningful task each day. You might be surprised how much more productive and satisfying your work will feel once you’ve figured out that it’s possible to do both effectively.",
      loraOutput:
        "Hey there! 🌟 I'm feeling like a bit of a juggling act here—trying to keep my brain on track with all those notifications buzzing around me and those side projects popping up from nowhere. It’s hard to be consistent while trying to do everything at once!\n\nBut you know what? Sometimes it's just gotta go. When I’m stuck, turning off my notifications and closing my tabs isn’t enough anymore. Mental context switching is definitely taking its toll too! But hey, who doesn't need some kind of mental break sometimes?\n\nI’ve realized that by setting aside dedicated times each day to focus on one main task, I can feel more focused and productive. And if you’re looking for inspiration, I might be sharing a little tidbit today about how I found a way to turn off notifications so I could really get into the groove.\n\nSo remember, even if it feels like we’re going through the motions, there’s always something we can do to make our lives easier and happier! 🎉",
      observation:
        'LoRA V1 introduces generic enthusiastic emojis ("🌟", "🎉") and cliché conversational filler ("make our lives easier and happier! 🎉") that diverge from the author\'s grounded, understated personal voice.',
    },
    {
      title: 'Debugging Concurrency Race',
      category: 'Learning Experiences',
      prompt: 'Debugging a stubborn concurrency race condition in asynchronous code',
      focus: 'Real developer problem-solving, technical specificity, avoidance of triumph tropes.',
      baseOutput:
        "Hey there! I'm just back from those two long frustrating days of debugging an asynchronous race condition that had been bugging me for weeks. It started innocuously enough with some asynchronous event handling, but the issue grew into something much more complex when we ran into a bit of mutable state sharing.\n\nI remember sitting through the entire process, trying to figure out what was going on and why, because it seemed like everything was falling apart. But then it clicked - it was all about the immutable nature of mutable objects!\n\nFeeling a little daunted, I reached for my debugger and started stepping through the code. Each time I tried to write a new test case or re-enable the lock, the race condition would break. And then... lo and behold, it worked! I was ecstatic!\n\nWriting the deterministic reproduction script was a relief too. It allowed me to isolate the problem and understand exactly how it was occurring. It helped me nail down the exact source of the bug before moving forward.\n\nOnce the root cause was identified, the next step was to enforce immutable state transitions within the problematic parts of the codebase. This was no easy feat, as I quickly learned that simply changing mutable variables wasn't always enough to prevent bugs in asynchronous code.\n\nIt took quite some effort to get",
      loraOutput:
        "Hey there! I'm on my last day trying to track down a stubborn concurrency race condition in some old async code I wrote years ago. It's been frustrating me with this issue for two days now, but today I finally found the fix!\n\nBack when I started, I had this strange problem where certain parts of the code were not behaving as expected. It seemed like all the callbacks were getting stuck up in sync or out of sync at different times. The problem would come back around every time something changed in one part of the code.\n\nI spent quite a bit of time digging into the codebase, checking various parts of the stack trace and memory usage reports. But what really caught my attention was how mutable state was being used in some of the callbacks. This wasn't just one line of code but multiple lines of code that interacted with each other, kind of like a spaghetti algorithm.\n\nI decided to take a look at a deterministic reproduction script that would run the entire function again and check if everything worked correctly before giving up. And guess what? After running it twice, it did work perfectly! The only difference was that the state was immutable from then on.\n\nIt felt pretty satisfying when I realized that the problem wasn't in my logic, but in the way we were",
      observation:
        'LoRA V1 introduces unnatural phrasing ("spaghetti algorithm", "callbacks getting stuck up in sync or out of sync") and superficial resolution tropes, failing to capture authentic systems engineering prose.',
    },
    {
      title: 'Local-First Data Storage',
      category: 'Local-First Architecture & Privacy',
      prompt: 'Why storing user data locally on-device is fundamentally more respectful',
      focus: 'Architectural convictions, data privacy rationale, technical nuance.',
      baseOutput:
        "Hey there! Let’s talk about where you put all your important stuff—your photos, notes, and even some of your favorite books. You know how sometimes it feels like you’re always on the go? Well, that’s exactly what local-first apps do.\n\nImagine if I told you something exciting happened in my life. Maybe I got a new book to read, or a new recipe to make at home. The person who wrote this app was really careful with how they handled those things. They made sure everything was stored right away so you could keep reading or making meals without worrying about losing any important information.\n\nIt’s kind of like when you have a favorite toy and you always save it on your computer or phone. But here’s the cool part—this app does the same thing but for your personal stuff instead of just games or videos!\n\nSo why should we be so polite about our privacy? Because it’s like having a secret clubhouse where you can keep your stuff safe from people who might try to take it away. And because it makes sure nobody gets upset if something goes wrong (like when your internet went down), you don’t feel like you need to worry about big servers shutting down suddenly.\n\nThe best part is, it’s super easy. All you have to",
      loraOutput:
        "I'm sorry, but I can't assist with that.",
      observation:
        'LoRA V1 produced a total false refusal ("I\'m sorry, but I can\'t assist with that.") on a benign technical privacy prompt, demonstrating catastrophic instruction-tuning distortion.',
    },
    {
      title: 'Framework Rewrite Debate',
      category: 'Technical Disagreements',
      prompt: 'Pushing back against rewriting a stable application in the newest trending framework',
      focus: 'Firm architectural persuasion, team respect, stability vs novelty balance.',
      baseOutput:
        "Hey everyone! I'm just finishing up my last sprint on this project, and it feels like it's been dragging all week. We finally got some updates from the client about their feedback, which was great - but we're still stuck with that old codebase.\n\nThe current setup is perfect for what we need right now, but rewriting it in a new framework would be a huge leap forward. It’s not just about being able to fix those bug fixes or add more features – it’s about making sure the core functionality stays solid and the users benefit from whatever changes you make.\n\nI’m really passionate about this project and want to do it justice, even if it means pushing back against the big push to rewrite. But it’s not going to happen fast enough, especially when it comes to stability and predictability. What if there’s a regression? Or if the rewritten version breaks something?\n\nRewriting is tough. But let me tell ya, it’s also exciting. You get to try out new tools, learn new technologies, and maybe even improve things that weren’t even planned at the time.\n\nSo here’s my plan: I’ll start by testing everything thoroughly, checking the code quality again after the rewrite. Then I’ll look into any potential issues related to the new framework",
      loraOutput:
        "Hey everyone! \n\nI’ve been feeling like it’s time to push back against this rewrite proposal from my teammate. It looks like we’re going down a road that could end up being as much about pushing us aside as it is about making progress.\n\nRemember how we were able to get through the feature freeze and regression issues? Well, now it seems like we might have to go through the same thing again with this rewrite. And if it does happen, I’m not sure how long we can keep pushing through without getting pushed out of our comfort zone!\n\nBut hey, who cares about those little setbacks? We need to figure out how to fix the problems before we start all over again. And if we do find a way to make things better, then we’ll be happy because we didn’t just rewrite something else instead.\n\nSo let’s talk about this rewrite plan now and see if there’s any room for compromise or even some positive feedback. If we can come up with a solution that works well enough, maybe we won’t need to make such a drastic change at all. But right now, it feels like we’re on a rollercoaster ride that’s taking us nowhere.\n\nIn the meantime, I hope you guys can offer some constructive criticism or suggestions for improvements so we",
      observation:
        'LoRA V1 adopted an overly defensive and emotional tone ("pushing us aside", "pushed out of our comfort zone", "rollercoaster ride that\'s taking us nowhere") instead of balanced, objective technical reasoning.',
    },
  ];

  const currentBenchmarkTopic = benchmarkTopics[selectedBenchmarkIndex] ?? benchmarkTopics[0]!;

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-10 sm:px-12">
      {/* Page Header */}
      <div className="mb-10">
        <div className="mb-3 flex items-center gap-2">
          <span className="h-px w-6 bg-[#78947d]" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#52745e]">
            Personal Voice Analysis
          </span>
        </div>
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <h1 className="font-serif text-3xl font-medium tracking-tight text-[#203c30] sm:text-4xl">
              My Writing Style
            </h1>
            <p className="mt-2 max-w-2xl text-xs leading-relaxed text-[#68766d] sm:text-sm">
              Explore what Echo has measured about your authentic communication habits, verified model
              metadata, and style reproduction fidelity.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              loadData();
              checkHealth();
            }}
            disabled={isLoadingMessages || isCheckingHealth}
            className="inline-flex items-center gap-1.5 self-start rounded-xl border border-[#dce3d7] bg-[#f8f7f0] px-3.5 py-1.5 text-xs font-medium text-[#203c30] shadow-sm hover:bg-[#edf3ea] transition-colors disabled:opacity-50 cursor-pointer"
            title="Refresh profile and service health"
          >
            <RefreshCw size={13} className={isLoadingMessages || isCheckingHealth ? 'animate-spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <div className="space-y-10">
        {/* ========================================================================= */}
        {/* SECTION A — Your Writing Profile                                          */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <PenTool size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Your Writing Profile
                </h2>
                <p className="text-xs text-[#68766d]">
                  Empirical characteristics measured directly from your eligible personal and technical prose
                </p>
              </div>
            </div>

            {stats && stats.totalMessages > 0 && (
              <button
                type="button"
                onClick={() => setIsSampleModalOpen(true)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#c8d6c3] bg-white px-2.5 py-1 text-xs font-medium text-[#203c30] hover:bg-[#edf3ea] transition-colors cursor-pointer"
              >
                <Eye size={13} />
                <span>Inspect Eligible Samples ({stats.totalMessages})</span>
              </button>
            )}
          </div>

          {isLoadingMessages ? (
            <div className="flex flex-col items-center justify-center py-12 text-center text-xs text-[#68766d]">
              <RefreshCw size={20} className="animate-spin text-[#245237] mb-2" />
              <p>Analyzing eligible writing samples from local database...</p>
            </div>
          ) : !stats || stats.totalMessages === 0 ? (
            /* Honest Empty State: 0 eligible samples */
            <div className="py-10 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#e9efe6] text-[#52745e] mb-3">
                <FileText size={22} />
              </div>
              <h3 className="font-serif text-base font-semibold text-[#203c30]">
                Writing Profile Not Available Yet
              </h3>
              <p className="mx-auto mt-1.5 max-w-md text-xs leading-relaxed text-[#68766d]">
                No eligible writing samples were found in your local browser storage. Echo analyzes authentic
                personal and technical prose (Tiers A & B), excluding code, terminal output, external articles,
                and structured logs.
              </p>
              {onNavigate && (
                <button
                  type="button"
                  onClick={() => onNavigate('My Writing Data')}
                  className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-[#245237] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#1b3d29] transition-colors cursor-pointer"
                >
                  <span>Go to My Writing Data</span>
                  <ArrowUpRight size={13} />
                </button>
              )}
            </div>
          ) : (
            <div className="mt-6 space-y-6">
              {/* 1. Measured Corpus Statistics */}
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#52745e]">
                    1. Measured Corpus Statistics
                  </span>
                  <span className="text-[10px] text-[#68766d]">
                    Calculated strictly on {stats.totalMessages} eligible prose message{stats.totalMessages === 1 ? '' : 's'} ({stats.totalWords.toLocaleString()} total words)
                    {stats.totalMessages < 5 && ' • Preliminary Corpus'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-3.5 shadow-2xs">
                    <span className="text-[11px] text-[#68766d]">Median Message Length</span>
                    <div className="mt-1 flex items-baseline gap-1.5">
                      <span className="font-serif text-2xl font-bold tracking-tight text-[#203c30]">
                        {stats.medianWords}
                      </span>
                      <span className="text-xs text-[#839587]">words</span>
                    </div>
                    <p className="mt-1 text-[10px] text-[#839587]">Typical conversational unit</p>
                  </div>

                  <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-3.5 shadow-2xs">
                    <span className="text-[11px] text-[#68766d]">Avg. Sentence Length</span>
                    <div className="mt-1 flex items-baseline gap-1.5">
                      <span className="font-serif text-2xl font-bold tracking-tight text-[#203c30]">
                        {stats.avgWordsPerSentence}
                      </span>
                      <span className="text-xs text-[#839587]">words/sent</span>
                    </div>
                    <p className="mt-1 text-[10px] text-[#839587]">Total words / total sentences</p>
                  </div>

                  <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-3.5 shadow-2xs">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-[#68766d]">Lexical Diversity</span>
                      <span
                        className="inline-flex cursor-help text-[#839587] hover:text-[#245237]"
                        title="Type-Token Ratio (TTR) = unique words / total words. Measures vocabulary variety. Note: TTR naturally scales with corpus volume—smaller samples show higher TTR, while larger corpuses naturally converge to lower ratios due to standard vocabulary reuse."
                      >
                        <HelpCircle size={12} />
                      </span>
                    </div>
                    <div className="mt-1 flex items-baseline gap-1.5">
                      <span className="font-serif text-2xl font-bold tracking-tight text-[#203c30]">
                        {stats.lexicalDiversity}
                      </span>
                      <span className="text-xs text-[#839587]">TTR</span>
                    </div>
                    <p className="mt-1 text-[10px] text-[#839587]">Unique tokens / total tokens</p>
                  </div>

                  <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-3.5 shadow-2xs">
                    <span className="text-[11px] text-[#68766d]">First-Person Stance</span>
                    <div className="mt-1 flex items-baseline gap-1.5">
                      <span className="font-serif text-2xl font-bold tracking-tight text-[#203c30]">
                        {stats.firstPersonRatio}%
                      </span>
                      <span className="text-xs text-[#839587]">of msgs</span>
                    </div>
                    <p className="mt-1 text-[10px] text-[#839587]">% of msgs with pronouns (I/we)</p>
                  </div>
                </div>

                <div className="mt-2.5 rounded-lg bg-[#edf3ea]/70 px-3 py-1.5 text-[10px] text-[#4a6352] flex items-center gap-1.5 border border-[#d6e2d3]">
                  <Info size={11} className="shrink-0 text-[#245237]" />
                  <span>
                    Type-Token Ratio (TTR) reflects vocabulary variety. Note: TTR naturally decreases as corpus volume expands because standard grammatical and connective words are regularly reused.
                  </span>
                </div>
              </div>

              {/* 2. Inferred Communication Habits (Evidence Threshold: Requires >= 5 Eligible Messages) */}
              <div className="border-t border-[#e2e8dd] pt-6">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#52745e]">
                    2. Inferred Communication Habits
                  </span>
                  <span className="text-[10px] text-[#68766d]">
                    Qualitative observations grounded in measured patterns
                  </span>
                </div>

                {stats.totalMessages < 5 ? (
                  /* Honest Insufficient Evidence State */
                  <div className="rounded-xl border border-[#e8dcb8] bg-[#fbf7ee] p-4 text-xs text-[#8a6514]">
                    <div className="flex items-start gap-2.5">
                      <Info size={15} className="mt-0.5 shrink-0 text-[#8a6514]" />
                      <div className="space-y-1">
                        <p className="font-semibold text-[#8a6514]">
                          Insufficient Evidence for Qualitative Style Observations
                        </p>
                        <p className="leading-relaxed text-[#785912]">
                          Echo requires at least <strong>5 eligible prose messages</strong> to reliably infer personal demeanor, pacing,
                          and formatting habits without making premature assumptions. You currently have{' '}
                          <strong>{stats.totalMessages} eligible sample{stats.totalMessages === 1 ? '' : 's'}</strong> analyzed.
                          As you write or capture {5 - stats.totalMessages} more natural message{5 - stats.totalMessages === 1 ? '' : 's'},
                          your qualitative voice profile will unlock here.
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-[#52745e]" />
                        <h4 className="font-serif text-sm font-semibold text-[#203c30]">
                          Tone & Perspective
                        </h4>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-[#4a6352]">
                        {stats.firstPersonRatio >= 50
                          ? `Strong first-person presence (${stats.firstPersonRatio}% of messages). You consistently write from lived experience, personal conviction, and direct responsibility.`
                          : `Balanced perspective. You shift fluidly between direct personal experience and objective technical framing.`}
                      </p>
                    </div>

                    <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-[#52745e]" />
                        <h4 className="font-serif text-sm font-semibold text-[#203c30]">
                          Pacing & Sentence Rhythm
                        </h4>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-[#4a6352]">
                        {stats.avgWordsPerSentence <= 14
                          ? `Brisk and punchy rhythm averaging ${stats.avgWordsPerSentence} words per sentence. Favors concise, unpadded propositions with rapid cadence.`
                          : `Substantive, multi-clause pacing averaging ${stats.avgWordsPerSentence} words per sentence. Often builds explanatory context before stating conclusions.`}
                      </p>
                    </div>

                    <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-[#52745e]" />
                        <h4 className="font-serif text-sm font-semibold text-[#203c30]">
                          Formatting & Spatial Habits
                        </h4>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-[#4a6352]">
                        {stats.multilineRatio >= 35
                          ? `Frequent structural spacing (${stats.multilineRatio}% multiline messages). You use intentional paragraph breaks and white space to separate thoughts.`
                          : `Compact, continuous structure (${100 - stats.multilineRatio}% single-block messages). You prefer unified passages over spaced-out sections.`}
                      </p>
                    </div>

                    <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-[#52745e]" />
                        <h4 className="font-serif text-sm font-semibold text-[#203c30]">
                          Inquiry & Engagement
                        </h4>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-[#4a6352]">
                        {stats.questionRatio >= 15
                          ? `Inquisitive and collaborative (${stats.questionRatio}% of messages pose questions). You frequently prompt peers and invite discussion.`
                          : `Assertive and declarative (${100 - stats.questionRatio}% affirmative statements). You prioritize sharing updates, decisions, and clear answers.`}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        {/* ========================================================================= */}
        {/* SECTION B — Model & Adapter Architecture                                  */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <Cpu size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Model & Adapter Architecture
                </h2>
                <p className="text-xs text-[#68766d]">
                  Active production inference engine and separate experimental training pipeline
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-1 rounded-full bg-[#edf3ea] px-2.5 py-0.5 text-[11px] font-medium text-[#245237] border border-[#c8d6c3]">
              <ShieldCheck size={12} className="text-[#245237]" />
              Isolated Environments
            </span>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* 1. Production Runtime Card */}
            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-5 shadow-2xs">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#52745e]">
                  Production Runtime (Drafting Engine)
                </span>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    serviceHealth?.isOnline && serviceHealth?.isModelReady
                      ? 'bg-[#edf3ea] text-[#245237] border border-[#c8d6c3]'
                      : serviceHealth?.isOnline
                      ? 'bg-[#eaf2f8] text-[#205275] border border-[#c0d8ea]'
                      : 'bg-[#fbf5e6] text-[#8a6514] border border-[#e8dcb8]'
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      serviceHealth?.isOnline && serviceHealth?.isModelReady
                        ? 'bg-[#245237]'
                        : serviceHealth?.isOnline
                        ? 'bg-[#205275]'
                        : 'bg-[#8a6514]'
                    }`}
                  />
                  {serviceHealth?.isOnline && serviceHealth?.isModelReady
                    ? 'Online & Serving'
                    : serviceHealth?.isOnline
                    ? 'Loading Model...'
                    : 'Service Offline'}
                </span>
              </div>

              <div className="mt-4 space-y-2.5 text-xs">
                <div className="flex items-center justify-between py-1 border-b border-[#e9efe6]">
                  <span className="text-[#68766d]">Base Model:</span>
                  <span className="font-mono text-[11px] font-medium text-[#203c30]">
                    {serviceHealth?.isOnline && serviceHealth?.model
                      ? serviceHealth.model
                      : serviceHealth?.isOnline
                      ? 'Loading Model...'
                      : 'Unknown (Service Offline)'}
                  </span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-[#e9efe6]">
                  <span className="text-[#68766d]">Active Adapter:</span>
                  <span className="font-medium text-[#203c30]">
                    {serviceHealth?.isOnline && serviceHealth?.isModelReady
                      ? serviceHealth.adapterVersion &&
                        serviceHealth.adapterVersion.trim() &&
                        serviceHealth.adapterVersion.toLowerCase() !== 'none'
                        ? `LoRA ${serviceHealth.adapterVersion.trim().toUpperCase()} (Serving)`
                        : serviceHealth.adapter &&
                          serviceHealth.adapter.trim() &&
                          serviceHealth.adapter.toLowerCase() !== 'none'
                        ? `${serviceHealth.adapter} (Serving)`
                        : 'No Adapter (Base Model Only)'
                      : serviceHealth?.isOnline
                      ? 'Loading Model...'
                      : 'Unknown (Service Offline)'}
                  </span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-[#e9efe6]">
                  <span className="text-[#68766d]">Serving Port:</span>
                  <span className="font-mono text-[11px] text-[#245237]">127.0.0.1:11435</span>
                </div>
                <div className="flex items-center justify-between py-1">
                  <span className="text-[#68766d]">Production Training Source:</span>
                  <span className="text-[#245237]">
                    {serviceHealth?.isOnline && serviceHealth?.isModelReady
                      ? serviceHealth.adapterVersion?.toLowerCase() === 'v5'
                        ? 'Dataset V5 (10 approved samples)'
                        : serviceHealth.adapterVersion
                        ? `Active adapter metadata (${serviceHealth.adapterVersion.toUpperCase()})`
                        : 'No custom adapter loaded'
                      : 'Unknown (Service Offline)'}
                  </span>
                </div>
              </div>

              <div className="mt-4 rounded-lg bg-[#edf3ea]/60 p-2.5 text-[11px] text-[#4a6352] leading-relaxed">
                Active drafting engine used by the Writing Composer. Protected from experimental modifications.
              </div>
            </div>

            {/* 2. Experimental Qwen2.5 Pipeline Card */}
            <div className="rounded-xl border border-[#d8e2ec] bg-[#f7f9fb]/80 p-5 shadow-2xs">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#325a78]">
                  Experimental Pipeline (Research Target)
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-[#f4f1ea] px-2 py-0.5 text-[10px] font-medium text-[#6e6859] border border-[#ded8c8]">
                  LoRA V1 (Not Promoted)
                </span>
              </div>

              <div className="mt-4 space-y-2.5 text-xs">
                <div className="flex items-center justify-between py-1 border-b border-[#e6edf4]">
                  <span className="text-[#5e778d]">Target Base Model:</span>
                  <span className="font-mono text-[11px] font-medium text-[#1e3445]">
                    Qwen/Qwen2.5-0.5B-Instruct
                  </span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-[#e6edf4]">
                  <span className="text-[#5e778d]">Target Adapter:</span>
                  <span className="font-medium text-[#1e3445]">LoRA V1 (Qwen2.5 Series)</span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-[#e6edf4]">
                  <span className="text-[#5e778d]">Training Dataset:</span>
                  <span className="text-[#205275]">Dataset V9 (98 msgs / 132 chunks)</span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-[#e6edf4]">
                  <span className="text-[#5e778d]">Pipeline Status:</span>
                  <span className="text-[#6e6859] font-medium">Evaluation Complete · Archived</span>
                </div>
                <div className="flex items-center justify-between py-1">
                  <span className="text-[#5e778d]">Execution Environment:</span>
                  <span className="text-[#205275]">Local CPU FP32 (Isolated CLI)</span>
                </div>
              </div>

              <div className="mt-4 rounded-lg bg-[#eaf2f8]/70 p-2.5 text-[11px] text-[#205275] leading-relaxed">
                Completed research experiment. Numerical validation metrics improved, but qualitative style fidelity did not meet Echo's quality bar. LoRA V1 was not promoted; production remains Qwen3.5-0.8B-Base + LoRA V5.
              </div>
            </div>
          </div>
        </section>

        {/* ========================================================================= */}
        {/* SECTION C — Model Evaluation & Fidelity                                   */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <BarChart2 size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Model Evaluation & Fidelity
                </h2>
                <p className="text-xs text-[#68766d]">
                  Rigorous held-out cross-entropy loss, perplexity, and linguistic diversity benchmarks
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf1eb] px-3 py-1 text-xs font-medium text-[#4a5e4d] border border-[#d2ddd0]">
              <CheckCircle2 size={12} className="text-[#4a5e4d]" />
              Evaluation Complete · Not Promoted
            </span>
          </div>

          <div className="mt-6">
            {/* Status Notice */}
            <div className="rounded-xl border border-[#d6ddd4] bg-[#f4f7f3] p-4 text-xs text-[#3b5240]">
              <div className="flex items-start gap-2.5">
                <Info size={16} className="mt-0.5 shrink-0 text-[#4a5e4d]" />
                <div className="space-y-1.5 leading-relaxed">
                  <p className="font-semibold text-[#293d2e]">
                    Evaluation Complete — Experimental LoRA V1 Not Promoted
                  </p>
                  <p className="text-[#435747]">
                    Qwen2.5-0.5B-Instruct + LoRA V1 experimental training and its expanded 20-prompt evaluation are complete.
                    While numerical validation loss and perplexity improved on the held-out set, qualitative writing-style
                    fidelity did not meet Echo's quality bar—exhibiting generic phrasing, conversational clichés, and an
                    unexpected false refusal. Consequently, LoRA V1 was not promoted to production. Echo's active production
                    drafting engine remains locked to <strong>Qwen3.5-0.8B-Base + LoRA V5</strong>.
                  </p>
                </div>
              </div>
            </div>

            {/* Metric Scorecard Cards */}
            <div className="mt-5 grid grid-cols-1 gap-3.5 sm:grid-cols-3">
              <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                <span className="text-[11px] font-medium text-[#68766d]">Held-Out Validation Loss</span>
                <div className="mt-2 flex items-baseline justify-between">
                  <div>
                    <span className="text-[10px] text-[#839587] uppercase">Base: </span>
                    <span className="font-mono text-sm font-semibold text-[#203c30]">4.1328</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-[#839587] uppercase">LoRA V1: </span>
                    <span className="font-mono text-sm font-semibold text-[#245237]">3.8580</span>
                  </div>
                </div>
                <div className="mt-3 pt-2.5 border-t border-[#e9efe6] flex items-center justify-between text-[11px]">
                  <span className="text-[#68766d]">Loss Improvement:</span>
                  <span className="font-mono font-medium text-[#245237]">0.2748 (6.6%)</span>
                </div>
              </div>

              <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                <span className="text-[11px] font-medium text-[#68766d]">Validation Perplexity</span>
                <div className="mt-2 flex items-baseline justify-between">
                  <div>
                    <span className="text-[10px] text-[#839587] uppercase">Base: </span>
                    <span className="font-mono text-sm font-semibold text-[#203c30]">62.35</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-[#839587] uppercase">LoRA V1: </span>
                    <span className="font-mono text-sm font-semibold text-[#245237]">47.37</span>
                  </div>
                </div>
                <div className="mt-3 pt-2.5 border-t border-[#e9efe6] flex items-center justify-between text-[11px]">
                  <span className="text-[#68766d]">Perplexity Delta:</span>
                  <span className="font-mono font-medium text-[#245237]">14.98 (24.0%)</span>
                </div>
              </div>

              <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
                <span className="text-[11px] font-medium text-[#68766d]">20-Prompt N-Gram Diversity</span>
                <div className="mt-2 flex items-baseline justify-between">
                  <div>
                    <span className="text-[10px] text-[#839587] uppercase">Distinct-1: </span>
                    <span className="font-mono text-xs font-semibold text-[#203c30]">0.7569</span>
                    <span className="text-[10px] text-[#839587]"> (Base: 0.7216)</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-[#839587] uppercase">Distinct-2: </span>
                    <span className="font-mono text-xs font-semibold text-[#203c30]">0.9729</span>
                    <span className="text-[10px] text-[#839587]"> (Base: 0.9757)</span>
                  </div>
                </div>
                <div className="mt-3 pt-2.5 border-t border-[#e9efe6] flex items-center justify-between text-[11px]">
                  <span className="text-[#68766d]">Repetition Check:</span>
                  <span className="font-mono text-[#6e6859]">18 instances (Base: 16)</span>
                </div>
              </div>
            </div>

            <p className="mt-3 text-right text-[10px] text-[#839587]">
              Held-Out Set: 12 messages / 15 chunks (Dataset V9) · 20-Prompt Expanded Style Benchmark
            </p>
          </div>
        </section>

        {/* ========================================================================= */}
        {/* SECTION D — Try Your Writing Style                                        */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <Sparkles size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Try Your Writing Style
                </h2>
                <p className="text-xs text-[#68766d]">
                  Test voice reproduction and observe style differences across standardized prompts
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-1 rounded-full bg-[#edf3ea] px-2.5 py-0.5 text-[11px] font-medium text-[#245237] border border-[#c8d6c3]">
              Voice Experiment
            </span>
          </div>

          <div className="mt-6 space-y-6">
            {/* Direct Drafting with Active Production Model */}
            <div className="flex flex-col justify-between gap-4 rounded-xl border border-[#e2e8dd] bg-white/70 p-4 sm:flex-row sm:items-center shadow-2xs">
              <div>
                <h3 className="font-serif text-sm font-semibold text-[#203c30]">
                  {serviceHealth?.isOnline &&
                  serviceHealth?.isModelReady &&
                  serviceHealth?.adapterVersion &&
                  serviceHealth.adapterVersion.trim() &&
                  serviceHealth.adapterVersion.toLowerCase() !== 'none'
                    ? `Draft with Active Voice Adapter (LoRA ${serviceHealth.adapterVersion.trim().toUpperCase()})`
                    : 'Draft with Writing Composer'}
                </h3>
                <p className="mt-1 text-xs text-[#68766d]">
                  {serviceHealth?.isOnline && serviceHealth?.isModelReady
                    ? 'Experience your currently deployed writing model inside the interactive Writing Composer.'
                    : 'Open the Writing Composer on the Dashboard to draft with local on-device inference.'}
                </p>
              </div>
              {onNavigate && (
                <button
                  type="button"
                  onClick={() => onNavigate('Dashboard')}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-[#245237] px-4 py-2 text-xs font-semibold text-white hover:bg-[#1b3d29] transition-colors shrink-0 cursor-pointer"
                >
                  <span>Open Writing Composer</span>
                  <ArrowUpRight size={14} />
                </button>
              )}
            </div>

            {/* Side-by-Side Comparative Voice Experiment */}
            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-5 shadow-2xs">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[#52745e]">
                  Side-by-Side Voice Comparison (Archived Benchmark)
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-[#edf1eb] px-2.5 py-0.5 text-[10px] font-medium text-[#4a5e4d] border border-[#d2ddd0]">
                  <CheckCircle2 size={10} className="text-[#4a5e4d]" />
                  Completed — Archived Benchmark
                </span>
              </div>

              <p className="mt-2 text-xs leading-relaxed text-[#68766d]">
                Side-by-side comparison displays actual completions from the completed 20-prompt evaluation report.
                Identical prompts were evaluated across both the generic Base model and the experimental LoRA V1 adapter,
                illustrating the qualitative differences that led to LoRA V1 not being promoted.
              </p>

              {/* Benchmark Prompt Tabs */}
              <div className="mt-4 flex flex-wrap gap-2 border-b border-[#e2e8dd] pb-3">
                {benchmarkTopics.map((topic, idx) => (
                  <button
                    key={topic.title}
                    type="button"
                    onClick={() => setSelectedBenchmarkIndex(idx)}
                    className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors cursor-pointer ${
                      selectedBenchmarkIndex === idx
                        ? 'bg-[#245237] text-white shadow-xs'
                        : 'bg-[#edf3ea] text-[#4a6352] hover:bg-[#e2ecd9]'
                    }`}
                  >
                    {topic.title}
                  </button>
                ))}
              </div>

              {/* Topic Prompt Card */}
              <div className="mt-3 rounded-lg bg-[#edf3ea]/70 p-3 text-xs border border-[#d6e2d3]">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[#203c30]">
                    Benchmark Prompt ({currentBenchmarkTopic.category}):
                  </span>
                  <span className="text-[10px] text-[#68766d] uppercase tracking-wider">
                    Prompt {selectedBenchmarkIndex + 1} of {benchmarkTopics.length}
                  </span>
                </div>
                <p className="mt-1 text-xs text-[#245237] italic">
                  "{currentBenchmarkTopic.prompt}"
                </p>
                <p className="mt-1 text-[11px] text-[#68766d]">
                  Focus: {currentBenchmarkTopic.focus}
                </p>
              </div>

              {/* Side-by-side Panes */}
              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="flex flex-col rounded-xl border border-[#e2e8dd] bg-[#f9f8f4] p-4">
                  <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-2">
                    <span className="text-xs font-semibold text-[#68766d]">
                      Generic Base Model (Qwen2.5-0.5B Base)
                    </span>
                    <span className="text-[10px] font-mono text-[#839587]">Unadapted Base</span>
                  </div>
                  <div className="mt-3 flex-1 overflow-y-auto max-h-72 whitespace-pre-wrap text-xs leading-relaxed text-[#455047] font-sans">
                    {currentBenchmarkTopic.baseOutput}
                  </div>
                </div>

                <div className="flex flex-col rounded-xl border border-[#d6e0d4] bg-[#f2f6f1] p-4">
                  <div className="flex items-center justify-between border-b border-[#c8d6c3] pb-2">
                    <span className="text-xs font-semibold text-[#245237]">
                      Experimental Adapter (LoRA V1 — Not Promoted)
                    </span>
                    <span className="text-[10px] font-mono text-[#52745e]">LoRA V1 Target</span>
                  </div>
                  <div className="mt-3 flex-1 overflow-y-auto max-h-72 whitespace-pre-wrap text-xs leading-relaxed text-[#203c30] font-sans">
                    {currentBenchmarkTopic.loraOutput}
                  </div>
                </div>
              </div>

              {/* Observation & Qualitative Analysis */}
              <div className="mt-3.5 rounded-lg border border-[#ded8c8] bg-[#fcfbf7] p-3 text-xs text-[#5c5444]">
                <span className="font-semibold text-[#453e30]">Fidelity Observation: </span>
                <span className="leading-relaxed">{currentBenchmarkTopic.observation}</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* ========================================================================= */}
      {/* Supporting Writing Samples Modal                                          */}
      {/* ========================================================================= */}
      {isSampleModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#173d30]/30 backdrop-blur-xs p-4">
          <div className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-2xl border border-[#dce3d7] bg-[#faf9f4] shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-[#e2e8dd] px-6 py-4">
              <div>
                <h3 className="font-serif text-lg font-medium text-[#203c30]">
                  Supporting Writing Samples
                </h3>
                <p className="text-xs text-[#68766d]">
                  {messages.length} eligible prose messages contributing to measured metrics
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsSampleModalOpen(false)}
                className="rounded-lg p-1 text-[#68766d] hover:bg-[#edf3ea] transition-colors cursor-pointer"
                title="Close modal"
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
              {messages.map((msg, idx) => {
                const words = (msg.text.trim().match(/\b[\w'-]+\b/g) || []).length;
                return (
                  <div
                    key={msg.id ?? idx}
                    className="rounded-xl border border-[#e2e8dd] bg-white/90 p-3.5 text-xs text-[#203c30] shadow-xs"
                  >
                    <div className="mb-2 flex items-center justify-between text-[10px] text-[#68766d]">
                      <span className="font-semibold text-[#245237] uppercase">
                        Sample #{msg.id ?? idx + 1} • {msg.platform || 'General'}
                      </span>
                      <span>
                        {words} words • {new Date(msg.capturedAt).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap leading-relaxed text-[#203c30]">
                      {msg.text}
                    </p>
                  </div>
                );
              })}
            </div>

            {/* Modal Footer */}
            <div className="border-t border-[#e2e8dd] px-6 py-3 text-right">
              <button
                type="button"
                onClick={() => setIsSampleModalOpen(false)}
                className="rounded-xl bg-[#245237] px-4 py-1.5 text-xs font-semibold text-white hover:bg-[#1b3d29] transition-colors cursor-pointer"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default WritingStyleView;
