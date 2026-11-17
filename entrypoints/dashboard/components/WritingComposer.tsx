import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ArrowUpRight,
  Globe2,
  Sparkles,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  AlertCircle,
  AlertTriangle,
  Info,
  Brain,
  Trash2,
} from 'lucide-react';

import {
  faXTwitter,
  faLinkedinIn,
} from '@fortawesome/free-brands-svg-icons';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

import {
  checkGenerationServiceHealth,
  generatePersonalizedDraft,
  type GenerationServiceHealth,
  type GenerationResponse,
  type GenerationLengthMode,
  type GenerationPlatform,
} from '../../utils/generationClient';

import {
  retrieveRelevantKnowledge,
  type RetrievedKnowledgeResult,
} from '../../utils/knowledgeRetrieval';

import {
  copyToClipboard,
} from '../../utils/extensionInfo';

import { DEFAULT_OLLAMA_MODEL } from '../../utils/ollamaClient';

function WritingComposer() {
  const [topic, setTopic] = useState('');
  const [leadIn, setLeadIn] = useState('');
  const [platform, setPlatform] = useState('X');
  const [lengthMode, setLengthMode] = useState<GenerationLengthMode>('medium');

  // Generation & Service States
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [serviceStatus, setServiceStatus] = useState<GenerationServiceHealth | null>(null);
  const [isCheckingService, setIsCheckingService] = useState(false);

  // Knowledge & Output States
  const [retrievedKnowledge, setRetrievedKnowledge] = useState<RetrievedKnowledgeResult | null>(null);
  const [isKnowledgeExpanded, setIsKnowledgeExpanded] = useState(false);
  const [generatedDraft, setGeneratedDraft] = useState('');
  const [generationMeta, setGenerationMeta] = useState<GenerationResponse | null>(null);
  const [isCopied, setIsCopied] = useState(false);

  const platforms = [
    {
      name: 'X',
      icon: faXTwitter,
    },
    {
      name: 'LinkedIn',
      icon: faLinkedinIn,
    },
    {
      name: 'General',
      icon: null,
    },
  ];

  const isMountedRef = useRef(true);
  const isCheckingRef = useRef(false);

  const checkHealth = useCallback(async () => {
    if (isCheckingRef.current) {
      return;
    }
    isCheckingRef.current = true;
    if (isMountedRef.current) {
      setIsCheckingService(true);
    }

    try {
      const health = await checkGenerationServiceHealth();
      if (isMountedRef.current) {
        setServiceStatus(health);
      }
    } catch {
      if (isMountedRef.current) {
        setServiceStatus({
          isOnline: false,
          isModelReady: false,
          status: 'offline',
          error: 'Unable to reach local generation service.',
        });
      }
    } finally {
      isCheckingRef.current = false;
      if (isMountedRef.current) {
        setIsCheckingService(false);
      }
    }
  }, []);

  // Track component mount status for cleanup
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Initial health check on component mount
  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Automatic retry when service is offline or loading model
  useEffect(() => {
    // Only poll every 5 seconds if the service is offline or still loading the model
    if (serviceStatus !== null && serviceStatus.status !== 'ready') {
      const interval = setInterval(() => {
        checkHealth();
      }, 5000);

      return () => clearInterval(interval);
    }
    // When service is ready, avoid unnecessary frequent polling
  }, [serviceStatus?.status, checkHealth]);

  async function handleGenerate() {
    const trimmedTopic = topic.trim();
    if (!trimmedTopic) {
      return;
    }

    setIsGenerating(true);
    setGenerationError(null);
    setIsCopied(false);

    try {
      // 1. Retrieve relevant personal knowledge from IndexedDB with graceful degradation
      let knowledgeResult: RetrievedKnowledgeResult | null = null;
      try {
        knowledgeResult = await retrieveRelevantKnowledge(trimmedTopic);
        if (isMountedRef.current) {
          setRetrievedKnowledge(knowledgeResult);
        }
      } catch (kbErr) {
        console.warn('Echo WritingComposer: Personal knowledge retrieval failed:', kbErr);
        knowledgeResult = {
          items: [],
          formattedStrings: [],
          keywordsUsed: [],
          hasRelevantKnowledge: false,
        };
        if (isMountedRef.current) {
          setRetrievedKnowledge(knowledgeResult);
        }
      }

      // 2. Request personalized draft from Python service
      const response = await generatePersonalizedDraft({
        topic: trimmedTopic,
        personalKnowledge: knowledgeResult?.formattedStrings ?? [],
        leadIn: leadIn.trim(),
        mode: 'sampled',
        lengthMode,
        platform: platform as GenerationPlatform,
        temperature: 0.65,
        repetitionPenalty: 1.15,
      });

      // 3. Set generated text as editable draft
      if (isMountedRef.current) {
        setGeneratedDraft(response.completeDraft);
        setGenerationMeta(response);
      }
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error ? err.message : 'An error occurred during generation.';
      if (isMountedRef.current) {
        setGenerationError(errorMsg);
      }
    } finally {
      if (isMountedRef.current) {
        setIsGenerating(false);
      }
    }
  }

  async function handleCopy() {
    if (!generatedDraft) return;

    try {
      await navigator.clipboard.writeText(generatedDraft);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2200);
    } catch {
      // Fallback if clipboard API is constrained
      const textArea = document.createElement('textarea');
      textArea.value = generatedDraft;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2200);
    }
  }

  const isServiceReady = serviceStatus?.status === 'ready';
  const isOffline = serviceStatus?.status === 'offline';
  const isLoadingModel = serviceStatus?.status === 'loading';

  return (
    <section className="relative px-12 pb-16 pt-2">
      <div className="rounded-2xl border border-[#dce3d7] bg-[#fbfaf6]/95 p-8 shadow-sm backdrop-blur-sm">
        {/* Header */}
        <div className="flex flex-col justify-between gap-4 border-b border-[#e5ebe0] pb-6 sm:flex-row sm:items-center">
          <div>
            <h2 className="font-serif text-[28px] leading-tight text-[#203c30]">
              Create something meaningful
            </h2>
            <p className="mt-1 text-[13px] text-[#68766d]">
              Draft with your authentic voice,{' '}
              <span className="italic text-[#839587]">grounded in who you are.</span>
            </p>
          </div>

          {/* Model Status Pill */}
          <div className="flex items-center gap-2 self-start sm:self-auto">
            {isCheckingService ? (
              <span className="flex items-center gap-1.5 rounded-full bg-[#f0f3ed] px-3 py-1 text-xs text-[#68766d]">
                <RefreshCw size={12} className="animate-spin" />
                Checking service...
              </span>
            ) : isServiceReady ? (
              <div className="flex items-center gap-2 rounded-full border border-[#d2e2d5] bg-[#edf4ee] px-3 py-1 text-xs text-[#2c533e]">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#528762] opacity-50" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-[#3b6d4e]" />
                </span>
                <span className="font-medium">
                  {serviceStatus?.adapter
                    ? `${serviceStatus.adapter.toUpperCase()} Active`
                    : 'Base Model Ready'}
                </span>
              </div>
            ) : isLoadingModel ? (
              <div className="flex items-center gap-2 rounded-full border border-[#fae2c8] bg-[#fef5ec] px-3 py-1 text-xs text-[#9c581a]">
                <Loader2 size={12} className="animate-spin text-[#c27329]" />
                <span>Loading weights into memory...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-full border border-[#ebd6d6] bg-[#fbf2f2] px-3 py-1 text-xs text-[#8c3535]">
                <span className="h-2 w-2 rounded-full bg-[#bf4343]" />
                <span>Service Offline</span>
              </div>
            )}

            <button
              type="button"
              onClick={checkHealth}
              disabled={isCheckingService}
              className="rounded-lg p-1.5 text-[#89978a] hover:bg-[#edf2ea] hover:text-[#203c30] transition-colors disabled:opacity-50"
              title="Refresh Service Status"
            >
              <RefreshCw size={13} className={isCheckingService ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>

        {/* Composer Controls */}
        <div className="mt-6 space-y-6">
          {/* Platform & Length Selectors */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Platform Selector */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-[#68766d]">
                Platform:
              </span>
              <div className="flex items-center gap-1.5">
                {platforms.map((p) => {
                  const isSelected = platform === p.name;
                  return (
                    <button
                      key={p.name}
                      type="button"
                      onClick={() => setPlatform(p.name)}
                      className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
                        isSelected
                          ? 'border-[#52745e] bg-[#eef3eb] text-[#203c30] ring-1 ring-[#52745e]/20'
                          : 'border-[#dce3d7] bg-white text-[#68766d] hover:border-[#9ab19e] hover:text-[#203c30]'
                      }`}
                    >
                      {p.icon ? (
                        <FontAwesomeIcon icon={p.icon} className="text-[11px]" />
                      ) : (
                        <Globe2 size={12} />
                      )}
                      <span>{p.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Length Mode Selector */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-[#68766d]">
                Length:
              </span>
              <div className="inline-flex rounded-lg border border-[#dce3d7] bg-white p-0.5 text-xs">
                {(['short', 'medium', 'long'] as GenerationLengthMode[]).map((mode) => {
                  const isSelected = lengthMode === mode;
                  return (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setLengthMode(mode)}
                      className={`rounded-md px-2.5 py-1 capitalize transition-all ${
                        isSelected
                          ? 'bg-[#203c30] font-medium text-[#f7f6ed] shadow-2xs'
                          : 'text-[#68766d] hover:text-[#203c30]'
                      }`}
                    >
                      {mode}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Topic & Premise Input */}
          <div className="space-y-2">
            <label
              htmlFor="composer-topic"
              className="block text-xs font-semibold text-[#203c30]"
            >
              What would you like to write about?
            </label>
            <div className="relative rounded-2xl border border-[#dce3d7] bg-[#fbfaf6] p-5 shadow-2xs focus-within:border-[#7ba085] focus-within:bg-white transition-all">
              <textarea
                id="composer-topic"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Share a thought, reflection, technical opinion, or update... (e.g., 'Reflections on why local-first architectures protect human autonomy')"
                rows={3}
                className="w-full resize-none bg-transparent text-sm leading-relaxed text-[#203c30] placeholder-[#9ca79e] outline-none"
              />

              <div className="mt-3 flex items-center justify-between border-t border-[#f0f3ed] pt-3 text-[11px] text-[#89978a]">
                <span>
                  {topic.trim() ? topic.trim().split(/\s+/).length : 0} words
                </span>
                <span className="text-[#718576]">
                  Personal knowledge facts will be retrieved automatically
                </span>
              </div>
            </div>
          </div>

          {/* Optional Lead-In Input */}
          <div className="space-y-1.5">
            <label
              htmlFor="composer-leadin"
              className="flex items-center justify-between text-xs text-[#526356]"
            >
              <span className="font-medium">Opening Sentence / Lead-In (Optional)</span>
              <span className="text-[10px] text-[#89978a]">Helps guide the exact starting cadence</span>
            </label>
            <input
              id="composer-leadin"
              type="text"
              value={leadIn}
              onChange={(e) => setLeadIn(e.target.value)}
              placeholder="e.g., 'Over the past month, I noticed something interesting...'"
              className="w-full rounded-xl border border-[#dce3d7] bg-[#fbfaf6] px-4 py-2.5 text-xs text-[#203c30] placeholder-[#a6b1a8] outline-none focus:border-[#7ba085] focus:bg-white transition-all"
            />
          </div>

          {/* Action Row */}
          <div className="flex flex-col-reverse justify-between gap-4 pt-2 sm:flex-row sm:items-center">
            {/* Knowledge Preview Toggle */}
            <div>
              {retrievedKnowledge && retrievedKnowledge.items.length > 0 && (
                <button
                  type="button"
                  onClick={() => setIsKnowledgeExpanded(!isKnowledgeExpanded)}
                  className="flex items-center gap-1.5 text-xs text-[#456b4f] hover:text-[#203c30] transition-colors"
                >
                  <Brain size={14} className="text-[#528762]" />
                  <span>
                    {retrievedKnowledge.items.length} personal fact
                    {retrievedKnowledge.items.length === 1 ? '' : 's'} grounded
                  </span>
                  {isKnowledgeExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>
              )}
            </div>

            {/* Generate Button */}
            <button
              type="button"
              onClick={handleGenerate}
              disabled={isGenerating || !topic.trim() || isOffline}
              className={`flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-all shadow-sm sm:w-auto ${
                isGenerating || !topic.trim() || isOffline
                  ? 'bg-[#839b89] cursor-not-allowed text-[#e5ebe4]'
                  : 'bg-[#234b37] text-[#f7f6ed] hover:bg-[#2d5d44] hover:shadow-md'
              }`}
            >
              {isGenerating ? (
                <>
                  <Loader2 size={16} className="animate-spin text-white" />
                  <span>Drafting in your voice...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Generate Draft</span>
                  <ArrowUpRight size={14} className="opacity-80" />
                </>
              )}
            </button>
          </div>

          {/* Grounded Knowledge Drawer */}
          {isKnowledgeExpanded && retrievedKnowledge && retrievedKnowledge.items.length > 0 && (
            <div className="rounded-xl border border-[#dce3d7] bg-[#f5f8f3] p-4 text-xs">
              <div className="mb-2 flex items-center justify-between font-semibold text-[#294d36]">
                <div className="flex items-center gap-1.5">
                  <Brain size={14} />
                  <span>Active Personal Knowledge Facts:</span>
                </div>
                <span className="text-[10px] font-normal text-[#5c7263]">
                  Matches: {retrievedKnowledge.keywordsUsed.join(', ')}
                </span>
              </div>
              <ul className="space-y-1.5 pl-1">
                {retrievedKnowledge.items.map((fact) => (
                  <li
                    key={fact.id}
                    className="flex items-start gap-2 text-[#3d5945]"
                  >
                    <span className="mt-1 h-1.5 w-1.5 rounded-full bg-[#528762] shrink-0" />
                    <span>
                      <strong className="text-[#203c30]">{fact.subject}</strong>{' '}
                      {fact.predicate}{' '}
                      <span className="font-medium text-[#203c30]">{fact.value}</span>
                      {fact.contextScope ? ` (${fact.contextScope})` : ''}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Generation Error Banner */}
          {generationError && (
            <div className="flex items-start gap-3 rounded-xl border border-[#edd6d6] bg-[#fdf4f4] p-4 text-xs text-[#8c3535]">
              <AlertCircle size={16} className="mt-0.5 shrink-0 text-[#b33e3e]" />
              <div>
                <p className="font-semibold text-[#7d2c2c]">Draft Generation Notice</p>
                <p className="mt-0.5 leading-relaxed text-[#944343]">{generationError}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Generated Output Card */}
      {generatedDraft && (
        <div className="mt-8 rounded-2xl border border-[#dce3d7] bg-[#fbfaf6]/95 shadow-sm backdrop-blur-sm overflow-hidden">
          {/* Draft Card Header */}
          <div className="flex items-center justify-between border-b border-[#e5ebe0] bg-[#f6f9f4]/80 px-6 py-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#e3ecdf] text-[#345c41]">
                <Sparkles size={14} />
              </div>
              <div>
                <h3 className="font-serif text-sm font-semibold text-[#203c30]">
                  Your Personalized Draft
                </h3>
                {generationMeta?.elapsedSec && (
                  <p className="text-[10px] text-[#788879]">
                    Generated in {generationMeta.elapsedSec}s · {generationMeta.tokensGenerated}{' '}
                    tokens ({generationMeta.tokensPerSec} tok/s)
                  </p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-full bg-[#eaf1e8] px-2.5 py-1 text-[10px] font-medium text-[#44664e]">
                Directly Editable
              </span>

              <button
                type="button"
                onClick={() => {
                  setGeneratedDraft('');
                  setGenerationMeta(null);
                }}
                className="rounded-lg p-1.5 text-[#89978a] hover:bg-[#edf2ea] hover:text-[#556957]"
                title="Discard Draft"
                aria-label="Discard Draft"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </div>

          {/* Grounding Status Notice / Warning Banner */}
          {generationMeta?.groundingCheck?.isGrounded === false && (
            <div className="flex items-start gap-3 border-b border-[#edd4c4] bg-[#fff6ee] px-6 py-3.5 text-[#883d19]">
              <AlertTriangle size={16} className="mt-0.5 shrink-0 text-[#c25e2e]" />
              <div className="space-y-1 text-[12px] leading-relaxed">
                <span className="font-semibold text-[#803816]">
                  Contains unverified claims
                </span>
                <p className="text-[11px] text-[#9c5631]">
                  This draft contains claims or details that Echo could not verify against your personal knowledge. Please review and edit carefully before sharing.
                </p>
                {generationMeta.groundingCheck.flaggedIssues &&
                  generationMeta.groundingCheck.flaggedIssues.length > 0 && (
                    <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] text-[#9c5631]">
                      {generationMeta.groundingCheck.flaggedIssues.map((issue, idx) => (
                        <li key={idx}>{issue}</li>
                      ))}
                    </ul>
                  )}
              </div>
            </div>
          )}

          {generationMeta?.groundingCheck?.isGrounded === true &&
            generationMeta.groundingCheck.fallbackUsed === true && (
              <div className="flex items-center gap-2.5 border-b border-[#d8e4dc] bg-[#f2f7f4] px-6 py-2.5 text-[#2c533e]">
                <Info size={14} className="shrink-0 text-[#3d7056]" />
                <p className="text-[11px] text-[#3d6550]">
                  Echo used fallback generation to adhere strictly to your verified personal facts.
                </p>
              </div>
            )}

          {/* Editable Textarea */}
          <div className="p-6">
            <textarea
              value={generatedDraft}
              onChange={(e) => setGeneratedDraft(e.target.value)}
              rows={5}
              placeholder="Your generated draft will appear here..."
              className="w-full resize-y bg-transparent font-serif text-[15px] leading-relaxed text-[#23382c] outline-none focus:ring-0"
            />

            <div className="mt-3 flex items-center justify-between border-t border-[#f0f3ed] pt-3 text-[11px] text-[#89968a]">
              <span>
                {generatedDraft.trim() ? generatedDraft.trim().split(/\s+/).length : 0} words ·{' '}
                {generatedDraft.length} characters
              </span>
              <span className="italic text-[#9aa79b]">
                Click text above to edit before copying
              </span>
            </div>
          </div>

          {/* Draft Footer Actions */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#e6ebe2] bg-[#fafbf8] px-6 py-3.5">
            <p className="text-[10px] text-[#8a968b]">
              Stays entirely on your device. Never published or uploaded automatically.
            </p>

            <button
              type="button"
              onClick={handleCopy}
              className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-[12px] font-semibold transition-all shadow-sm ${
                isCopied
                  ? 'bg-[#3b6d4e] text-white'
                  : 'bg-[#234b37] text-[#f7f6ed] hover:bg-[#2d5d44]'
              }`}
            >
              {isCopied ? (
                <>
                  <Check size={14} />
                  <span>Copied to Clipboard!</span>
                </>
              ) : (
                <>
                  <Copy size={14} />
                  <span>Copy Draft</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

export default WritingComposer;
