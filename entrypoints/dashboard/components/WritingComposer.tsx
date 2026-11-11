import { useState, useEffect } from 'react';
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
} from '../../utils/generationClient';

import {
  retrieveRelevantKnowledge,
  type RetrievedKnowledgeResult,
} from '../../utils/knowledgeRetrieval';

import {
  getExtensionId,
  buildServiceStartupCommand,
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
  const [isCommandCopied, setIsCommandCopied] = useState(false);

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

  // Poll service health on component mount
  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    setIsCheckingService(true);
    try {
      const health = await checkGenerationServiceHealth();
      setServiceStatus(health);
    } catch {
      setServiceStatus({
        isOnline: false,
        isModelReady: false,
        status: 'offline',
        error: 'Unable to reach local generation service.',
      });
    } finally {
      setIsCheckingService(false);
    }
  }

  async function handleGenerate() {
    const trimmedTopic = topic.trim();
    if (!trimmedTopic) {
      return;
    }

    setIsGenerating(true);
    setGenerationError(null);
    setIsCopied(false);

    try {
      // 1. Retrieve relevant personal knowledge from IndexedDB
      const knowledgeResult = await retrieveRelevantKnowledge(trimmedTopic);
      setRetrievedKnowledge(knowledgeResult);

      // 2. Request personalized draft from Python service
      const response = await generatePersonalizedDraft({
        topic: trimmedTopic,
        personalKnowledge: knowledgeResult.formattedStrings,
        leadIn: leadIn.trim(),
        mode: 'sampled',
        lengthMode,
        temperature: 0.65,
        repetitionPenalty: 1.15,
      });

      // 3. Set generated text as editable draft
      setGeneratedDraft(response.completeDraft);
      setGenerationMeta(response);
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error ? err.message : 'An error occurred during generation.';
      setGenerationError(errorMsg);
    } finally {
      setIsGenerating(false);
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

  async function handleCopyStartupCommand() {
    const cmd = buildServiceStartupCommand();
    const success = await copyToClipboard(cmd);
    if (success) {
      setIsCommandCopied(true);
      setTimeout(() => setIsCommandCopied(false), 2200);
    }
  }

  const isServiceReady = serviceStatus?.status === 'ready';
  const isOffline = serviceStatus?.status === 'offline';
  const isLoadingModel = serviceStatus?.status === 'loading';

  return (
    <section className="relative z-10 space-y-6 px-12 pb-16">
      {/* Service Offline / Degraded Notice */}
      {isOffline && (
        <div className="space-y-3 rounded-2xl border border-[#e2d5c5] bg-[#fffaf3] p-4 text-[#8a5d3b] shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <AlertCircle size={18} className="mt-0.5 shrink-0 text-[#b46d3f]" />
              <div className="text-[12px] leading-5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-[#6d4323]">Local Generation Service is offline</span>
                  {getExtensionId() && (
                    <span className="rounded bg-[#f0e4d2] px-1.5 py-0.5 font-mono text-[10px] text-[#7a4e2e]">
                      ID: {getExtensionId().slice(0, 8)}...
                    </span>
                  )}
                </div>
                <p className="mt-0.5 text-[11px] text-[#9a6e4d]">
                  Echo requires CORS authorization for your specific Chrome extension origin. Run the command below to start the service:
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={checkHealth}
              disabled={isCheckingService}
              className="flex items-center gap-1.5 rounded-lg border border-[#d9c4b0] bg-white px-3 py-1.5 text-[11px] font-medium text-[#7a4e2e] transition hover:bg-[#fbf7f0] disabled:opacity-50"
            >
              <RefreshCw
                size={12}
                className={isCheckingService ? 'animate-spin' : ''}
              />
              {isCheckingService ? 'Checking...' : 'Check Connection'}
            </button>
          </div>

          {/* Interactive Startup Command Display */}
          <div className="flex items-center justify-between gap-2 rounded-xl border border-[#ecdccb] bg-[#fdf8f0] px-3 py-2">
            <code className="select-all font-mono text-[11px] text-[#5c371b] break-all">
              {buildServiceStartupCommand()}
            </code>
            <button
              type="button"
              onClick={handleCopyStartupCommand}
              className="flex shrink-0 items-center gap-1 rounded-lg border border-[#d9c4b0] bg-white px-2.5 py-1 text-[11px] font-medium text-[#7a4e2e] shadow-xs transition hover:bg-[#f7efe3] active:scale-95 cursor-pointer"
              title="Copy service startup command"
            >
              {isCommandCopied ? (
                <>
                  <Check size={12} className="text-[#3b754e]" />
                  <span className="text-[#3b754e]">Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          {/* Architecture note: Ollama vs Python service */}
          <div className="flex items-center gap-1.5 text-[11px] text-[#a57a5a]">
            <Info size={12} className="shrink-0" />
            <span>
              Echo uses two separate local services: <strong>Ollama</strong> (<code className="font-mono text-[10px]">{DEFAULT_OLLAMA_MODEL}</code> on port 11434) for private knowledge graph extraction, and this <strong>Python service</strong> (port 11435) for personalized LoRA generation.
            </span>
          </div>
        </div>
      )}

      {/* Main Composer Box */}
      <div className="overflow-hidden rounded-2xl border border-[#dce2d7] bg-[#fffef9]/90 shadow-[0_12px_40px_rgba(35,60,45,0.07)] backdrop-blur-md">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#e9e9df] px-6 py-5">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles size={16} className="text-[#66836d]" />

              <h2 className="text-[14px] font-semibold tracking-tight text-[#294435]">
                Create something meaningful
              </h2>
            </div>

            <p className="mt-1.5 pl-6 text-[11px] text-[#899087]">
              Start with an idea. We'll help you find the words.
            </p>
          </div>

          {/* Service Status Indicator Badge */}
          <div className="flex items-center gap-2">
            {isServiceReady ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf1e9] px-3 py-1 text-[10px] font-medium text-[#46694e]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#529362]" />
                LoRA V2 Ready
              </span>
            ) : isLoadingModel ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-[#fbf5e6] px-3 py-1 text-[10px] font-medium text-[#94762c]">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#cfa12f]" />
                Model Loading...
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-[#f8ecec] px-3 py-1 text-[10px] font-medium text-[#a14343]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#c74c4c]" />
                Service Offline
              </span>
            )}

            <button
              type="button"
              onClick={checkHealth}
              disabled={isCheckingService}
              title="Refresh Service Status"
              aria-label="Refresh Service Status"
              className="rounded-lg p-1 text-[#899087] transition hover:bg-[#eeefe8] hover:text-[#304438]"
            >
              <RefreshCw
                size={12}
                className={isCheckingService ? 'animate-spin' : ''}
              />
            </button>
          </div>
        </div>

        {/* Topic Input */}
        <div className="px-6 pt-5 pb-3">
          <div className="mb-2 flex items-center justify-between">
            <label
              htmlFor="writing-topic"
              className="block text-[10px] font-semibold uppercase tracking-[0.16em] text-[#788578]"
            >
              What's on your mind?
            </label>
            <span className="text-[10px] text-[#9ea89d]">
              {topic.length}/500
            </span>
          </div>

          <textarea
            id="writing-topic"
            value={topic}
            maxLength={500}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Share an idea, a project you built, a learning experience, or something you'd like to post about..."
            rows={3}
            disabled={isGenerating}
            className="w-full resize-none bg-transparent text-[14px] leading-7 text-[#304438] outline-none placeholder:text-[#a4aaa0] disabled:opacity-60"
          />
        </div>

        {/* Lead-in Input (Optional) */}
        <div className="border-t border-[#f0f2eb] px-6 py-3">
          <div className="mb-1 flex items-center justify-between">
            <label
              htmlFor="writing-leadin"
              className="block text-[10px] font-semibold uppercase tracking-[0.16em] text-[#788578]"
            >
              Optional Lead-In Phrase
            </label>
            <span className="text-[10px] text-[#9ea89d]">
              {leadIn.length}/100
            </span>
          </div>

          <input
            id="writing-leadin"
            type="text"
            value={leadIn}
            maxLength={100}
            onChange={(e) => setLeadIn(e.target.value)}
            placeholder="e.g. Over the past few months, I've been focusing on..."
            disabled={isGenerating}
            className="w-full bg-transparent text-[12px] text-[#304438] outline-none placeholder:text-[#a4aaa0] disabled:opacity-60"
          />
        </div>

        {/* Personal Knowledge Preview Banner */}
        {retrievedKnowledge && (
          <div className="border-t border-[#e9e9df] bg-[#fafbf7] px-6 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain size={14} className="text-[#55725a]" />
                {retrievedKnowledge.hasRelevantKnowledge ? (
                  <span className="text-[11px] font-medium text-[#3b5943]">
                    Found {retrievedKnowledge.items.length} relevant personal fact
                    {retrievedKnowledge.items.length === 1 ? '' : 's'} in your writing data
                  </span>
                ) : (
                  <span className="text-[11px] text-[#7a8579]">
                    No matching personal facts found; generating with learned writing style only.
                  </span>
                )}
              </div>

              {retrievedKnowledge.hasRelevantKnowledge && (
                <button
                  type="button"
                  onClick={() => setIsKnowledgeExpanded(!isKnowledgeExpanded)}
                  className="flex items-center gap-1 text-[11px] font-medium text-[#52745e] hover:underline"
                >
                  {isKnowledgeExpanded ? 'Hide facts' : 'View facts'}
                  {isKnowledgeExpanded ? (
                    <ChevronUp size={13} />
                  ) : (
                    <ChevronDown size={13} />
                  )}
                </button>
              )}
            </div>

            {/* Expandable Knowledge Chips */}
            {isKnowledgeExpanded && retrievedKnowledge.hasRelevantKnowledge && (
              <div className="mt-3 flex flex-wrap gap-2 pt-2 border-t border-[#e8ebe3]">
                {retrievedKnowledge.formattedStrings.map((fact, idx) => (
                  <div
                    key={idx}
                    className="rounded-lg border border-[#dbe2d6] bg-white px-2.5 py-1 text-[11px] text-[#415a48]"
                  >
                    {fact}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Generation Error Banner */}
        {generationError && (
          <div className="border-t border-[#f0dede] bg-[#fff5f5] px-6 py-3 text-[12px] text-[#9b3a3a]">
            <span className="font-semibold">Generation failed: </span>
            {generationError}
          </div>
        )}

        {/* Zero-Fact Safety Notice */}
        {retrievedKnowledge?.isAutobiographicalWithoutFacts && (
          <div className="flex items-center gap-2.5 border-t border-[#e2d5c5] bg-[#fffaf3] px-6 py-2.5 text-[11px] text-[#8a5d3b]">
            <Info size={14} className="shrink-0 text-[#b46d3f]" />
            <span>
              <strong className="font-semibold">Notice:</strong> No verified personal notes exist in Echo for this topic. Echo will provide a general reflection to prevent fabricating personal experiences.
            </span>
          </div>
        )}

        {/* Footer Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-[#e9e9df] bg-[#fafaf5]/80 px-6 py-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Platform selector */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#899087]">
                Write for
              </span>

              <div className="flex items-center gap-1 rounded-xl bg-[#eeefe8] p-1">
                {platforms.map((item) => {
                  const isActive = platform === item.name;

                  return (
                    <button
                      key={item.name}
                      type="button"
                      disabled={isGenerating}
                      onClick={() => setPlatform(item.name)}
                      className={`flex items-center gap-2 rounded-lg px-3 py-2 text-[11px] font-medium transition-all ${
                        isActive
                          ? 'bg-white text-[#315b42] shadow-sm'
                          : 'text-[#818b80] hover:text-[#315b42]'
                      } disabled:opacity-60`}
                    >
                      {item.icon ? (
                        <FontAwesomeIcon
                          icon={item.icon}
                          className="text-[13px]"
                        />
                      ) : (
                        <Globe2 size={14} strokeWidth={1.8} />
                      )}

                      {item.name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Length selector */}
            <div className="flex items-center gap-2">
              <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#899087]">
                Length
              </span>

              <div className="flex items-center gap-1 rounded-xl bg-[#eeefe8] p-1">
                {(['short', 'medium', 'long'] as const).map((mode) => {
                  const isActive = lengthMode === mode;
                  const labelMap = {
                    short: 'Short (1–2s)',
                    medium: 'Medium (Post)',
                    long: 'Long (Detailed)',
                  };

                  return (
                    <button
                      key={mode}
                      type="button"
                      disabled={isGenerating}
                      onClick={() => setLengthMode(mode)}
                      className={`rounded-lg px-2.5 py-1.5 text-[11px] font-medium transition-all ${
                        isActive
                          ? 'bg-white text-[#315b42] shadow-sm'
                          : 'text-[#818b80] hover:text-[#315b42]'
                      } disabled:opacity-60`}
                    >
                      {labelMap[mode]}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Generate button */}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!topic.trim() || isGenerating || !isServiceReady}
            className="group flex items-center gap-3 rounded-xl bg-[#234b37] px-5 py-3 text-[12px] font-semibold text-[#f7f6ed] shadow-[0_4px_12px_rgba(35,75,55,0.15)] transition-all hover:bg-[#315d43] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isGenerating ? (
              <>
                <Loader2 size={15} className="animate-spin text-[#d8e6dc]" />
                <span>Generating draft...</span>
              </>
            ) : (
              <>
                <Sparkles size={15} />
                <span>Generate</span>
                <ArrowUpRight
                  size={14}
                  className="transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Generated Draft Review & Direct Editor Card */}
      {generatedDraft && (
        <div className="overflow-hidden rounded-2xl border border-[#d2ded1] bg-[#fffefc] shadow-[0_12px_36px_rgba(35,60,45,0.08)]">
          {/* Draft Header */}
          <div className="flex flex-wrap items-center justify-between border-b border-[#e6ebe2] bg-[#f8faf6] px-6 py-4">
            <div className="flex items-center gap-2.5">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#e3ecdf] text-[#345941]">
                <Sparkles size={13} />
              </span>
              <div>
                <h3 className="text-[13px] font-semibold text-[#234432]">
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
