import { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  Database,
  ShieldCheck,
  RefreshCw,
  Info,
  CheckCircle2,
  AlertTriangle,
  Server,
  Layers,
  ArrowUpRight,
} from 'lucide-react';

import {
  checkGenerationServiceHealth,
  type GenerationServiceHealth,
} from '../../utils/generationClient';
import {
  getWritingMessages,
  getPersonalKnowledge,
} from '../../utils/database';

interface SettingsViewProps {
  onNavigate?: (section: 'Dashboard' | 'My Writing Data' | 'Personal Knowledge' | 'My Writing Style' | 'Settings') => void;
}

function getAppVersion(): string {
  try {
    if (typeof browser !== 'undefined' && browser.runtime?.getManifest) {
      const manifest = browser.runtime.getManifest();
      if (manifest?.version) return manifest.version;
    }
    const globalChrome = (globalThis as unknown as { chrome?: { runtime?: { getManifest?: () => { version?: string } } } }).chrome;
    if (globalChrome?.runtime?.getManifest) {
      const manifest = globalChrome.runtime.getManifest();
      if (manifest?.version) return manifest.version;
    }
  } catch {
    // Ignore context / environment errors
  }
  return '0.0.0';
}

export function SettingsView({ onNavigate }: SettingsViewProps) {
  const [serviceHealth, setServiceHealth] = useState<GenerationServiceHealth | null>(null);
  const [isCheckingService, setIsCheckingService] = useState(false);
  const [messageCount, setMessageCount] = useState<number | null>(null);
  const [knowledgeCount, setKnowledgeCount] = useState<number | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(true);

  const checkService = useCallback(async () => {
    setIsCheckingService(true);
    try {
      const health = await checkGenerationServiceHealth();
      setServiceHealth(health);
    } catch {
      setServiceHealth({
        isOnline: false,
        isModelReady: false,
        status: 'offline',
        error: 'Unable to reach local generation service on 127.0.0.1:11435.',
      });
    } finally {
      setIsCheckingService(false);
    }
  }, []);

  const loadStorageStats = useCallback(async () => {
    setIsLoadingStats(true);
    try {
      const [messages, knowledge] = await Promise.all([
        getWritingMessages(),
        getPersonalKnowledge(),
      ]);
      setMessageCount(messages.length);
      setKnowledgeCount(knowledge.filter((k) => k.status === 'active').length);
    } catch (err) {
      console.warn('Echo Settings: Failed to load storage stats:', err);
    } finally {
      setIsLoadingStats(false);
    }
  }, []);

  useEffect(() => {
    checkService();
    loadStorageStats();
  }, [checkService, loadStorageStats]);

  const appVersion = getAppVersion();

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-10 sm:px-12">
      {/* Page Header */}
      <div className="mb-10">
        <div className="mb-3 flex items-center gap-2">
          <span className="h-px w-6 bg-[#78947d]" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#52745e]">
            System & Environment
          </span>
        </div>
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <h1 className="font-serif text-3xl font-medium tracking-tight text-[#203c30] sm:text-4xl">
              Settings
            </h1>
            <p className="mt-2 max-w-2xl text-xs leading-relaxed text-[#68766d] sm:text-sm">
              Review your local inference engine status, inspect verified on-device storage metrics, and view application details.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              checkService();
              loadStorageStats();
            }}
            disabled={isCheckingService || isLoadingStats}
            className="inline-flex items-center gap-1.5 self-start rounded-xl border border-[#dce3d7] bg-[#f8f7f0] px-3.5 py-1.5 text-xs font-medium text-[#203c30] shadow-sm hover:bg-[#edf3ea] transition-colors disabled:opacity-50 cursor-pointer"
            title="Refresh diagnostics and storage metrics"
          >
            <RefreshCw
              size={13}
              className={isCheckingService || isLoadingStats ? 'animate-spin' : ''}
            />
            <span>Refresh Diagnostics</span>
          </button>
        </div>
      </div>

      <div className="space-y-10">
        {/* ========================================================================= */}
        {/* SECTION A — Local Inference Diagnostics                                   */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <Cpu size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Local Inference Diagnostics
                </h2>
                <p className="text-xs text-[#68766d]">
                  Connection and model serving state for on-device drafting
                </p>
              </div>
            </div>

            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border ${
                serviceHealth?.isOnline && serviceHealth?.isModelReady
                  ? 'bg-[#edf3ea] text-[#245237] border-[#c8d6c3]'
                  : serviceHealth?.isOnline
                  ? 'bg-[#eaf2f8] text-[#205275] border-[#c0d8ea]'
                  : 'bg-[#fbf5e6] text-[#8a6514] border-[#e8dcb8]'
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

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Production Endpoint</span>
                <Server size={14} className="text-[#52745e]" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-mono text-sm font-semibold text-[#203c30]">
                  127.0.0.1:11435
                </span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Dedicated Python HTTP generation service running locally on CPU.
              </p>
            </div>

            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Active Production Model</span>
                <Cpu size={14} className="text-[#52745e]" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-mono text-sm font-semibold text-[#203c30]">
                  {serviceHealth?.isOnline && serviceHealth?.model
                    ? serviceHealth.model
                    : serviceHealth?.isOnline
                    ? 'Loading Model...'
                    : 'Unknown (Service Offline)'}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Base model architecture loaded for style-grounded generation.
              </p>
            </div>

            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Active Style Adapter</span>
                <Layers size={14} className="text-[#52745e]" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-medium text-sm text-[#203c30]">
                  {serviceHealth?.isOnline && serviceHealth?.isModelReady
                    ? serviceHealth.adapterVersion &&
                      serviceHealth.adapterVersion.trim() &&
                      serviceHealth.adapterVersion.toLowerCase() !== 'none'
                      ? `LoRA ${serviceHealth.adapterVersion.trim().toUpperCase()} (Active)`
                      : serviceHealth.adapter &&
                        serviceHealth.adapter.trim() &&
                        serviceHealth.adapter.toLowerCase() !== 'none'
                      ? `${serviceHealth.adapter} (Active)`
                      : 'Base Model Only'
                    : serviceHealth?.isOnline
                    ? 'Loading Adapter...'
                    : 'Unknown (Service Offline)'}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Fine-tuned LoRA weights applied to match your authentic writing voice.
              </p>
            </div>

            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Inference Execution Mode</span>
                <ShieldCheck size={14} className="text-[#52745e]" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-medium text-sm text-[#245237]">
                  Local CPU FP32
                </span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Standard on-device execution with no GPU requirement or remote dependency.
              </p>
            </div>
          </div>

          {!serviceHealth?.isOnline && (
            <div className="mt-5 rounded-xl border border-[#e8dcb8] bg-[#fbf7ee] p-4 text-xs text-[#8a6514]">
              <div className="flex items-start gap-2.5">
                <AlertTriangle size={15} className="mt-0.5 shrink-0 text-[#8a6514]" />
                <div>
                  <p className="font-semibold text-[#8a6514]">
                    Generation Service is Currently Offline
                  </p>
                  <p className="mt-1 leading-relaxed text-[#785912]">
                    To enable writing generation, start the local Python generation service in your terminal.
                    Your existing writing messages and personal knowledge remain safely preserved in local storage.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ========================================================================= */}
        {/* SECTION B — Storage & Privacy (Read-Only)                                 */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <Database size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  Storage & Privacy
                </h2>
                <p className="text-xs text-[#68766d]">
                  Summary of personal data stored privately in your browser
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-1 rounded-full bg-[#edf3ea] px-2.5 py-0.5 text-[11px] font-medium text-[#245237] border border-[#c8d6c3]">
              <ShieldCheck size={12} className="text-[#245237]" />
              Read-Only
            </span>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Stored Writing Messages</span>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('My Writing Data')}
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-[#245237] hover:underline cursor-pointer"
                  >
                    <span>View Data</span>
                    <ArrowUpRight size={12} />
                  </button>
                )}
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-serif text-3xl font-semibold text-[#203c30]">
                  {isLoadingStats ? '...' : messageCount ?? 0}
                </span>
                <span className="text-xs text-[#68766d]">messages</span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Captured message samples used for personal writing style analysis.
              </p>
            </div>

            <div className="rounded-xl border border-[#e2e8dd] bg-white/70 p-4 shadow-2xs">
              <div className="flex items-center justify-between text-xs text-[#68766d]">
                <span className="font-medium">Personal Knowledge Facts</span>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('Personal Knowledge')}
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-[#245237] hover:underline cursor-pointer"
                  >
                    <span>View Facts</span>
                    <ArrowUpRight size={12} />
                  </button>
                )}
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-serif text-3xl font-semibold text-[#203c30]">
                  {isLoadingStats ? '...' : knowledgeCount ?? 0}
                </span>
                <span className="text-xs text-[#68766d]">active facts</span>
              </div>
              <p className="mt-1.5 text-[11px] text-[#68766d]">
                Structured facts extracted to ground generated drafts in your authentic background.
              </p>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-[#e2e8dd] bg-white/60 p-4 text-xs leading-relaxed text-[#4a6352]">
            <div className="flex items-start gap-2.5">
              <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-[#245237]" />
              <div>
                <p className="font-semibold text-[#203c30]">
                  100% On-Device Browser Storage
                </p>
                <p className="mt-1 text-[#68766d]">
                  All messages and facts are stored inside your browser's local IndexedDB database.
                  Echo does not send your data to external databases, and settings cannot wipe or modify records from this view.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ========================================================================= */}
        {/* SECTION C — About Echo                                                    */}
        {/* ========================================================================= */}
        <section className="rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between border-b border-[#e2e8dd] pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9efe6] text-[#245237]">
                <Info size={18} />
              </div>
              <div>
                <h2 className="font-serif text-lg font-medium text-[#203c30]">
                  About Echo
                </h2>
                <p className="text-xs text-[#68766d]">
                  Architecture and verified local privacy guarantees
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-1 rounded-full bg-[#edf3ea] px-2.5 py-0.5 text-[11px] font-mono font-medium text-[#245237] border border-[#c8d6c3]">
              v{appVersion}
            </span>
          </div>

          <div className="mt-6 space-y-4 text-xs leading-relaxed text-[#4a6352]">
            <p>
              <strong className="text-[#203c30]">Echo</strong> is an AI writing assistant designed to capture your natural voice and draft communications that sound genuinely like you.
            </p>

            <div className="rounded-xl border border-[#e2e8dd] bg-white/60 p-4 space-y-2">
              <h4 className="font-serif text-sm font-semibold text-[#203c30]">
                Local-First Architecture Details
              </h4>
              <ul className="list-disc pl-4 space-y-1.5 text-[11px] text-[#68766d]">
                <li>
                  <strong className="text-[#203c30]">Data Persistence:</strong> Writing samples and extracted facts are stored in local browser IndexedDB storage.
                </li>
                <li>
                  <strong className="text-[#203c30]">Style Inference:</strong> Local CPU drafting connects to a dedicated Python generation service on <code className="font-mono text-[10px] text-[#245237]">127.0.0.1:11435</code>.
                </li>
                <li>
                  <strong className="text-[#203c30]">Knowledge Extraction:</strong> Background fact extraction connects to your local Ollama daemon on <code className="font-mono text-[10px] text-[#245237]">localhost:11434</code>.
                </li>
                <li>
                  <strong className="text-[#203c30]">Zero Cloud Telemetry:</strong> No personal writing samples, style weights, or knowledge facts are transmitted to remote third-party cloud servers.
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default SettingsView;

