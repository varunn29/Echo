import { useState, useEffect } from 'react';
import {
  Sparkles,
  Loader2,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  Layers,
  Cpu,
  Copy,
  Check,
} from 'lucide-react';

import {
  getTrainingReadiness,
  type TrainingReadinessResult,
  DEFAULT_TRAINING_READINESS_THRESHOLD,
} from '../../utils/database';
import {
  getTrainingStatus,
  type TrainingStatusResponse,
} from '../../utils/trainingClient';
import {
  getActiveTrainingJob,
  pollActiveTrainingJob,
  type ActiveTrainingJobRecord,
} from '../../utils/trainingTrigger';
import {
  buildServiceStartupCommand,
  copyToClipboard,
} from '../../utils/extensionInfo';

interface TrainingStatusCardProps {
  className?: string;
}

export function TrainingStatusCard({ className = '' }: TrainingStatusCardProps) {
  const [readiness, setReadiness] = useState<TrainingReadinessResult | null>(null);
  const [serviceStatus, setServiceStatus] = useState<TrainingStatusResponse | null>(null);
  const [activeJob, setActiveJob] = useState<ActiveTrainingJobRecord | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCommandCopied, setIsCommandCopied] = useState(false);

  async function refreshStatus() {
    setIsRefreshing(true);
    try {
      // 1. Poll any active in-flight training job
      const polledJob = await pollActiveTrainingJob();
      setActiveJob(polledJob);

      // 2. Query current database readiness
      const currentReadiness = await getTrainingReadiness();
      setReadiness(currentReadiness);

      // 3. Query training service status
      const currentStatus = await getTrainingStatus();
      setServiceStatus(currentStatus);
    } catch (err) {
      console.warn('Echo: Failed to refresh training status:', err);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }

  useEffect(() => {
    refreshStatus();

    // Poll every 5 seconds if a job is actively running
    const interval = setInterval(() => {
      if (
        serviceStatus?.status === 'preparing' ||
        serviceStatus?.status === 'training' ||
        serviceStatus?.status === 'validating' ||
        serviceStatus?.status === 'activating' ||
        activeJob?.status === 'active'
      ) {
        refreshStatus();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [serviceStatus?.status, activeJob?.status]);

  const threshold = readiness?.threshold ?? DEFAULT_TRAINING_READINESS_THRESHOLD;
  const eligibleCount = readiness?.eligibleCount ?? 0;
  const progressPercent = Math.min(100, Math.round((eligibleCount / threshold) * 100));
  const isThresholdMet = eligibleCount >= threshold;

  // Determine active display state
  const isOnline = serviceStatus?.isOnline ?? false;
  const currentStage = serviceStatus?.status ?? 'offline';

  const isTrainingActive =
    currentStage === 'preparing' ||
    currentStage === 'training' ||
    currentStage === 'validating' ||
    currentStage === 'activating' ||
    activeJob?.status === 'active';

  return (
    <div
      className={`rounded-2xl border border-[#dce3d7] bg-[#f8f7f0]/85 p-6 shadow-sm backdrop-blur-sm transition-all ${className}`}
    >
      <div className="flex items-center justify-between pb-4 border-b border-[#e5ebe0]">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e3ecdf] text-[#345c41]">
            <Sparkles size={18} />
          </div>
          <div>
            <h3 className="font-serif text-lg font-medium text-[#203c30]">
              Personal Style Learning
            </h3>
            <p className="text-xs text-[#68766d]">
              Automatic local LoRA training from your eligible writing samples
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Status Badge */}
          {!isOnline ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#f3eee0] px-3 py-1 text-xs font-medium text-[#7d6b38] border border-[#e8dfc8]">
              <AlertTriangle size={13} />
              Service Offline
            </span>
          ) : isTrainingActive ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#e1edf7] px-3 py-1 text-xs font-medium text-[#205b82] border border-[#cce0ef]">
              <Loader2 size={13} className="animate-spin" />
              Learning Style...
            </span>
          ) : currentStage === 'completed' || activeJob?.status === 'completed' ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#e4f1e4] px-3 py-1 text-xs font-medium text-[#2e6b36] border border-[#cfdfcf]">
              <CheckCircle2 size={13} />
              Style Updated
            </span>
          ) : currentStage === 'error' || activeJob?.status === 'failed' ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#f9e7e7] px-3 py-1 text-xs font-medium text-[#8c3232] border border-[#f0cfcf]">
              <AlertCircle size={13} />
              Training Paused
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#e8f1e5] px-3 py-1 text-xs font-medium text-[#3b6647] border border-[#d6e4d2]">
              <Cpu size={13} />
              Standing By
            </span>
          )}

          <button
            type="button"
            onClick={refreshStatus}
            disabled={isRefreshing}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-[#68766d] hover:bg-[#eae8dc] transition-colors disabled:opacity-50"
            title="Refresh training status"
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Progress towards 25-sample threshold */}
      <div className="mt-5">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="font-medium text-[#465a4c]">
            Eligible Samples Progress
          </span>
          <span className="font-semibold text-[#203c30]">
            {eligibleCount} / {threshold} samples ({progressPercent}%)
          </span>
        </div>

        <div className="h-2 w-full overflow-hidden rounded-full bg-[#e5ebe0]">
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isThresholdMet ? 'bg-[#4a7756]' : 'bg-[#7ba085]'
            }`}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* State-specific explanatory messaging */}
      <div className="mt-4 rounded-xl bg-white/70 p-3.5 border border-[#e2e8dd] text-xs leading-relaxed">
        {!isOnline ? (
          <div className="space-y-2 text-[#735824]">
            <div className="flex items-start gap-2">
              <AlertTriangle size={15} className="shrink-0 mt-0.5 text-[#b08732]" />
              <div>
                <p className="font-medium">Local training service is offline</p>
                <p className="mt-0.5 text-[#7a6843]">
                  Your writing samples are safely saved in your browser. Start the service on <code className="rounded bg-[#f0e8d5] px-1 py-0.5 font-mono text-[11px]">127.0.0.1:11435</code> with your extension origin allowed:
                </p>
              </div>
            </div>
            <div className="flex items-center justify-between gap-2 rounded-lg border border-[#e8dfc7] bg-[#fcf9ee] px-2.5 py-1.5">
              <code className="select-all font-mono text-[10px] text-[#6d5118] break-all">
                {buildServiceStartupCommand()}
              </code>
              <button
                type="button"
                onClick={async () => {
                  const ok = await copyToClipboard(buildServiceStartupCommand());
                  if (ok) {
                    setIsCommandCopied(true);
                    setTimeout(() => setIsCommandCopied(false), 2200);
                  }
                }}
                className="flex shrink-0 items-center gap-1 rounded border border-[#d4c39f] bg-white px-2 py-0.5 text-[10px] font-medium text-[#735824] hover:bg-[#fbf7ed] cursor-pointer"
                title="Copy service startup command"
              >
                {isCommandCopied ? <Check size={10} className="text-[#3b754e]" /> : <Copy size={10} />}
                <span>{isCommandCopied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>
        ) : isTrainingActive ? (
          <div className="flex items-start gap-2 text-[#245474]">
            <Loader2 size={15} className="shrink-0 mt-0.5 animate-spin text-[#35729c]" />
            <div>
              <p className="font-medium">
                Training in progress: {serviceStatus?.stage || 'training'}
              </p>
              <p className="mt-0.5 text-[#3b637d]">
                Echo is training your personal LoRA adapter locally. Existing generation and message collection remain available.
              </p>
            </div>
          </div>
        ) : currentStage === 'completed' || activeJob?.status === 'completed' ? (
          <div className="flex items-start gap-2 text-[#296332]">
            <CheckCircle2 size={15} className="shrink-0 mt-0.5 text-[#3b8747]" />
            <div>
              <p className="font-medium">Your personal writing style is active</p>
              <p className="mt-0.5 text-[#3f6d46]">
                Adapter version{' '}
                <span className="font-semibold">
                  {serviceStatus?.adapterVersion || activeJob?.adapterVersion || 'v2'}
                </span>{' '}
                is successfully loaded and ready for drafting.
              </p>
            </div>
          </div>
        ) : currentStage === 'error' || activeJob?.status === 'failed' ? (
          <div className="flex items-start gap-2 text-[#7f2626]">
            <AlertCircle size={15} className="shrink-0 mt-0.5 text-[#aa3333]" />
            <div>
              <p className="font-medium">Previous training run could not complete</p>
              <p className="mt-0.5 text-[#853939]">
                {activeJob?.error || serviceStatus?.error || 'Training encountered an error.'} Your samples have been kept pending and will retry automatically.
              </p>
            </div>
          </div>
        ) : isThresholdMet ? (
          <div className="flex items-start gap-2 text-[#2d5f38]">
            <Sparkles size={15} className="shrink-0 mt-0.5 text-[#478754]" />
            <div>
              <p className="font-medium">Training threshold reached!</p>
              <p className="mt-0.5 text-[#426a4a]">
                You have collected {eligibleCount} eligible samples. Echo will automatically update your personal adapter.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-2 text-[#465a4c]">
            <Layers size={15} className="shrink-0 mt-0.5 text-[#6c8672]" />
            <div>
              <p className="font-medium">Collecting your personal style</p>
              <p className="mt-0.5 text-[#5e7163]">
                Echo automatically screens for natural writing and excludes code, terminal commands, and short fragments.
                Need {threshold - eligibleCount} more eligible sample{threshold - eligibleCount === 1 ? '' : 's'} to trigger automatic learning.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TrainingStatusCard;
