import { useEffect, useState, useMemo } from 'react';
import {
  Trash2,
  X,
  AlertTriangle,
  Download,
  Brain,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Loader2,
  FileText,
  Clock,
  RotateCcw,
  Plus,
  Pencil,
  Search,
} from 'lucide-react';

import {
  getWritingMessages,
  deleteWritingMessage,
  getPersonalKnowledge,
  addManualPersonalKnowledge,
  updatePersonalKnowledge,
  deletePersonalKnowledge,
  type WritingMessage,
  type PersonalKnowledgeItem,
  type KnowledgeCategory,
  type KnowledgeStability,
  type KnowledgeConfidence,
  type KnowledgeImportance,
} from '../../utils/database';
import {
  checkOllamaStatus,
  type OllamaStatus,
  DEFAULT_OLLAMA_MODEL,
} from '../../utils/ollamaClient';
import {
  importSampleWritingSamples,
  type IngestionResult,
  SAMPLE_WRITING_SAMPLES,
} from '../../utils/sampleWritingIngestion';
import type { KnowledgeCoordinatorStatus } from '../../utils/knowledgeCoordinator';

interface KnowledgeFormData {
  id?: number;
  category: KnowledgeCategory;
  subject: string;
  predicate: string;
  value: string;
  contextScope: string;
  stability: KnowledgeStability;
  confidence: KnowledgeConfidence;
  importance: KnowledgeImportance;
}

interface WritingDataProps {
  initialTab?: 'messages' | 'knowledge';
  onTabChange?: (tab: 'messages' | 'knowledge') => void;
}

const DEFAULT_KNOWLEDGE_FORM: KnowledgeFormData = {
  category: 'skills',
  subject: '',
  predicate: '',
  value: '',
  contextScope: '',
  stability: 'stable',
  confidence: 'high',
  importance: 'primary',
};

export function WritingData({ initialTab = 'messages', onTabChange }: WritingDataProps = {}) {
  const [messages, setMessages] = useState<WritingMessage[]>([]);
  const [knowledgeItems, setKnowledgeItems] = useState<PersonalKnowledgeItem[]>([]);
  const [activeTab, setActiveTab] = useState<'messages' | 'knowledge'>(initialTab);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedMessageId, setSelectedMessageId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Search & Filter states for Messages
  const [messageSearch, setMessageSearch] = useState('');
  const [platformFilter, setPlatformFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'trained'>('all');

  // Search & Filter states for Knowledge
  const [knowledgeSearch, setKnowledgeSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [confidenceFilter, setConfidenceFilter] = useState<string>('all');

  useEffect(() => {
    if (initialTab) {
      setActiveTab(initialTab);
    }
  }, [initialTab]);

  const filteredMessages = useMemo(() => {
    return messages.filter((msg) => {
      if (messageSearch.trim()) {
        const query = messageSearch.toLowerCase();
        const matchesText = (msg.text || '').toLowerCase().includes(query);
        const matchesPlatform = (msg.platform || '').toLowerCase().includes(query);
        if (!matchesText && !matchesPlatform) return false;
      }
      if (platformFilter !== 'all') {
        if ((msg.platform || '').toLowerCase() !== platformFilter.toLowerCase()) return false;
      }
      if (statusFilter !== 'all') {
        if (msg.trainingStatus !== statusFilter) return false;
      }
      return true;
    });
  }, [messages, messageSearch, platformFilter, statusFilter]);

  const filteredKnowledge = useMemo(() => {
    return knowledgeItems.filter((item) => {
      if (knowledgeSearch.trim()) {
        const query = knowledgeSearch.toLowerCase();
        const matchesSubject = (item.subject || '').toLowerCase().includes(query);
        const matchesPredicate = (item.predicate || '').toLowerCase().includes(query);
        const matchesValue = (item.value || '').toLowerCase().includes(query);
        const matchesScope = (item.contextScope || '').toLowerCase().includes(query);
        if (!matchesSubject && !matchesPredicate && !matchesValue && !matchesScope) return false;
      }
      if (categoryFilter !== 'all') {
        if (item.category !== categoryFilter) return false;
      }
      if (confidenceFilter !== 'all') {
        if (item.confidence !== confidenceFilter) return false;
      }
      return true;
    });
  }, [knowledgeItems, knowledgeSearch, categoryFilter, confidenceFilter]);

  // Sample Writing Ingestion states
  const [isImporting, setIsImporting] = useState(false);
  const [ingestionSummary, setIngestionSummary] = useState<IngestionResult | null>(null);
  const [ingestionError, setIngestionError] = useState('');

  // Ollama & Coordinator states
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [isCheckingOllama, setIsCheckingOllama] = useState(false);
  const [coordinatorStatus, setCoordinatorStatus] = useState<KnowledgeCoordinatorStatus | null>(null);
  const [isActionSubmitting, setIsActionSubmitting] = useState(false);
  const [actionFeedback, setActionFeedback] = useState('');
  const [actionError, setActionError] = useState('');

  // Personal Knowledge Curation states (Add / Edit)
  const [isKnowledgeModalOpen, setIsKnowledgeModalOpen] = useState(false);
  const [knowledgeModalMode, setKnowledgeModalMode] = useState<'add' | 'edit'>('add');
  const [knowledgeFormData, setKnowledgeFormData] = useState<KnowledgeFormData>(DEFAULT_KNOWLEDGE_FORM);
  const [isSavingKnowledge, setIsSavingKnowledge] = useState(false);
  const [knowledgeFormError, setKnowledgeFormError] = useState('');

  // Personal Knowledge Curation states (Delete)
  const [deletingKnowledgeItem, setDeletingKnowledgeItem] = useState<PersonalKnowledgeItem | null>(null);
  const [isDeletingKnowledge, setIsDeletingKnowledge] = useState(false);
  const [knowledgeDeleteError, setKnowledgeDeleteError] = useState('');

  function openAddKnowledgeModal() {
    setKnowledgeModalMode('add');
    setKnowledgeFormData(DEFAULT_KNOWLEDGE_FORM);
    setKnowledgeFormError('');
    setIsKnowledgeModalOpen(true);
  }

  function openEditKnowledgeModal(item: PersonalKnowledgeItem) {
    if (item.status !== 'active') return;
    setKnowledgeModalMode('edit');
    setKnowledgeFormData({
      id: item.id,
      category: item.category,
      subject: item.subject,
      predicate: item.predicate,
      value: item.value,
      contextScope: item.contextScope || '',
      stability: item.stability,
      confidence: item.confidence,
      importance: item.importance,
    });
    setKnowledgeFormError('');
    setIsKnowledgeModalOpen(true);
  }

  async function handleSaveKnowledge(e: React.FormEvent) {
    e.preventDefault();
    if (isSavingKnowledge) return;

    const subjectTrimmed = knowledgeFormData.subject.trim();
    const predicateTrimmed = knowledgeFormData.predicate.trim();
    const valueTrimmed = knowledgeFormData.value.trim();

    if (!subjectTrimmed) {
      setKnowledgeFormError('Subject is required.');
      return;
    }
    if (!predicateTrimmed) {
      setKnowledgeFormError('Predicate is required.');
      return;
    }
    if (!valueTrimmed) {
      setKnowledgeFormError('Value is required.');
      return;
    }

    setIsSavingKnowledge(true);
    setKnowledgeFormError('');

    try {
      if (knowledgeModalMode === 'add') {
        await addManualPersonalKnowledge({
          category: knowledgeFormData.category,
          subject: subjectTrimmed,
          predicate: predicateTrimmed,
          value: valueTrimmed,
          contextScope: knowledgeFormData.contextScope.trim() || undefined,
          stability: knowledgeFormData.stability,
          confidence: knowledgeFormData.confidence,
          importance: knowledgeFormData.importance,
        });
      } else {
        if (!knowledgeFormData.id) {
          throw new Error('Missing fact ID for update.');
        }
        await updatePersonalKnowledge(knowledgeFormData.id, {
          category: knowledgeFormData.category,
          subject: subjectTrimmed,
          predicate: predicateTrimmed,
          value: valueTrimmed,
          contextScope: knowledgeFormData.contextScope.trim(),
          stability: knowledgeFormData.stability,
          confidence: knowledgeFormData.confidence,
          importance: knowledgeFormData.importance,
        });
      }

      await loadData();
      setIsKnowledgeModalOpen(false);
      setKnowledgeFormData(DEFAULT_KNOWLEDGE_FORM);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setKnowledgeFormError(msg);
    } finally {
      setIsSavingKnowledge(false);
    }
  }

  function openDeleteKnowledgeModal(item: PersonalKnowledgeItem) {
    setDeletingKnowledgeItem(item);
    setKnowledgeDeleteError('');
  }

  async function handleConfirmDeleteKnowledge() {
    if (!deletingKnowledgeItem || isDeletingKnowledge) return;
    if (typeof deletingKnowledgeItem.id !== 'number') return;

    setIsDeletingKnowledge(true);
    setKnowledgeDeleteError('');

    try {
      await deletePersonalKnowledge(deletingKnowledgeItem.id, 'user_manually_rejected');
      await loadData();
      setDeletingKnowledgeItem(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setKnowledgeDeleteError(msg);
    } finally {
      setIsDeletingKnowledge(false);
    }
  }

  async function loadData() {
    try {
      const [savedMessages, savedKnowledge] = await Promise.all([
        getWritingMessages(),
        getPersonalKnowledge({ status: 'active' }),
      ]);
      setMessages(savedMessages);
      setKnowledgeItems(savedKnowledge);
    } catch (err) {
      console.error('Echo: Failed to load data:', err);
      setError('Unable to load your writing data.');
    } finally {
      setIsLoading(false);
    }
  }

  async function checkOllama() {
    setIsCheckingOllama(true);
    try {
      const status = await checkOllamaStatus();
      setOllamaStatus(status);
    } catch {
      setOllamaStatus({
        isRunning: false,
        isModelReady: false,
        modelName: DEFAULT_OLLAMA_MODEL,
        availability: 'offline',
        error: 'Unable to connect to Ollama on http://localhost:11434.',
      });
    } finally {
      setIsCheckingOllama(false);
    }
  }

  async function fetchCoordinatorStatus() {
    try {
      if (typeof browser !== 'undefined' && browser.runtime?.sendMessage) {
        const response = await browser.runtime.sendMessage({
          type: 'GET_KNOWLEDGE_COORDINATOR_STATUS',
        });
        if (response?.success && response.status) {
          setCoordinatorStatus(response.status);
        }
      }
    } catch (err) {
      console.warn('Echo: Failed to fetch coordinator status:', err);
    }
  }

  useEffect(() => {
    loadData();
    checkOllama();
    fetchCoordinatorStatus();

    function handleVisibilityChange() {
      if (document.visibilityState === 'visible') {
        loadData();
        checkOllama();
        fetchCoordinatorStatus();
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Poll only while extraction is actively running in background
  useEffect(() => {
    if (!coordinatorStatus?.isProcessing) return;

    const timer = setInterval(() => {
      fetchCoordinatorStatus();
      loadData();
    }, 2000);

    return () => clearInterval(timer);
  }, [coordinatorStatus?.isProcessing]);

  async function handleStartExtraction() {
    if (isActionSubmitting || coordinatorStatus?.isProcessing) return;
    setIsActionSubmitting(true);
    setActionError('');
    setActionFeedback('');

    try {
      if (typeof browser !== 'undefined' && browser.runtime?.sendMessage) {
        const response = await browser.runtime.sendMessage({
          type: 'START_KNOWLEDGE_EXTRACTION',
        });

        if (response?.success) {
          const res = response.result;
          if (res?.status === 'completed') {
            setActionFeedback(`Extraction complete: ${res.processedCount} message(s) evaluated.`);
          } else if (res?.status === 'already_running') {
            setActionFeedback('Extraction is already in progress.');
          } else if (res?.status === 'service_offline') {
            setActionError('Ollama is offline. Start Ollama to begin extraction.');
          } else if (res?.status === 'service_unknown') {
            setActionError(res.error || 'Ollama is responding slowly. Please check Ollama and try again.');
          } else if (res?.status === 'service_blocked') {
            setActionError(`Extraction blocked: ${res.error || 'Model not installed or origin rejected.'}`);
          } else if (res?.status === 'empty') {
            setActionFeedback('No pending messages waiting for extraction.');
          }
          await fetchCoordinatorStatus();
          await loadData();
          if (res?.processedCount > 0) {
            setActiveTab('knowledge');
          }
        } else {
          setActionError(response?.error || 'Failed to start knowledge extraction.');
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setActionError(msg);
    } finally {
      setIsActionSubmitting(false);
    }
  }

  async function handleRetryFailed() {
    if (isActionSubmitting) return;
    setIsActionSubmitting(true);
    setActionError('');
    setActionFeedback('');

    try {
      if (typeof browser !== 'undefined' && browser.runtime?.sendMessage) {
        const response = await browser.runtime.sendMessage({
          type: 'RETRY_FAILED_KNOWLEDGE_MESSAGES',
        });

        if (response?.success) {
          setActionFeedback(`Requeued ${response.requeuedCount} failed message(s) for extraction.`);
          await fetchCoordinatorStatus();
          await loadData();
        } else {
          setActionError(response?.error || 'Failed to retry failed messages.');
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setActionError(msg);
    } finally {
      setIsActionSubmitting(false);
    }
  }

  async function handleReconsiderSkipped() {
    if (isActionSubmitting) return;
    setIsActionSubmitting(true);
    setActionError('');
    setActionFeedback('');

    try {
      if (typeof browser !== 'undefined' && browser.runtime?.sendMessage) {
        const response = await browser.runtime.sendMessage({
          type: 'RECONSIDER_SKIPPED_KNOWLEDGE_MESSAGES',
        });

        if (response?.success) {
          setActionFeedback(
            `Reconsidered skipped messages: ${response.requeuedCount} re-enqueued for extraction.`
          );
          await fetchCoordinatorStatus();
          await loadData();
        } else {
          setActionError(response?.error || 'Failed to reconsider skipped messages.');
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setActionError(msg);
    } finally {
      setIsActionSubmitting(false);
    }
  }

  async function handleImportSampleWriting() {
    if (isImporting || isActionSubmitting) return;
    setIsImporting(true);
    setIngestionError('');
    setIngestionSummary(null);
    setActionFeedback('');

    try {
      const result = await importSampleWritingSamples();
      setIngestionSummary(result);
      await loadData();

      // Trigger extraction through coordinator for newly imported messages
      if (result.importedCount > 0) {
        await handleStartExtraction();
      } else {
        await fetchCoordinatorStatus();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setIngestionError(msg);
    } finally {
      setIsImporting(false);
    }
  }

  function handleExport() {
    if (messages.length === 0 && knowledgeItems.length === 0) {
      return;
    }

    const exportData = {
      exportedAt: new Date().toISOString(),
      totalMessages: messages.length,
      totalPersonalKnowledge: knowledgeItems.length,
      messages: messages.map(function (message) {
        return {
          id: message.id,
          platform: message.platform,
          text: message.text,
          createdAt:
            message.createdAt instanceof Date
              ? message.createdAt.toISOString()
              : new Date(message.createdAt).toISOString(),
          capturedAt:
            message.capturedAt instanceof Date
              ? message.capturedAt.toISOString()
              : new Date(message.capturedAt).toISOString(),
          trainingStatus: message.trainingStatus,
        };
      }),
      personalKnowledge: knowledgeItems.map(function (item) {
        return {
          id: item.id,
          category: item.category,
          subject: item.subject,
          predicate: item.predicate,
          value: item.value,
          contextScope: item.contextScope,
          stability: item.stability,
          status: item.status,
          confidence: item.confidence,
          importance: item.importance,
          lastConfirmedAt:
            item.lastConfirmedAt instanceof Date
              ? item.lastConfirmedAt.toISOString()
              : new Date(item.lastConfirmedAt).toISOString(),
        };
      }),
    };

    const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });

    const downloadUrl = URL.createObjectURL(jsonBlob);
    const downloadLink = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    downloadLink.href = downloadUrl;
    downloadLink.download = `echo-backup-${timestamp}.json`;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(downloadUrl);
  }

  function openDeleteConfirmation(id: number | undefined) {
    if (id === undefined) {
      return;
    }
    setSelectedMessageId(id);
  }

  async function handleDelete() {
    if (selectedMessageId === null) {
      return;
    }

    setIsDeleting(true);
    setError('');

    try {
      await deleteWritingMessage(selectedMessageId);
      // Reload both messages and active knowledge to reflect privacy scrubbing
      await loadData();
      await fetchCoordinatorStatus();
      setSelectedMessageId(null);
    } catch (err) {
      console.error('Echo: Failed to delete message:', err);
      setError('Unable to delete this message. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <>
      {/* Page Header */}
      <section className="px-12 py-12">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl text-[#203c30]">
              My Writing Data
            </h1>
            <p className="mt-3 text-sm text-[#68766d]">
              Review captured writing messages and verified personal knowledge extracted on your device.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {!isLoading && (messages.length > 0 || knowledgeItems.length > 0) && (
              <button
                type="button"
                onClick={handleExport}
                className="inline-flex items-center gap-2 rounded-xl border border-[#c6d3c2] bg-white/90 px-4 py-2.5 text-xs font-semibold text-[#203c30] shadow-sm transition-colors hover:border-[#9ab094] hover:bg-white cursor-pointer"
              >
                <Download size={15} />
                Export Backup (JSON)
              </button>
            )}
          </div>
        </div>

        {/* Ollama & Extraction Status Card */}
        <div className="mt-8 rounded-2xl border border-[#dce3d9] bg-white/70 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#203c30]/10 text-[#203c30]">
                <Brain size={22} />
              </div>
              <div>
                <h3 className="text-base font-semibold text-[#203c30]">
                  Personal Knowledge Extraction
                </h3>
                <div className="mt-1 flex items-center gap-2 text-xs">
                  {isCheckingOllama ? (
                    <span className="flex items-center gap-1.5 text-[#68766d]">
                      <Loader2 size={13} className="animate-spin" />
                      Checking Ollama...
                    </span>
                  ) : ollamaStatus?.isRunning && ollamaStatus?.isModelReady ? (
                    <span className="flex items-center gap-1.5 font-medium text-emerald-700">
                      <CheckCircle2 size={14} />
                      Ollama Connected ({DEFAULT_OLLAMA_MODEL})
                    </span>
                  ) : ollamaStatus?.availability === 'unknown' ? (
                    <span className="flex items-center gap-1.5 font-medium text-amber-700">
                      <Clock size={14} />
                      {ollamaStatus?.error || 'Ollama is responding slowly; checking again.'}
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 font-medium text-amber-700">
                      <AlertCircle size={14} />
                      {ollamaStatus?.error || 'Ollama Offline'}
                    </span>
                  )}

                  <button
                    type="button"
                    onClick={() => {
                      checkOllama();
                      fetchCoordinatorStatus();
                    }}
                    disabled={isCheckingOllama || isActionSubmitting || coordinatorStatus?.isProcessing}
                    className="ml-2 text-[#68766d] underline hover:text-[#203c30] cursor-pointer"
                    title="Re-check connection and status"
                  >
                    Check
                  </button>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleImportSampleWriting}
                disabled={isImporting || isActionSubmitting || coordinatorStatus?.isProcessing}
                className="inline-flex items-center gap-2 rounded-xl border border-[#c6d3c2] bg-white/90 px-4 py-2.5 text-xs font-semibold text-[#203c30] shadow-sm transition-colors hover:border-[#9ab094] hover:bg-white disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
                title="Import sample writing messages to test knowledge extraction"
              >
                {isImporting ? (
                  <>
                    <Loader2 size={15} className="animate-spin" />
                    Importing Sample Writing...
                  </>
                ) : (
                  <>
                    <FileText size={15} />
                    Import Sample Writing ({SAMPLE_WRITING_SAMPLES.length})
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={handleStartExtraction}
                disabled={
                  isActionSubmitting ||
                  coordinatorStatus?.isProcessing ||
                  !ollamaStatus?.isRunning ||
                  !ollamaStatus?.isModelReady ||
                  messages.length === 0
                }
                className="inline-flex items-center gap-2 rounded-xl bg-[#203c30] px-5 py-2.5 text-xs font-semibold text-white shadow-sm transition-all hover:bg-[#2d5242] disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
              >
                {isActionSubmitting || coordinatorStatus?.isProcessing ? (
                  <>
                    <Loader2 size={15} className="animate-spin" />
                    {coordinatorStatus?.isProcessing ? 'Extracting in Background...' : 'Starting Extraction...'}
                  </>
                ) : (
                  <>
                    <Sparkles size={15} />
                    Extract Personal Knowledge
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Coordinator Status Banner / Metrics */}
          {coordinatorStatus && (
            <div className="mt-5 border-t border-[#dce3d9]/70 pt-4">
              {/* Metrics Pill Row */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="rounded-lg bg-[#203c30]/5 px-2.5 py-1 font-medium text-[#203c30]">
                  Evaluated: <strong className="font-semibold">{coordinatorStatus.extractedCount}</strong>
                </span>
                <span className="rounded-lg bg-amber-50 px-2.5 py-1 font-medium text-amber-800">
                  Pending: <strong className="font-semibold">{coordinatorStatus.pendingCount}</strong>
                </span>
                <span className="rounded-lg bg-slate-100 px-2.5 py-1 font-medium text-slate-700">
                  Skipped: <strong className="font-semibold">{coordinatorStatus.skippedCount}</strong>
                </span>
                {coordinatorStatus.failedTemporaryCount > 0 && (
                  <span className="rounded-lg bg-orange-50 px-2.5 py-1 font-medium text-orange-700">
                    Retrying: <strong className="font-semibold">{coordinatorStatus.failedTemporaryCount}</strong>
                  </span>
                )}
                {coordinatorStatus.failedPermanentCount > 0 && (
                  <span className="rounded-lg bg-red-50 px-2.5 py-1 font-medium text-red-700">
                    Failed: <strong className="font-semibold">{coordinatorStatus.failedPermanentCount}</strong>
                  </span>
                )}
              </div>

              {/* State-specific Notice */}
              <div className="mt-3">
                {coordinatorStatus.isProcessing || coordinatorStatus.inFlightMessageId !== null ? (
                  <div className="flex items-center gap-2 text-xs font-medium text-[#356b45]">
                    <Loader2 size={14} className="animate-spin text-[#356b45]" />
                    <span>
                      Extraction currently running
                      {coordinatorStatus.inFlightMessageId ? ` (processing message #${coordinatorStatus.inFlightMessageId})` : ''}
                      {coordinatorStatus.pendingCount > 0 ? ` — ${coordinatorStatus.pendingCount} message(s) in queue` : ''}.
                    </span>
                  </div>
                ) : coordinatorStatus.serviceStatus === 'offline' ? (
                  <div className="flex items-center gap-2 text-xs text-amber-800">
                    <AlertCircle size={14} className="text-amber-600" />
                    <span>
                      Ollama is currently offline. Background extraction is paused and will automatically resume once Ollama is online.
                    </span>
                  </div>
                ) : coordinatorStatus.serviceStatus === 'unknown' ? (
                  <div className="flex items-center gap-2 text-xs text-amber-800">
                    <Clock size={14} className="text-amber-600" />
                    <span>
                      {coordinatorStatus.serviceError ||
                        'Ollama is responding slowly. Background extraction will automatically resume once Ollama is responsive.'}
                    </span>
                  </div>
                ) : coordinatorStatus.serviceStatus === 'blocked_configuration' ? (
                  <div className="flex items-center gap-2 text-xs text-red-700">
                    <AlertCircle size={14} className="text-red-600" />
                    <span>
                      Extraction blocked: {coordinatorStatus.serviceError || 'Model missing or origin rejected'}.
                    </span>
                  </div>
                ) : coordinatorStatus.pendingCount > 0 ? (
                  <div className="flex items-center gap-2 text-xs text-[#68766d]">
                    <Clock size={14} className="text-[#68766d]" />
                    <span>
                      {coordinatorStatus.pendingCount} message(s) waiting for background extraction.
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-[#456b4f]">
                    <CheckCircle2 size={14} className="text-[#356b45]" />
                    <span>All collected messages have been evaluated by the background coordinator.</span>
                  </div>
                )}
              </div>

              {/* Recovery Actions Row (Failed Permanent & Skipped) */}
              {(coordinatorStatus.failedPermanentCount > 0 || coordinatorStatus.skippedCount > 0) && (
                <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-[#dce3d9]/70 pt-3">
                  {coordinatorStatus.failedPermanentCount > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-red-700">
                        {coordinatorStatus.failedPermanentCount} message(s) permanently failed:
                      </span>
                      <button
                        type="button"
                        onClick={handleRetryFailed}
                        disabled={isActionSubmitting || coordinatorStatus.isProcessing}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-2.5 py-1 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
                      >
                        <RefreshCw size={12} className={isActionSubmitting ? 'animate-spin' : ''} />
                        Retry Failed
                      </button>
                    </div>
                  )}

                  {coordinatorStatus.skippedCount > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-[#68766d]">
                        {coordinatorStatus.skippedCount} message(s) skipped by filter:
                      </span>
                      <button
                        type="button"
                        onClick={handleReconsiderSkipped}
                        disabled={isActionSubmitting || coordinatorStatus.isProcessing}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-[#c6d3c2] bg-white px-2.5 py-1 text-xs font-semibold text-[#203c30] hover:bg-[#eef1eb] disabled:opacity-50"
                      >
                        <RotateCcw size={12} className={isActionSubmitting ? 'animate-spin' : ''} />
                        Reconsider Skipped
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Action Feedback Notice */}
          {actionFeedback && (
            <div className="mt-4 rounded-xl border border-[#c6d3c2] bg-emerald-50 p-3.5 text-xs text-emerald-800">
              <p className="font-semibold">{actionFeedback}</p>
            </div>
          )}

          {/* Action Error Notice */}
          {actionError && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3.5 text-xs text-red-800">
              <p className="font-semibold">{actionError}</p>
            </div>
          )}

          {/* Ingestion Status Notice */}
          {ingestionSummary && (
            <div className="mt-4 rounded-xl border border-[#cfe1f0] bg-[#e8f1f8] p-3.5 text-xs text-[#205b82]">
              <p className="font-semibold">Sample Writing Ingestion:</p>
              <p className="mt-1">
                {ingestionSummary.importedCount > 0
                  ? `Successfully imported ${ingestionSummary.importedCount} sample message(s) into writing data (${ingestionSummary.existingCount} already existed).`
                  : `All ${ingestionSummary.existingCount} sample messages are already present in your writing data.`}
              </p>
            </div>
          )}

          {/* Ingestion Error Notice */}
          {ingestionError && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3.5 text-xs text-red-800">
              <p className="font-semibold">Ingestion failed:</p>
              <p className="mt-1">{ingestionError}</p>
            </div>
          )}
        </div>

        {/* Navigation Tabs */}
        <div className="mt-8 flex gap-2 border-b border-[#dce3d9] pb-4">
          <button
            type="button"
            onClick={() => {
              setActiveTab('messages');
              onTabChange?.('messages');
            }}
            className={`flex items-center gap-2 rounded-xl px-5 py-2.5 text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'messages'
                ? 'bg-[#203c30] text-white shadow-sm'
                : 'text-[#68766d] hover:bg-white/60 hover:text-[#203c30]'
            }`}
          >
            <FileText size={15} />
            <span>Collected Messages</span>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] ${
                activeTab === 'messages'
                  ? 'bg-white/20 text-white'
                  : 'bg-[#203c30]/10 text-[#203c30]'
              }`}
            >
              {messages.length}
            </span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab('knowledge');
              onTabChange?.('knowledge');
            }}
            className={`flex items-center gap-2 rounded-xl px-5 py-2.5 text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'knowledge'
                ? 'bg-[#203c30] text-white shadow-sm'
                : 'text-[#68766d] hover:bg-white/60 hover:text-[#203c30]'
            }`}
          >
            <Brain size={15} />
            <span>Extracted Personal Knowledge</span>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] ${
                activeTab === 'knowledge'
                  ? 'bg-white/20 text-white'
                  : 'bg-[#203c30]/10 text-[#203c30]'
              }`}
            >
              {knowledgeItems.length}
            </span>
          </button>
        </div>

        {/* Global Loading / Errors */}
        {isLoading && (
          <p className="mt-8 text-sm text-[#68766d]">
            Loading your data...
          </p>
        )}

        {error && (
          <p className="mt-8 text-sm text-red-600">
            {error}
          </p>
        )}

        {/* TAB 1: Collected Messages */}
        {activeTab === 'messages' && !isLoading && !error && (
          <>
            {messages.length === 0 ? (
              <p className="mt-8 text-sm text-[#68766d]">
                No messages collected yet. Your submitted writing samples will appear here.
              </p>
            ) : (
              <div className="mt-6 space-y-4">
                {/* Search & Filter Toolbar */}
                <div className="flex flex-wrap items-center gap-3 rounded-xl border border-[#dce3d9] bg-white/70 p-3">
                  <div className="relative flex-1 min-w-[200px]">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#68766d]" />
                    <input
                      type="text"
                      value={messageSearch}
                      onChange={(e) => setMessageSearch(e.target.value)}
                      placeholder="Search messages by content or platform..."
                      className="w-full rounded-lg bg-white border border-[#dce3d9] pl-9 pr-3 py-1.5 text-xs text-[#203c30] placeholder:text-[#9ca79e] outline-none focus:border-[#203c30]"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <select
                      value={platformFilter}
                      onChange={(e) => setPlatformFilter(e.target.value)}
                      className="rounded-lg bg-white border border-[#dce3d9] px-2.5 py-1.5 text-xs text-[#203c30] outline-none focus:border-[#203c30] cursor-pointer"
                    >
                      <option value="all">All Platforms</option>
                      <option value="X">X (Twitter)</option>
                      <option value="LinkedIn">LinkedIn</option>
                      <option value="General">General</option>
                    </select>
                    <select
                      value={statusFilter}
                      onChange={(e) => setStatusFilter(e.target.value as any)}
                      className="rounded-lg bg-white border border-[#dce3d9] px-2.5 py-1.5 text-xs text-[#203c30] outline-none focus:border-[#203c30] cursor-pointer"
                    >
                      <option value="all">All Statuses</option>
                      <option value="pending">Pending</option>
                      <option value="trained">Trained</option>
                    </select>
                    {(messageSearch || platformFilter !== 'all' || statusFilter !== 'all') && (
                      <button
                        type="button"
                        onClick={() => {
                          setMessageSearch('');
                          setPlatformFilter('all');
                          setStatusFilter('all');
                        }}
                        className="rounded-lg px-2.5 py-1.5 text-xs text-[#203c30] hover:bg-[#eef1eb] cursor-pointer"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>

                {filteredMessages.length === 0 ? (
                  <div className="rounded-xl border border-[#dce3d9] bg-white/70 p-8 text-center text-xs text-[#68766d]">
                    No messages match your search or filter.
                  </div>
                ) : (
                  filteredMessages.map((message) => (
                    <article
                      key={message.id}
                      className="rounded-xl border border-[#dce3d9] bg-white/80 p-5 shadow-sm"
                    >
                      <div className="flex items-center justify-between">
                        <span className="rounded-full bg-[#356b45]/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-[#356b45]">
                          {message.platform}
                        </span>
                        <span className="text-xs text-[#68766d]">
                          {new Date(message.capturedAt).toLocaleString()}
                        </span>
                      </div>

                      <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-[#203c30]">
                        {message.text}
                      </p>

                      <div className="mt-5 flex items-center justify-between border-t border-[#f0f2eb] pt-4">
                        <span className="text-xs font-medium text-[#68766d]">
                          Training status
                        </span>

                        {message.trainingStatus === 'pending' ? (
                          <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-800">
                            Pending training
                          </span>
                        ) : (
                          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-800">
                            Trained
                          </span>
                        )}
                      </div>

                      <div className="mt-4 flex justify-end">
                        <button
                          type="button"
                          onClick={() => openDeleteConfirmation(message.id)}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-[#e5e7df] px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-50 transition-colors cursor-pointer"
                        >
                          <Trash2 size={13} />
                          Delete
                        </button>
                      </div>
                    </article>
                  ))
                )}
              </div>
            )}
          </>
        )}

        {/* TAB 2: Personal Knowledge */}
        {activeTab === 'knowledge' && !isLoading && !error && (
          <>
            <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-[#203c30]">
                  Personal Knowledge
                </h3>
                <p className="text-xs text-[#68766d]">
                  Facts about your skills, projects, and preferences used to ground your personalized drafts.
                </p>
              </div>

              <button
                type="button"
                onClick={openAddKnowledgeModal}
                className="inline-flex items-center gap-2 rounded-xl bg-[#203c30] px-4 py-2 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-[#2d5242] cursor-pointer"
              >
                <Plus size={14} />
                Add Knowledge
              </button>
            </div>

            {/* Search & Filter Toolbar for Knowledge */}
            <div className="mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-[#dce3d9] bg-white/70 p-3">
              <div className="relative flex-1 min-w-[200px]">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#68766d]" />
                <input
                  type="text"
                  value={knowledgeSearch}
                  onChange={(e) => setKnowledgeSearch(e.target.value)}
                  placeholder="Search facts by subject, predicate, or value..."
                  className="w-full rounded-lg bg-white border border-[#dce3d9] pl-9 pr-3 py-1.5 text-xs text-[#203c30] placeholder:text-[#9ca79e] outline-none focus:border-[#203c30]"
                />
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="rounded-lg bg-white border border-[#dce3d9] px-2.5 py-1.5 text-xs text-[#203c30] outline-none focus:border-[#203c30] cursor-pointer"
                >
                  <option value="all">All Categories</option>
                  <option value="skills">Skills</option>
                  <option value="projects">Projects</option>
                  <option value="preferences">Preferences</option>
                  <option value="background">Background</option>
                  <option value="style">Style</option>
                  <option value="other">Other</option>
                </select>
                <select
                  value={confidenceFilter}
                  onChange={(e) => setConfidenceFilter(e.target.value)}
                  className="rounded-lg bg-white border border-[#dce3d9] px-2.5 py-1.5 text-xs text-[#203c30] outline-none focus:border-[#203c30] cursor-pointer"
                >
                  <option value="all">All Confidences</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                {(knowledgeSearch || categoryFilter !== 'all' || confidenceFilter !== 'all') && (
                  <button
                    type="button"
                    onClick={() => {
                      setKnowledgeSearch('');
                      setCategoryFilter('all');
                      setConfidenceFilter('all');
                    }}
                    className="rounded-lg px-2.5 py-1.5 text-xs text-[#203c30] hover:bg-[#eef1eb] cursor-pointer"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {knowledgeItems.length === 0 ? (
              <div className="mt-8 rounded-2xl border border-dashed border-[#c6d3c2] p-10 text-center bg-white/40">
                <Sparkles size={28} className="mx-auto text-[#68766d]" />
                <h4 className="mt-3 text-base font-semibold text-[#203c30]">
                  No Knowledge Items Yet
                </h4>
                <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-[#68766d]">
                  Add a personal fact manually or click "Extract Personal Knowledge" above to have Ollama analyze your writing data.
                </p>
                <div className="mt-5">
                  <button
                    type="button"
                    onClick={openAddKnowledgeModal}
                    className="inline-flex items-center gap-2 rounded-xl bg-[#203c30] px-4 py-2 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-[#2d5242] cursor-pointer"
                  >
                    <Plus size={14} />
                    Add Knowledge Fact
                  </button>
                </div>
              </div>
            ) : filteredKnowledge.length === 0 ? (
              <div className="mt-6 rounded-xl border border-[#dce3d9] bg-white/70 p-8 text-center text-xs text-[#68766d]">
                No personal facts match your search or filter.
              </div>
            ) : (
              <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
                {filteredKnowledge.map((item) => (
                  <div
                    key={item.id}
                    className="flex flex-col justify-between rounded-xl border border-[#dce3d9] bg-white/80 p-5 shadow-sm"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="rounded-full bg-[#356b45]/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-[#356b45]">
                            {item.category}
                          </span>

                          {item.sourceType === 'manual' ? (
                            <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-[11px] font-medium text-blue-800">
                              Manually added
                            </span>
                          ) : item.editedByUser ? (
                            <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-[11px] font-medium text-amber-800">
                              Edited
                            </span>
                          ) : (
                            <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-[11px] font-medium text-emerald-800">
                              Extracted
                            </span>
                          )}
                        </div>

                        <span className="text-[11px] font-medium text-[#68766d]">
                          {item.confidence} confidence
                        </span>
                      </div>

                      <h4 className="mt-3 text-base font-semibold text-[#203c30]">
                        {item.subject}
                      </h4>

                      <p className="mt-1 text-xs font-medium text-[#405247]">
                        <span className="italic text-[#68766d]">{item.predicate}:</span>{' '}
                        {item.value}
                      </p>

                      {item.contextScope && (
                        <p className="mt-2 text-[11px] text-[#68766d]">
                          Scope: <span className="font-medium text-[#405247]">{item.contextScope}</span>
                        </p>
                      )}
                    </div>

                    <div>
                      <div className="mt-4 flex items-center justify-between border-t border-[#f0f2eb] pt-3 text-[11px] text-[#86968c]">
                        <span>Importance: {item.importance}</span>
                        <span>
                          {item.sourceType === 'manual'
                            ? `Added: ${new Date(item.createdAt).toLocaleDateString()}`
                            : `Confirmed: ${new Date(item.lastConfirmedAt).toLocaleDateString()}`}
                        </span>
                      </div>

                      <div className="mt-3 flex items-center justify-end gap-2 border-t border-[#f0f2eb] pt-2.5">
                        <button
                          type="button"
                          onClick={() => openEditKnowledgeModal(item)}
                          aria-label={`Edit ${item.subject}`}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-[#dce3d9] bg-white px-3 py-1.5 text-xs font-semibold text-[#405247] transition-colors hover:bg-[#f0f2eb] cursor-pointer"
                        >
                          <Pencil size={13} />
                          Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => openDeleteKnowledgeModal(item)}
                          aria-label={`Delete ${item.subject}`}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-700 transition-colors hover:bg-red-100 cursor-pointer"
                        >
                          <Trash2 size={13} />
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </section>

      {/* Add / Edit Knowledge Modal */}
      {isKnowledgeModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#14231c]/50 px-4 backdrop-blur-sm">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="knowledge-dialog-title"
            className="w-full max-w-lg rounded-2xl border border-[#e5e7df] bg-[#faf9f4] p-7 shadow-2xl max-h-[90vh] overflow-y-auto"
          >
            <div className="flex items-start justify-between">
              <div>
                <h2
                  id="knowledge-dialog-title"
                  className="font-serif text-2xl text-[#203c30]"
                >
                  {knowledgeModalMode === 'add' ? 'Add Personal Knowledge' : 'Edit Personal Knowledge'}
                </h2>
                <p className="mt-1 text-xs text-[#68766d]">
                  {knowledgeModalMode === 'add'
                    ? 'Manually record a fact about your skills, projects, or preferences for draft grounding.'
                    : 'Update existing fact details. Source provenance and creation timestamp are preserved.'}
                </p>
              </div>

              <button
                type="button"
                onClick={() => setIsKnowledgeModalOpen(false)}
                disabled={isSavingKnowledge}
                aria-label="Close dialog"
                className="rounded-lg p-2 text-[#68766d] transition-colors hover:bg-[#eeeee7] hover:text-[#203c30] disabled:opacity-50 cursor-pointer"
              >
                <X size={19} />
              </button>
            </div>

            {knowledgeFormError && (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                <p className="font-semibold">{knowledgeFormError}</p>
              </div>
            )}

            <form onSubmit={handleSaveKnowledge} className="mt-5 space-y-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="knowledge-category" className="block text-xs font-semibold text-[#405247]">
                    Category *
                  </label>
                  <select
                    id="knowledge-category"
                    value={knowledgeFormData.category}
                    onChange={(e) =>
                      setKnowledgeFormData({ ...knowledgeFormData, category: e.target.value as KnowledgeCategory })
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-3 py-2 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none cursor-pointer"
                  >
                    <option value="skills">Skills</option>
                    <option value="projects">Projects</option>
                    <option value="goals">Goals</option>
                    <option value="habits">Habits</option>
                    <option value="preferences">Preferences</option>
                    <option value="opinions">Opinions</option>
                    <option value="states">States</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="knowledge-context-scope" className="block text-xs font-semibold text-[#405247]">
                    Context Scope (optional)
                  </label>
                  <input
                    type="text"
                    id="knowledge-context-scope"
                    placeholder="e.g. backend, mobile, general"
                    value={knowledgeFormData.contextScope}
                    onChange={(e) =>
                      setKnowledgeFormData({ ...knowledgeFormData, contextScope: e.target.value })
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-3 py-2 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="knowledge-subject" className="block text-xs font-semibold text-[#405247]">
                  Subject *
                </label>
                <input
                  type="text"
                  id="knowledge-subject"
                  placeholder="e.g. TypeScript, Task Tracker, Morning Routine"
                  value={knowledgeFormData.subject}
                  onChange={(e) =>
                    setKnowledgeFormData({ ...knowledgeFormData, subject: e.target.value })
                  }
                  className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-3 py-2 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="knowledge-predicate" className="block text-xs font-semibold text-[#405247]">
                  Predicate *
                </label>
                <input
                  type="text"
                  id="knowledge-predicate"
                  placeholder="e.g. uses, built, prefers, plans to learn"
                  value={knowledgeFormData.predicate}
                  onChange={(e) =>
                    setKnowledgeFormData({ ...knowledgeFormData, predicate: e.target.value })
                  }
                  className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-3 py-2 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="knowledge-value" className="block text-xs font-semibold text-[#405247]">
                  Value *
                </label>
                <textarea
                  id="knowledge-value"
                  rows={2}
                  placeholder="e.g. for backend APIs and distributed services"
                  value={knowledgeFormData.value}
                  onChange={(e) =>
                    setKnowledgeFormData({ ...knowledgeFormData, value: e.target.value })
                  }
                  className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-3 py-2 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div>
                  <label htmlFor="knowledge-stability" className="block text-xs font-semibold text-[#405247]">
                    Stability
                  </label>
                  <select
                    id="knowledge-stability"
                    value={knowledgeFormData.stability}
                    onChange={(e) =>
                      setKnowledgeFormData({ ...knowledgeFormData, stability: e.target.value as KnowledgeStability })
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-2.5 py-1.5 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none cursor-pointer"
                  >
                    <option value="stable">Stable</option>
                    <option value="evolving">Evolving</option>
                    <option value="temporary">Temporary</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="knowledge-confidence" className="block text-xs font-semibold text-[#405247]">
                    Confidence
                  </label>
                  <select
                    id="knowledge-confidence"
                    value={knowledgeFormData.confidence}
                    onChange={(e) =>
                      setKnowledgeFormData({ ...knowledgeFormData, confidence: e.target.value as KnowledgeConfidence })
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-2.5 py-1.5 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none cursor-pointer"
                  >
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="knowledge-importance" className="block text-xs font-semibold text-[#405247]">
                    Importance
                  </label>
                  <select
                    id="knowledge-importance"
                    value={knowledgeFormData.importance}
                    onChange={(e) =>
                      setKnowledgeFormData({ ...knowledgeFormData, importance: e.target.value as KnowledgeImportance })
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#dce3d9] bg-white px-2.5 py-1.5 text-xs text-[#203c30] focus:border-[#203c30] focus:outline-none cursor-pointer"
                  >
                    <option value="primary">Primary</option>
                    <option value="secondary">Secondary</option>
                    <option value="incidental">Incidental</option>
                  </select>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-3 pt-3 border-t border-[#e5e7df]">
                <button
                  type="button"
                  onClick={() => setIsKnowledgeModalOpen(false)}
                  disabled={isSavingKnowledge}
                  className="rounded-lg border border-[#dce3d9] px-4 py-2 text-xs font-semibold text-[#405247] transition-colors hover:bg-[#eeeee7] disabled:opacity-50 cursor-pointer"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={isSavingKnowledge}
                  className="inline-flex items-center gap-2 rounded-lg bg-[#203c30] hover:bg-[#2d5242] px-5 py-2 text-xs font-semibold text-white shadow-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
                >
                  {isSavingKnowledge ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      Saving...
                    </>
                  ) : knowledgeModalMode === 'add' ? (
                    <>
                      <Plus size={13} />
                      Save Knowledge
                    </>
                  ) : (
                    'Update Knowledge'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Knowledge Confirmation Modal */}
      {deletingKnowledgeItem !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#14231c]/50 px-4 backdrop-blur-sm">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-knowledge-dialog-title"
            className="w-full max-w-md rounded-2xl border border-[#e5e7df] bg-[#faf9f4] p-7 shadow-2xl"
          >
            <div className="flex items-start justify-between">
              <div className="flex size-12 items-center justify-center rounded-full bg-red-100 text-red-700">
                <AlertTriangle size={23} />
              </div>

              <button
                type="button"
                onClick={() => setDeletingKnowledgeItem(null)}
                disabled={isDeletingKnowledge}
                aria-label="Close confirmation"
                className="rounded-lg p-2 text-[#68766d] transition-colors hover:bg-[#eeeee7] hover:text-[#203c30] disabled:opacity-50 cursor-pointer"
              >
                <X size={19} />
              </button>
            </div>

            <h2
              id="delete-knowledge-dialog-title"
              className="mt-5 font-serif text-2xl text-[#203c30]"
            >
              Delete this knowledge fact?
            </h2>

            <p className="mt-3 text-sm leading-6 text-[#68766d]">
              This fact will be archived and excluded from future personalized draft retrieval.
              Any linked message evidence records will be invalidated.
            </p>

            <div className="mt-4 rounded-xl border border-[#dce3d9] bg-white p-3 text-xs text-[#203c30]">
              <span className="font-semibold">{deletingKnowledgeItem.subject}</span>{' '}
              <span className="text-[#68766d]">({deletingKnowledgeItem.predicate}):</span>{' '}
              <span>{deletingKnowledgeItem.value}</span>
            </div>

            {knowledgeDeleteError && (
              <div className="mt-3 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                <p className="font-semibold">{knowledgeDeleteError}</p>
              </div>
            )}

            <div className="mt-7 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setDeletingKnowledgeItem(null)}
                disabled={isDeletingKnowledge}
                className="rounded-lg border border-[#dce3d9] px-5 py-2.5 text-sm font-semibold text-[#405247] transition-colors hover:bg-[#eeeee7] disabled:opacity-50 cursor-pointer"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleConfirmDeleteKnowledge}
                disabled={isDeletingKnowledge}
                className="inline-flex items-center gap-2 rounded-lg bg-red-700 hover:bg-red-800 px-5 py-2.5 text-sm font-semibold text-white transition-colors disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
              >
                <Trash2 size={15} />
                {isDeletingKnowledge ? 'Deleting...' : 'Delete Fact'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {selectedMessageId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#14231c]/50 px-4 backdrop-blur-sm">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-dialog-title"
            className="w-full max-w-md rounded-2xl border border-[#e5e7df] bg-[#faf9f4] p-7 shadow-2xl"
          >
            <div className="flex items-start justify-between">
              <div className="flex size-12 items-center justify-center rounded-full bg-red-100 text-red-700">
                <AlertTriangle size={23} />
              </div>

              <button
                type="button"
                onClick={() => setSelectedMessageId(null)}
                disabled={isDeleting}
                aria-label="Close confirmation"
                className="rounded-lg p-2 text-[#68766d] transition-colors hover:bg-[#eeeee7] hover:text-[#203c30] disabled:opacity-50 cursor-pointer"
              >
                <X size={19} />
              </button>
            </div>

            <h2
              id="delete-dialog-title"
              className="mt-5 font-serif text-2xl text-[#203c30]"
            >
              Delete this message?
            </h2>

            <p className="mt-3 text-sm leading-6 text-[#68766d]">
              This message will be permanently removed from your writing data.
              Any linked personal knowledge evidence and derived facts will be privacy-scrubbed and archived.
            </p>

            <div className="mt-8 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setSelectedMessageId(null)}
                disabled={isDeleting}
                className="rounded-lg border border-[#dce3d9] px-5 py-2.5 text-sm font-semibold text-[#405247] transition-colors hover:bg-[#eeeee7] disabled:opacity-50 cursor-pointer"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleDelete}
                disabled={isDeleting}
                className="inline-flex items-center gap-2 rounded-lg bg-red-700 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
              >
                <Trash2 size={15} />
                {isDeleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default WritingData;
