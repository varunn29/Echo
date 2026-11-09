
import { useEffect, useState } from 'react';
import { Trash2, X, AlertTriangle, Download } from 'lucide-react';

import {
  getWritingMessages,
  deleteWritingMessage,
  type WritingMessage,
} from '../../utils/database';

function WritingData() {
  const [messages, setMessages] = useState<WritingMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedMessageId, setSelectedMessageId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(function () {
    async function loadMessages() {
      try {
        const savedMessages = await getWritingMessages();
        setMessages(savedMessages);
      } catch (error) {
        console.error('Echo: Failed to load messages:', error);
        setError('Unable to load your writing data.');
      } finally {
        setIsLoading(false);
      }
    }

    loadMessages();
  }, []);

  function handleExport() {
    if (messages.length === 0) {
      return;
    }

    const exportData = {
      exportedAt: new Date().toISOString(),
      totalMessages: messages.length,
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
    };

    const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const downloadUrl = URL.createObjectURL(jsonBlob);
    const downloadLink = document.createElement('a');
    const dateStamp = new Date().toISOString().split('T')[0];

    downloadLink.href = downloadUrl;
    downloadLink.download = `echo-writing-data-${dateStamp}.json`;
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

      setMessages(function (currentMessages) {
        return currentMessages.filter(function (message) {
          return message.id !== selectedMessageId;
        });
      });

      setSelectedMessageId(null);
    } catch (error) {
      console.error('Echo: Failed to delete message:', error);
      setError('Unable to delete this message. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <>
      <section className="px-12 py-12">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl text-[#203c30]">
              My Writing Data
            </h1>

            <p className="mt-3 text-sm text-[#68766d]">
              Your collected writing, stored privately on this device.
            </p>
          </div>

          {!isLoading && messages.length > 0 && (
            <button
              type="button"
              onClick={handleExport}
              className="inline-flex items-center gap-2 rounded-xl border border-[#c6d3c2] bg-white/90 px-4 py-2.5 text-xs font-semibold text-[#203c30] shadow-sm transition-colors hover:border-[#9ab094] hover:bg-white"
            >
              <Download size={15} />
              Export Data (JSON)
            </button>
          )}
        </div>

        {isLoading && (
          <p className="mt-8 text-sm text-[#68766d]">
            Loading your messages...
          </p>
        )}

        {error && (
          <p className="mt-8 text-sm text-red-600">
            {error}
          </p>
        )}

        {!isLoading && !error && messages.length === 0 && (
          <p className="mt-8 text-sm text-[#68766d]">
            No messages collected yet. Your submitted ChatGPT messages will appear here.
          </p>
        )}

        {!isLoading && messages.length > 0 && (
          <div className="mt-8 space-y-4">
            {messages.map(function (message) {
              return (
                <article
                  key={message.id}
                  className="rounded-xl border border-[#dce3d9] bg-[#f8f6ef]/90 p-5"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-[#203c30]">
                      {message.platform}
                    </span>

                    <span className="text-xs text-[#68766d]">
                      {message.capturedAt.toLocaleString()}
                    </span>
                  </div>

                  <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-[#405247]">
                    {message.text}
                  </p>

                  <div className="mt-5 flex items-center justify-between border-t border-[#dce3d9] pt-4">
                    <span className="text-xs font-medium text-[#68766d]">
                      Training status
                    </span>

                    {message.trainingStatus === 'pending' ? (
                      <span className="rounded-full bg-[#f5ead2] px-3 py-1 text-xs font-semibold text-[#936b25]">
                        Pending training
                      </span>
                    ) : (
                      <span className="rounded-full bg-[#dcecdf] px-3 py-1 text-xs font-semibold text-[#356b45]">
                        Trained
                      </span>
                    )}
                  </div>

                  <div className="mt-4 flex justify-end">
                    <button
                      type="button"
                      onClick={function () {
                        openDeleteConfirmation(message.id);
                      }}
                      className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm font-semibold text-red-700 transition-colors hover:border-red-300 hover:bg-red-100"
                    >
                      <Trash2 size={15} />
                      Delete
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>

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
                onClick={function () {
                  setSelectedMessageId(null);
                }}
                disabled={isDeleting}
                aria-label="Close confirmation"
                className="rounded-lg p-2 text-[#68766d] transition-colors hover:bg-[#eeeee7] hover:text-[#203c30] disabled:opacity-50"
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
              This action cannot be undone.
            </p>

            <div className="mt-8 flex justify-end gap-3">
              <button
                type="button"
                onClick={function () {
                  setSelectedMessageId(null);
                }}
                disabled={isDeleting}
                className="rounded-lg border border-[#dce3d9] px-5 py-2.5 text-sm font-semibold text-[#405247] transition-colors hover:bg-[#eeeee7] disabled:opacity-50"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleDelete}
                disabled={isDeleting}
                className="inline-flex items-center gap-2 rounded-lg bg-red-700 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-60"
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
