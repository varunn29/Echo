
import { useEffect, useState } from 'react';
import {
  getWritingMessages,
  type WritingMessage,
} from '../../utils/database';

function WritingData() {
  const [messages, setMessages] = useState<WritingMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

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

  return (
    <section className="px-12 py-12">
      <h1 className="font-serif text-4xl text-[#203c30]">
        My Writing Data
      </h1>

      <p className="mt-3 text-sm text-[#68766d]">
        Your collected writing, stored privately on this device.
      </p>

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

      {!isLoading && !error && messages.length > 0 && (
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
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default WritingData;
