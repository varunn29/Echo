import { useState } from 'react';
import {
  ArrowUpRight,
  Globe2,
  Sparkles,
} from 'lucide-react';

import {
  faXTwitter,
  faLinkedinIn,
} from '@fortawesome/free-brands-svg-icons';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

function WritingComposer() {
  const [topic, setTopic] = useState('');
  const [platform, setPlatform] = useState('X');

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

  function handleGenerate() {
    if (!topic.trim()) {
      return;
    }

    console.log({
      topic: topic,
      platform: platform,
    });
  }

  return (
    <section className="relative z-10 px-12">
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

          <span className="hidden rounded-full bg-[#edf1e9] px-3 py-1.5 text-[10px] font-medium text-[#55725a] sm:inline-flex">
            Your voice, your way
          </span>
        </div>

        {/* Topic input */}
        <div className="px-6 py-5">
          <label
            htmlFor="writing-topic"
            className="mb-3 block text-[10px] font-semibold uppercase tracking-[0.16em] text-[#788578]"
          >
            What's on your mind?
          </label>

          <textarea
            id="writing-topic"
            value={topic}
            onChange={function (event) {
              setTopic(event.target.value);
            }}
            placeholder="Share an idea, a thought, an experience, or something you'd like to talk about..."
            rows={3}
            className="w-full resize-none bg-transparent text-[14px] leading-7 text-[#304438] outline-none placeholder:text-[#a4aaa0]"
          />
        </div>

        {/* Footer */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-[#e9e9df] bg-[#fafaf5]/80 px-6 py-4">

          {/* Platform selector */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#899087]">
              Write for
            </span>

            <div className="flex items-center gap-1 rounded-xl bg-[#eeefe8] p-1">
              {platforms.map(function (item) {
                const isActive = platform === item.name;

                return (
                  <button
                    key={item.name}
                    type="button"
                    onClick={function () {
                      setPlatform(item.name);
                    }}
                    className={`flex items-center gap-2 rounded-lg px-3 py-2 text-[11px] font-medium transition-all ${
                      isActive
                        ? 'bg-white text-[#315b42] shadow-sm'
                        : 'text-[#818b80] hover:text-[#315b42]'
                    }`}
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

          {/* Generate button */}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!topic.trim()}
            className="group flex items-center gap-3 rounded-xl bg-[#234b37] px-5 py-3 text-[12px] font-semibold text-[#f7f6ed] shadow-[0_4px_12px_rgba(35,75,55,0.15)] transition-all hover:bg-[#315d43] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Sparkles size={15} />

            Generate

            <ArrowUpRight
              size={14}
              className="transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
            />
          </button>

        </div>
      </div>
    </section>
  );
}

export default WritingComposer;