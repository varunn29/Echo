import {
  AudioLines,
  ArrowUpRight,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

function App() {
  function openDashboard() {
    browser.tabs.create({
      url: browser.runtime.getURL('/dashboard.html'),
    });
  }

  return (
    <main className="w-[360px] overflow-hidden bg-[#f8f7f0] text-[#294435]">

      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#e5e7dc] px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#234b37] text-[#f4f2e9]">
            <AudioLines size={21} strokeWidth={1.8} />
          </div>

          <div>
            <h1 className="text-[21px] font-semibold tracking-tight">
              Echo<span className="text-[#78947d]">.</span>
            </h1>
            <p className="text-[9px] font-medium uppercase tracking-[0.2em] text-[#899087]">
              Personal writing studio
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 rounded-full border border-[#dce3d7] px-2.5 py-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[#71966e]" />
          <span className="text-[9px] font-medium text-[#55725a]">
            Local first
          </span>
        </div>
      </div>

      {/* Introduction */}
      <section className="px-6 pb-6 pt-8">
        <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-xl bg-[#e9eee3] text-[#52745e]">
          <Sparkles size={17} strokeWidth={1.7} />
        </div>

        <h2 className="font-serif text-[30px] leading-[1.2] tracking-tight">
          Your words.
          <br />
          <span className="italic text-[#66836d]">Your voice.</span>
        </h2>

        <p className="mt-3 max-w-[280px] text-[12px] leading-[1.8] text-[#788378]">
          A personal writing assistant that learns your style and helps you
          express your thoughts naturally.
        </p>

        {/* Dashboard action */}
        <button
          type="button"
          onClick={openDashboard}
          className="group mt-7 flex w-full items-center justify-between rounded-xl bg-[#234b37] px-5 py-4 text-[#f7f6ed] shadow-[0_5px_16px_rgba(35,75,55,0.14)] transition-colors hover:bg-[#315d43]"
        >
          <span className="text-[13px] font-semibold">
            Open your workspace
          </span>

          <ArrowUpRight
            size={18}
            className="transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
          />
        </button>
      </section>

      {/* Privacy footer */}
      <div className="border-t border-[#e5e7dc] bg-[#f1f1e9] px-6 py-4">
        <div className="flex items-center gap-2">
          <ShieldCheck size={16} className="shrink-0 text-[#66836d]" />

          <div>
            <p className="text-[10px] font-semibold text-[#45634d]">
              Your writing stays yours.
            </p>
            <p className="mt-0.5 text-[9px] text-[#899087]">
              Designed with privacy at its core.
            </p>
          </div>
        </div>
      </div>

    </main>
  );
}

export default App;