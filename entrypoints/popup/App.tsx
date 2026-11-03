
function App() {
  function openDashboard() {
    browser.tabs.create({
      url: browser.runtime.getURL('/dashboard.html'),
    });
  }

  return (
    <main className="w-[360px] min-h-[340px] bg-[#0B0F0E] p-5 text-white">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/15 text-xl font-bold text-emerald-400">
            E
          </div>
          <span className="text-lg font-bold tracking-[0.2em]">
            ECHO
          </span>
        </div>

        <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-400">
          LOCAL FIRST
        </span>
      </header>

      {/* Introduction */}
      <section className="mt-8">
        <h1 className="text-2xl font-semibold leading-tight tracking-tight">
          Your writing.
          <br />
          <span className="text-emerald-400">Your style.</span>
        </h1>

        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          A personal AI writing assistant that learns how you communicate.
        </p>
      </section>

      {/* Setup Status */}
      <section className="mt-7 rounded-xl border border-white/10 bg-white/[0.03] p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-zinc-200">
            Setup status
          </h2>

          <span className="rounded-full bg-amber-400/10 px-2.5 py-1 text-xs font-medium text-amber-400">
            In progress
          </span>
        </div>

        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          Your Echo workspace is being prepared. Writing collection and
          personalization will be available in upcoming updates.
        </p>
      </section>

      {/* Dashboard Button */}
      <button
        type="button"
        onClick={openDashboard}
        className="mt-5 w-full rounded-xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-[#07130C] transition hover:bg-emerald-400"
      >
        Open Dashboard ↗
      </button>

      {/* Footer */}
      <footer className="mt-6 border-t border-white/10 pt-4 text-center">
        <p className="text-xs text-zinc-500">
          Your writing stays on your device.
        </p>
      </footer>
    </main>
  );
}

export default App;
