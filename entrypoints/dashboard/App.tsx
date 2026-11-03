

function App() {
  return (
    <div className="min-h-screen bg-[#0B0F0E] text-white">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 flex w-64 flex-col border-r border-white/10 bg-[#0D1310] p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/15 text-xl font-bold text-emerald-400">
            E
          </div>

          <span className="text-xl font-bold tracking-[0.2em]">
            ECHO
          </span>
        </div>

        <div className="mt-12">
          <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-zinc-500">
            Workspace
          </p>

          <nav className="space-y-2">
            <a
              href="#dashboard"
              className="flex items-center gap-3 rounded-lg bg-emerald-500/10 px-4 py-3 text-sm font-medium text-emerald-400"
            >
              <span>▦</span>
              Dashboard
            </a>

            <a
              href="#writing-data"
              className="flex items-center gap-3 rounded-lg px-4 py-3 text-sm text-zinc-400 transition hover:bg-white/5 hover:text-white"
            >
              <span>✎</span>
              My Writing Data
            </a>

            <a
              href="#writing-style"
              className="flex items-center gap-3 rounded-lg px-4 py-3 text-sm text-zinc-400 transition hover:bg-white/5 hover:text-white"
            >
              <span>✧</span>
              My Writing Style
            </a>
          </nav>
        </div>

        <div className="mt-10">
          <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-zinc-500">
            Preferences
          </p>

          <a
            href="#settings"
            className="flex items-center gap-3 rounded-lg px-4 py-3 text-sm text-zinc-400 transition hover:bg-white/5 hover:text-white"
          >
            <span>⚙</span>
            Settings
          </a>
        </div>

        <div className="mt-auto rounded-xl border border-white/10 bg-white/[0.03] p-4">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">
              Local-first
            </span>
          </div>

          <p className="mt-2 text-xs leading-relaxed text-zinc-500">
            Your writing data stays on your device.
          </p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64 min-h-screen p-10">
        {/* Top Bar */}
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div>
            <p className="text-sm text-zinc-500">
              Workspace / Dashboard
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-500/15 text-sm font-semibold text-emerald-400">
              U
            </div>
            <span className="text-sm text-zinc-300">My Workspace</span>
          </div>
        </header>

        {/* Welcome Section */}
        <section className="mt-10">
          <h1 className="text-3xl font-semibold tracking-tight">
            Good afternoon.
          </h1>

          <p className="mt-2 text-zinc-400">
            Let's create something that sounds like you.
          </p>
        </section>

        {/* Generation Card */}
        <section className="mt-10 max-w-4xl rounded-2xl border border-white/10 bg-[#111916] p-7">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">
                Create something new
              </h2>

              <p className="mt-1 text-sm text-zinc-400">
                Turn your ideas into writing that sounds like you.
              </p>
            </div>

            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-500/10 text-xl text-emerald-400">
              ✧
            </div>
          </div>

          <div className="mt-7">
            <label
              htmlFor="topic"
              className="mb-3 block text-sm font-medium text-zinc-300"
            >
              What would you like to write about?
            </label>

            <textarea
              id="topic"
              rows={4}
              placeholder="Enter a topic, idea, or thought..."
              className="w-full resize-none rounded-xl border border-white/10 bg-[#0B100E] p-4 text-sm text-white outline-none transition placeholder:text-zinc-600 focus:border-emerald-500/50"
            />
          </div>

          <div className="mt-5 flex items-center justify-between">
            <div>
              <label
                htmlFor="platform"
                className="mb-2 block text-xs font-medium text-zinc-400"
              >
                Platform
              </label>

              <select
                id="platform"
                className="rounded-lg border border-white/10 bg-[#0B100E] px-4 py-2.5 text-sm text-zinc-300 outline-none focus:border-emerald-500/50"
              >
                <option value="x">X (Twitter)</option>
                <option value="linkedin">LinkedIn</option>
                <option value="general">General</option>
              </select>
            </div>

            <button
              type="button"
              disabled
              className="mt-6 flex cursor-not-allowed items-center gap-2 rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-[#07130C] opacity-60"
            >
              <span>✧</span>
              Generate
            </button>
          </div>
        </section>

        {/* Getting Started */}
        <section className="mt-10 max-w-4xl">
          <h2 className="text-lg font-semibold">
            Getting started
          </h2>

          <p className="mt-1 text-sm text-zinc-500">
            Your personalized writing workspace is being prepared.
          </p>

          <div className="mt-5 grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <div className="text-xl text-emerald-400">✎</div>

              <h3 className="mt-4 text-sm font-medium">
                Build your writing dataset
              </h3>

              <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                Echo will learn from writing you choose to provide.
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <div className="text-xl text-emerald-400">✧</div>

              <h3 className="mt-4 text-sm font-medium">
                Personalize your AI
              </h3>

              <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                Your personal writing adapter will power personalized
                generation once training is available.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
