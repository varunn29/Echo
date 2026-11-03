import {
  AudioLines,
  House,
  Database,
  WandSparkles,
  Settings,
  ShieldCheck,
  ChevronRight,
} from 'lucide-react';

function Sidebar() {
  return (
    <aside className="relative flex h-full w-[224px] shrink-0 flex-col overflow-hidden bg-[#173d30] px-4 py-6 text-[#f4f2e9]">

      {/* Subtle background detail */}
      <div className="pointer-events-none absolute -right-24 top-36 h-64 w-64 rounded-full bg-[#315b46]/30 blur-3xl" />

      {/* Brand */}
      <div className="relative flex items-center gap-3 px-2">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#e9eddf]/10 text-[#dce8d8]">
          <AudioLines size={23} strokeWidth={1.8} />
        </div>

        <div>
          <h1 className="text-[23px] font-semibold tracking-tight">
            Echo<span className="text-[#a9c5a9]">.</span>
          </h1>
          <p className="mt-[-2px] text-[9px] font-medium uppercase tracking-[0.2em] text-[#a9bdb0]">
            Writing studio
          </p>
        </div>
      </div>

      {/* Navigation */}
      <div className="relative mt-14">
        <p className="mb-4 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#91aa9a]">
          Workspace
        </p>

        <nav className="space-y-1">
          <button
            className="flex w-full items-center gap-3 rounded-xl bg-[#e9eddf]/10 px-3 py-3 text-left text-sm font-medium text-[#f4f2e9] ring-1 ring-inset ring-[#e9eddf]/10"
          >
            <House size={18} strokeWidth={1.8} />
            <span>Dashboard</span>
            <ChevronRight size={15} className="ml-auto text-[#a9c5a9]" />
          </button>

          <button
            className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-[#b7c8bd] transition-colors hover:bg-[#e9eddf]/[0.07] hover:text-white"
          >
            <Database size={18} strokeWidth={1.8} />
            <span>My Writing Data</span>
          </button>

          <button
            className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-[#b7c8bd] transition-colors hover:bg-[#e9eddf]/[0.07] hover:text-white"
          >
            <WandSparkles size={18} strokeWidth={1.8} />
            <span>My Writing Style</span>
          </button>
        </nav>
      </div>

      {/* Settings */}
      <div className="relative mt-10">
        <p className="mb-4 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#91aa9a]">
          Preferences
        </p>

        <button
          className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-[#b7c8bd] transition-colors hover:bg-[#e9eddf]/[0.07] hover:text-white"
        >
          <Settings size={18} strokeWidth={1.8} />
          <span>Settings</span>
        </button>
      </div>

      <div className="flex-1" />

      {/* Local-first information */}
      <div className="relative rounded-2xl border border-[#e9eddf]/10 bg-[#e9eddf]/[0.06] p-4">
        <div className="flex items-center gap-2">
          <ShieldCheck size={17} className="text-[#b8d2a9]" />
          <span className="text-xs font-semibold text-[#e5ecdf]">
            Local first
          </span>
        </div>

        <p className="mt-2 text-[11px] leading-[1.7] text-[#a9bdb0]">
          Your writing and personal data stay on your device.
        </p>

        <div className="mt-3 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-[#a8ce91]" />
          <span className="text-[10px] font-medium text-[#b8d2a9]">
            Private by design
          </span>
        </div>
      </div>

      {/* Footer */}
      <div className="relative mt-5 border-t border-[#e9eddf]/10 pt-4">
        <p className="px-2 text-[10px] text-[#91aa9a]">
          ECHO · PERSONAL WRITING ASSISTANT
        </p>
      </div>

    </aside>
  );
}

export default Sidebar;