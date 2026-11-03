import { ArrowUpRight, Sparkles } from 'lucide-react';

function WelcomeSection() {
  return (
    <section className="relative shrink-0 px-12 pb-8 pt-12">

      <div className="flex items-center justify-between">
        <div>
          <div className="mb-5 flex items-center gap-2">
            <span className="h-px w-7 bg-[#78947d]" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#52745e]">
              Your personal writing space
            </span>
          </div>

          <h1 className="font-serif text-[46px] leading-[1.12] tracking-[-1.5px] text-[#203c30]">
            Words that sound
            <br />
            <span className="italic text-[#66836d]">like you.</span>
          </h1>

          <p className="mt-4 max-w-md text-[13px] leading-6 text-[#68766d]">
            Your thoughts, your voice, your style. A little help turning
            what you have in mind into something worth sharing.
          </p>
        </div>

        <div className="mb-2 hidden items-center gap-2 rounded-full border border-[#dce3d7] bg-[#f8f7f0]/75 px-4 py-2.5 sm:flex">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#7caa7c] opacity-40" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-[#638b68]" />
          </span>

          <span className="text-[11px] font-medium tracking-wide text-[#45664e]">
            Your studio is ready
          </span>
        </div>
      </div>

      <div className="mt-8 flex items-center gap-2 border-t border-[#6f896f]/20 pt-4">
        <Sparkles size={13} className="text-[#78947d]" />
        <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-[#758176]">
          Made for your voice
        </span>

        <ArrowUpRight size={13} className="ml-1 text-[#78947d]" />
      </div>

    </section>
  );
}

export default WelcomeSection;