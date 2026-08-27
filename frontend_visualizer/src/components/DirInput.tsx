"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { FolderOpen, ArrowRight } from "lucide-react";

export default function DirInput({
  initialDir,
  compact = false,
}: {
  initialDir: string;
  compact?: boolean;
}) {
  const router = useRouter();
  const [value, setValue] = useState(initialDir);
  const [pending, setPending] = useState(false);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setPending(true);
    const sp = new URLSearchParams();
    sp.set("dir", value.trim());
    router.push(`/?${sp.toString()}`);
    setTimeout(() => setPending(false), 400);
  }

  return (
    <form onSubmit={submit} className="w-full">
      <label className="block text-[11px] uppercase tracking-[0.14em] text-stone-500 font-medium mb-1.5">
        Session directory
      </label>
      <div
        className={`group flex items-center gap-2 rounded-xl border bg-white transition-all
          ${pending ? "border-amber-300 ring-2 ring-amber-100" : "border-stone-200 focus-within:border-stone-400 focus-within:ring-2 focus-within:ring-stone-100"}
          ${compact ? "px-3 py-1.5" : "px-3.5 py-2"}`}
      >
        <FolderOpen className="w-4 h-4 text-stone-400 shrink-0" />
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          className={`flex-1 bg-transparent outline-none font-mono text-stone-800 placeholder-stone-400 ${
            compact ? "text-[13px]" : "text-sm"
          }`}
          placeholder="~/Downloads/recorder_sessions/…"
        />
        <button
          type="submit"
          disabled={pending}
          className="inline-flex items-center gap-1 rounded-lg bg-stone-900 text-white px-3 py-1.5 text-xs font-medium hover:bg-stone-700 transition-colors disabled:opacity-60"
        >
          Load
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </form>
  );
}
