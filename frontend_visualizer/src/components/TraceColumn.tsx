"use client";

import { useEffect, useRef } from "react";
import { ListOrdered } from "lucide-react";
import { cn } from "@/lib/utils";
import { threadColor } from "@/lib/colors";
import type { Activity } from "@/lib/types";

type Props = {
  activities: Activity[];
  threadByObjective: Record<string, string>;
  activeThreadId: string | null;
  related: Set<string>;
  selectedActivityId: string | null;
  onSelectActivity: (id: string) => void;
  filterToRelated: boolean;
};

export default function TraceColumn({
  activities,
  threadByObjective,
  activeThreadId,
  related,
  selectedActivityId,
  onSelectActivity,
  filterToRelated,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  useEffect(() => {
    if (!selectedActivityId) return;
    const el = itemRefs.current.get(selectedActivityId);
    if (el && containerRef.current) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedActivityId]);

  // When selection lives in a model node, scroll to first related activity.
  useEffect(() => {
    if (selectedActivityId) return;
    if (related.size === 0) return;
    const firstId = activities.find((a) => related.has(a.activity_id))?.activity_id;
    if (!firstId) return;
    const el = itemRefs.current.get(firstId);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [related, selectedActivityId, activities]);

  return (
    <section className="flex flex-col h-full min-h-0">
      <header className="flex items-center justify-between border-b border-stone-200 bg-white/70 px-5 py-4 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-amber-100 bg-amber-50">
            <ListOrdered className="h-4 w-4 text-amber-700" />
          </div>
          <div>
            <h2 className="text-base font-semibold leading-5 tracking-tight text-stone-900">
              Activity trace
            </h2>
            <p className="text-[13px] leading-5 text-stone-500">
              {activities.length} steps in chronological order
            </p>
          </div>
        </div>
      </header>

      <div ref={containerRef} data-export-scroll className="flex-1 overflow-y-auto space-y-2 px-3 py-3">
        {activities.map((a, idx) => {
          const threadId = threadByObjective[a.activity_id];
          const tc = threadColor(threadId);
          const isRelated = related.has(a.activity_id);
          const isSelected = selectedActivityId === a.activity_id;
          const isInactiveThread = activeThreadId !== null && threadId !== activeThreadId;
          const dim =
            (filterToRelated && related.size > 0 && !isRelated) ||
            (!filterToRelated && related.size > 0 && !isRelated && !isInactiveThread);

          if (filterToRelated && related.size > 0 && !isRelated) return null;

          return (
            <button
              key={a.activity_id}
              ref={(el) => {
                if (el) itemRefs.current.set(a.activity_id, el);
                else itemRefs.current.delete(a.activity_id);
              }}
              onClick={() => onSelectActivity(a.activity_id)}
              className={cn(
                "group w-full text-left rounded-lg border transition-all",
                "px-3.5 py-3",
                isSelected
                  ? "border-amber-300 bg-amber-50/70 shadow-sm"
                  : isRelated
                  ? cn("border-stone-300 bg-white shadow-sm", tc.ring, "ring-1")
                  : "border-stone-200 bg-white hover:border-stone-300 hover:bg-stone-50",
                dim && !isSelected && !isRelated && "opacity-50",
                isInactiveThread && !isRelated && !isSelected && "opacity-60",
              )}
            >
              <div className="flex items-start gap-3">
                <div className="flex shrink-0 flex-col items-center pt-1">
                  <span className="font-mono text-xs leading-none text-stone-500">
                    {String(idx + 1).padStart(3, "0")}
                  </span>
                  <span
                    className={cn(
                      "mt-1.5 w-2 h-2 rounded-full",
                      threadId ? tc.dot : "bg-stone-300",
                    )}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="mb-1.5 flex items-center gap-1.5">
                    {threadId && (
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5 font-mono text-xs font-medium",
                          tc.soft,
                          tc.text,
                        )}
                      >
                        {threadId}
                      </span>
                    )}
                    <span className="truncate font-mono text-xs text-stone-500">
                      {a.activity_id}
                    </span>
                  </div>
                  <p className="text-sm leading-5 text-stone-800">
                    {a.objective}
                  </p>
                  {isSelected && (
                    <p className="mt-2 text-[13px] leading-relaxed text-stone-500">
                      {a.additional_context}
                    </p>
                  )}
                  <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-stone-500">
                    <span>actions {a.start_action_idx}–{a.end_action_idx}</span>
                    <span aria-hidden>·</span>
                    <span>{a.semantic_action_count} semantic action{a.semantic_action_count === 1 ? "" : "s"}</span>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
        {filterToRelated && related.size === 0 && (
          <div className="px-4 py-12 text-center text-sm text-stone-400">
            Nothing selected yet. Click a model node or trace step.
          </div>
        )}
      </div>
    </section>
  );
}
