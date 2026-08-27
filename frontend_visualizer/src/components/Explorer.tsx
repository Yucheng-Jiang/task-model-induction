"use client";

import { useMemo, useState } from "react";
import { Sparkle, Layers3, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { threadColor } from "@/lib/colors";
import type { SessionData } from "@/lib/types";
import DirInput from "./DirInput";
import ExportHtmlButton from "./ExportHtmlButton";
import TraceColumn from "./TraceColumn";
import ObjectiveColumn from "./ObjectiveColumn";
import ProcedureColumn from "./ProcedureColumn";
import UnifiedModelColumn from "./UnifiedModelColumn";

type Selection =
  | { kind: "objective"; id: string }
  | { kind: "objModelNode"; id: string; refs: Set<string> }
  | { kind: "procNode"; id: string; refs: Set<string> }
  | null;

declare global {
  interface Window {
    __EXPLORER_EXPORT_STATE__?:
      | {
          activeThreadId?: string;
          filterTrace?: boolean;
          viewMode?: "pre-reconciliation" | "unified";
          selectedObjectiveId?: string | null;
          selectedObjModelNodeId?: string | null;
          selectedProcNodeId?: string | null;
        }
      | undefined;
  }
}

export default function Explorer({ data }: { data: SessionData }) {
  const exportedState =
    typeof window !== "undefined" ? window.__EXPLORER_EXPORT_STATE__ : undefined;
  const defaultThreadId =
    data.threads.reduce(
      (best, t) =>
        t.localObjectiveIds.length > (best?.localObjectiveIds.length ?? -1) ? t : best,
      data.threads[0],
    )?.id ?? data.threads[0]?.id ?? "C1";

  const [activeThreadId, setActiveThreadId] = useState<string>(
    exportedState?.activeThreadId ?? defaultThreadId,
  );
  const [selection, setSelection] = useState<Selection>(
    exportedState?.selectedObjectiveId
      ? { kind: "objective", id: exportedState.selectedObjectiveId }
      : null,
  );
  const [filterTrace, setFilterTrace] = useState(Boolean(exportedState?.filterTrace));
  const [viewMode, setViewMode] = useState<"pre-reconciliation" | "unified">(
    exportedState?.viewMode ?? "unified",
  );

  const activeThread = useMemo(
    () => data.threads.find((t) => t.id === activeThreadId) ?? data.threads[0],
    [data.threads, activeThreadId],
  );

  // Compute the "related set" of activity ids derived from current selection.
  const related = useMemo<Set<string>>(() => {
    if (!selection) return new Set();
    if (selection.kind === "objective") return new Set([selection.id]);
    return selection.refs;
  }, [selection]);

  const selectedActivityId = selection?.kind === "objective" ? selection.id : null;
  const selectedObjModelNodeId =
    selection?.kind === "objModelNode" ? selection.id : null;
  const selectedProcNodeId = selection?.kind === "procNode" ? selection.id : null;

  // Positions to highlight in density bars
  const highlightPositions = useMemo(() => {
    const out: number[] = [];
    for (const id of related) {
      const idx = data.activityIndex[id];
      if (idx !== undefined) out.push(idx);
    }
    return out;
  }, [related, data.activityIndex]);

  function selectActivity(id: string) {
    setSelection({ kind: "objective", id });
    const threadId = data.threadByObjective[id];
    if (threadId) setActiveThreadId(threadId);
  }

  function selectObjModelNode(id: string, refs: Set<string>) {
    setSelection((prev) =>
      prev?.kind === "objModelNode" && prev.id === id ? null : { kind: "objModelNode", id, refs },
    );
  }

  function selectProcNode(id: string, refs: Set<string>) {
    setSelection((prev) =>
      prev?.kind === "procNode" && prev.id === id ? null : { kind: "procNode", id, refs },
    );
  }

  function clearSelection() {
    setSelection(null);
  }

  const totalLocal = data.activities.length;


  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[var(--background)] text-stone-900">
      {/* Top header */}
      <header className="shrink-0 border-b border-stone-200 bg-white/85 backdrop-blur-md">
        <div className="flex items-center gap-4 px-6 py-3">
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-stone-900 to-stone-700 flex items-center justify-center shadow-sm">
              <Layers3 className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight">Task Trace Explorer</h1>
              <p className="text-xs leading-tight text-stone-500">
                {totalLocal} activities · {data.threads.length} task threads
              </p>
            </div>
          </div>
          <div className="flex-1 max-w-2xl">
            <DirInput initialDir={data.dir} compact />
          </div>
          <div className="flex items-center gap-1 shrink-0 rounded-lg border border-stone-200 bg-stone-50 p-0.5">
            <button
              onClick={() => setViewMode("pre-reconciliation")}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-all",
                viewMode === "pre-reconciliation"
                  ? "bg-white shadow-sm text-stone-900 border border-stone-200"
                  : "text-stone-500 hover:text-stone-700",
              )}
            >
              Pre-reconciliation
            </button>
            <button
              onClick={() => setViewMode("unified")}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-all",
                viewMode === "unified"
                  ? "bg-white shadow-sm text-stone-900 border border-stone-200"
                  : "text-stone-500 hover:text-stone-700",
              )}
            >
              Unified
            </button>
          </div>
          <div data-export-ignore>
            <ExportHtmlButton
              dir={data.dir}
              activeThreadId={activeThreadId}
            />
          </div>
          <div className="hidden 2xl:flex items-center text-xs text-stone-400 gap-2 shrink-0 font-mono">
            <Sparkle className="w-3 h-3" />
            click any item to link views
          </div>
        </div>

        {/* Thread tabs */}
        <div className="px-6 pb-3 flex items-center gap-2 overflow-x-auto">
          {data.threads.map((t) => {
            const tc = threadColor(t.id);
            const isActive = t.id === activeThreadId;
            return (
              <button
                key={t.id}
                onClick={() => {
                  setActiveThreadId(t.id);
                  setSelection(null);
                }}
                className={cn(
                  "group flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all whitespace-nowrap",
                  isActive
                    ? cn("border-stone-300 bg-white shadow-sm")
                    : "border-transparent hover:border-stone-200 hover:bg-white/60",
                )}
              >
                <span className={cn("w-2 h-2 rounded-full", tc.dot)} />
                <span
                  className={cn(
                    "font-mono text-xs font-medium",
                    isActive ? "text-stone-900" : "text-stone-500",
                  )}
                >
                  {t.id}
                </span>
                <span
                  className={cn(
                    "max-w-[260px] truncate text-[13px]",
                    isActive ? "text-stone-800" : "text-stone-500 group-hover:text-stone-700",
                  )}
                >
                  {t.label}
                </span>
                <span className="font-mono text-[11px] text-stone-400">
                  {t.localObjectiveIds.length}
                </span>
              </button>
            );
          })}
          <div className="ml-auto flex items-center gap-2 shrink-0">
            {selection && (
              <button
                onClick={clearSelection}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-900"
              >
                <X className="w-3 h-3" />
                clear selection
              </button>
            )}
            <label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-stone-500">
              <input
                type="checkbox"
                checked={filterTrace}
                onChange={(e) => setFilterTrace(e.target.checked)}
                className="accent-stone-900"
              />
              filter trace to selection
            </label>
          </div>
        </div>
      </header>

      {/* Main grid */}
      <div
        className={cn(
          "flex-1 min-h-0 grid grid-cols-1 divide-x divide-stone-200",
          viewMode === "pre-reconciliation"
            ? "lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)_minmax(0,1.05fr)]"
            : "lg:grid-cols-[minmax(320px,0.9fr)_minmax(540px,1.7fr)]",
        )}
      >
        <TraceColumn
          activities={data.activities}
          threadByObjective={data.threadByObjective}
          activeThreadId={activeThreadId}
          related={related}
          selectedActivityId={selectedActivityId}
          onSelectActivity={selectActivity}
          filterToRelated={filterTrace}
        />

        {viewMode === "pre-reconciliation" ? (
          <>
            {activeThread && (
              <ObjectiveColumn
                threadId={activeThread.id}
                threadLabel={activeThread.label}
                threadObjective={activeThread.task_thread_objective}
                root={activeThread.objectiveModel}
                totalActivities={totalLocal}
                activityIndex={data.activityIndex}
                related={related}
                highlightPositions={highlightPositions}
                selectedNodeId={selectedObjModelNodeId}
                selectedActivityId={selectedActivityId}
                onSelectNode={selectObjModelNode}
              />
            )}
            {activeThread && (
              <ProcedureColumn
                model={activeThread.procedureModel}
                threadId={activeThread.id}
                totalActivities={totalLocal}
                activityIndex={data.activityIndex}
                related={related}
                highlightPositions={highlightPositions}
                selectedNodeId={selectedProcNodeId}
                selectedActivityId={selectedActivityId}
                onSelectNode={selectProcNode}
              />
            )}
          </>
        ) : (
          activeThread && (
            <UnifiedModelColumn
              key={activeThread.id}
              model={activeThread.unifiedModel}
              threadId={activeThread.id}
              threadLabel={activeThread.label}
              threadObjective={activeThread.task_thread_objective}
              totalActivities={totalLocal}
              activityIndex={data.activityIndex}
              related={related}
              highlightPositions={highlightPositions}
              selectedNodeId={selectedObjModelNodeId}
              selectedActivityId={selectedActivityId}
              onSelectNode={selectObjModelNode}
            />
          )
        )}
      </div>
    </div>
  );
}
