"use client";

import { useMemo, useState } from "react";
import {
  Repeat,
  GitMerge,
  RotateCw,
  ChevronDown,
  Info,
  Layers3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { threadColor } from "@/lib/colors";
import { expandSegmentRefs } from "@/lib/segments";
import DensityBar from "./DensityBar";
import type { UnifiedTaskModel, UnifiedModelNode } from "@/lib/types";

type Props = {
  model: UnifiedTaskModel | null;
  threadId: string;
  threadLabel: string;
  threadObjective: string;
  totalActivities: number;
  activityIndex: Record<string, number>;
  related: Set<string>;
  highlightPositions: number[];
  selectedNodeId: string | null;
  selectedActivityId: string | null;
  onSelectNode: (id: string, refs: Set<string>) => void;
};

// Plain-language chips for the non-default operators only. SEQ (steps in
// order) is the default reading of an outline and doesn't need a label.
const LOOP_STYLE: Record<
  string,
  { cls: string; icon: React.ComponentType<{ className?: string }> }
> = {
  FOR:    { cls: "bg-amber-50 text-amber-700 border-amber-200/60",    icon: Repeat },
  WHILE:  { cls: "bg-rose-50 text-rose-700 border-rose-200/60",       icon: RotateCw },
  CHOICE: { cls: "bg-violet-50 text-violet-700 border-violet-200/60", icon: GitMerge },
  PARALLEL: { cls: "bg-teal-50 text-teal-700 border-teal-200/60", icon: Layers3 },
};

function operatorChipText(node: UnifiedModelNode): string | null {
  const proc = node.procedure;
  switch (proc.operator) {
    case "FOR": {
      const v = proc.bindings?.iteration_variable as string | undefined;
      return v ? `Repeat for each ${v.replaceAll("_", " ")}` : "Repeat for each item";
    }
    case "WHILE":
      return "Repeat until complete";
    case "CHOICE":
      return "Choose one path";
    case "PARALLEL":
      return "Can happen together";
    default:
      return null;
  }
}

function OperatorChip({ node }: { node: UnifiedModelNode }) {
  const text = operatorChipText(node);
  if (!text) return null;
  const s = LOOP_STYLE[node.procedure.operator];
  if (!s) return null;
  const Icon = s.icon;
  return (
    <span
      className={cn(
        "inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md border px-2 text-xs font-medium leading-none whitespace-nowrap",
        s.cls,
      )}
    >
      <Icon className="w-3 h-3" />
      {text}
    </span>
  );
}

// ── Flat-list types ──────────────────────────────────────────────────────────

type FlatRow = {
  node: UnifiedModelNode;
  depth: number;
  refs: Set<string>;       // aggregate refs for this node + all descendants
  positions: number[];     // positions in the global activity list
};

// Build annotated flat rows for all nodes (pre-computes refs + positions)
function buildAnnotations(
  node: UnifiedModelNode,
  activityIndex: Record<string, number>,
  depth: number,
  out: Map<string, FlatRow>,
): Set<string> {
  const childRefSets: Set<string>[] = [];
  for (const child of node.decomposition) {
    childRefSets.push(buildAnnotations(child, activityIndex, depth + 1, out));
  }
  const selfRefs = new Set(expandSegmentRefs(node.activity_refs));
  const allRefs = new Set<string>(selfRefs);
  for (const s of childRefSets) for (const id of s) allRefs.add(id);
  const positions: number[] = [];
  for (const id of allRefs) {
    const idx = activityIndex[id];
    if (idx !== undefined) positions.push(idx);
  }
  out.set(node.id, { node, depth, refs: allRefs, positions });
  return allRefs;
}

// Depth-first in-order traversal, only emitting nodes that are visible
function flattenInOrder(
  node: UnifiedModelNode,
  expanded: Set<string>,
  annotations: Map<string, FlatRow>,
  out: FlatRow[],
) {
  const ann = annotations.get(node.id);
  if (ann) out.push(ann);
  if (expanded.has(node.id)) {
    for (const child of node.decomposition) {
      flattenInOrder(child, expanded, annotations, out);
    }
  }
}

const INDENT = 20;
const ROW_PAD = 18;

// ── Inline detail panel ──────────────────────────────────────────────────────

function NodeDetail({
  overview = false,
  row,
  selectedNodeId,
  selectedActivityId,
  onSelectNode,
}: {
  overview?: boolean;
  row: FlatRow;
  selectedNodeId: string | null;
  selectedActivityId: string | null;
  onSelectNode: (id: string, refs: Set<string>) => void;
}) {
  const { node } = row;
  const proc = node.procedure;
  const bodySteps = proc.body ?? [];

  return (
    <div
      data-node-detail={node.id}
      className={cn(
        "border-t border-stone-100 bg-stone-50/45 pb-4 pl-[3.25rem] pr-4 pt-3",
        overview ? "grid gap-x-6 gap-y-4 md:grid-cols-2" : "space-y-4",
      )}
    >
      {node.summary && (
        <p
          className={cn(
            "text-sm text-stone-600 leading-relaxed",
            overview && "md:col-span-2",
          )}
        >
          {node.summary}
        </p>
      )}

      {proc.operator === "FOR" && proc.bindings && (
        <div className={cn("text-[13px] space-y-2", overview && "md:col-span-2")}>
          <div className="font-medium text-amber-700">
            Repeated once for each{" "}
            {((proc.bindings.iteration_variable as string | undefined) ?? "item").replaceAll("_", " ")}
            :
          </div>
          {Array.isArray(proc.bindings.collection) &&
            (proc.bindings.collection as unknown[]).length > 0 && (
              <div className="flex flex-wrap gap-1">
                {(proc.bindings.collection as unknown[]).map((item, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center rounded-md bg-amber-100/70 px-2 py-1 text-xs text-amber-800"
                  >
                    {String(item)}
                  </span>
                ))}
              </div>
            )}
        </div>
      )}

      {proc.operator === "WHILE" && proc.condition && (
        <p
          className={cn(
            "text-[13px] leading-relaxed text-rose-700",
            overview && "md:col-span-2",
          )}
        >
          <span className="font-medium">Repeats until: </span>
          <span className="text-rose-800/80">{proc.condition}</span>
        </p>
      )}

      {bodySteps.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-stone-500">
            How it was done
          </div>
          <div className="space-y-0.5">
            {bodySteps.map((step, i) => {
              const stepRefs = new Set(expandSegmentRefs(step.activity_refs));
              const stepId = `${node.id}::step::${i}`;
              const isStepSelected = selectedNodeId === stepId;
              const isStepPath =
                selectedActivityId !== null && stepRefs.has(selectedActivityId);
              const hasRefs = stepRefs.size > 0;
              return (
                <button
                  key={i}
                  type="button"
                  disabled={!hasRefs}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (hasRefs) onSelectNode(stepId, stepRefs);
                  }}
                  className={cn(
                    "w-full rounded-md px-2 py-1.5 text-left flex items-start gap-2.5 transition-colors",
                    isStepSelected
                      ? "bg-amber-100/70"
                      : isStepPath
                      ? "bg-purple-100/50"
                      : hasRefs
                      ? "hover:bg-stone-100 cursor-pointer"
                      : "cursor-default",
                  )}
                >
                  <span className="w-5 shrink-0 text-right text-xs leading-5 tabular-nums text-stone-500">
                    {i + 1}.
                  </span>
                  <span className="min-w-0 flex-1 text-sm leading-5 text-stone-700">
                    {step.name}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {node.observed_outcome?.description && (
        <div>
          <div className="mb-1.5 text-xs font-medium uppercase tracking-[0.12em] text-stone-500">
            Outcome
          </div>
          <p className="text-sm text-stone-600 leading-relaxed">
            {node.observed_outcome.description}
          </p>
        </div>
      )}

      {proc.evidence_summary && (
        <div className={cn(overview && "md:col-span-2")}>
          <div className="mb-1.5 text-xs font-medium uppercase tracking-[0.12em] text-stone-500">
            Evidence
          </div>
          <p className="text-sm leading-relaxed text-stone-500">
            {proc.evidence_summary}
          </p>
        </div>
      )}
    </div>
  );
}

// ── Column ───────────────────────────────────────────────────────────────────

export default function UnifiedModelColumn({
  model,
  threadId,
  threadLabel,
  threadObjective,
  totalActivities,
  activityIndex,
  related,
  highlightPositions,
  selectedNodeId,
  selectedActivityId,
  onSelectNode,
}: Props) {
  const root = model?.root ?? null;
  const taskColor = threadColor(threadId);

  // Pre-compute all annotations (refs + positions) once per model
  const annotations = useMemo(() => {
    const map = new Map<string, FlatRow>();
    if (root) buildAnnotations(root, activityIndex, 0, map);
    return map;
  }, [root, activityIndex]);

  // Visible steps always include their details. The only disclosure state is
  // whether a branch's child steps are visible. Explicit branch choices
  // override the temporary path reveal used when someone selects an activity
  // in the trace.
  const [expansionOverrides, setExpansionOverrides] = useState<Map<string, boolean>>(
    () => new Map(),
  );

  const autoExpanded = useMemo(() => {
    const next = new Set<string>();
    if (!selectedActivityId) return next;
    for (const [id, ann] of annotations) {
      if (ann.node.decomposition.length > 0 && ann.refs.has(selectedActivityId)) {
        next.add(id);
      }
    }
    return next;
  }, [selectedActivityId, annotations]);

  const effectiveExpanded = useMemo(() => {
    const next = new Set<string>();
    for (const [id, ann] of annotations) {
      if (ann.node.decomposition.length === 0) continue;
      const override = expansionOverrides.get(id);
      // The root starts with its outline visible, but folding that outline
      // never hides the root card or its details.
      const defaultExpanded = id === root?.id || autoExpanded.has(id);
      if (override ?? defaultExpanded) next.add(id);
    }
    return next;
  }, [root, annotations, expansionOverrides, autoExpanded]);

  // Flat visible rows
  const rows = useMemo(() => {
    const out: FlatRow[] = [];
    if (root) flattenInOrder(root, effectiveExpanded, annotations, out);
    return out;
  }, [root, effectiveExpanded, annotations]);

  function toggleBranch(id: string) {
    const nextValue = !effectiveExpanded.has(id);
    setExpansionOverrides((prev) => {
      const next = new Map(prev);
      next.set(id, nextValue);
      return next;
    });
  }

  // Selecting a model item replaces a selected trace activity. Preserve the
  // path that was revealed for that activity so the clicked item cannot vanish.
  function selectModelNode(id: string, refs: Set<string>) {
    if (selectedActivityId) {
      setExpansionOverrides((prev) => {
        const next = new Map(prev);
        for (const expandedId of effectiveExpanded) {
          next.set(expandedId, true);
        }
        return next;
      });
    }
    onSelectNode(id, refs);
  }

  return (
    <section className="flex flex-col h-full min-h-0">
      <header className="flex items-start gap-3 border-b border-stone-200 bg-white/70 px-6 py-4 backdrop-blur shrink-0">
        <div
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-stone-200",
            taskColor.soft,
          )}
        >
          <Layers3 className={cn("h-4 w-4", taskColor.text)} />
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold leading-5 tracking-tight text-stone-900">
            What was done
          </h2>
          <p className="truncate text-[13px] leading-5 text-stone-500">{threadLabel}</p>
        </div>
      </header>

      <div className="shrink-0 border-b border-stone-200 bg-stone-50/70 px-6 py-4">
        <p className="mb-1 text-xs font-medium uppercase tracking-[0.12em] text-stone-500">
          Overall goal
        </p>
        <p className="max-w-3xl text-[15px] font-medium leading-6 text-stone-800">
          {threadObjective}
        </p>
      </div>

      <div data-export-scroll className="flex-1 overflow-y-auto py-3">
        {!model && (
          <div className="px-4 py-12 text-center text-sm text-stone-400">
            No unified model available for this thread.
          </div>
        )}

        {rows.map((row) => {
          const { node, depth, refs, positions } = row;
          const hasChildren = node.decomposition.length > 0;
          const isExpanded = effectiveExpanded.has(node.id);
          const isSelected = selectedNodeId === node.id;
          const isPath = selectedActivityId !== null && refs.has(selectedActivityId);
          const relatedCount = [...refs].filter((id) => related.has(id)).length;
          const isRoot = depth === 0;
          const isSection = depth === 1;
          const operatorText = operatorChipText(node);
          const displayObjective = isRoot ? "Task outline" : node.objective;

          return (
            <div
              key={node.id}
              className={cn("relative mb-3 pr-5", isSection && "mt-4")}
              style={{ paddingLeft: ROW_PAD + depth * INDENT }}
            >
              {/* Hierarchy guides sit outside the card; the thicker color rail
                  belongs to the card and visually joins its title and detail. */}
              {Array.from({ length: depth }, (_, d) => (
                <span
                  key={d}
                  aria-hidden
                  className="absolute bottom-0 top-0 w-px bg-stone-200/80"
                  style={{ left: ROW_PAD + d * INDENT + 9 }}
                />
              ))}

              <article
                data-tree-node-id={node.id}
                data-tree-depth={depth}
                data-branch-expanded={hasChildren ? isExpanded : undefined}
                data-detail-open="true"
                aria-label={displayObjective}
                className={cn(
                  "relative max-w-3xl overflow-hidden rounded-xl border bg-white shadow-[0_1px_2px_rgba(28,25,23,0.06)] transition-all",
                  isSelected
                    ? "border-amber-300 ring-2 ring-amber-100"
                    : isPath
                    ? "border-purple-200 ring-1 ring-purple-100"
                    : "border-stone-200 hover:border-stone-300",
                )}
              >
                <span
                  aria-hidden
                  data-card-rail
                  className={cn("absolute inset-y-0 left-0 z-10 w-1", taskColor.dot)}
                />

                <div
                  className={cn(
                    "relative flex items-start gap-2 pb-2.5 pl-4 pr-4 pt-3",
                    isSelected
                      ? "bg-amber-50/70"
                      : isPath
                      ? "bg-purple-50/45"
                      : "bg-white",
                  )}
                >

                {/* Leading marks describe hierarchy; the labeled control below
                    is the one place that changes child-step visibility. */}
                {isRoot ? (
                  <span
                    aria-hidden
                    className={cn(
                      "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center",
                      taskColor.text,
                    )}
                  >
                    <Layers3 className="h-4 w-4" />
                  </span>
                ) : (
                  <span
                    aria-hidden
                    className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center"
                  >
                    <span
                      className={cn(
                        hasChildren
                          ? "h-2 w-2 rounded-sm border border-stone-300 bg-stone-50"
                          : "h-1.5 w-1.5 rounded-full bg-stone-300",
                      )}
                    />
                  </span>
                )}

                {isRoot ? (
                  <div className="min-w-0 max-w-3xl flex-1">
                    <div className="flex items-start gap-3">
                      <div
                        role="heading"
                        aria-level={3}
                        className="min-w-0 flex-1 text-base font-semibold leading-6 text-stone-900"
                      >
                        {displayObjective}
                      </div>
                      <span
                        className={cn(
                          "mt-0.5 inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md px-2 text-xs font-medium",
                          taskColor.soft,
                          taskColor.text,
                        )}
                      >
                        <Info className="h-3.5 w-3.5" />
                        Overview
                      </span>
                    </div>

                    <div className="mt-1.5 flex min-h-6 flex-wrap items-center gap-x-2.5 gap-y-1">
                      <span className="text-xs text-stone-500">
                        {refs.size} step{refs.size === 1 ? "" : "s"}
                      </span>
                      {hasChildren && (
                        <span className="text-xs text-stone-500">
                          Details stay visible; child steps use the control below
                        </span>
                      )}
                      {relatedCount > 0 && (
                        <span className="text-xs font-medium text-amber-700">
                          {relatedCount} match{relatedCount === 1 ? "" : "es"}
                        </span>
                      )}
                    </div>

                    {positions.length > 0 && (
                      <div className="mt-2">
                        <DensityBar
                          total={totalActivities}
                          positions={positions}
                          highlightPositions={highlightPositions}
                          colorClass={taskColor.bar}
                          highlightClass="bg-amber-500"
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <button
                    type="button"
                    data-node-select
                    aria-label={`Highlight related activity for ${displayObjective}`}
                    aria-pressed={isSelected}
                    onClick={() => selectModelNode(node.id, refs)}
                    className="min-w-0 max-w-3xl flex-1 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-offset-2"
                  >
                    <div className="flex items-start">
                      <span
                        role="heading"
                        aria-level={Math.min(depth + 3, 6)}
                        className={cn(
                          "min-w-0 flex-1",
                          isSection
                            ? "text-[15px] font-semibold leading-6 text-stone-900"
                            : "text-sm font-medium leading-5 text-stone-700",
                        )}
                      >
                        {displayObjective}
                      </span>
                    </div>

                    {(operatorText || isSection || relatedCount > 0) && (
                      <div className="mt-1.5 flex min-h-6 flex-wrap items-center gap-x-2.5 gap-y-1">
                        {operatorText && <OperatorChip node={node} />}
                        {isSection && (
                          <span className="text-xs text-stone-500">
                            {refs.size} step{refs.size === 1 ? "" : "s"}
                          </span>
                        )}
                        {relatedCount > 0 && (
                          <span className="text-xs font-medium text-amber-700">
                            {relatedCount} match{relatedCount === 1 ? "" : "es"}
                          </span>
                        )}
                      </div>
                    )}

                    {positions.length > 0 && (
                      <div className="mt-2">
                        <DensityBar
                          total={totalActivities}
                          positions={positions}
                          highlightPositions={highlightPositions}
                          colorClass={taskColor.bar}
                          highlightClass="bg-amber-500"
                        />
                      </div>
                    )}
                  </button>
                )}
                </div>

                <NodeDetail
                  overview={isRoot}
                  row={row}
                  selectedNodeId={selectedNodeId}
                  selectedActivityId={selectedActivityId}
                  onSelectNode={selectModelNode}
                />

                {hasChildren && (
                  <button
                    type="button"
                    data-tree-toggle
                    aria-label={`${isExpanded ? "Hide" : "Show"} ${node.decomposition.length} child ${node.decomposition.length === 1 ? "step" : "steps"} for ${displayObjective}; this step's details stay visible`}
                    aria-expanded={isExpanded}
                    onClick={() => toggleBranch(node.id)}
                    className={cn(
                      "group relative flex w-full items-center justify-between border-t border-stone-100 px-4 py-2.5 pl-[3.25rem] text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-purple-400",
                      isExpanded
                        ? "bg-stone-50/70 hover:bg-stone-100"
                        : "bg-white hover:bg-stone-50",
                    )}
                  >
                    <span className="flex min-w-0 items-center gap-2.5">
                      <span className="text-sm font-medium text-stone-700">
                        {isExpanded ? "Hide" : "Show"} {node.decomposition.length} child{" "}
                        {node.decomposition.length === 1 ? "step" : "steps"}
                      </span>
                      <span className="hidden text-xs text-stone-400 sm:inline">
                        {isExpanded ? "currently visible" : "details stay visible"}
                      </span>
                    </span>
                    <ChevronDown
                      className={cn(
                        "h-4 w-4 shrink-0 text-stone-500 transition-transform",
                        isExpanded && "rotate-180",
                      )}
                      aria-hidden
                    />
                  </button>
                )}
              </article>
            </div>
          );
        })}
      </div>
    </section>
  );
}
