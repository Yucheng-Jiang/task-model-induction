"use client";

import { useMemo, useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Target } from "lucide-react";
import { cn } from "@/lib/utils";
import { expandSegmentRefs } from "@/lib/segments";
import DensityBar from "./DensityBar";
import type { ObjectiveModelNode } from "@/lib/types";

type Props = {
  threadId: string;
  threadLabel: string;
  threadObjective: string;
  root: ObjectiveModelNode | null;
  totalActivities: number;
  activityIndex: Record<string, number>;
  related: Set<string>;
  highlightPositions: number[];
  selectedNodeId: string | null;
  selectedActivityId: string | null;
  onSelectNode: (id: string, refs: Set<string>) => void;
};

type Annotated = {
  node: ObjectiveModelNode;
  refs: Set<string>;
  positions: number[];
  depth: number;
  pathIds: string[];
};

function annotate(
  node: ObjectiveModelNode | null,
  objIndex: Record<string, number>,
  depth: number,
  pathIds: string[],
  out: Map<string, Annotated>,
): Set<string> {
  if (!node) return new Set();
  const selfRefs = new Set(expandSegmentRefs(node.subgoal_segments));
  const nextPath = [...pathIds, node.id];
  const childRefSets = (node.decomposition ?? []).map((child) =>
    annotate(child, objIndex, depth + 1, nextPath, out),
  );
  const allRefs = new Set<string>(selfRefs);
  for (const s of childRefSets) for (const id of s) allRefs.add(id);
  const positions: number[] = [];
  for (const id of allRefs) {
    const idx = objIndex[id];
    if (idx !== undefined) positions.push(idx);
  }
  out.set(node.id, { node, refs: allRefs, positions, depth, pathIds: nextPath });
  return allRefs;
}

function flattenInOrder(
  node: ObjectiveModelNode,
  annotationById: Map<string, Annotated>,
  expanded: Set<string>,
  out: Annotated[],
) {
  const ann = annotationById.get(node.id);
  if (ann) out.push(ann);
  if (expanded.has(node.id)) {
    for (const c of node.decomposition ?? []) {
      flattenInOrder(c, annotationById, expanded, out);
    }
  }
}

const INDENT = 18;
const ROW_PAD = 14;

export default function ObjectiveColumn({
  threadLabel,
  threadObjective,
  root,
  totalActivities,
  activityIndex,
  related,
  highlightPositions,
  selectedNodeId,
  selectedActivityId,
  onSelectNode,
}: Props) {
  const annotationById = useMemo(() => {
    const map = new Map<string, Annotated>();
    annotate(root, activityIndex, 0, [], map);
    return map;
  }, [root, activityIndex]);

  const initialExpanded = useMemo(() => {
    const set = new Set<string>();
    if (root) set.add(root.id);
    if (root) for (const child of root.decomposition ?? []) set.add(child.id);
    return set;
  }, [root]);

  const [expanded, setExpanded] = useState<Set<string>>(initialExpanded);

  useEffect(() => {
    setExpanded(initialExpanded);
  }, [initialExpanded]);

  // Auto-expand ancestors of any node that contains the selected objective.
  useEffect(() => {
    if (!selectedActivityId) return;
    const toExpand = new Set(expanded);
    for (const ann of annotationById.values()) {
      if (ann.refs.has(selectedActivityId)) {
        for (const id of ann.pathIds) toExpand.add(id);
      }
    }
    setExpanded(toExpand);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedActivityId]);

  const rows = useMemo(() => {
    const out: Annotated[] = [];
    if (root) flattenInOrder(root, annotationById, expanded, out);
    return out;
  }, [root, annotationById, expanded]);

  function toggle(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <section className="flex flex-col h-full min-h-0">
      <header className="flex items-start gap-2.5 px-5 py-3 border-b border-stone-200 bg-white/70 backdrop-blur shrink-0">
        <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center shrink-0">
          <Target className="w-3.5 h-3.5 text-indigo-700" />
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="text-[13px] font-semibold tracking-tight text-stone-900">
            Goals
          </h2>
          <p className="text-[11px] text-stone-500 leading-snug truncate">{threadLabel}</p>
        </div>
      </header>

      <div className="px-5 py-3 border-b border-stone-200 bg-stone-50/60 shrink-0">
        <p className="text-[12px] text-stone-700 leading-relaxed max-w-prose">{threadObjective}</p>
      </div>

      <div data-export-scroll className="flex-1 overflow-y-auto py-2">
        {!root && (
          <div className="text-[12px] text-stone-400 px-4 py-12 text-center">
            No objective model available for this thread.
          </div>
        )}
        {rows.map((row) => {
          const hasChildren = (row.node.decomposition ?? []).length > 0;
          const isExpanded = expanded.has(row.node.id);
          const { refs, positions, depth } = row;
          const relatedCount = [...refs].filter((id) => related.has(id)).length;
          const isSelected = selectedNodeId === row.node.id;
          const isPath =
            selectedActivityId !== null && refs.has(selectedActivityId);
          const isRoot = depth === 0;
          const isSection = depth === 1;

          return (
            <div key={row.node.id} className={cn(isSection && "mt-1.5")}>
              <button
                type="button"
                onClick={() => onSelectNode(row.node.id, refs)}
                className={cn(
                  "group relative w-full text-left flex items-start gap-1.5 pr-4 transition-colors",
                  isRoot || isSection ? "py-2" : "py-[5px]",
                  isSelected
                    ? "bg-amber-50/80"
                    : isPath
                    ? "bg-indigo-50/50"
                    : "hover:bg-stone-50",
                )}
                style={{ paddingLeft: ROW_PAD + depth * INDENT }}
              >
                {/* indent guides */}
                {Array.from({ length: depth }, (_, d) => (
                  <span
                    key={d}
                    aria-hidden
                    className="absolute top-0 bottom-0 w-px bg-stone-200/80"
                    style={{ left: ROW_PAD + d * INDENT + 9 }}
                  />
                ))}

                {/* expand/collapse chevron */}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (hasChildren) toggle(row.node.id);
                  }}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && hasChildren) {
                      e.stopPropagation();
                      toggle(row.node.id);
                    }
                  }}
                  className={cn(
                    "mt-px w-5 h-5 rounded flex items-center justify-center shrink-0",
                    hasChildren
                      ? "text-stone-400 hover:bg-stone-200/60 hover:text-stone-600"
                      : "text-stone-300",
                  )}
                >
                  {hasChildren ? (
                    isExpanded ? (
                      <ChevronDown className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5" />
                    )
                  ) : (
                    <span className="w-[5px] h-[5px] rounded-full bg-stone-300" />
                  )}
                </span>

                <div className="flex-1 min-w-0 max-w-prose">
                  <div className="flex items-start gap-2">
                    <p
                      className={cn(
                        "flex-1 min-w-0 leading-normal",
                        isRoot
                          ? "text-[13px] font-semibold text-stone-900"
                          : isSection
                          ? "text-[13px] font-medium text-stone-900"
                          : "text-[12.5px] text-stone-600",
                        !isSelected && "line-clamp-2",
                      )}
                    >
                      {row.node.objective}
                    </p>
                    {row.node.summary && (
                      <span
                        className={cn(
                          "shrink-0 inline-flex items-center gap-0.5 text-[10px] mt-1 transition-opacity",
                          isSelected
                            ? "text-amber-700"
                            : "text-stone-400 opacity-0 group-hover:opacity-100",
                        )}
                      >
                        details
                        {isSelected ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRight className="w-3 h-3" />
                        )}
                      </span>
                    )}
                  </div>

                  {(isRoot || isSection || relatedCount > 0) && (
                    <div className="flex items-center gap-2 mt-0.5">
                      {(isRoot || isSection) && (
                        <span className="text-[10.5px] text-stone-400">
                          {refs.size} step{refs.size === 1 ? "" : "s"}
                        </span>
                      )}
                      {relatedCount > 0 && (
                        <span className="text-[10.5px] text-amber-700">
                          {relatedCount} match{relatedCount === 1 ? "" : "es"}
                        </span>
                      )}
                    </div>
                  )}

                  {positions.length > 0 && (isRoot || isSelected) && (
                    <div className="mt-1.5">
                      <DensityBar
                        total={totalActivities}
                        positions={positions}
                        highlightPositions={highlightPositions}
                        colorClass="bg-indigo-300"
                        highlightClass="bg-amber-500"
                      />
                    </div>
                  )}
                </div>
              </button>

              {isSelected && row.node.summary && (
                <div
                  className="mr-4 mb-2 mt-0.5 max-w-prose rounded-lg border border-stone-200 bg-stone-50/70 px-3.5 py-3"
                  style={{ marginLeft: ROW_PAD + depth * INDENT + 26 }}
                >
                  <p className="text-[12px] text-stone-600 leading-relaxed">
                    {row.node.summary}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
