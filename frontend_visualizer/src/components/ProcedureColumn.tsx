"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Repeat,
  GitMerge,
  RotateCw,
  ChevronDown,
  ChevronRight,
  Workflow,
  Circle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { expandSegmentRefs } from "@/lib/segments";
import DensityBar from "./DensityBar";
import type {
  ProcedureModel,
  ProcedureNode,
  ProcedureStep,
  ProcedureControlStep,
  ProcedureActivityLeaf,
  ProcedureAbstractStep,
} from "@/lib/types";
import { procedureBodySteps } from "@/lib/types";

type Props = {
  model: ProcedureModel | null;
  threadId: string;
  totalActivities: number;
  activityIndex: Record<string, number>;
  related: Set<string>;
  highlightPositions: number[];
  selectedNodeId: string | null;
  selectedActivityId: string | null;
  onSelectNode: (id: string, refs: Set<string>) => void;
};

// Plain-language chips for the non-default operators only. SEQ (steps in
// order) is the default reading of a step list and doesn't need a label.
const OPERATOR_STYLE: Record<
  string,
  { label: string; cls: string; icon: React.ComponentType<{ className?: string }> }
> = {
  FOR:    { label: "repeats per item",     cls: "bg-amber-50 text-amber-700 border-amber-100",   icon: Repeat },
  WHILE:  { label: "repeats until done",   cls: "bg-rose-50 text-rose-700 border-rose-100",      icon: RotateCw },
  CHOICE: { label: "one of several paths", cls: "bg-violet-50 text-violet-700 border-violet-100", icon: GitMerge },
};

function OperatorBadge({ operator }: { operator: string }) {
  const s = OPERATOR_STYLE[operator];
  if (!s) return null;
  const Icon = s.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium shrink-0",
        s.cls,
      )}
    >
      <Icon className="w-3 h-3" />
      {s.label}
    </span>
  );
}

// ── FOR / WHILE context headers ─────────────────────────────────────────────

function ForHeader({ bindings }: { bindings: Record<string, unknown> | null | undefined }) {
  if (!bindings) return null;
  const iterVar = bindings.iteration_variable as string | undefined;
  const collection = bindings.collection;
  const items: string[] = Array.isArray(collection)
    ? collection.map(String)
    : typeof collection === "string"
    ? [collection]
    : [];
  return (
    <div className="rounded bg-amber-50 border border-amber-100 px-2.5 py-2 text-[11px] space-y-1.5">
      <div className="font-mono text-amber-700 font-medium">
        for each <span className="italic">{iterVar ?? "item"}</span>:
      </div>
      {items.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {items.map((item, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-mono text-[10px]"
            >
              <span className="text-amber-500">[{i + 1}]</span>
              {item}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function WhileHeader({ condition }: { condition: string | null | undefined }) {
  if (!condition) return null;
  return (
    <div className="rounded bg-rose-50 border border-rose-100 px-2.5 py-1.5 text-[11px]">
      <span className="font-mono text-rose-600 font-medium">until: </span>
      <span className="text-rose-800 italic">{condition}</span>
    </div>
  );
}

function BodyLabel({ operator, bindings }: { operator: string; bindings?: Record<string, unknown> | null }) {
  if (operator === "FOR") {
    const iterVar = (bindings?.iteration_variable as string | undefined) ?? "item";
    return (
      <div className="text-[10px] uppercase tracking-[0.14em] text-amber-600 font-medium">
        Body — runs once per {iterVar}
      </div>
    );
  }
  if (operator === "WHILE") {
    return (
      <div className="text-[10px] uppercase tracking-[0.14em] text-rose-600 font-medium">
        Loop body — repeats until condition
      </div>
    );
  }
  return null;
}

// ── Activity leaf row ────────────────────────────────────────────────────────

function ActivityLeafRow({ step }: { step: ProcedureActivityLeaf }) {
  return (
    <div className="flex items-start gap-2 py-1 px-2">
      <Circle className="mt-1 w-2 h-2 text-stone-300 shrink-0" />
      <div className="flex-1 min-w-0 text-[12px] leading-snug line-clamp-2">
        {step.name ? (
          <span className="text-stone-700">{step.name}</span>
        ) : (
          <span className="text-stone-500">{step.description}</span>
        )}
      </div>
    </div>
  );
}

// ── Shared tree context (passed down to avoid prop drilling) ─────────────────

type TreeCtx = {
  byId: Map<string, ProcedureNode>;
  refsByNode: Map<string, Set<string>>;
  positionsByNode: Map<string, number[]>;
  totalActivities: number;
  activityIndex: Record<string, number>;
  related: Set<string>;
  highlightPositions: number[];
  selectedNodeId: string | null;
  selectedActivityId: string | null;
  expanded: Set<string>;
  onToggle: (id: string) => void;
  onSelectNode: (id: string, refs: Set<string>) => void;
  cardRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
};

// ── Abstract named step row (FOR/WHILE body template step with activity_refs) ─

function AbstractStepRow({
  step,
  ctx,
}: {
  step: ProcedureAbstractStep;
  ctx: TreeCtx;
}) {
  const refs = useMemo(
    () => new Set(expandSegmentRefs(step.activity_refs)),
    [step.activity_refs],
  );
  const syntheticId = `__abs__${step.name}`;
  const isSelected = ctx.selectedNodeId === syntheticId;
  const isPath = ctx.selectedActivityId !== null && refs.has(ctx.selectedActivityId);

  return (
    <button
      type="button"
      onClick={() => ctx.onSelectNode(syntheticId, refs)}
      className={cn(
        "w-full text-left px-2 py-1.5 rounded border transition-all",
        isSelected
          ? "border-amber-200 bg-amber-50/70"
          : isPath
          ? "border-teal-100 bg-teal-50/30"
          : "border-transparent hover:border-stone-100 hover:bg-stone-50/60",
      )}
    >
      <div className="flex items-start gap-1.5">
        <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-stone-300 shrink-0" />
        <div className="flex-1 min-w-0">
          <span className="text-[12px] font-medium text-stone-800 leading-snug">
            {step.name}
          </span>
          {step.description && (
            <span className="text-[11px] text-stone-500 leading-snug">
              {" — "}
              {step.description}
            </span>
          )}
          {isSelected && step.activity_refs.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1">
              {step.activity_refs.map((ref, i) => (
                <span
                  key={i}
                  className="inline-flex items-center px-1.5 py-0.5 rounded border font-mono text-[10px] bg-amber-100 border-amber-200 text-amber-800"
                >
                  {ref.replace(/activity_(\d+)/g, "act_$1")}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

// ── Routes a step to the right renderer ─────────────────────────────────────

function StepItem({ step, ctx, depth }: { step: ProcedureStep; ctx: TreeCtx; depth: number }) {
  if ("activity_id" in step) {
    return <ActivityLeafRow step={step} />;
  }
  if (!("operator" in step) && "activity_refs" in step) {
    return <AbstractStepRow step={step as ProcedureAbstractStep} ctx={ctx} />;
  }
  const ctrl = step as ProcedureControlStep;
  if (ctrl.procedure_node_id) {
    const node = ctx.byId.get(ctrl.procedure_node_id);
    if (node) {
      return <ProcedureNodeCard node={node} ctx={ctx} depth={depth} />;
    }
  }
  return <InlineStepBlock step={ctrl} ctx={ctx} depth={depth} />;
}

// ── Inline control construct (no entry in procedure_nodes) ───────────────────

function InlineStepBlock({
  step,
  ctx,
  depth,
}: {
  step: ProcedureControlStep;
  ctx: TreeCtx;
  depth: number;
}) {
  const [open, setOpen] = useState(depth < 2);
  const childSteps: ProcedureStep[] =
    procedureBodySteps(step.body) .length > 0
      ? procedureBodySteps(step.body)
      : step.steps ?? [];

  return (
    <div className="mt-1 rounded border border-stone-100 bg-stone-50/40">
      <button
        type="button"
        className="w-full text-left px-2 py-1.5 flex items-center gap-2"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="text-stone-400 shrink-0">
          {open ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )}
        </span>
        <OperatorBadge operator={step.operator} />
        {step.name ? (
          <span className="text-[12px] font-medium text-stone-800 truncate">
            {step.name}
          </span>
        ) : (
          !OPERATOR_STYLE[step.operator] && (
            <span className="text-[12px] text-stone-500">Steps</span>
          )
        )}
        {step.condition && (
          <span className="text-[11px] text-stone-400 italic truncate">
            until: {step.condition}
          </span>
        )}
      </button>

      {open && (
        <div className="pl-4 pr-2 pb-1.5 border-l-2 border-stone-100 ml-3 space-y-1.5">
          {step.description && (
            <p className="text-[11px] text-stone-500 leading-snug py-1">{step.description}</p>
          )}
          {step.operator === "FOR" && <ForHeader bindings={step.bindings} />}
          {step.operator === "WHILE" && <WhileHeader condition={step.condition} />}
          {childSteps.length > 0 && (
            <div className="space-y-0.5">
              <BodyLabel operator={step.operator} bindings={step.bindings} />
              {childSteps.map((s, i) => (
                <StepItem key={i} step={s} ctx={ctx} depth={depth + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Named procedure node card (entries in procedure_nodes) ───────────────────

function ProcedureNodeCard({
  node,
  ctx,
  depth,
}: {
  node: ProcedureNode;
  ctx: TreeCtx;
  depth: number;
}) {
  const {
    refsByNode,
    positionsByNode,
    totalActivities,
    related,
    highlightPositions,
    selectedNodeId,
    selectedActivityId,
    expanded,
    onToggle,
    onSelectNode,
    cardRefs,
    byId,
  } = ctx;

  const refs = refsByNode.get(node.id) ?? new Set<string>();
  const positions = positionsByNode.get(node.id) ?? [];
  const relatedCount = [...refs].filter((id) => related.has(id)).length;
  const isSelected = selectedNodeId === node.id;
  const isPath = selectedActivityId !== null && refs.has(selectedActivityId);
  const isRoot = depth === 0;
  const isExpanded = expanded.has(node.id);
  const steps: ProcedureStep[] = procedureBodySteps(node.body);

  return (
    <div
      id={`proc-${node.id}`}
      ref={(el) => {
        if (el) cardRefs.current.set(node.id, el);
        else cardRefs.current.delete(node.id);
      }}
      className={cn(
        "rounded-lg border transition-all bg-white",
        depth > 0 && "mt-1.5",
        isSelected
          ? "border-amber-300 bg-amber-50/60 shadow-sm"
          : isPath
          ? "border-teal-200 bg-teal-50/30"
          : "border-stone-200 hover:border-stone-300",
      )}
    >
      <button
        type="button"
        onClick={() => onSelectNode(node.id, refs)}
        className="w-full text-left px-3 py-2.5"
      >
        <div className="flex items-start gap-2">
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              onToggle(node.id);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.stopPropagation();
                onToggle(node.id);
              }
            }}
            className="mt-0.5 w-5 h-5 rounded text-stone-500 hover:bg-stone-100 flex items-center justify-center shrink-0"
          >
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </span>
          <div className="flex-1 min-w-0">
            <p
              className={cn(
                "leading-snug",
                isRoot
                  ? "text-[13px] font-semibold text-stone-900"
                  : "text-[12.5px] font-medium text-stone-800",
                !isSelected && "line-clamp-2",
              )}
            >
              {node.name}
            </p>
            {(OPERATOR_STYLE[node.operator] || isRoot || relatedCount > 0) && (
              <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                <OperatorBadge operator={node.operator} />
                {isRoot && (
                  <span className="text-[10px] text-stone-400">
                    {refs.size} episode{refs.size === 1 ? "" : "s"}
                  </span>
                )}
                {relatedCount > 0 && (
                  <span className="text-[10px] text-amber-700">
                    {relatedCount} match{relatedCount === 1 ? "" : "es"}
                  </span>
                )}
              </div>
            )}
            {isSelected && node.description && (
              <p className="text-[12px] text-stone-600 leading-relaxed mt-1.5">
                {node.description}
              </p>
            )}
            {positions.length > 0 && (isRoot || isSelected) && (
              <div className="mt-2">
                <DensityBar
                  total={totalActivities}
                  positions={positions}
                  highlightPositions={highlightPositions}
                  colorClass="bg-teal-300"
                  highlightClass="bg-amber-500"
                />
              </div>
            )}
          </div>
        </div>
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-stone-100 space-y-3">
          {node.operator === "FOR" && <ForHeader bindings={node.bindings} />}
          {node.operator === "WHILE" && <WhileHeader condition={node.condition} />}
          {steps.length > 0 && (
            <div className="space-y-0.5">
              <BodyLabel operator={node.operator} bindings={node.bindings} />
              {steps.map((s, i) => (
                <StepItem key={i} step={s} ctx={ctx} depth={depth + 1} />
              ))}
            </div>
          )}

          {isSelected && node.dataflow && node.dataflow.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.14em] text-stone-400 font-medium mb-1">
                Dataflow
              </div>
              <ul className="space-y-0.5 text-[12px] text-stone-600 list-disc list-inside marker:text-stone-300">
                {node.dataflow.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}

          {isSelected && node.effects && node.effects.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.14em] text-stone-400 font-medium mb-1">
                Effects
              </div>
              <ul className="space-y-0.5 text-[12px] text-stone-600 list-disc list-inside marker:text-stone-300">
                {node.effects.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </div>
          )}

          {isSelected && node.evidence_summary && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.14em] text-stone-400 font-medium mb-1">
                Evidence
              </div>
              <p className="text-[12px] text-stone-600 leading-relaxed">{node.evidence_summary}</p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

// ── Column root ──────────────────────────────────────────────────────────────

export default function ProcedureColumn({
  model,
  threadId,
  totalActivities,
  activityIndex,
  related,
  highlightPositions,
  selectedNodeId,
  selectedActivityId,
  onSelectNode,
}: Props) {
  const nodes = model?.procedure_nodes ?? [];
  const rootId = model?.root_procedure_id;
  const rootNode = useMemo(
    () => nodes.find((n) => n.id === rootId) ?? null,
    [nodes, rootId],
  );

  const byId = useMemo(() => {
    const m = new Map<string, ProcedureNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  const refsByNode = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const n of nodes) m.set(n.id, new Set(expandSegmentRefs(n.activity_refs)));
    return m;
  }, [nodes]);

  const positionsByNode = useMemo(() => {
    const m = new Map<string, number[]>();
    for (const [id, refs] of refsByNode) {
      const positions: number[] = [];
      for (const r of refs) {
        const idx = activityIndex[r];
        if (idx !== undefined) positions.push(idx);
      }
      m.set(id, positions);
    }
    return m;
  }, [refsByNode, activityIndex]);

  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(rootId ? [rootId] : []),
  );

  useEffect(() => {
    setExpanded(new Set(rootId ? [rootId] : []));
  }, [rootId, threadId]);

  // Auto-expand all named nodes that contain the selected activity
  useEffect(() => {
    if (!selectedActivityId) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      for (const [id, refs] of refsByNode) {
        if (refs.has(selectedActivityId)) next.add(id);
      }
      return next;
    });
  }, [selectedActivityId, refsByNode]);

  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Scroll to first node containing the selected activity
  useEffect(() => {
    if (!selectedActivityId) return;
    const first = nodes.find((n) => refsByNode.get(n.id)?.has(selectedActivityId));
    if (!first) return;
    const el = cardRefs.current.get(first.id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selectedActivityId, nodes, refsByNode]);

  function toggle(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const ctx: TreeCtx = {
    byId,
    refsByNode,
    positionsByNode,
    totalActivities,
    activityIndex,
    related,
    highlightPositions,
    selectedNodeId,
    selectedActivityId,
    expanded,
    onToggle: toggle,
    onSelectNode,
    cardRefs,
  };

  return (
    <section className="flex flex-col h-full min-h-0">
      <header className="flex items-start gap-2.5 px-5 py-3 border-b border-stone-200 bg-white/70 backdrop-blur shrink-0">
        <div className="w-7 h-7 rounded-lg bg-teal-50 border border-teal-100 flex items-center justify-center shrink-0">
          <Workflow className="w-3.5 h-3.5 text-teal-700" />
        </div>
        <div className="min-w-0">
          <h2 className="text-[13px] font-semibold tracking-tight text-stone-900">
            How it was done
          </h2>
          <p className="text-[11px] text-stone-500 leading-snug">
            {nodes.length} procedure block{nodes.length === 1 ? "" : "s"}
          </p>
        </div>
      </header>

      <div data-export-scroll className="flex-1 overflow-y-auto px-3 py-3">
        {!model && (
          <div className="text-[12px] text-stone-400 px-4 py-12 text-center">
            No procedure model for this thread.
          </div>
        )}
        {rootNode && <ProcedureNodeCard node={rootNode} ctx={ctx} depth={0} />}
      </div>
    </section>
  );
}
