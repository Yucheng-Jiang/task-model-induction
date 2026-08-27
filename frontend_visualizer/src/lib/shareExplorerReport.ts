import type { Activity, SessionData, ThreadBundle, UnifiedModelNode } from "./types";
import { expandSegmentRefs } from "./segments";

type BadgeTone =
  | "slate"
  | "amber"
  | "teal"
  | "sky"
  | "rose"
  | "violet"
  | "fuchsia"
  | "orange"
  | "indigo"
  | "emerald"
  | "stone";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function badge(label: string, tone: BadgeTone = "slate"): string {
  return `<span class="share-badge badge-${tone}">${escapeHtml(label)}</span>`;
}

function threadTone(threadId: string): BadgeTone {
  const tones: Record<string, BadgeTone> = {
    C1: "indigo",
    C2: "emerald",
    C3: "rose",
    C4: "amber",
    C5: "sky",
    C6: "fuchsia",
    C7: "violet",
    C8: "teal",
    C9: "orange",
  };
  return tones[threadId] ?? "stone";
}

function aggregateRefs(node: UnifiedModelNode, refs = new Set<string>()): Set<string> {
  for (const ref of expandSegmentRefs(node.activity_refs)) refs.add(ref);
  for (const child of node.decomposition) aggregateRefs(child, refs);
  return refs;
}

function refsAttribute(refs: Iterable<string>): string {
  return escapeHtml(Array.from(refs).join(" "));
}

function renderActivityTimeline(
  refs: Set<string>,
  activityIndex: Record<string, number>,
  totalActivities: number,
  tone: BadgeTone,
): string {
  if (totalActivities === 0 || refs.size === 0) return "";

  const marks = Array.from(refs)
    .map((activityId) => ({ activityId, position: activityIndex[activityId] }))
    .filter(
      (mark): mark is { activityId: string; position: number } =>
        Number.isInteger(mark.position) && mark.position >= 0,
    )
    .sort((a, b) => a.position - b.position);

  if (marks.length === 0) return "";

  const markWidth = Math.max(0.4, (1 / totalActivities) * 100);
  return `<div class="share-activity-timeline">
    <div class="share-timeline-label">
      <span>Activity timeline</span>
      <span>${marks.length} of ${totalActivities} steps</span>
    </div>
    <div class="share-timeline" role="img" aria-label="Activity timeline: ${marks.length} of ${totalActivities} activities">
      ${marks
        .map(
          ({ activityId, position }) =>
            `<span class="share-timeline-mark dot-${tone}" data-share-timeline-mark data-share-timeline-activity="${escapeHtml(activityId)}" style="left:${(position / totalActivities) * 100}%;width:${markWidth}%"></span>`,
        )
        .join("")}
    </div>
  </div>`;
}

function operatorLabel(node: UnifiedModelNode): { label: string; tone: BadgeTone } | null {
  switch (node.procedure.operator) {
    case "FOR": {
      const variable = node.procedure.bindings?.iteration_variable;
      return {
        label: variable
          ? `Repeat for each ${String(variable).replaceAll("_", " ")}`
          : "Repeat for each item",
        tone: "amber",
      };
    }
    case "WHILE":
      return { label: "Repeat until complete", tone: "rose" };
    case "CHOICE":
      return { label: "Choose one path", tone: "violet" };
    case "PARALLEL":
      return { label: "Can happen together", tone: "teal" };
    default:
      return null;
  }
}

function renderUnifiedDetails(node: UnifiedModelNode): string {
  const bodySteps = node.procedure.body ?? [];
  const collection = node.procedure.bindings?.collection;
  const collectionItems = Array.isArray(collection)
    ? collection
        .map((item) => `<span class="share-value-chip">${escapeHtml(String(item))}</span>`)
        .join("")
    : "";

  const detailParts = [
    node.summary ? `<p class="share-unified-summary">${escapeHtml(node.summary)}</p>` : "",
    node.procedure.description && node.procedure.description !== node.summary
      ? `<p class="share-unified-copy">${escapeHtml(node.procedure.description)}</p>`
      : "",
    node.procedure.operator === "FOR" && collectionItems
      ? `<section class="share-detail-section share-detail-wide">
          <h4>Runs once for each item</h4>
          <div class="share-value-list">${collectionItems}</div>
        </section>`
      : "",
    node.procedure.condition
      ? `<section class="share-detail-section share-detail-wide">
          <h4>Repeats until</h4>
          <p>${escapeHtml(node.procedure.condition)}</p>
        </section>`
      : "",
    bodySteps.length
      ? `<section class="share-detail-section">
          <h4>How it was done</h4>
          <ol class="share-body-steps">${bodySteps
            .map(
              (step) => `<li>
                <span>${escapeHtml(step.name)}</span>
                ${step.description ? `<small>${escapeHtml(step.description)}</small>` : ""}
              </li>`,
            )
            .join("")}</ol>
        </section>`
      : "",
    node.observed_outcome?.description
      ? `<section class="share-detail-section">
          <h4>Outcome</h4>
          <p>${escapeHtml(node.observed_outcome.description)}</p>
        </section>`
      : "",
    node.procedure.evidence_summary
      ? `<section class="share-detail-section share-detail-wide">
          <h4>Evidence</h4>
          <p>${escapeHtml(node.procedure.evidence_summary)}</p>
        </section>`
      : "",
  ].filter(Boolean);

  return detailParts.length ? detailParts.join("") : `<p class="share-empty">No additional details recorded.</p>`;
}

function renderUnifiedNode(
  node: UnifiedModelNode,
  threadId: string,
  railTone: BadgeTone,
  activityIndex: Record<string, number>,
  totalActivities: number,
  isRoot = false,
): string {
  const refs = aggregateRefs(node);
  const children = node.decomposition ?? [];
  const childRefs = new Set<string>();
  for (const child of children) aggregateRefs(child, childRefs);

  const key = `${threadId}::${node.id}`;
  const refsData = refsAttribute(refs);
  const keyData = escapeHtml(key);
  const displayObjective = isRoot ? "Task outline" : node.objective;
  const operator = operatorLabel(node);
  const timeline = renderActivityTimeline(refs, activityIndex, totalActivities, railTone);
  const childCount = `${children.length} child ${children.length === 1 ? "step" : "steps"}`;
  const meta = isRoot
    ? `${badge("Overview", railTone)}
       <span>${refs.size} ${refs.size === 1 ? "step" : "steps"}</span>
       ${children.length ? '<span class="share-detail-hint">Details stay visible; child steps use the control below</span>' : ""}`
    : `${operator ? badge(operator.label, operator.tone) : ""}
       ${refs.size ? `<span>${refs.size} ${refs.size === 1 ? "step" : "steps"}</span>` : ""}`;

  const nodeMarker = isRoot
    ? `<span class="share-root-mark" aria-hidden="true"><span></span><span></span><span></span></span>`
    : `<span class="share-tree-marker ${children.length ? "is-branch" : ""}" aria-hidden="true"></span>`;

  const showLabel = `Show ${childCount}`;
  const hideLabel = `Hide ${childCount}`;
  const initiallyExpanded = isRoot;
  const childToggle = children.length
    ? `<button
        type="button"
        class="share-child-toggle ${initiallyExpanded ? "is-expanded" : ""}"
        data-share-tree-toggle
        data-share-tree-key="${keyData}"
        data-share-show-label="${escapeHtml(showLabel)}"
        data-share-hide-label="${escapeHtml(hideLabel)}"
        aria-label="${escapeHtml(initiallyExpanded ? hideLabel : showLabel)}; this step's details stay visible"
        aria-expanded="${initiallyExpanded ? "true" : "false"}"
      >
        <span data-share-tree-toggle-label>${escapeHtml(initiallyExpanded ? hideLabel : showLabel)}</span>
        <span class="share-child-toggle-end">
          <span class="share-child-toggle-hint" data-share-tree-toggle-hint>${initiallyExpanded ? "currently visible" : "details stay visible"}</span>
          <span class="share-chevron" aria-hidden="true"></span>
        </span>
      </button>`
    : "";

  const heading = isRoot
    ? `<div class="share-unified-card-head">
        ${nodeMarker}
        <div class="share-unified-heading-wrap">
          <h3>${escapeHtml(displayObjective)}</h3>
          <div class="share-unified-meta">${meta}</div>
          ${timeline}
        </div>
      </div>`
    : `<div class="share-unified-card-head">
        ${nodeMarker}
        <button type="button" class="share-model-select share-unified-heading-wrap" data-share-model-select data-share-model-key="${keyData}" data-share-model-refs="${refsData}" aria-label="Highlight related activity for ${escapeHtml(displayObjective)}">
          <h3>${escapeHtml(displayObjective)}</h3>
          <div class="share-unified-meta">${meta}</div>
          ${timeline}
        </button>
      </div>`;

  const childrenRegion = !children.length
    ? ""
    : `<div
        class="share-unified-children ${isRoot ? "share-root-children" : ""}"
        data-share-child-region
        data-share-tree-key="${keyData}"
        data-share-child-refs="${refsAttribute(childRefs)}"
        ${isRoot ? "" : "hidden"}
      >
        ${children
          .map((child) => renderUnifiedNode(child, threadId, railTone, activityIndex, totalActivities))
          .join("")}
      </div>`;

  return `<div class="share-unified-node">
    <article class="share-unified-card rail-${railTone}" data-share-model-card data-share-model-key="${keyData}" data-share-model-refs="${refsData}">
      ${heading}
      <div class="share-unified-card-body ${isRoot ? "share-overview-body" : ""}">
        ${renderUnifiedDetails(node)}
      </div>
      ${childToggle}
    </article>
    ${childrenRegion}
  </div>`;
}

function renderTraceActivity(
  activity: Activity,
  index: number,
  threadId: string,
): string {
  const tone = threadTone(threadId);
  const semanticActionCount = activity.semantic_action_count;
  return `<button
    type="button"
    class="share-trace-card"
    data-share-trace-card
    data-share-trace-id="${escapeHtml(activity.activity_id)}"
    data-share-trace-thread="${escapeHtml(threadId)}"
    aria-label="Select ${escapeHtml(activity.objective)}"
  >
    <div class="share-trace-index">
      <span>${String(index + 1).padStart(3, "0")}</span>
      <span class="share-thread-dot dot-${tone}" aria-hidden="true"></span>
    </div>
    <div class="share-trace-content">
      <div class="share-trace-meta">
        ${badge(threadId, tone)}
        <span class="share-activity-id">${escapeHtml(activity.activity_id)}</span>
      </div>
      <p class="share-trace-objective">${escapeHtml(activity.objective)}</p>
      <div class="share-trace-stats">
        <span>actions ${activity.start_action_idx}–${activity.end_action_idx}</span>
        <span aria-hidden="true">·</span>
        <span>${semanticActionCount} semantic action${semanticActionCount === 1 ? "" : "s"}</span>
      </div>
    </div>
  </button>`;
}

function renderTaskPanel(
  thread: ThreadBundle,
  index: number,
  active: boolean,
  activityIndex: Record<string, number>,
  totalActivities: number,
): string {
  const tone = threadTone(thread.id);
  const panelId = `share-unified-panel-${index}`;
  return `<section
    id="${panelId}"
    class="share-unified-task"
    role="tabpanel"
    data-share-unified-task="${escapeHtml(thread.id)}"
    ${active ? "" : "hidden"}
  >
    <header class="share-pane-header share-unified-header">
      <div class="share-pane-icon pane-${tone}" aria-hidden="true">
        <span class="share-stack-icon"><span></span><span></span><span></span></span>
      </div>
      <div class="share-pane-title-wrap">
        <h2>What was done</h2>
        <p>${escapeHtml(thread.label)}</p>
      </div>
    </header>
    <div class="share-overall-goal">
      <p>Overall goal</p>
      <div>${escapeHtml(thread.task_thread_objective)}</div>
    </div>
    <div class="share-unified-scroll">
      ${
        thread.unifiedModel
          ? renderUnifiedNode(thread.unifiedModel.root, thread.id, tone, activityIndex, totalActivities, true)
          : `<p class="share-no-model">No unified model available for this task.</p>`
      }
    </div>
  </section>`;
}

function renderTaskTab(thread: ThreadBundle, index: number, active: boolean): string {
  const tone = threadTone(thread.id);
  return `<button
    type="button"
    class="share-task-tab"
    role="tab"
    data-share-tab="${escapeHtml(thread.id)}"
    aria-controls="share-unified-panel-${index}"
    aria-selected="${active ? "true" : "false"}"
  >
    <span class="share-thread-dot dot-${tone}" aria-hidden="true"></span>
    <span class="share-tab-id">${escapeHtml(thread.id)}</span>
    <span class="share-tab-label">${escapeHtml(thread.label)}</span>
    <span class="share-tab-count">${thread.localObjectiveIds.length}</span>
  </button>`;
}

/**
 * A portable, Unified-only counterpart to the live explorer. It embeds every
 * task so recipients can switch tabs without needing the local session files.
 */
export function renderShareExplorer(data: SessionData, initialThreadId?: string): string {
  const defaultThread =
    data.threads.reduce(
      (best, thread) =>
        thread.localObjectiveIds.length > (best?.localObjectiveIds.length ?? -1) ? thread : best,
      data.threads[0],
    ) ?? data.threads[0];
  const activeThread = data.threads.find((thread) => thread.id === initialThreadId) ?? defaultThread;
  const activeThreadId = activeThread?.id ?? "";
  const trace = data.activities
    .map((activity, index) => {
      const threadId = data.threadByObjective[activity.activity_id] ?? "unassigned";
      return renderTraceActivity(activity, index, threadId);
    })
    .join("");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Task results · Task Trace Explorer</title>
    <style>
      :root {
        color-scheme: light;
        --page: #f8f7f5;
        --surface: #ffffff;
        --surface-muted: #fafaf9;
        --text: #1c1917;
        --muted: #78716c;
        --border: #e7e5e4;
        --border-strong: #d6d3d1;
        --focus: #a855f7;
      }
      * { box-sizing: border-box; }
      html, body { height: 100%; }
      body {
        margin: 0;
        background: var(--page);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      button, input { font: inherit; }
      button { color: inherit; }
      [hidden] { display: none !important; }
      .share-app {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 600px;
        overflow: hidden;
      }
      .share-header {
        background: rgba(255, 255, 255, .88);
        border-bottom: 1px solid var(--border);
        flex: 0 0 auto;
      }
      .share-header-main {
        align-items: center;
        display: flex;
        gap: 16px;
        padding: 12px 24px;
      }
      .share-brand { align-items: center; display: flex; flex: 0 0 auto; gap: 10px; }
      .share-brand-mark {
        align-items: center;
        background: linear-gradient(135deg, #292524, #57534e);
        border-radius: 11px;
        box-shadow: 0 1px 2px rgba(28, 25, 23, .15);
        display: inline-flex;
        height: 32px;
        justify-content: center;
        width: 32px;
      }
      .share-brand-mark svg { height: 17px; stroke: white; width: 17px; }
      .share-brand h1 { font-size: 15px; letter-spacing: -.01em; line-height: 1.15; margin: 0; }
      .share-brand p { color: var(--muted); font-size: 12px; line-height: 1.15; margin: 3px 0 0; }
      .share-link-hint { color: #a8a29e; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; margin-left: auto; }
      .share-tabs-row {
        align-items: center;
        display: flex;
        gap: 8px;
        padding: 0 24px 12px;
      }
      .share-tab-list { align-items: center; display: flex; flex: 1 1 auto; gap: 8px; min-width: 0; overflow-x: auto; }
      .share-task-tab {
        align-items: center;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        cursor: pointer;
        display: inline-flex;
        flex: 0 0 auto;
        gap: 8px;
        max-width: 340px;
        min-width: 0;
        padding: 6px 12px;
        transition: background .12s ease, border-color .12s ease, box-shadow .12s ease;
      }
      .share-task-tab:hover { background: rgba(255, 255, 255, .7); border-color: var(--border); }
      .share-task-tab[aria-selected="true"] { background: #fff; border-color: var(--border-strong); box-shadow: 0 1px 2px rgba(28, 25, 23, .08); }
      .share-task-tab:focus-visible, .share-child-toggle:focus-visible, .share-model-select:focus-visible, .share-trace-card:focus-visible, .share-clear:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
      .share-thread-dot { border-radius: 999px; display: inline-block; flex: 0 0 auto; height: 8px; width: 8px; }
      .dot-indigo { background: #6366f1; } .dot-emerald { background: #10b981; } .dot-rose { background: #f43f5e; }
      .dot-amber { background: #f59e0b; } .dot-sky { background: #0ea5e9; } .dot-fuchsia { background: #d946ef; }
      .dot-violet { background: #8b5cf6; } .dot-teal { background: #14b8a6; } .dot-orange { background: #f97316; }
      .dot-stone { background: #a8a29e; }
      .share-tab-id { color: #57534e; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; font-weight: 650; }
      .share-task-tab[aria-selected="true"] .share-tab-id { color: #1c1917; }
      .share-tab-label { color: #57534e; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .share-task-tab[aria-selected="true"] .share-tab-label { color: #292524; }
      .share-tab-count { color: #a8a29e; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
      .share-tab-controls { align-items: center; display: flex; flex: 0 0 auto; gap: 10px; margin-left: auto; }
      .share-clear { background: transparent; border: 0; border-radius: 6px; color: var(--muted); cursor: pointer; font-size: 12px; padding: 5px 8px; }
      .share-clear:hover { background: #f5f5f4; color: #1c1917; }
      .share-filter { align-items: center; color: var(--muted); cursor: pointer; display: flex; font-size: 12px; gap: 6px; white-space: nowrap; }
      .share-filter input { accent-color: #1c1917; }
      .share-main { display: grid; flex: 1 1 auto; grid-template-columns: minmax(320px, .9fr) minmax(540px, 1.7fr); min-height: 0; }
      .share-pane, .share-unified-task { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
      .share-trace-pane { border-right: 1px solid var(--border); }
      .share-pane-header { align-items: center; background: rgba(255, 255, 255, .72); border-bottom: 1px solid var(--border); display: flex; flex: 0 0 auto; gap: 12px; padding: 16px 20px; }
      .share-pane-icon { align-items: center; border: 1px solid #fde68a; border-radius: 12px; display: flex; flex: 0 0 auto; height: 36px; justify-content: center; width: 36px; }
      .share-trace-icon { background: #fffbeb; color: #b45309; }
      .share-trace-icon::before { content: "≡"; font-size: 22px; font-weight: 700; line-height: 1; transform: rotate(90deg); }
      .pane-indigo { background: #eef2ff; border-color: #e0e7ff; color: #4338ca; } .pane-emerald { background: #ecfdf5; border-color: #d1fae5; color: #047857; }
      .pane-rose { background: #fff1f2; border-color: #ffe4e6; color: #be123c; } .pane-amber { background: #fffbeb; border-color: #fef3c7; color: #b45309; }
      .pane-sky { background: #f0f9ff; border-color: #e0f2fe; color: #0369a1; } .pane-fuchsia { background: #fdf4ff; border-color: #fae8ff; color: #a21caf; }
      .pane-violet { background: #f5f3ff; border-color: #ede9fe; color: #6d28d9; } .pane-teal { background: #f0fdfa; border-color: #ccfbf1; color: #0f766e; }
      .pane-orange { background: #fff7ed; border-color: #ffedd5; color: #c2410c; } .pane-stone { background: #fafaf9; border-color: #f5f5f4; color: #57534e; }
      .share-stack-icon, .share-root-mark { display: grid; gap: 2px; width: 16px; }
      .share-stack-icon span, .share-root-mark span { border: 1.5px solid currentColor; border-radius: 2px; display: block; height: 4px; transform: skewY(-18deg); }
      .share-pane-title-wrap { min-width: 0; }
      .share-pane-title-wrap h2 { color: #1c1917; font-size: 16px; letter-spacing: -.01em; line-height: 1.25; margin: 0; }
      .share-pane-title-wrap p { color: var(--muted); font-size: 13px; line-height: 1.35; margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .share-trace-list { display: flex; flex: 1 1 auto; flex-direction: column; gap: 8px; min-height: 0; overflow-y: auto; padding: 12px; }
      .share-trace-card { align-items: flex-start; appearance: none; background: #fff; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; display: flex; gap: 12px; padding: 12px 14px; text-align: left; transition: border-color .12s ease, box-shadow .12s ease, opacity .12s ease, background .12s ease; width: 100%; }
      .share-trace-card:hover { background: #fafaf9; border-color: var(--border-strong); }
      .share-trace-card.is-selected { background: #fffbeb; border-color: #fbbf24; box-shadow: 0 1px 2px rgba(180, 83, 9, .12); }
      .share-trace-card.is-related { border-color: #d8b4fe; box-shadow: 0 0 0 1px #f3e8ff; }
      .share-trace-card.is-inactive-thread { opacity: .60; }
      .share-trace-card.is-dimmed { opacity: .5; }
      .share-trace-index { align-items: center; color: var(--muted); display: flex; flex: 0 0 auto; flex-direction: column; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; gap: 7px; padding-top: 2px; width: 27px; }
      .share-trace-index .share-thread-dot { height: 7px; width: 7px; }
      .share-trace-content { min-width: 0; }
      .share-trace-meta { align-items: center; display: flex; flex-wrap: wrap; gap: 6px; }
      .share-badge { align-items: center; border-radius: 5px; display: inline-flex; font-size: 11px; font-weight: 650; line-height: 1.1; padding: 4px 6px; }
      .badge-slate { background: #e2e8f0; color: #475569; } .badge-amber { background: #fef3c7; color: #b45309; }
      .badge-teal { background: #ccfbf1; color: #0f766e; } .badge-sky { background: #e0f2fe; color: #0369a1; }
      .badge-rose { background: #ffe4e6; color: #be123c; } .badge-violet { background: #ede9fe; color: #6d28d9; }
      .badge-fuchsia { background: #fae8ff; color: #a21caf; } .badge-orange { background: #ffedd5; color: #c2410c; }
      .badge-indigo { background: #e0e7ff; color: #4338ca; } .badge-emerald { background: #d1fae5; color: #047857; }
      .badge-stone { background: #f5f5f4; color: #57534e; }
      .share-activity-id { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .share-trace-objective { color: #292524; font-size: 14px; line-height: 1.45; margin: 7px 0 0; }
      .share-trace-stats { color: var(--muted); display: flex; flex-wrap: wrap; font-size: 12px; gap: 6px; line-height: 1.4; margin-top: 8px; }
      .share-trace-empty { color: #a8a29e; font-size: 13px; margin: 12px 0 0; text-align: center; }
      .share-unified-header { padding-left: 24px; padding-right: 24px; }
      .share-overall-goal { background: rgba(250, 250, 249, .75); border-bottom: 1px solid var(--border); flex: 0 0 auto; padding: 14px 24px; }
      .share-overall-goal p { color: var(--muted); font-size: 12px; font-weight: 650; letter-spacing: .12em; margin: 0 0 4px; text-transform: uppercase; }
      .share-overall-goal div { color: #292524; font-size: 15px; font-weight: 550; line-height: 1.5; max-width: 880px; }
      .share-unified-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 12px 20px 18px; }
      .share-unified-node { margin-top: 12px; }
      .share-unified-card { --share-card-content-left: 52px; background: #fff; border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 1px 2px rgba(28, 25, 23, .06); overflow: hidden; position: relative; }
      .share-unified-card::before { bottom: 0; content: ""; left: 0; position: absolute; top: 0; width: 4px; z-index: 2; }
      .rail-indigo::before { background: #6366f1; } .rail-emerald::before { background: #10b981; } .rail-rose::before { background: #f43f5e; }
      .rail-amber::before { background: #f59e0b; } .rail-sky::before { background: #0ea5e9; } .rail-fuchsia::before { background: #d946ef; }
      .rail-violet::before { background: #8b5cf6; } .rail-teal::before { background: #14b8a6; } .rail-orange::before { background: #f97316; }
      .rail-stone::before { background: #a8a29e; }
      .share-unified-card.is-selected { border-color: #fbbf24; box-shadow: 0 0 0 2px #fef3c7; }
      .share-unified-card.is-path { border-color: #d8b4fe; box-shadow: 0 0 0 1px #f3e8ff; }
      .share-unified-card-head { align-items: flex-start; display: flex; gap: 8px; padding: 12px 16px 10px; }
      .share-root-mark { align-items: center; color: #78716c; display: inline-flex; flex: 0 0 28px; height: 28px; justify-content: center; margin-top: 1px; width: 28px; }
      .share-tree-marker { align-items: center; display: inline-flex; flex: 0 0 28px; height: 28px; justify-content: center; margin-top: 1px; width: 28px; }
      .share-tree-marker::after { background: #d6d3d1; border-radius: 999px; content: ""; display: block; height: 6px; width: 6px; }
      .share-tree-marker.is-branch::after { background: #fafaf9; border: 1px solid #d6d3d1; border-radius: 3px; height: 8px; width: 8px; }
      .share-unified-heading-wrap { min-width: 0; width: 100%; }
      .share-model-select { background: transparent; border: 0; cursor: pointer; padding: 0; text-align: left; }
      .share-model-select:hover h3 { color: #57534e; }
      .share-unified-card h3 { color: #292524; font-size: 15px; font-weight: 600; line-height: 1.45; margin: 0; }
      .share-unified-meta { align-items: center; color: var(--muted); display: flex; flex-wrap: wrap; font-size: 12px; gap: 8px; line-height: 1.4; margin-top: 7px; }
      .share-detail-hint { color: #a8a29e; }
      .share-activity-timeline { margin-top: 9px; }
      .share-timeline-label { color: #a8a29e; display: flex; font-size: 11px; justify-content: space-between; line-height: 1.3; margin-bottom: 5px; }
      .share-timeline { background: #f0eeeb; border-radius: 999px; height: 6px; overflow: hidden; position: relative; }
      .share-timeline-mark { bottom: 0; opacity: .72; position: absolute; top: 0; }
      .share-timeline-mark.is-highlighted { background: #f59e0b; opacity: 1; }
      .share-unified-card-body { background: rgba(250, 250, 249, .7); border-top: 1px solid #f0eeeb; display: grid; gap: 14px 24px; grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 14px 16px 16px var(--share-card-content-left); }
      .share-unified-summary, .share-unified-copy, .share-detail-wide { grid-column: 1 / -1; }
      .share-unified-summary { color: #57534e; font-size: 14px; line-height: 1.6; margin: 0; }
      .share-unified-copy { color: #78716c; font-size: 13px; line-height: 1.5; margin: 0; }
      .share-detail-section h4 { color: #78716c; font-size: 11px; letter-spacing: .1em; margin: 0 0 6px; text-transform: uppercase; }
      .share-detail-section p { color: #57534e; font-size: 13px; line-height: 1.5; margin: 0; }
      .share-value-list { display: flex; flex-wrap: wrap; gap: 6px; }
      .share-value-chip { background: #fef3c7; border-radius: 6px; color: #92400e; font-size: 12px; padding: 5px 8px; }
      .share-body-steps { color: #57534e; display: grid; font-size: 13px; gap: 6px; line-height: 1.45; margin: 0; padding-left: 20px; }
      .share-body-steps small { color: var(--muted); display: block; font-size: 12px; line-height: 1.45; margin-top: 2px; }
      .share-child-toggle { align-items: center; appearance: none; background: #fff; border: 0; border-top: 1px solid #f0eeeb; cursor: pointer; display: flex; font-size: 14px; font-weight: 600; justify-content: space-between; padding: 10px 16px 10px var(--share-card-content-left); text-align: left; transition: background .12s ease; width: 100%; }
      .share-child-toggle:hover { background: #fafaf9; }
      .share-child-toggle > span:first-child { color: #57534e; }
      .share-child-toggle-end { align-items: center; display: inline-flex; gap: 10px; margin-left: 16px; }
      .share-child-toggle-hint { color: #a8a29e; font-size: 12px; font-weight: 400; white-space: nowrap; }
      .share-chevron { border-bottom: 1.5px solid #78716c; border-right: 1.5px solid #78716c; display: block; height: 7px; transform: rotate(45deg); transition: transform .12s ease; width: 7px; }
      .share-child-toggle.is-expanded .share-chevron { transform: rotate(225deg); }
      .share-unified-children { border-left: 1px solid #dedbd5; margin: 10px 0 0 24px; padding-left: 16px; }
      .share-root-children { margin-top: 14px; }
      .share-empty, .share-no-model { color: var(--muted); font-size: 13px; font-style: italic; line-height: 1.5; margin: 0; }
      .share-no-model { padding: 36px 20px; text-align: center; }
      @media (max-width: 960px) {
        .share-app { height: auto; min-height: 100%; overflow: visible; }
        .share-header-main { padding-left: 16px; padding-right: 16px; }
        .share-tabs-row { padding-left: 16px; padding-right: 16px; }
        .share-link-hint { display: none; }
        .share-main { display: block; }
        .share-trace-pane { border-bottom: 1px solid var(--border); border-right: 0; max-height: 55vh; }
        .share-unified-task { min-height: 70vh; }
        .share-unified-header, .share-overall-goal { padding-left: 18px; padding-right: 18px; }
        .share-unified-scroll { padding-left: 14px; padding-right: 14px; }
      }
      @media (max-width: 620px) {
        .share-tab-controls { display: none; }
        .share-unified-card-body { grid-template-columns: 1fr; }
        .share-detail-section { grid-column: 1 / -1; }
        .share-unified-children { margin-left: 14px; padding-left: 10px; }
        .share-detail-hint { display: none; }
        .share-child-toggle-hint { display: none; }
      }
    </style>
  </head>
  <body>
    <div class="share-app" data-share-explorer data-initial-thread="${escapeHtml(activeThreadId)}">
      <header class="share-header">
        <div class="share-header-main">
          <div class="share-brand">
            <span class="share-brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5"/><path d="m4 16.5 8 4.5 8-4.5"/></svg>
            </span>
            <div>
              <h1>Task Trace Explorer</h1>
              <p>${data.activities.length} activities · ${data.threads.length} task threads</p>
            </div>
          </div>
          <div class="share-link-hint">click any item to link views</div>
        </div>
        <div class="share-tabs-row">
          <div role="tablist" aria-label="Task threads" class="share-tab-list">
            ${data.threads.map((thread, index) => renderTaskTab(thread, index, thread.id === activeThreadId)).join("")}
          </div>
          <div class="share-tab-controls">
            <button type="button" class="share-clear" data-share-clear hidden>× clear selection</button>
            <label class="share-filter"><input type="checkbox" data-share-filter /> filter trace to selection</label>
          </div>
        </div>
      </header>
      <main class="share-main">
        <section class="share-pane share-trace-pane" aria-label="Activity trace">
          <header class="share-pane-header">
            <div class="share-pane-icon share-trace-icon" aria-hidden="true"></div>
            <div class="share-pane-title-wrap">
              <h2>Activity trace</h2>
              <p>${data.activities.length} steps in chronological order</p>
            </div>
          </header>
          <div class="share-trace-list" data-share-trace-list>
            ${trace}
            <p class="share-trace-empty" data-share-filter-empty hidden>Nothing selected yet. Click a model node or trace step.</p>
          </div>
        </section>
        ${data.threads
          .map((thread, index) =>
            renderTaskPanel(
              thread,
              index,
              thread.id === activeThreadId,
              data.activityIndex,
              data.activities.length,
            ),
          )
          .join("")}
      </main>
    </div>
    <script>
      (() => {
        const app = document.querySelector("[data-share-explorer]");
        if (!app) return;

        const tabs = Array.from(app.querySelectorAll("[data-share-tab]"));
        const panels = Array.from(app.querySelectorAll("[data-share-unified-task]"));
        const traceCards = Array.from(app.querySelectorAll("[data-share-trace-card]"));
        const modelCards = Array.from(app.querySelectorAll("[data-share-model-card]"));
        const childRegions = Array.from(app.querySelectorAll("[data-share-child-region]"));
        const timelineMarks = Array.from(app.querySelectorAll("[data-share-timeline-mark]"));
        const filter = app.querySelector("[data-share-filter]");
        const clear = app.querySelector("[data-share-clear]");
        const empty = app.querySelector("[data-share-filter-empty]");
        let activeThread = app.dataset.initialThread || tabs[0]?.dataset.shareTab || "";
        let selectedActivity = "";
        let selectedModel = "";
        let selectedRefs = new Set();

        const refsFor = (element, name) => new Set((element.dataset[name] || "").split(" ").filter(Boolean));
        const contains = (refs, value) => refs.has(value);

        function setBranchOpen(key, open) {
          childRegions.forEach((region) => {
            if (region.dataset.shareTreeKey !== key) return;
            region.hidden = !open;
          });
          app.querySelectorAll("[data-share-tree-toggle]").forEach((toggle) => {
            if (toggle.dataset.shareTreeKey !== key) return;
            toggle.classList.toggle("is-expanded", open);
            toggle.setAttribute("aria-expanded", String(open));
            const label = open
              ? toggle.dataset.shareHideLabel || "Hide child steps"
              : toggle.dataset.shareShowLabel || "Show child steps";
            toggle.setAttribute("aria-label", label + "; this step's details stay visible");
            const labelNode = toggle.querySelector("[data-share-tree-toggle-label]");
            if (labelNode) labelNode.textContent = label;
            const hintNode = toggle.querySelector("[data-share-tree-toggle-hint]");
            if (hintNode) hintNode.textContent = open ? "currently visible" : "details stay visible";
          });
        }

        function revealActivityPath(activityId) {
          childRegions.forEach((region) => {
            if (contains(refsFor(region, "shareChildRefs"), activityId)) {
              setBranchOpen(region.dataset.shareTreeKey || "", true);
            }
          });
        }

        function sync() {
          const hasSelection = Boolean(selectedActivity || selectedModel);
          tabs.forEach((tab) => {
            const active = tab.dataset.shareTab === activeThread;
            tab.setAttribute("aria-selected", String(active));
            tab.tabIndex = active ? 0 : -1;
          });
          panels.forEach((panel) => {
            panel.hidden = panel.dataset.shareUnifiedTask !== activeThread;
          });

          traceCards.forEach((card) => {
            const id = card.dataset.shareTraceId || "";
            const thread = card.dataset.shareTraceThread || "";
            const isSelected = Boolean(selectedActivity) && id === selectedActivity;
            const isRelated = !selectedActivity && selectedRefs.has(id);
            const isInactive = thread !== activeThread && !isSelected && !isRelated;
            const isDimmed = hasSelection && !isSelected && !isRelated && thread === activeThread;
            card.classList.toggle("is-selected", isSelected);
            card.classList.toggle("is-related", isRelated);
            card.classList.toggle("is-inactive-thread", isInactive);
            card.classList.toggle("is-dimmed", isDimmed);
            card.hidden = Boolean(filter?.checked && hasSelection && !isSelected && !isRelated);
          });

          modelCards.forEach((card) => {
            const key = card.dataset.shareModelKey || "";
            const refs = refsFor(card, "shareModelRefs");
            card.classList.toggle("is-selected", Boolean(selectedModel) && key === selectedModel);
            card.classList.toggle("is-path", Boolean(selectedActivity) && contains(refs, selectedActivity));
          });
          timelineMarks.forEach((mark) => {
            const activityId = mark.dataset.shareTimelineActivity || "";
            const isHighlighted = selectedActivity
              ? activityId === selectedActivity
              : selectedRefs.has(activityId);
            mark.classList.toggle("is-highlighted", isHighlighted);
          });
          if (clear) clear.hidden = !hasSelection;
          if (empty) empty.hidden = !(filter?.checked && !hasSelection);
        }

        function activateThread(threadId, clearSelection = true) {
          activeThread = threadId;
          if (clearSelection) {
            selectedActivity = "";
            selectedModel = "";
            selectedRefs = new Set();
          }
          sync();
        }

        tabs.forEach((tab, index) => {
          tab.addEventListener("click", () => activateThread(tab.dataset.shareTab || ""));
          tab.addEventListener("keydown", (event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
            event.preventDefault();
            const next = (index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
            tabs[next].focus();
            activateThread(tabs[next].dataset.shareTab || "");
          });
        });

        app.querySelectorAll("[data-share-tree-toggle]").forEach((toggle) => {
          toggle.addEventListener("click", () => {
            const key = toggle.dataset.shareTreeKey || "";
            const region = childRegions.find((candidate) => candidate.dataset.shareTreeKey === key);
            setBranchOpen(key, Boolean(region?.hidden));
          });
        });

        app.querySelectorAll("[data-share-model-select]").forEach((button) => {
          button.addEventListener("click", () => {
            const key = button.dataset.shareModelKey || "";
            if (selectedModel === key) {
              selectedModel = "";
              selectedRefs = new Set();
            } else {
              selectedActivity = "";
              selectedModel = key;
              selectedRefs = refsFor(button, "shareModelRefs");
            }
            sync();
          });
        });

        traceCards.forEach((card) => {
          card.addEventListener("click", () => {
            selectedActivity = card.dataset.shareTraceId || "";
            selectedModel = "";
            selectedRefs = new Set(selectedActivity ? [selectedActivity] : []);
            activateThread(card.dataset.shareTraceThread || activeThread, false);
            revealActivityPath(selectedActivity);
            sync();
          });
        });

        filter?.addEventListener("change", sync);
        clear?.addEventListener("click", () => {
          selectedActivity = "";
          selectedModel = "";
          selectedRefs = new Set();
          sync();
        });
        sync();
      })();
    </script>
  </body>
</html>`;
}
