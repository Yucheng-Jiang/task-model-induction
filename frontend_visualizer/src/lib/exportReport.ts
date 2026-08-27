import path from "path";
import type {
  ObjectiveModelNode,
  ProcedureBody,
  ProcedureModel,
  ProcedureStep,
  SessionData,
  ThreadBundle,
  UnifiedModelNode,
} from "./types";
import { procedureBodySteps } from "./types";
import { expandSegmentRefs } from "./segments";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

type BadgeTone =
  | "slate"
  | "stone"
  | "amber"
  | "blue"
  | "indigo"
  | "emerald"
  | "teal"
  | "sky"
  | "rose"
  | "violet"
  | "fuchsia"
  | "orange";

function badge(label: string, tone: BadgeTone = "slate"): string {
  return `<span class="badge badge-${tone}">${escapeHtml(label)}</span>`;
}

function renderObjectiveNode(node: ObjectiveModelNode): string {
  const children = node.decomposition ?? [];
  const subgoals = node.subgoal_segments?.length
    ? `<ul class="chip-list">${node.subgoal_segments
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join("")}</ul>`
    : "";

  return `
    <details class="tree-node" open>
      <summary>
        <span class="tree-title">${escapeHtml(node.objective)}</span>
        ${badge(node.id, "blue")}
      </summary>
      <div class="tree-body">
        <p class="summary">${escapeHtml(node.summary || "No summary available.")}</p>
        ${subgoals}
        ${children.length ? children.map(renderObjectiveNode).join("") : ""}
      </div>
    </details>
  `;
}

function renderProcedureBody(body: ProcedureBody): string {
  const steps = procedureBodySteps(body);
  if (!steps.length) return `<p class="empty">No nested body.</p>`;
  return `<div class="step-list">${steps.map(renderProcedureStep).join("")}</div>`;
}

function renderProcedureStep(step: ProcedureStep): string {
  if ("activity_id" in step) {
    return `
      <div class="step-card">
        <div class="step-head">
          ${badge(step.activity_id, "amber")}
          <strong>${escapeHtml(step.name ?? "Activity leaf")}</strong>
        </div>
        ${
          step.description
            ? `<p class="summary">${escapeHtml(step.description)}</p>`
            : `<p class="empty">No description.</p>`
        }
      </div>
    `;
  }

  if ("operator" in step) {
    return `
      <details class="step-card" open>
        <summary class="step-head">
          ${badge(step.operator, "teal")}
          <strong>${escapeHtml(step.name ?? "Control step")}</strong>
        </summary>
        <div class="tree-body">
          ${
            step.description
              ? `<p class="summary">${escapeHtml(step.description)}</p>`
              : `<p class="empty">No description.</p>`
          }
          ${step.condition ? `<p><strong>Condition:</strong> ${escapeHtml(step.condition)}</p>` : ""}
          ${step.steps?.length ? `<div class="step-list">${step.steps.map(renderProcedureStep).join("")}</div>` : ""}
          ${renderProcedureBody(step.body ?? null)}
        </div>
      </details>
    `;
  }

  if ("procedure_node_id" in step) {
    const procedureNodeId =
      typeof step.procedure_node_id === "string" ? step.procedure_node_id : "unknown node";
    return `
      <div class="step-card">
        <div class="step-head">
          ${badge("procedure ref", "teal")}
          <strong>${escapeHtml(procedureNodeId)}</strong>
        </div>
        <p class="summary">References a named procedure node defined elsewhere in the model.</p>
      </div>
    `;
  }

  const refCount = Array.isArray((step as { activity_refs?: unknown }).activity_refs)
    ? ((step as { activity_refs: unknown[] }).activity_refs.length)
    : 0;

  return `
    <div class="step-card">
      <div class="step-head">
        ${badge(`${refCount} refs`, "slate")}
        <strong>${escapeHtml(step.name ?? "Abstract step")}</strong>
      </div>
      ${
        step.description
          ? `<p class="summary">${escapeHtml(step.description)}</p>`
          : `<p class="empty">No description.</p>`
      }
    </div>
  `;
}

function renderProcedureModel(model: ProcedureModel | null): string {
  if (!model) return `<p class="empty">Procedure model not available.</p>`;

  const nodes = model.procedure_nodes
    .map(
      (node) => `
        <details class="tree-node" ${node.id === model.root_procedure_id ? "open" : ""}>
          <summary>
            <span class="tree-title">${escapeHtml(node.name || node.id)}</span>
            ${badge(node.operator, "teal")}
            ${badge(`${node.activity_refs.length} refs`, "amber")}
          </summary>
          <div class="tree-body">
            <p class="summary">${escapeHtml(node.description || "No description available.")}</p>
            ${
              node.condition
                ? `<p><strong>Condition:</strong> ${escapeHtml(node.condition)}</p>`
                : ""
            }
            ${
              node.evidence_summary
                ? `<p><strong>Evidence:</strong> ${escapeHtml(node.evidence_summary)}</p>`
                : ""
            }
            ${renderProcedureBody(node.body)}
          </div>
        </details>
      `,
    )
    .join("");

  return `<div class="stack">${nodes}</div>`;
}

function aggregateUnifiedRefs(node: UnifiedModelNode, refs = new Set<string>()): Set<string> {
  for (const ref of expandSegmentRefs(node.activity_refs)) refs.add(ref);
  for (const child of node.decomposition) aggregateUnifiedRefs(child, refs);
  return refs;
}

function unifiedOperator(node: UnifiedModelNode): { label: string; tone: BadgeTone } {
  switch (node.procedure.operator) {
    case "FOR": {
      const variable = node.procedure.bindings?.iteration_variable;
      return {
        label: variable ? `For each ${String(variable).replaceAll("_", " ")}` : "For each item",
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
      return { label: "In order", tone: "sky" };
  }
}

type ThreadRailTone =
  | "indigo"
  | "emerald"
  | "rose"
  | "amber"
  | "sky"
  | "fuchsia"
  | "violet"
  | "teal"
  | "orange"
  | "stone";

function threadRailTone(threadId: string): ThreadRailTone {
  const tones: Record<string, ThreadRailTone> = {
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

function renderUnifiedNode(
  node: UnifiedModelNode,
  depth = 0,
  railTone: ThreadRailTone = "stone",
): string {
  const isRoot = depth === 0;
  const children = node.decomposition ?? [];
  const operator = unifiedOperator(node);
  const bodySteps = node.procedure.body ?? [];
  const aggregateRefCount = aggregateUnifiedRefs(node).size;
  const collection = node.procedure.bindings?.collection;
  const collectionItems = Array.isArray(collection)
    ? collection
        .map((item) => `<span class="value-chip">${escapeHtml(String(item))}</span>`)
        .join("")
    : "";

  const detailParts = [
    node.summary ? `<p class="unified-summary">${escapeHtml(node.summary)}</p>` : "",
    node.procedure.description && node.procedure.description !== node.summary
      ? `<p class="unified-copy">${escapeHtml(node.procedure.description)}</p>`
      : "",
    node.procedure.operator === "FOR" && collectionItems
      ? `<section class="detail-section detail-section-wide">
          <h4>Runs once for each item</h4>
          <div class="value-list">${collectionItems}</div>
        </section>`
      : "",
    node.procedure.condition
      ? `<section class="detail-section detail-section-wide">
          <h4>Repeats until</h4>
          <p>${escapeHtml(node.procedure.condition)}</p>
        </section>`
      : "",
    bodySteps.length
      ? `<section class="detail-section">
          <h4>How it was done</h4>
          <ol class="body-steps">${bodySteps
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
      ? `<section class="detail-section">
          <h4>Outcome</h4>
          <p>${escapeHtml(node.observed_outcome.description)}</p>
        </section>`
      : "",
    node.procedure.evidence_summary
      ? `<section class="detail-section detail-section-wide">
          <h4>Evidence</h4>
          <p>${escapeHtml(node.procedure.evidence_summary)}</p>
        </section>`
      : "",
  ].filter(Boolean);

  const renderedChildren = children
    .map((child) => renderUnifiedNode(child, depth + 1, railTone))
    .join("");
  const childrenRegion = !children.length
    ? ""
    : isRoot
      ? `<div class="unified-children root-children">${renderedChildren}</div>`
      : `<details class="child-disclosure">
          <summary>
            <span class="disclosure-icon" aria-hidden="true"></span>
            <span class="show-label">Show</span><span class="hide-label">Hide</span>
            ${children.length} child ${children.length === 1 ? "step" : "steps"}
          </summary>
          <div class="unified-children">${renderedChildren}</div>
        </details>`;

  return `
    <div class="unified-node">
      <article class="unified-card rail-${railTone}">
        <div class="unified-card-head">
          <div class="unified-heading-wrap">
            <h3>${escapeHtml(isRoot ? "Task outline" : node.objective)}</h3>
            <div class="unified-meta">
              ${isRoot ? badge("Overview", "violet") : badge(operator.label, operator.tone)}
              <span>${aggregateRefCount} ${aggregateRefCount === 1 ? "step" : "steps"}</span>
            </div>
          </div>
        </div>
        <div class="unified-card-body ${isRoot ? "overview-body" : ""}">
          ${detailParts.length ? detailParts.join("") : `<p class="empty">No additional details recorded.</p>`}
        </div>
      </article>
      ${childrenRegion}
    </div>
  `;
}

type StaticReportOptions = {
  activeThreadId?: string;
  unifiedOnly?: boolean;
};

function renderThread(thread: ThreadBundle, options: StaticReportOptions): string {
  const models = options.unifiedOnly
    ? `<article class="panel panel-wide unified-panel">
        <div class="panel-heading">
          <div>
            <div class="eyebrow">Unified result</div>
            <h3>What was done</h3>
          </div>
          <span class="panel-note">Titles and details stay together. Controls only show or hide child steps.</span>
        </div>
        ${
          thread.unifiedModel
            ? renderUnifiedNode(thread.unifiedModel.root, 0, threadRailTone(thread.id))
            : `<p class="empty">Unified model not available.</p>`
        }
      </article>`
    : `<article class="panel">
        <h3>Objective model</h3>
        ${
          thread.objectiveModel
            ? renderObjectiveNode(thread.objectiveModel)
            : `<p class="empty">Objective model not available.</p>`
        }
      </article>
      <article class="panel">
        <h3>Procedure model</h3>
        ${renderProcedureModel(thread.procedureModel)}
      </article>
      <article class="panel panel-wide">
        <h3>Unified model</h3>
        ${
          thread.unifiedModel
            ? renderUnifiedNode(thread.unifiedModel.root, 0, threadRailTone(thread.id))
            : `<p class="empty">Unified model not available.</p>`
        }
      </article>`;

  return `
    <section class="thread-section" id="thread-${escapeHtml(thread.id)}">
      <div class="section-header">
        <div>
          <div class="eyebrow">Overall goal</div>
          <h2>${escapeHtml(thread.label)}</h2>
          <p class="summary">${escapeHtml(thread.task_thread_objective)}</p>
        </div>
        <div class="badge-row">
          ${badge(thread.id, threadRailTone(thread.id))}
          ${badge(`${thread.localObjectiveIds.length} activities`, "amber")}
        </div>
      </div>
      <div class="panel-grid">
        ${models}
      </div>
    </section>
  `;
}

function renderActivities(
  data: SessionData,
  includedThreadIds: Set<string>,
  includeAdditionalContext: boolean,
): string {
  return data.activities
    .map((activity, index) => {
      const threadId = data.threadByObjective[activity.activity_id] ?? "unassigned";
      if (!includedThreadIds.has(threadId)) return "";
      return `
        <tr>
          <td>${index + 1}</td>
          <td>${badge(threadId, threadRailTone(threadId))}</td>
          <td><code>${escapeHtml(activity.activity_id)}</code></td>
          <td>${escapeHtml(activity.objective)}</td>
          ${includeAdditionalContext ? `<td>${escapeHtml(activity.additional_context || "")}</td>` : ""}
          <td>${escapeHtml(String(activity.event_count))}</td>
        </tr>
      `;
    })
    .join("");
}

export function exportedHtmlFilename(data: SessionData): string {
  const base = path.basename(data.resolvedDir).replace(/[^a-zA-Z0-9._-]+/g, "-");
  return `${base || "task-trace-explorer"}.html`;
}

export function renderStandaloneReport(
  data: SessionData,
  options: StaticReportOptions = {},
): string {
  const activeThread = options.activeThreadId
    ? data.threads.find((thread) => thread.id === options.activeThreadId)
    : undefined;
  const includedThreads = activeThread ? [activeThread] : data.threads;
  const includedThreadIds = new Set(includedThreads.map((thread) => thread.id));
  const includedActivityCount = data.activities.filter((activity) =>
    includedThreadIds.has(data.threadByObjective[activity.activity_id] ?? "unassigned"),
  ).length;
  const sessionName = path.basename(data.resolvedDir) || "Task trace";
  const includeAdditionalContext = !options.unifiedOnly;
  const threadNav = includedThreads
    .map(
      (thread) => `
        <a class="nav-chip" href="#thread-${escapeHtml(thread.id)}">
          ${escapeHtml(thread.id)} · ${escapeHtml(thread.label)}
        </a>
      `,
    )
    .join("");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(activeThread?.label ?? sessionName)} · Task result</title>
    <style>
      :root {
        --bg: #f6f3ed;
        --surface: #fffdf9;
        --surface-2: #f1ece2;
        --text: #1f2937;
        --muted: #6b7280;
        --border: #ddd6c8;
        --blue: #1d4ed8;
        --blue-bg: #dbeafe;
        --amber: #b45309;
        --amber-bg: #fef3c7;
        --teal: #0f766e;
        --teal-bg: #ccfbf1;
        --sky: #0369a1;
        --sky-bg: #e0f2fe;
        --rose: #be123c;
        --rose-bg: #ffe4e6;
        --violet: #6d28d9;
        --violet-bg: #ede9fe;
        --slate: #475569;
        --slate-bg: #e2e8f0;
      }
      * { box-sizing: border-box; }
      html { scroll-behavior: smooth; }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: linear-gradient(180deg, #f8f5ee 0%, #f3efe7 100%);
        color: var(--text);
      }
      .page {
        max-width: 1440px;
        margin: 0 auto;
        padding: 28px;
      }
      .hero, .panel, .trace-panel {
        background: rgba(255, 253, 249, 0.92);
        border: 1px solid var(--border);
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
      }
      .hero {
        padding: 24px;
        position: sticky;
        top: 0;
        backdrop-filter: blur(14px);
        z-index: 10;
      }
      h1, h2, h3 { margin: 0; }
      h1 { font-size: 28px; }
      h2 { font-size: 22px; }
      h3 { font-size: 16px; margin-bottom: 12px; }
      p { margin: 0; line-height: 1.5; }
      .summary { color: var(--muted); margin-top: 8px; }
      .hero-grid, .meta-grid, .panel-grid {
        display: grid;
        gap: 16px;
      }
      .hero-grid, .meta-grid { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-top: 18px; }
      .meta-grid { margin-top: 12px; }
      .meta-card {
        background: var(--surface-2);
        border-radius: 14px;
        padding: 14px;
      }
      .meta-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
      }
      .nav-row, .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
      }
      .nav-chip, .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 600;
        text-decoration: none;
      }
      .nav-chip {
        color: var(--text);
        background: white;
        border: 1px solid var(--border);
      }
      .badge-slate { color: var(--slate); background: var(--slate-bg); }
      .badge-amber { color: var(--amber); background: var(--amber-bg); }
      .badge-blue { color: var(--blue); background: var(--blue-bg); }
      .badge-indigo { color: #4338ca; background: #e0e7ff; }
      .badge-emerald { color: #047857; background: #d1fae5; }
      .badge-teal { color: var(--teal); background: var(--teal-bg); }
      .badge-sky { color: var(--sky); background: var(--sky-bg); }
      .badge-rose { color: var(--rose); background: var(--rose-bg); }
      .badge-violet { color: var(--violet); background: var(--violet-bg); }
      .badge-fuchsia { color: #a21caf; background: #fae8ff; }
      .badge-orange { color: #c2410c; background: #ffedd5; }
      .badge-stone { color: #57534e; background: #f5f5f4; }
      .trace-panel {
        margin-top: 18px;
        padding: 20px;
        overflow: auto;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        min-width: 900px;
      }
      th, td {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #ebe5da;
        vertical-align: top;
      }
      th {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      code {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12px;
      }
      .thread-section {
        margin-top: 22px;
      }
      .section-header {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
        margin-bottom: 14px;
      }
      .panel-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .panel {
        padding: 18px;
      }
      .panel-wide {
        grid-column: 1 / -1;
      }
      .eyebrow {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.11em;
        margin-bottom: 7px;
        text-transform: uppercase;
      }
      .panel-heading {
        align-items: start;
        display: flex;
        gap: 20px;
        justify-content: space-between;
        margin-bottom: 16px;
      }
      .panel-note {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
        max-width: 380px;
        text-align: right;
      }
      .unified-panel { padding: 22px; }
      .unified-node { margin-top: 12px; }
      .unified-card {
        background: #fff;
        border: 1px solid #e7e5e4;
        border-radius: 14px;
        box-shadow: 0 1px 2px rgba(28, 25, 23, 0.06);
        overflow: hidden;
        position: relative;
      }
      .unified-card::before {
        bottom: 0;
        content: "";
        left: 0;
        position: absolute;
        top: 0;
        width: 5px;
        z-index: 2;
      }
      .rail-indigo::before { background: #6366f1; }
      .rail-emerald::before { background: #10b981; }
      .rail-rose::before { background: #f43f5e; }
      .rail-amber::before { background: #f59e0b; }
      .rail-sky::before { background: #0ea5e9; }
      .rail-fuchsia::before { background: #d946ef; }
      .rail-violet::before { background: #8b5cf6; }
      .rail-teal::before { background: #14b8a6; }
      .rail-orange::before { background: #f97316; }
      .rail-stone::before { background: #a8a29e; }
      .unified-card-head {
        align-items: start;
        display: flex;
        gap: 12px;
        padding: 14px 18px 12px 22px;
      }
      .unified-heading-wrap { min-width: 0; width: 100%; }
      .unified-card h3 {
        color: #292524;
        font-size: 15px;
        line-height: 1.45;
        margin: 0;
      }
      .unified-meta {
        align-items: center;
        color: var(--muted);
        display: flex;
        flex-wrap: wrap;
        font-size: 12px;
        gap: 8px;
        margin-top: 8px;
      }
      .unified-card-body {
        background: rgba(250, 250, 249, 0.72);
        border-top: 1px solid #f0eeeb;
        display: grid;
        gap: 15px 24px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        padding: 15px 18px 17px 22px;
      }
      .unified-summary, .unified-copy, .detail-section-wide {
        grid-column: 1 / -1;
      }
      .unified-summary {
        color: #57534e;
        font-size: 14px;
        line-height: 1.6;
      }
      .unified-copy {
        color: #78716c;
        font-size: 13px;
      }
      .detail-section h4 {
        color: #78716c;
        font-size: 11px;
        letter-spacing: 0.1em;
        margin: 0 0 7px;
        text-transform: uppercase;
      }
      .detail-section p {
        color: #57534e;
        font-size: 13px;
      }
      .value-list { display: flex; flex-wrap: wrap; gap: 6px; }
      .value-chip {
        background: var(--amber-bg);
        border-radius: 7px;
        color: var(--amber);
        font-size: 12px;
        padding: 5px 8px;
      }
      .body-steps {
        color: #57534e;
        display: grid;
        font-size: 13px;
        gap: 7px;
        margin: 0;
        padding-left: 20px;
      }
      .body-steps li { padding-left: 3px; }
      .body-steps small {
        color: var(--muted);
        display: block;
        line-height: 1.45;
        margin-top: 2px;
      }
      .unified-children {
        border-left: 1px solid #dedbd5;
        margin: 10px 0 0 24px;
        padding-left: 16px;
      }
      .root-children { margin-top: 14px; }
      .child-disclosure { margin: 8px 0 0 18px; }
      .child-disclosure > summary {
        align-items: center;
        border-radius: 8px;
        color: #57534e;
        cursor: pointer;
        display: inline-flex;
        font-size: 12px;
        font-weight: 650;
        gap: 5px;
        list-style: none;
        padding: 6px 9px;
      }
      .child-disclosure > summary:hover { background: #ece9e4; }
      .child-disclosure > summary::-webkit-details-marker { display: none; }
      .disclosure-icon {
        border-bottom: 4px solid transparent;
        border-left: 6px solid #78716c;
        border-top: 4px solid transparent;
        height: 0;
        margin-right: 2px;
        transition: transform 120ms ease;
        width: 0;
      }
      .child-disclosure[open] .disclosure-icon { transform: rotate(90deg); }
      .hide-label { display: none; }
      .child-disclosure[open] .show-label { display: none; }
      .child-disclosure[open] .hide-label { display: inline; }
      .trace-disclosure { margin-top: 20px; }
      .trace-disclosure > summary {
        background: rgba(255, 253, 249, 0.92);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: #44403c;
        cursor: pointer;
        font-size: 13px;
        font-weight: 650;
        list-style: none;
        padding: 12px 15px;
      }
      .trace-disclosure > summary::-webkit-details-marker { display: none; }
      .tree-node, .step-card {
        border: 1px solid #e8e0d2;
        border-radius: 14px;
        background: #fff;
        margin-top: 10px;
      }
      .tree-node > summary, .step-head {
        list-style: none;
        cursor: pointer;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        padding: 12px 14px;
      }
      .tree-node > summary::-webkit-details-marker { display: none; }
      .tree-title {
        font-weight: 700;
      }
      .tree-body {
        padding: 0 14px 14px;
      }
      .step-list, .stack {
        display: grid;
        gap: 10px;
      }
      .chip-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 0;
        margin: 12px 0 0;
        list-style: none;
      }
      .chip-list li {
        padding: 6px 10px;
        border-radius: 999px;
        background: #f4f4f5;
        font-size: 12px;
      }
      .empty {
        color: var(--muted);
        font-style: italic;
      }
      @media (max-width: 960px) {
        .page { padding: 16px; }
        .hero { position: static; }
        .panel-grid { grid-template-columns: 1fr; }
        .section-header { flex-direction: column; }
        .panel-heading { flex-direction: column; }
        .panel-note { text-align: left; }
        .unified-card-body { grid-template-columns: 1fr; }
        .detail-section { grid-column: 1 / -1; }
        .unified-children { margin-left: 13px; padding-left: 10px; }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <h1>Task result</h1>
        <p class="summary">A self-contained report for ${escapeHtml(activeThread?.label ?? sessionName)}. This file opens on its own and can be shared directly.</p>
        <div class="hero-grid">
          ${activeThread ? "" : `<div class="meta-card">
            <div class="meta-label">Session</div>
            <div>${escapeHtml(sessionName)}</div>
          </div>`}
          <div class="meta-card">
            <div class="meta-label">Activities</div>
            <div>${includedActivityCount}</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Task ${includedThreads.length === 1 ? "thread" : "threads"}</div>
            <div>${includedThreads.length}</div>
          </div>
        </div>
        ${includedThreads.length > 1 ? `<div class="nav-row">${threadNav}</div>` : ""}
      </section>

      ${includedThreads.map((thread) => renderThread(thread, options)).join("")}

      <details class="trace-disclosure">
        <summary>View supporting activity trace · ${includedActivityCount} activities</summary>
        <section class="trace-panel">
          <h2>Activity trace</h2>
          <p class="summary">The chronological observations behind this result.</p>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Thread</th>
                <th>Activity ID</th>
                <th>Objective</th>
                ${includeAdditionalContext ? "<th>Additional context</th>" : ""}
                <th>Events</th>
              </tr>
            </thead>
            <tbody>${renderActivities(data, includedThreadIds, includeAdditionalContext)}</tbody>
          </table>
        </section>
      </details>
    </div>
  </body>
</html>`;
}
