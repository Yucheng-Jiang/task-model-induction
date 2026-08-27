# Task Model Visualizer

A local Next.js app for reading the task models the pipeline produces. It reads
a session directory straight off disk — no upload, no database.

```bash
npm install
npm run dev
```

Open `localhost:3000` and enter a session path, or link to one directly:

```
http://localhost:3000/?dir=~/Downloads/recorder_sessions/session_001
```

Set `SESSION_DIR` to change the default the page opens with.

## What it shows

Pick a task thread, then read it in one of two views. Both keep the trace
alongside the model, and selecting a node highlights the activities that
evidence it — the fastest way to check whether an induced claim is actually
supported by what happened.

**Unified** (default) pairs the trace with stage 6's reconciled model:

| Column | Source |
|---|---|
| **Trace** | `activity.jsonl` — the activities, in order |
| **Unified** | `task_thread_task_model/` — objectives with control-flow annotations |

**Pre-reconciliation** puts stages 4 and 5 side by side, so you can see what
alignment changed:

| Column | Source |
|---|---|
| **Trace** | `activity.jsonl` |
| **Objective** | `task_thread_objective_model/` — the goal decomposition |
| **Procedure** | `task_thread_procedure_model/` — the control flow |

## Exporting

**Share HTML**, in the toolbar, inlines the whole explorer into a single file
with every thread tab. It opens anywhere, no server needed.

`POST /export-folder` returns the same explorer as a portable multi-file bundle
for hosting. It has no toolbar button — call the route directly with a JSON body
of `{ dir, exportState }`.

## Requirements

The session must have been through at least stage 3, since the explorer loads
its thread list from `derived_task_thread_objectives/manifest.json`. Objective,
procedure, and unified columns fill in as stages 4, 5, and 6 complete, so a
partially processed session renders fine.

Needs Node 18+.
