#!/usr/bin/env python3
"""Induce procedure models for derived task-thread objective roots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from task_model_induction.codex_cli_sandbox import CodexCliSandbox
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.reporting.model_readable_report import write_procedure_collection_markdown, write_procedure_markdown
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.schemas import ProcedureModelInductionOutput, ProcedureTaskModel
    from task_model_induction.utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        estimate_litellm_usage_cost_usd,
        litellm_completion,
        direct_llm_input_too_large,
        litellm_model_config,
        normalize_litellm_usage,
        run_with_retries,
        utc_now_iso,
        write_json_atomic,
    )
    from task_model_induction.validate.validate_procedure_model import procedure_schema, validate_procedure_output
except ModuleNotFoundError:
    from codex_cli_sandbox import CodexCliSandbox
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.model_readable_report import write_procedure_collection_markdown, write_procedure_markdown
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import ProcedureModelInductionOutput, ProcedureTaskModel
    from utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        estimate_litellm_usage_cost_usd,
        litellm_completion,
        direct_llm_input_too_large,
        litellm_model_config,
        normalize_litellm_usage,
        run_with_retries,
        utc_now_iso,
        write_json_atomic,
    )
    from validate.validate_procedure_model import procedure_schema, validate_procedure_output


STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_GENERATION = "procedure generation"
STAGE_MERGE = "merge output"
STAGES = [STAGE_LOAD_INPUTS, STAGE_PREFLIGHT, STAGE_GENERATION, STAGE_MERGE]


def procedure_output_schema_text() -> str:
    return json.dumps(procedure_schema(), indent=2, ensure_ascii=True)


def validator_script_text() -> str:
    return (Path(__file__).resolve().parent / "validate" / "validate_procedure_model.py").read_text(encoding="utf-8")


def while_condition_grounding_guidance() -> str:
    return """### Grounded WHILE exit conditions

Every WHILE must make its stopping rule operational. A phrase such as "until done",
"until satisfied", "until the UI is acceptable", or "until it works" is invalid.
Provide both fields below:

- `condition`: a positive, binary objective-state predicate. Do not prefix it with
  "until" or describe the iteration process.
- `condition_grounding`: exactly
  `{"predicate": <same text as condition>, "verifier": <concrete check and expected signal>, "evidence_refs": [<activity refs inside this WHILE>], "observed_status": "satisfied|unsatisfied|unknown"}`.

`condition_grounding.predicate` must exactly equal `condition`. The verifier must say
how an observer can decide the predicate from a command result, artifact, UI state, or
user-visible outcome. Evidence refs must point to the loop activities that show the
final check or the best available evidence. Use `satisfied` only when the trace shows
the predicate becoming true; use `unsatisfied` when it remains false, and `unknown`
when the trace ends without establishing either state.

Example: instead of "until the example data is ready", use condition "The
`example_data` folder contains the extracted DataSTORM dataset and the dataset entry is
visible in the workspace", verifier "Inspect `example_data` in the Explorer and confirm
the named dataset entry appears", evidence refs covering the unzip and final Explorer
inspection, and observed_status `satisfied`."""


def procedure_induction_prompt(
    *,
    input_file: str = "input/task_thread_objectives.json",
    schema_file: str = "input/procedure_model_schema.json",
    validator_file: str = "input/validate_procedure_model.py",
    output_file: str = "output/procedure_model.json",
) -> str:
    return f"""You induce a procedure model from a task-thread objective JSON by applying the Structured Programming Theorem (Böhm–Jacopini, 1966).

## Theoretical grounding

The Structured Programming Theorem proves that any computable procedure can be expressed using exactly three control constructs:
1. **Sequence** — steps executed one after another in order.
2. **Selection** — a choice between mutually exclusive alternative paths.
3. **Iteration** — a body repeated either over a named collection (for-each) or until a condition holds (while).

Decompose the observed activity trace into a tree built from these constructs. Every repetition or branching in the trace must be captured by the appropriate construct — not collapsed into a flat sequence. A flat SEQ where iteration or selection is evident is a modeling error.

## Input files
- `{input_file}`: task-thread objective JSON with `canonical_root_id`, `task_thread_objective`, and `activities` records. Each `activities` entry is one activity episode — the atomic unit of trace analysis.
- `{schema_file}`: the exact JSON schema for the required output.
- `{validator_file}`: deterministic validator you must run before finishing.

## Required top-level output
{{
  "version": "0.1",
  "root_procedure_id": "<id of the root procedure node>",
  "procedure_nodes": []
}}

## The four operators

| Operator | Construct       | When to use |
|----------|-----------------|-------------|
| `SEQ`    | Sequence        | Steps in fixed order with no repetition and no branching. Use only when no other operator applies. |
| `FOR`    | For-each iteration | The same body applied to each member of a named collection (files, accounts, stages, artifacts, users, …). |
| `WHILE`  | While iteration | A body repeated until an observable objective-state condition is satisfied. |
| `CHOICE` | Selection       | Mutually exclusive alternatives; only one branch observed. |

**Iteration is the default for any repetition.** FOR for named collections; WHILE for condition-driven loops. Using SEQ to represent repetition is incorrect.

Activity leaves (only valid inside SEQ and CHOICE bodies):
- `{{"activity_id": "activity_0003"}}` — name and description filled automatically; do not invent them.

FOR and WHILE bodies are ABSTRACT TEMPLATES — named steps with `name`, `description`, and `activity_refs`. NO `activity_id` leaves inside FOR or WHILE bodies. Each abstract step must explicitly map to the activity episodes it covers across all iterations/passes via its own `activity_refs`. The node-level `activity_refs` covers the union across all step-level refs (all episodes in the node).

Operator selection rules:
- `FOR`: the same body applies to each of several named items (files, accounts, stages, artifacts). Write the body as an abstract SEQ of named steps. Each abstract step must have `activity_refs` listing ALL activities from ALL iterations where that step occurred. The node-level `activity_refs` is the union across all steps.
- `WHILE`: body repeats ≥ 2 times in the trace until condition is satisfied. Write the body as a flat abstract SEQ of named steps (minimal repeating actions per pass). Each abstract step must have `activity_refs` listing ALL activities from ALL passes where that step occurred. The node-level `activity_refs` is the union across all steps. Do NOT nest FOR or WHILE inside a WHILE body unless the full inner iteration genuinely repeats every pass. If an initial FOR traversal precedes a repair loop, model as `SEQ [ FOR, WHILE ]` not `WHILE [ FOR ]`.
- `SEQ`: ordered, non-repeating steps; body may contain activity_id leaves and/or nested constructs.
- `CHOICE`: mutually exclusive alternatives; only one branch observed.

{while_condition_grounding_guidance()}

Each procedure node must include:
- `operator` from the four primitives only.
- `bindings`: for `FOR`, use exactly `{{"iteration_variable": "<name>", "collection": ["<item>", ...]}}`. Omit it for other operators.
- `body`: for `SEQ`, `FOR`, `WHILE` — the child steps or body procedure.
- `condition` and `condition_grounding`: for `WHILE` — the exact observable exit predicate and its verifier, trace evidence, and observed status.
- `dataflow` and `effects` when useful.
- `activity_refs`: episode IDs from the source data. IDs use the format `activity_NNNN`. Use compact ranges like `activity_0000-activity_0004` for contiguous episodes. For FOR/WHILE, must span all iterations/passes.
- `evidence_summary`.

Coverage rules:
- Every activity episode in the input must be covered by at least one procedure node's `activity_refs` or inline `activity_id` leaf.
- Composite nodes cover the union of their children's episodes.
- Prefer one primary owning node per episode.

Workflow:
1. Read `{input_file}` and `{schema_file}`.
2. Write the candidate JSON to `{output_file}`.
3. Run `python {validator_file} {output_file} {input_file} --text`.
4. If validation fails, use the feedback to repair `{output_file}` and rerun validation.
5. Ensure full activity episode coverage via `procedure_nodes[].activity_refs`.
6. **FOR scan**: after validation passes, scan each SEQ node's body steps for missed FOR patterns. Look at consecutive steps in order — if 2+ named items (accounts, users, files, stages, etc.) are processed with the same steps, convert that group to a FOR node with explicit `bindings`. Rerun validation after any restructuring.
7. Finish only after validation passes.

Write only the final JSON file at `{output_file}`. The final answer can be a short note that validation passed."""


def procedure_generation_prompt_text() -> str:
    return f"""You induce a procedure model from a task-thread objective JSON by applying the Structured Programming Theorem (Böhm–Jacopini, 1966).
Return only a valid JSON object.

## Theoretical grounding

The Structured Programming Theorem proves that any computable procedure can be expressed using exactly three control constructs:
1. **Sequence** — steps executed one after another in order.
2. **Selection** — a choice between mutually exclusive alternative paths.
3. **Iteration** — a body repeated either over a named collection (for-each) or until a condition holds (while).

Your task is to decompose the observed activity trace into a tree built from these constructs. Every branching, repetition, or iteration in the trace must be captured by the appropriate construct — not collapsed into a flat sequence. A flat SEQ where iteration or selection is evident is a modeling error.

## Input

A sequence of activity episodes — the atomic units of trace analysis. Each episode corresponds to one `activities` entry in the source data.

## Required top-level output
{{
  "version": "0.1",
  "root_procedure_id": "<id of the root procedure node>",
  "procedure_nodes": []
}}

## The four operators

These four operators are the complete and closed vocabulary — no others are permitted:

| Operator | Construct       | When to use |
|----------|-----------------|-------------|
| `SEQ`    | Sequence        | Steps that occur in a fixed order with no repetition and no branching. Use only when no other operator applies. |
| `FOR`    | For-each iteration | The same procedure body is applied to each member of a named, enumerable collection (files, accounts, workflow phases, artifacts, users, …). |
| `WHILE`  | While iteration | A body is repeated until an observable objective-state condition is satisfied (tests pass, reviewer can sign in, UI is stable, …). |
| `CHOICE` | Selection       | Mutually exclusive alternative paths to the same goal; only one branch is observed in the trace. |

**Iteration is the default for any repetition.** If the trace shows the same steps applied to multiple items, that is FOR. If the trace shows steps repeated until a condition holds, that is WHILE. Representing either of these as SEQ is incorrect.

Activity leaves (only valid inside SEQ and CHOICE bodies):
- `{{"activity_id": "activity_0003"}}` — name and description are filled in automatically; do not invent them.

FOR and WHILE bodies are ABSTRACT TEMPLATES — they describe what happens per item / per pass using named steps with `name`, `description`, and `activity_refs`. Do NOT place activity_id leaves inside a FOR or WHILE body. Each abstract step must explicitly map to the activity episodes it covers across all iterations/passes via its own `activity_refs`. The node-level `activity_refs` covers the union across all step-level refs.

Operator selection — for each node, identify which operator the trace evidence best supports:

- **FOR**: The same procedure body applies to each item in a named collection. Signal patterns:
  - Multiple named files, accounts, stages, or artifacts processed the same way (e.g., "open the CSV, markdown, JSON, and report files" → FOR file_type in [CSV, markdown, JSON, report])
  - Multiple named users authenticated or configured with the same steps
  - Multiple workflow phases or named stages processed with the same prepare/execute/verify pattern
  - Any repeated sub-task applied to each member of a named set
  Use FOR with `bindings.iteration_variable` naming the variable and `bindings.collection` enumerating the items. Write the body as an abstract SEQ of named steps (no activity_id leaves). Each abstract step must have `activity_refs` listing ALL activities from ALL iterations where that step occurred. The node-level `activity_refs` is the union across all step-level refs.

- **WHILE**: The body is a single abstract pass that **must be observed to repeat ≥ 2 times** in the trace — each pass makes progress toward the exit condition. Signal patterns:
  - Run → observe failure → fix → rerun (the fix-rerun cycle repeats)
  - Configuration attempt → verify → adjust → retry (adjustment cycle repeats)
  - Login or setup attempt repeated until access is confirmed
  Write the body as a flat abstract SEQ of named steps (the minimal repeating actions per pass). Each abstract step must have `activity_refs` listing ALL activities from ALL passes where that step occurred. State the condition in objective terms. The node-level `activity_refs` is the union across all step-level refs.

  **Critical nesting rule**: Do NOT place a FOR or another WHILE directly inside a WHILE body unless the full iteration genuinely repeats in every single pass.
  - If the trace shows: an initial collection traversal (FOR) followed by a separate repair loop (WHILE), model them as **sequential siblings**: `SEQ [ FOR(initial-traversal), WHILE(repair-loop) ]`.
  - If the trace shows: every repair pass re-traverses the full collection, then `WHILE(FOR(...))` is correct — but the inner FOR must have complete `bindings` with the collection enumerated.
  - An unnamed or unbound FOR inside a WHILE body is always wrong.

- **CHOICE**: Mutually exclusive alternative paths to the same goal; only one branch observed.

- **SEQ**: Ordered, non-repeating steps with no iteration and no branching. Use only when no other operator fits. SEQ bodies may contain activity_id leaves and/or nested control constructs.

{while_condition_grounding_guidance()}

Each procedure node must include:
- `operator`: one of the four primitives.
- `bindings`: for `FOR`, use exactly `{{"iteration_variable": "<name>", "collection": ["<item>", ...]}}`. Omit it for other operators.
- `body`: for `SEQ`, `FOR`, `WHILE` — specify child steps or the loop body. Steps may be control constructs or activity leaves.
- `condition` and `condition_grounding`: for `WHILE` — the exact observable exit predicate and its verifier, trace evidence, and observed status.
- `dataflow` and `effects` when causally meaningful.
- `activity_refs`: episode IDs from the source `activities` list. Use the ID format from the source data (e.g., `activity_0003`). Use compact ranges (e.g., `activity_0003-activity_0007`) for composite nodes.
- `evidence_summary`: one or two sentences grounding the node in specific trace evidence.

Coverage rules:
- Every activity episode in the input must appear in at least one node's `activity_refs` or as an inline `activity_id` leaf.
- Composite nodes cover the union of their children's episodes.
- Prefer one primary owning node per episode.

Required output schema:
{procedure_output_schema_text()}
"""


def procedure_for_scan_prompt_text() -> str:
    return f"""You are scanning a validated procedure model for missed FOR patterns inside SEQ nodes. Return only a valid JSON object — the corrected model, or the original if nothing needs changing.

A FOR pattern is missed when a SEQ body contains consecutive steps that apply the same procedure to each item in a named, enumerable collection — but the model treats them as a plain sequence instead.

## What to look for (scan each SEQ node's body steps in order)

Signal: two or more named items (accounts, files, workflow phases, artifacts, servers) each processed with the same sequence of steps, one item after another.

Concrete examples:
- resource configured for item A → tested as A → resource configured for item B → tested as B → FOR item in [A, B]
- Phase 1 walked through with prepare/execute/verify → Phase 2 walked through the same way → FOR phase in [phase1, phase2]
- CSV file opened/processed → JSON file opened/processed the same way → FOR file_type in [CSV, JSON]

## Rules

- Only convert when the collection is explicitly named and enumerable (you can list the items).
- The same abstract steps must apply to EVERY item — not just loosely similar steps.
- A collection of ≥ 2 named items qualifies.
- Convert only the identified consecutive steps to a FOR node; leave the rest of the SEQ unchanged.
- FOR body must be abstract named steps with `activity_refs` spanning ALL items — no `activity_id` leaves inside the FOR body.
- FOR requires exactly `bindings.iteration_variable` and a non-empty, explicitly enumerated `bindings.collection`.
- Node-level `activity_refs` for the FOR node = union of all step-level refs across all items.
- If the surrounding SEQ referenced those steps via `activity_id` leaves, replace the leaf group with a reference to the new FOR node.
- If you find nothing to convert, return the model unchanged.

Required output schema:
{procedure_output_schema_text()}
"""


def procedure_repair_prompt_text() -> str:
    return f"""Your previous procedure-model output failed deterministic validation.
Return only a valid JSON object.

Repair the procedure model using the original input, your previous result, and the validation feedback.

Repair checklist:
- Every activity episode in the input must appear in at least one node's `activity_refs` or as an inline `activity_id` leaf.
- All operators must be one of: `SEQ`, `FOR`, `WHILE`, `CHOICE`. No ACT, no other operators or macro labels.
- Leaf activities `{{"activity_id": "activity_0003"}}` — no `operator`, no `name`/`description` — are only valid inside SEQ and CHOICE bodies.
- FOR and WHILE bodies must NOT contain activity_id leaves. Use abstract named steps (`name`, `description`, and `activity_refs`) inside FOR/WHILE bodies.
- Each abstract step inside a FOR/WHILE body must have `activity_refs` listing ALL activities from ALL iterations/passes where that step occurred. A step with `name` and `description` but no `activity_refs` is a modeling error.
- `FOR` nodes must have exactly `bindings.iteration_variable` and a non-empty, explicitly enumerated `bindings.collection`. Node-level `activity_refs` is the union of all step-level refs across all iterations.
- `WHILE` nodes must have an operational `condition` plus `condition_grounding` with the same predicate, a concrete verifier, in-loop evidence refs, and `observed_status`. Reject phrases such as "until done", "until satisfied", "acceptable", or "works reliably". Node-level `activity_refs` is the union of all step-level refs across all passes. WHILE body must be a flat abstract SEQ — do NOT nest FOR or WHILE inside a WHILE unless the full inner iteration genuinely repeats every pass. If an initial collection traversal (FOR) precedes a repair loop (WHILE), they are siblings in a parent SEQ, not nested.
- An unnamed or unbound FOR inside any body is always a modeling error.
- For SEQ/CHOICE nodes: `activity_refs` must equal exactly the union of body activity_id leaves plus referenced child node `activity_refs`.
- Episode IDs use the format `activity_NNNN` from the source data.

Scan for missed FOR/WHILE patterns — SEQ is only for truly ordered, non-repeating, non-conditional steps:
- Any SEQ whose child steps apply the same abstract body to a named set of items → convert to FOR.
- Any cluster of repeated inspect → fix → retest steps → convert to WHILE.
- Named collection items (item A, item B, item C…), multiple accounts, multiple files, multiple artifacts → FOR.
- Repeated change-request → rerun → re-inspect cycles → WHILE.

Required output schema:
{procedure_output_schema_text()}
"""


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.monotonic)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0
    breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def elapsed_secs(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_secs": self.elapsed_secs(),
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": round(self.estimated_usd, 6),
            "cost_breakdown": self.cost_breakdown(),
        }

    def record_call(self, *, operation: str, model: str, usage: dict[str, int], estimated_usd: float | None) -> None:
        self.llm_requests += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                setattr(self, key, getattr(self, key) + int(value))
        if estimated_usd is not None:
            self.estimated_usd += estimated_usd
        bucket = self.breakdown.setdefault(
            operation,
            {
                "model": model,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        bucket["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        bucket["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        if estimated_usd is not None:
            bucket["estimated_usd"] += estimated_usd

    def cost_breakdown(self) -> dict[str, Any]:
        return {
            operation: {
                **payload,
                "estimated_usd": round(float(payload.get("estimated_usd") or 0.0), 6),
            }
            for operation, payload in self.breakdown.items()
        }


class ProcedureReporter(ConsoleProgressReporter):
    run_name = "procedure_model_induction"
    success_title = "Procedure Model Induction Complete"
    failure_title = "Procedure Model Induction Failed"
    default_failure_stage = STAGE_PREFLIGHT

    def __init__(self, *, no_console: bool = False) -> None:
        super().__init__(stages=STAGES, no_console=no_console)
        self._root_status_lock = Lock()

    def render(self) -> Any:
        if not all((self._Panel, self._Table, self._Text, self._Group, self._box)):
            return "Procedure model induction"
        return self._Panel(
            self._Group(
                self._Text("Procedure Model Induction", style="bold cyan"),
                self._stage_table(),
                self._roots_table(),
                self._metrics_table(),
            ),
            title="Running",
            border_style="cyan",
            box=self._box.ROUNDED,
        )

    def render_success(self, detail: str) -> Any:
        if not all((self._Panel, self._Group, self._box)):
            return detail
        return self._Panel(
            self._Group(self._summary_table(detail), self._metrics_table(), self._paths_table()),
            title=self.success_title,
            border_style="green",
            box=self._box.ROUNDED,
        )

    def render_failure(self, message: str) -> Any:
        if not all((self._Panel, self._Group, self._box)):
            return message
        return self._Panel(
            self._Group(self._summary_table(f"failed: {message}"), self._stage_table(), self._paths_table()),
            title=self.failure_title,
            border_style="red",
            box=self._box.ROUNDED,
        )

    def plain_summary(self) -> str:
        counters = self.state.counters
        metrics = self.state.metrics
        root_details = metrics.get("root_status_summary") or metrics.get("root_plan_summary")
        root_detail_text = f" roots=[{root_details}] " if root_details else " "
        return (
            f"elapsed={format_duration(time.monotonic() - self.started_at)} "
            f"direct_model={metrics.get('direct_model', '')} "
            f"codex_model={metrics.get('codex_model', '')} "
            f"roots={counters.get('roots', 0)}"
            f"{root_detail_text}"
            f"succeeded={counters.get('succeeded', 0)} "
            f"direct={counters.get('direct_llm', 0)} "
            f"codex={counters.get('codex_cli', 0)} "
            f"llm_requests={metrics.get('llm_requests', 0)} "
            f"tokens={metrics.get('total_tokens', 0)} "
            f"estimated_cost=${float(metrics.get('estimated_usd', 0.0)):.6f}"
        )

    def _summary_table(self, detail: str) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("status", detail)
        table.add_row("elapsed", format_duration(time.monotonic() - self.started_at))
        table.add_row("direct model", str(self.state.metrics.get("direct_model") or ""))
        table.add_row("codex model", str(self.state.metrics.get("codex_model") or ""))
        table.add_row("roots", str(self.state.counters.get("roots", 0)))
        table.add_row("succeeded", str(self.state.counters.get("succeeded", 0)))
        table.add_row("direct runs", str(self.state.counters.get("direct_llm", 0)))
        table.add_row("codex runs", str(self.state.counters.get("codex_cli", 0)))
        root_summary = str(self.state.metrics.get("root_plan_summary") or "").strip()
        if root_summary:
            table.add_row("root plan", root_summary)
        return table

    def _metrics_table(self) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        metrics = self.state.metrics
        for key, label in (
            ("llm_requests", "llm requests"),
            ("input_tokens", "input tokens"),
            ("output_tokens", "output tokens"),
            ("total_tokens", "tokens"),
        ):
            table.add_row(label, str(metrics.get(key, 0)))
        if metrics.get("parallelism"):
            table.add_row("parallelism", str(metrics.get("parallelism")))
        table.add_row("estimated cost", f"${float(metrics.get('estimated_usd', 0.0)):.6f}")
        return table

    def _stage_table(self) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_column(overflow="fold")
        markers = {"pending": "-", "active": ">", "done": "ok", "failed": "!!"}
        for stage in self.state.stages:
            table.add_row(markers.get(stage.status, "?"), stage.name, stage.detail)
        return table

    def _roots_table(self) -> Any:
        root_statuses = self.state.metrics.get("root_statuses")
        if not isinstance(root_statuses, list) or not root_statuses:
            return self._Text("roots: none", style="dim")
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_column(style="dim")
        for item in root_statuses:
            if not isinstance(item, dict):
                continue
            table.add_row(
                str(item.get("status", "")),
                str(item.get("name", "")),
                str(item.get("detail", "")),
            )
        return table

    def initialize_root_statuses(self, root_plans: list[tuple[Path, int | None, str]]) -> None:
        with self._root_status_lock:
            statuses: list[dict[str, str]] = []
            summary_parts: list[str] = []
            for path, count, mode in root_plans:
                activity_text = str(count) if count is not None else "unknown"
                detail = f"n={activity_text} mode={mode}"
                statuses.append({"name": path.name, "status": "queued", "detail": detail})
                summary_parts.append(f"{path.name}({detail},queued)")
            self.set_metric("root_statuses", statuses)
            self.set_metric("root_plan_summary", ", ".join(summary_parts))
            self.set_metric("root_status_summary", ", ".join(summary_parts))

    def update_root_status(self, path: Path, status: str, detail: str) -> None:
        with self._root_status_lock:
            current = self.state.metrics.get("root_statuses")
            statuses = list(current) if isinstance(current, list) else []
            updated = False
            for item in statuses:
                if isinstance(item, dict) and item.get("name") == path.name:
                    item["status"] = status
                    item["detail"] = detail
                    updated = True
                    break
            if not updated:
                statuses.append({"name": path.name, "status": status, "detail": detail})
            summary_parts = [
                f"{item.get('name', '')}({item.get('detail', '')},{item.get('status', '')})"
                for item in statuses
                if isinstance(item, dict)
            ]
            self.set_metric("root_statuses", statuses)
            self.set_metric("root_status_summary", ", ".join(summary_parts))

    def _paths_table(self) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        for label, path in self.state.paths.items():
            table.add_row(label, str(path))
        return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Induce procedure models from derived task-thread objectives.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Pipeline data directory.")
    parser.add_argument("--config", type=Path, default=None, help="Task model induction config path.")
    parser.add_argument(
        "--output",
        "--output_path",
        dest="output",
        help="Merged output JSON path. Default: configured task_model_with_procedures.json in the data directory.",
    )
    parser.add_argument("--preflight-only", "--preflight_only", dest="preflight_only", action="store_true")
    parser.add_argument("--no_console", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def activity_count(input_path: Path) -> int | None:
    try:
        source = read_json(input_path)
    except Exception:
        return None
    activities = source.get("activities")
    return len(activities) if isinstance(activities, list) else None


def format_root_plan_summary(root_plans: list[tuple[Path, int | None, str]]) -> str:
    parts: list[str] = []
    for path, count, mode in root_plans:
        activity_text = str(count) if count is not None else "?"
        mode_label = "direct" if mode == "direct_llm" else "codex"
        stem = path.stem
        if stem.endswith(".json"):
            stem = Path(stem).stem
        parts.append(f"{stem} ({activity_text}, {mode_label})")
    return " | ".join(parts)


def extract_json_from_response(response: str) -> dict[str, Any]:
    text = response.strip()
    try:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def normalize_usage(response: Any) -> dict[str, int]:
    return normalize_litellm_usage(response)


def estimated_completion_cost_usd(response: Any, model: str) -> float | None:
    return estimated_litellm_completion_cost_usd(response, model)


def call_procedure_llm(
    *,
    system_prompt: str,
    content: dict[str, Any],
    model: str,
    llm_timeout_secs: float,
    stats: RunStats,
    operation: str,
) -> dict[str, Any]:
    response = litellm_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ],
        temperature=0.0 if "gpt-5" not in model and "kimi" not in model else 1.0,
        timeout=llm_timeout_secs,
        request_timeout=llm_timeout_secs,
        response_format={"type": "json_object"},
    )
    stats.record_call(
        operation=operation,
        model=model,
        usage=normalize_usage(response),
        estimated_usd=estimated_completion_cost_usd(response, model),
    )
    return extract_json_from_response(completion_message_content(response))


def write_direct_meta(
    output_path: Path,
    *,
    input_path: Path,
    model: str,
    max_retries: int,
    retry_count: int,
    valid: bool,
    feedback: Any,
    stats: RunStats,
) -> None:
    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model": model,
        "max_retries": max_retries,
        "retry_count": retry_count,
        "preflight_only": False,
        "valid": valid,
        "errors": list(feedback.errors),
        "warnings": list(feedback.warnings),
        "cost": stats.as_dict(),
        "execution_mode": "direct_llm",
        "activity_count": activity_count(input_path),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)


def build_activity_lookup(source: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map activity_id → {name, description} from source activity data."""
    lookup: dict[str, dict[str, str]] = {}
    for act in source.get("activities") or []:
        if not isinstance(act, dict):
            continue
        aid = act.get("activity_id")
        if isinstance(aid, str):
            lookup[aid] = {
                "name": str(act.get("objective") or ""),
                "description": str(act.get("additional_context") or ""),
            }
    return lookup


def enrich_activity_leaves(obj: Any, lookup: dict[str, dict[str, str]]) -> Any:
    """Recursively overwrite name/description on activity leaf nodes from source data."""
    if isinstance(obj, dict):
        if "activity_id" in obj and "operator" not in obj:
            aid = obj["activity_id"]
            if aid in lookup:
                obj = {**obj, **lookup[aid]}
            return obj
        return {k: enrich_activity_leaves(v, lookup) for k, v in obj.items()}
    if isinstance(obj, list):
        return [enrich_activity_leaves(item, lookup) for item in obj]
    return obj


def _seq_nodes_needing_for_scan(candidate: dict[str, Any], min_leaf_steps: int = 8) -> list[dict[str, Any]]:
    """Return SEQ nodes whose body contains enough activity_id leaves to hide a FOR pattern."""
    result = []
    for node in candidate.get("procedure_nodes", []):
        if node.get("operator") != "SEQ":
            continue
        body = node.get("body") or {}
        steps = body.get("steps", []) if isinstance(body, dict) else []
        leaf_count = sum(1 for s in steps if isinstance(s, dict) and "activity_id" in s)
        if leaf_count >= min_leaf_steps:
            result.append(node)
    return result


def run_for_scan_pass(
    *,
    candidate: dict[str, Any],
    source: dict[str, Any],
    model: str,
    llm_timeout_secs: float,
    stats: RunStats,
    min_leaf_steps: int = 8,
) -> dict[str, Any]:
    """Run a targeted FOR-pattern scan on SEQ nodes with many leaf steps.

    Returns the improved candidate if the scan produces a valid model,
    otherwise returns the original candidate unchanged.
    """
    large_seq_nodes = _seq_nodes_needing_for_scan(candidate, min_leaf_steps)
    if not large_seq_nodes:
        return candidate
    scan_hints = [
        {
            "id": n.get("id"),
            "name": n.get("name"),
            "activity_refs": n.get("activity_refs", []),
            "leaf_step_count": sum(
                1 for s in (n.get("body") or {}).get("steps", [])
                if isinstance(s, dict) and "activity_id" in s
            ),
        }
        for n in large_seq_nodes
    ]
    scanned = call_procedure_llm(
        system_prompt=procedure_for_scan_prompt_text(),
        content={
            "original_input": source,
            "current_model": candidate,
            "seq_nodes_to_scan": scan_hints,
        },
        model=model,
        llm_timeout_secs=llm_timeout_secs,
        stats=stats,
        operation="for_scan",
    )
    scan_feedback = validate_procedure_output(scanned, source=source)
    if scan_feedback.valid:
        return scanned
    return candidate


def run_direct_procedure_stage(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    llm_timeout_secs: float,
    max_retries: int,
) -> dict[str, Any]:
    stats = RunStats()
    source = read_json(input_path)
    candidate = call_procedure_llm(
        system_prompt=procedure_generation_prompt_text(),
        content={"original_input": source},
        model=model,
        llm_timeout_secs=llm_timeout_secs,
        stats=stats,
        operation="generation",
    )
    feedback = validate_procedure_output(candidate, source=source)
    retry_count = 0
    while not feedback.valid and retry_count < max_retries:
        retry_count += 1
        candidate = call_procedure_llm(
            system_prompt=procedure_repair_prompt_text(),
            content={
                "original_input": source,
                "previous_result": candidate,
                "validation_feedback": feedback.as_text(),
            },
            model=model,
            llm_timeout_secs=llm_timeout_secs,
            stats=stats,
            operation="repair",
        )
        feedback = validate_procedure_output(candidate, source=source)
    if not feedback.valid:
        write_direct_meta(
            output_path,
            input_path=input_path,
            model=model,
            max_retries=max_retries,
            retry_count=retry_count,
            valid=False,
            feedback=feedback,
            stats=stats,
        )
        raise RuntimeError(f"procedure validation failed for {input_path}: {feedback.as_text()}")
    candidate = run_for_scan_pass(
        candidate=candidate,
        source=source,
        model=model,
        llm_timeout_secs=llm_timeout_secs,
        stats=stats,
    )
    candidate = enrich_activity_leaves(candidate, build_activity_lookup(source))
    payload = ProcedureTaskModel.model_validate(candidate).model_dump()
    write_json_atomic(output_path, payload)
    write_procedure_markdown(output_path, payload)
    write_direct_meta(
        output_path,
        input_path=input_path,
        model=model,
        max_retries=max_retries,
        retry_count=retry_count,
        valid=True,
        feedback=feedback,
        stats=stats,
    )
    return {
        "execution_mode": "direct_llm",
        "usage": {
            "llm_requests": stats.llm_requests,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "total_tokens": stats.total_tokens,
        },
        "estimated_usd": round(stats.estimated_usd, 6),
        "cost_breakdown": stats.cost_breakdown(),
        "activity_count": activity_count(input_path),
    }


def discover_procedure_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    manifest_path = input_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except Exception:
            manifest = {}
        roots = manifest.get("roots")
        if isinstance(roots, list):
            inputs: list[Path] = []
            for root in roots:
                if not isinstance(root, dict):
                    continue
                file_value = root.get("file")
                if not isinstance(file_value, str) or not file_value.strip():
                    continue
                path = Path(file_value).expanduser()
                if not path.is_absolute():
                    path = input_dir / path
                if path.is_file():
                    inputs.append(path.resolve())
            if inputs:
                return inputs
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != "manifest.json"
        and not path.name.endswith(".meta.json")
        and not path.name.endswith("_hierarchy.json")
    )


def run_codex_procedure_stage(
    *,
    input_path: Path,
    output_path: Path,
    codex_config: dict[str, Any],
    rebuild_image: bool,
) -> dict[str, Any]:
    source_text = input_path.read_text(encoding="utf-8")
    result = CodexCliSandbox().run_file_task(
        prompt=procedure_induction_prompt(),
        files={
            "task_thread_objectives.json": source_text,
            "procedure_model_schema.json": procedure_output_schema_text(),
            "validate_procedure_model.py": validator_script_text(),
        },
        output_file="procedure_model.json",
        output_files=("validation.txt",),
        codex_config=codex_config,
        rebuild_image=rebuild_image,
    )
    result["execution_mode"] = "codex_cli"
    result["activity_count"] = activity_count(input_path)
    usage = extract_codex_usage(result)
    estimated_usd = estimate_usage_cost_usd(usage, str(codex_config.get("model") or ""))
    if usage:
        result["usage"] = usage
    if estimated_usd is not None:
        result["estimated_usd"] = estimated_usd
    if not result.get("ok"):
        raise RuntimeError(codex_failure_message(result))
    output_content = result.get("output_content")
    if not isinstance(output_content, str) or not output_content.strip():
        raise RuntimeError("Codex run succeeded but returned no procedure_model.json")
    try:
        parsed = json.loads(output_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"returned procedure_model.json is invalid JSON: {exc}") from exc
    source = json.loads(source_text) if source_text.strip() else None
    feedback = validate_procedure_output(parsed, source=source)
    if not feedback.valid:
        raise RuntimeError("returned procedure_model.json failed local validation: " + feedback.as_text())
    if source:
        parsed = enrich_activity_leaves(parsed, build_activity_lookup(source))
    payload = ProcedureTaskModel.model_validate(parsed).model_dump()
    write_json_atomic(output_path, payload)
    write_procedure_markdown(output_path, payload)
    validation_text = (result.get("output_files") or {}).get("validation.txt")
    if isinstance(validation_text, str) and validation_text.strip():
        output_path.with_suffix(output_path.suffix + ".validation.txt").write_text(validation_text, encoding="utf-8")
    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "run_id": result.get("run_id"),
        "session_id": result.get("session_id"),
        "model": (result.get("codex") or {}).get("model") or codex_config.get("model"),
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "usage": usage,
        "estimated_usd": estimated_usd,
        "proxy_cost": result.get("proxy_cost"),
        "execution_mode": "codex_cli",
        "activity_count": result["activity_count"],
        "preflight_only": False,
        "validation": feedback.as_dict(),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)
    return result


def codex_failure_message(result: dict[str, Any]) -> str:
    codex_config = result.get("codex") or {}
    details: list[str] = []
    for key in ("run_id", "returncode"):
        if result.get(key) not in (None, 0, ""):
            details.append(f"{key}={result[key]}")
    for key in ("model", "model_reasoning_effort", "personality"):
        if codex_config.get(key):
            details.append(f"{key}={codex_config[key]}")
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    if stdout:
        details.append(f"stdout={stdout[-500:]}")
    if stderr:
        details.append(f"stderr={stderr[-500:]}")
    message = str(result.get("error") or "Codex procedure run failed")
    return f"{message} ({'; '.join(details)})" if details else message


def extract_codex_usage(result: dict[str, Any]) -> dict[str, int]:
    usages: list[dict[str, int]] = []
    for stream_name in ("stdout", "stderr"):
        stream = result.get(stream_name)
        if not isinstance(stream, str):
            continue
        for line in stream.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            usages.extend(find_usage_dicts(event))
    return max(usages, key=lambda item: item.get("total_tokens", 0)) if usages else {}


def find_usage_dicts(value: Any) -> list[dict[str, int]]:
    found: list[dict[str, int]] = []
    if isinstance(value, dict):
        normalized = normalize_litellm_usage(value)
        if normalized:
            found.append(normalized)
        for child in value.values():
            found.extend(find_usage_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_usage_dicts(child))
    return found


def estimate_usage_cost_usd(usage: dict[str, int], model: str) -> float | None:
    estimated = estimate_litellm_usage_cost_usd(usage, model)
    return round(estimated, 6) if estimated is not None else None


def sum_root_usage(root_results: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in root_results:
        usage = item.get("usage") if isinstance(item, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return {key: value for key, value in totals.items() if value}


def build_root_result(input_path: Path, output_path: Path, run_result: dict[str, Any]) -> dict[str, Any]:
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "ok": True,
        "run_id": run_result.get("run_id"),
        "session_id": run_result.get("session_id"),
        "execution_mode": run_result.get("execution_mode"),
        "usage": run_result.get("usage"),
        "estimated_usd": run_result.get("estimated_usd"),
        "proxy_cost": run_result.get("proxy_cost"),
        "activity_count": run_result.get("activity_count"),
    }
    if output_path.exists():
        result["procedure_task_model"] = json.loads(output_path.read_text(encoding="utf-8"))
    return result


def load_cached_procedure_root(
    *,
    input_path: Path,
    output_path: Path,
    execution_mode: str,
) -> dict[str, Any] | None:
    """Load a per-root cache only if it satisfies the current validator."""

    if not output_path.exists():
        return None
    try:
        source = read_json(input_path)
        candidate = read_json(output_path)
        feedback = validate_procedure_output(candidate, source=source)
        if not feedback.valid:
            return None
        ProcedureTaskModel.model_validate(candidate)
    except Exception:
        return None

    meta: dict[str, Any] = {}
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            loaded = read_json(meta_path)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    cost = meta.get("cost") if isinstance(meta.get("cost"), dict) else {}
    return {
        "execution_mode": str(meta.get("execution_mode") or execution_mode),
        "run_id": meta.get("run_id"),
        "session_id": meta.get("session_id"),
        "usage": meta.get("usage") or cost,
        "estimated_usd": meta.get("estimated_usd") or cost.get("estimated_usd"),
        "proxy_cost": meta.get("proxy_cost"),
        "activity_count": activity_count(input_path),
    }


def write_merged_procedure_task_model(
    *,
    output_path: Path,
    output_dir: Path,
    root_results: list[dict[str, Any]],
    model: str,
    preflight_only: bool,
) -> dict[str, Any]:
    usage = sum_root_usage(root_results)
    estimated = round(sum(float(item.get("estimated_usd") or 0.0) for item in root_results), 6)
    cost = {"source": "estimated_from_usage", "total_usd": estimated, "usage_from_codex_stdout": usage} if usage or estimated else None
    payload = {
        "meta": {
            "created_at": utc_now_iso(),
            "model": model,
            "output_dir": str(output_dir),
            "num_roots": len(root_results),
            "num_succeeded": sum(1 for item in root_results if item.get("ok")),
            "preflight_only": preflight_only,
            "cost": cost,
        },
        "roots": root_results,
    }
    payload = ProcedureModelInductionOutput.model_validate(payload).model_dump()
    write_json_atomic(output_path, payload)
    write_procedure_collection_markdown(output_path, payload)
    if cost is not None:
        write_json_atomic(output_path.with_suffix(output_path.suffix + ".cost.json"), cost)
        print("[cost] procedures " + json.dumps(cost, ensure_ascii=False), file=sys.stderr)
    return payload


def procedure_model_induction(
    *,
    data_dir: str | Path,
    input_dir: str,
    output_dir: str,
    output_file_name: str,
    output_path: str | Path | None,
    direct_model: str,
    direct_litellm_params: dict[str, Any] | None,
    codex_config: dict[str, Any],
    max_retries: int,
    direct_llm_max_activities: int,
    workers: int,
    llm_timeout_secs: float,
    reuse_cache: bool = False,
    preflight_only: bool = False,
    no_console: bool = False,
    rebuild_codex_sandbox: bool = False,
) -> dict[str, Any] | None:
    with ProcedureReporter(no_console=no_console) as reporter:
        try:
            reporter.set_metric("direct_model", direct_model)
            reporter.set_metric("codex_model", codex_config.get("model") or "")
            reporter.start_stage(STAGE_LOAD_INPUTS, "resolving paths")
            resolved_data_dir = Path(data_dir).expanduser()
            resolved_input_dir = resolved_data_dir / input_dir
            resolved_output_dir = resolved_data_dir / output_dir
            resolved_output = Path(output_path).expanduser() if output_path else resolved_data_dir / output_file_name
            if not resolved_output.is_absolute():
                resolved_output = resolved_data_dir / resolved_output
            procedure_inputs = discover_procedure_inputs(resolved_input_dir)
            reporter.add_path("input_dir", resolved_input_dir)
            reporter.add_path("output_dir", resolved_output_dir)
            reporter.add_path("output", resolved_output)
            reporter.set_counter("roots", len(procedure_inputs))
            reporter.finish_stage(STAGE_LOAD_INPUTS, f"loaded {len(procedure_inputs)} root inputs")

            reporter.start_stage(STAGE_PREFLIGHT, "validating planned work")
            if workers < 1:
                raise ValueError("workers must be >= 1")
            if not procedure_inputs:
                raise FileNotFoundError(f"no task-thread objective inputs found at {resolved_input_dir}")
            reporter.finish_stage(STAGE_PREFLIGHT, "ready")

            if preflight_only:
                root_results = [
                    {
                        "input_file": str(path),
                        "output_file": str(resolved_output_dir / path.name),
                        "ok": True,
                        "execution_mode": (
                            "direct_llm"
                            if (
                                (count := activity_count(path)) is not None
                                and count < direct_llm_max_activities
                                and not direct_llm_input_too_large(path)
                            )
                            else "codex_cli"
                        ),
                    }
                    for path in procedure_inputs
                ]
                reporter.set_counter("direct_llm", sum(1 for item in root_results if item["execution_mode"] == "direct_llm"))
                reporter.set_counter("codex_cli", sum(1 for item in root_results if item["execution_mode"] == "codex_cli"))
                reporter.set_counter("succeeded", len(root_results))
                payload = write_merged_procedure_task_model(
                    output_path=resolved_output,
                    output_dir=resolved_output_dir,
                    root_results=root_results,
                    model=direct_model if reporter.state.counters.get("codex_cli", 0) == 0 else str(codex_config.get("model") or ""),
                    preflight_only=True,
                )
                reporter.mark_stage_done(STAGE_GENERATION, "skipped")
                reporter.finish_stage(STAGE_MERGE, "wrote preflight output")
                reporter.final_success("preflight complete; no LLM or Codex calls were made")
                return payload

            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            root_plans: list[tuple[Path, int | None, str]] = []
            for path in procedure_inputs:
                count = activity_count(path)
                execution_mode = (
                    "direct_llm"
                    if count is not None
                    and count < direct_llm_max_activities
                    and not direct_llm_input_too_large(path)
                    else "codex_cli"
                )
                if execution_mode == "codex_cli" and count is not None and count < direct_llm_max_activities:
                    reporter.progress(
                        f"routing {path.name} to codex_cli: input too large for direct LLM"
                    )
                root_plans.append((path, count, execution_mode))
            mode_counts = {"direct_llm": 0, "codex_cli": 0}
            for path, count, mode in root_plans:
                mode_counts[mode] += 1
            reporter.initialize_root_statuses(root_plans)
            reporter.set_counter("direct_llm", mode_counts["direct_llm"])
            reporter.set_counter("codex_cli", mode_counts["codex_cli"])
            reporter.set_metric("direct_llm_max_activities", direct_llm_max_activities)
            reporter.progress("root plan loaded")

            reporter.start_stage(STAGE_GENERATION, "generating procedure models")
            max_workers = max(1, min(workers, len(procedure_inputs)))
            reporter.set_metric("parallelism", max_workers)
            root_results: list[dict[str, Any]] = []
            errors: list[str] = []

            def runner(path: Path, out_path: Path, count: int | None, execution_mode: str) -> dict[str, Any]:
                activity_text = str(count) if count is not None else "unknown"
                detail = f"n={activity_text} mode={execution_mode}"
                if reuse_cache:
                    cached = load_cached_procedure_root(
                        input_path=path,
                        output_path=out_path,
                        execution_mode=execution_mode,
                    )
                    if cached is not None:
                        reporter.update_root_status(path, "done", f"n={activity_text} cache")
                        reporter.progress(f"loaded cache for {path.stem}")
                        return cached
                reporter.update_root_status(path, "running", detail)
                reporter.progress(f"running {path.stem} ({activity_text}, {execution_mode})")
                if execution_mode == "direct_llm":
                    with litellm_model_config(model_alias=direct_model, litellm_params=direct_litellm_params):
                        result = run_direct_procedure_stage(
                            input_path=path,
                            output_path=out_path,
                            model=direct_model,
                            llm_timeout_secs=llm_timeout_secs,
                            max_retries=max_retries,
                        )
                else:
                    result = run_with_retries(
                        lambda: run_codex_procedure_stage(
                            input_path=path,
                            output_path=out_path,
                            codex_config=codex_config,
                            rebuild_image=rebuild_codex_sandbox,
                        ),
                        attempts=max_retries,
                        on_retry=lambda attempt, exc: reporter.progress(
                            f"codex retry {attempt}/{max_retries} for {path.name}: {str(exc)[:200]}"
                        ),
                    )
                reporter.update_root_status(path, "done", detail)
                reporter.progress(f"finished {path.stem} ({execution_mode})")
                return result

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_paths = {
                    executor.submit(
                        runner,
                        path,
                        resolved_output_dir / path.name,
                        count,
                        execution_mode,
                    ): (path, resolved_output_dir / path.name, execution_mode)
                    for path, count, execution_mode in root_plans
                }
                for future in as_completed(future_to_paths):
                    path, out_path, execution_mode = future_to_paths[future]
                    try:
                        run_result = future.result()
                        root_results.append(build_root_result(path, out_path, run_result))
                        reporter.increment("succeeded")
                    except Exception as exc:
                        activity_text = str(activity_count(path)) if activity_count(path) is not None else "unknown"
                        reporter.update_root_status(path, "failed", f"n={activity_text} mode={execution_mode}")
                        errors.append(f"{path.name}: {exc}")
                        root_results.append(
                            {
                                "input_file": str(path),
                                "output_file": str(out_path),
                                "ok": False,
                                "error": str(exc),
                                "execution_mode": execution_mode,
                                "activity_count": activity_count(path),
                            }
                        )
                    reporter.progress(f"completed {len(root_results)}/{len(procedure_inputs)} roots")

            root_results.sort(key=lambda item: str(item.get("input_file") or ""))
            usage = sum_root_usage(root_results)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                reporter.set_metric(key, usage.get(key, 0))
            reporter.set_metric(
                "llm_requests",
                sum(
                    int((item.get("usage") or {}).get("llm_requests") or 0)
                    for item in root_results
                    if isinstance(item, dict)
                ),
            )
            reporter.set_metric("estimated_usd", sum(float(item.get("estimated_usd") or 0.0) for item in root_results))
            reporter.finish_stage(STAGE_GENERATION, "completed")

            reporter.start_stage(STAGE_MERGE, "writing merged output")
            payload = write_merged_procedure_task_model(
                output_path=resolved_output,
                output_dir=resolved_output_dir,
                root_results=root_results,
                model=direct_model if mode_counts["codex_cli"] == 0 else str(codex_config.get("model") or ""),
                preflight_only=False,
            )
            reporter.finish_stage(STAGE_MERGE, "saved output")
            if errors:
                raise RuntimeError("one or more procedure root runs failed: " + "; ".join(errors))
            reporter.final_success(f"induced {len(root_results)} procedure roots")
            return payload
        except Exception as exc:
            reporter.fail_active_stage(exc)
            reporter.final_failure()
            raise


def main() -> int:
    args = parse_args()
    if args.data_dir is None:
        raise ValueError("--data_dir is required for step5 procedure model induction.")
    config_path = Path(args.config).expanduser().resolve() if args.config else resolve_config_path()
    config = load_config(config_path)
    if config.dotenv_path:
        try:
            from dotenv import load_dotenv

            dotenv_path = resolve_dotenv_path(config_path, config.dotenv_path)
            load_dotenv(dotenv_path, override=False)
        except ModuleNotFoundError:
            pass

    stage_config = config.procedure_induction_stage
    direct_branch = stage_config.direct_llm_branch
    codex_branch = stage_config.codex_branch
    codex_config = {
        "model": codex_branch.model,
        "model_reasoning_effort": codex_branch.model_reasoning_effort,
        "personality": codex_branch.personality,
        "model_provider": codex_branch.model_provider,
        "provider_name": codex_branch.provider_name,
        "command_timeout_seconds": codex_branch.command_timeout_seconds,
        "litellm_params": codex_branch.litellm_params,
    }
    procedure_model_induction(
        data_dir=args.data_dir,
        input_dir=stage_config.input_dir,
        output_dir=stage_config.output_dir,
        output_file_name=stage_config.output_file_name,
        output_path=args.output or stage_config.output_file_name,
        direct_model=direct_branch.model,
        direct_litellm_params=direct_branch.litellm_params,
        codex_config=codex_config,
        max_retries=stage_config.max_retries,
        direct_llm_max_activities=direct_branch.direct_llm_max_activities,
        workers=stage_config.workers,
        llm_timeout_secs=stage_config.llm_timeout_seconds,
        reuse_cache=stage_config.reuse_cache,
        preflight_only=args.preflight_only,
        no_console=args.no_console,
        rebuild_codex_sandbox=stage_config.rebuild_codex_sandbox,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
