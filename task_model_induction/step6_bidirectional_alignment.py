#!/usr/bin/env python3
"""Reconcile objective and procedure models into a unified task model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from task_model_induction.codex_cli_sandbox import CodexCliSandbox
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.schemas import (
        UnifiedTaskModel,
        UnifiedTaskModelMergedMeta,
        UnifiedTaskModelOutput,
        UnifiedTaskModelRootResult,
    )
    from task_model_induction.utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        estimate_litellm_usage_cost_usd,
        extract_max_usage_from_json_events,
        litellm_completion,
        direct_llm_input_too_large,
        litellm_model_config,
        normalize_litellm_usage,
        run_with_retries,
        read_json_object as read_json,
        sum_usage_dicts,
        utc_now_iso,
        write_json_atomic,
    )
    from task_model_induction.validate.validate_unified_model import unified_schema, validate_unified_output
except ModuleNotFoundError:
    from codex_cli_sandbox import CodexCliSandbox
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import (
        UnifiedTaskModel,
        UnifiedTaskModelMergedMeta,
        UnifiedTaskModelOutput,
        UnifiedTaskModelRootResult,
    )
    from utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        estimate_litellm_usage_cost_usd,
        extract_max_usage_from_json_events,
        litellm_completion,
        direct_llm_input_too_large,
        litellm_model_config,
        normalize_litellm_usage,
        run_with_retries,
        read_json_object as read_json,
        sum_usage_dicts,
        utc_now_iso,
        write_json_atomic,
    )
    from validate.validate_unified_model import unified_schema, validate_unified_output


STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_RECONCILIATION = "reconciliation"
STAGE_MERGE = "merge output"
STAGES = [STAGE_LOAD_INPUTS, STAGE_PREFLIGHT, STAGE_RECONCILIATION, STAGE_MERGE]

ACTIVITY_RE = re.compile(r"^activity_(?P<start>\d{4})(?:-activity_(?P<end>\d{4}))?$")
OBJECTIVE_ACTIVITY_RE = re.compile(
    r"^(?:activity|subgoal_segment)_(?P<start>\d{4})"
    r"(?:-(?:activity|subgoal_segment)_(?P<end>\d{4}))?$"
)
GROUNDING_FIELDS = (
    "deliverables",
    "success_criteria",
    "observed_outcome",
    "evidence_refs",
)


# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------

def unified_schema_text() -> str:
    return json.dumps(unified_schema(), indent=2, ensure_ascii=True)


def validator_script_text() -> str:
    return (Path(__file__).resolve().parent / "validate" / "validate_unified_model.py").read_text(encoding="utf-8")


def while_condition_grounding_guidance() -> str:
    return """For every WHILE, preserve or construct an operational exit condition:
- `condition` is a positive, binary objective-state predicate, not "until done",
  "until satisfied", "acceptable", "clear enough", or another subjective summary.
- `condition_grounding` contains exactly `predicate` (identical to `condition`),
  `verifier` (the concrete check and expected observable signal), `evidence_refs`
  (non-empty activity refs inside this node), and `observed_status` (`satisfied`,
  `unsatisfied`, or `unknown`).
- Mark `satisfied` only when cited trace evidence shows the exit state. If the work
  stops without establishing it, use `unsatisfied` or `unknown` rather than inventing
  success."""


def reconciliation_generation_prompt() -> str:
    return f"""You produce a unified task model by reconciling independently-induced objective and procedure models for the same activity trace. Return only a valid JSON object.

## What you produce

A unified tree where each node has two layers:

**Objective layer** — a domain-specific program in natural language. Captures domain invariants, concrete deliverables, verifiable success criteria, and the outcome actually established by the trace. Expressed as `objective`, `summary`, `deliverables`, `success_criteria`, `observed_outcome`, `evidence_refs`, and `decomposition` children.

**Procedure layer** — the faithful record of how the work was actually carried out in the observed trace: the real execution path including failures and corrections, at the grain that reveals meaningful structure rather than mechanical noise. Expressed as `procedure.operator`, `procedure.body`, and operator-specific fields.

Both layers are determined jointly from both input models. Neither is authoritative alone.

## Inputs

- `source`: task-thread JSON with an `activities` list; each entry is one atomic episode.
- `objective_model`: hierarchical objective model; `subgoal_segments` reference activity ranges; `decomposition` is the goal tree.
- `procedure_model`: control-flow procedure model; `procedure_nodes` have `operator`, `activity_refs`, and `body`.

## Procedure operators

Every procedure annotation has a `body`: an ordered list of named steps, each with `name`, optional `description`, and non-empty `activity_refs`.

| Operator | When to use | `body` content | `activity_refs` per step | `decomposition` children |
|---|---|---|---|---|
| `SEQ` | Ordered, non-repeating phases | One domain-operation step per phase | That step's activity range | Recursive child nodes |
| `WHILE` | Repeated until a condition holds | Abstract per-pass domain-operation template; annotate failed attempts | Spans ALL passes where that step occurs | Semantic sub-phases (distinct goal-focus areas the work moved through) — NOT pass-by-pass enumeration; empty if no meaningful sub-phase structure |
| `FOR` | Same operations applied to each item in a named collection | Abstract per-item domain-operation template | Spans ALL items/iterations where that step occurs | One child per semantically distinct named item; no children for fungible items |
| `CHOICE` | Mutually exclusive alternatives; one observed | Observed branch steps | Observed branch range | The observed branch as a child |

`WHILE` requires `condition` and evidence-backed `condition_grounding`.
`FOR` requires exactly `bindings.iteration_variable` and a non-empty explicit
`bindings.collection` (e.g., `{{"iteration_variable": "account", "collection": ["alice", "bob"]}}`).

{while_condition_grounding_guidance()}

## Grounded objective contract

The objective model already contains grounding fields. Preserve them recursively
instead of replacing them with free-text summaries:

- `deliverables`: concrete artifacts or states, including their expected state.
- `success_criteria`: independently checkable predicates and their verifier.
- `observed_outcome`: what the trace actually establishes; this is distinct from intended success.
- `evidence_refs`: source evidence supporting the node and all nested grounding fields.

When a node boundary is unchanged, copy these fields without loss. When splitting
or merging nodes, retain the relevant evidence-backed entries on the resulting
nodes; never invent an achieved outcome. Every node must have at least one
deliverable and one success criterion. Nested grounding evidence refs must be a
subset of the node's `evidence_refs`.

## Reconciliation: joint reasoning at every level

At each level, decide simultaneously: what phases exist, which operator governs each phase, and what the objective goal of each phase is. These are one integrated decision, not sequential steps.

### Reading the two models together

Each scenario calls for a different response:

**Both models draw the same boundary** → honor it.

**Procedure shows WHILE or FOR across a range** → strong structural signal: that range is one iterative phase. Do not split it into sequential siblings. The WHILE/FOR node is the parent; any semantic sub-phases of the iterative work become its decomposition children.

**Procedure shows only a flat SEQ with no internal structure** → weak structural signal; the procedure model has no opinion on internal boundaries within that range. Defer to the objective model's semantic clusters for phase decisions.

**Objective shows a clear semantic phase shift** → supports a new boundary even when the procedure model draws a continuous SEQ across it.

**Neither model shows structure for a range** → scan the source `activities` for that range directly. Look for consecutive activities applying the same steps to named items. If a clear FOR pattern emerges, use Change operator (SEQ → FOR with `bindings`).

### WHILE nodes: separating goal from execution strategy

When a WHILE governs iterative attempts toward a result, the node's `objective` and `summary` must express the **ultimate desired outcome** — not the iteration process itself. Trial-and-error is a procedure fact, not a goal.

In `procedure.body`, annotate which attempts failed and why (briefly), and make the successful resolution explicit — so a reader can distinguish dead ends from the working solution without re-reading the trace.

### Structural adjustments

Make an adjustment only when the evidence clearly warrants it; if the current structure is reasonable, leave it.

| Adjustment | When to apply |
|---|---|
| **Split** | A node covers activities the procedure assigns to two distinct phases |
| **Merge** | Siblings are jointly governed by one coherent procedure phase |
| **Insert parent** | Siblings all belong to the same procedure macro-phase |
| **Further decompose** | A leaf covers many activities and the procedure reveals internal structure |
| **Re-parent / move** | A child belongs to a different procedure phase than its siblings |
| **Change operator** | Named items processed identically (SEQ → FOR with `bindings`); repetition-until-condition (SEQ → WHILE) |

## Procedure body: domain operation descriptions

Every body step must describe the **domain operation** performed — the conceptual computation or transformation, stated in terms of the task's domain. Integrate the following into natural prose:

- **What operation** is performed (named in domain terms)
- **What inputs** are consumed (task artifacts, values, or context)
- **What transformation** is applied (algorithm, derivation, logical process, or constraint)
- **What output** is produced (result or artifact that subsequent steps or the node's objective depend on)

For WHILE/FOR phases, the body describes the abstract per-pass/per-item domain-operation template. Omit trace noise: incidental reads, bookkeeping, re-reading unchanged artifacts.

## Coverage and ID rules

- Every source activity must appear in at least one node's `activity_refs`.
- A node's `activity_refs` = union of its children's `activity_refs` (if it has children) or union of its body-step `activity_refs` (if it is a leaf).
- Sibling objective nodes must be disjoint and ordered by their source activities.
- A non-leaf procedure body may cover all or part of its node, but never activities outside it.
- Use compact ranges: `activity_NNNN` or `activity_NNNN-activity_MMMM`.
- Root id = the task-thread id (e.g., `C1`). Children: `C1.1`, `C1.2`, ...; grandchildren: `C1.1.1`, etc. Sequential, no gaps, no reuse.

## Validation workflow

1. Write the candidate JSON.
2. Run: `python validate_unified_model.py output/unified_model.json input/source.json --text`
3. Repair and rerun if validation fails. Finish only after validation passes.

## Required output schema

{unified_schema_text()}
"""


def reconciliation_repair_prompt() -> str:
    return f"""Your previous unified task model output failed deterministic validation. Return only a valid JSON object.

Repair the unified task model using the original inputs, your previous result, and the validation feedback.

Repair checklist:
- Set `version` to `0.2`.
- Preserve non-empty `deliverables` and `success_criteria`, plus `observed_outcome` and `evidence_refs`, on every node.
- Nested deliverable, criterion, and outcome evidence refs must be subsets of the node's `evidence_refs`.
- Every activity episode in the source must appear in at least one node's `activity_refs`.
- `procedure.operator` must be one of: SEQ, FOR, WHILE, CHOICE.
- WHILE nodes must have an operational `procedure.condition` and `procedure.condition_grounding`; the predicate texts must match exactly, the verifier must name a concrete check, evidence refs must stay inside the node, and observed status must reflect the trace.
- FOR nodes must have exactly `procedure.bindings.iteration_variable` and a non-empty `procedure.bindings.collection`.
- Every `procedure.body` step must have a non-empty `name` and non-empty `activity_refs`.
- Node ids must be unique and non-empty. Children must be nested under their parent id (e.g., `C1.2` under `C1`).
- Decomposition child ids must be sequential integers: C1.1, C1.2, C1.3, ...
- `activity_refs` format: `activity_NNNN` or `activity_NNNN-activity_MMMM`.

Required output schema:
{unified_schema_text()}
"""


def codex_reconciliation_prompt(
    *,
    input_source: str = "input/source.json",
    input_objective: str = "input/objective_model.json",
    input_procedure: str = "input/procedure_model.json",
    schema_file: str = "input/unified_schema.json",
    validator_file: str = "input/validate_unified_model.py",
    output_file: str = "output/unified_model.json",
) -> str:
    return f"""You produce a unified task model by reconciling independently-induced objective and procedure models for the same activity trace.

## Workspace

- `{input_source}`: task-thread JSON with `activities` list (each entry is one atomic episode).
- `{input_objective}`: hierarchical objective model.
- `{input_procedure}`: control-flow procedure model.
- `{schema_file}`: JSON schema for the required output.
- `{validator_file}`: deterministic validator; run before finishing.
- `{output_file}`: write your output here.

## What you produce

A unified tree where each node has two layers:

**Objective layer** — a domain-specific program in natural language capturing domain invariants, concrete deliverables, verifiable success criteria, and the outcome established by evidence. Expressed as `objective`, `summary`, `deliverables`, `success_criteria`, `observed_outcome`, `evidence_refs`, and `decomposition` children.

**Procedure layer** — the faithful record of how the work was actually carried out: the real execution path including failures and corrections, at the grain that reveals meaningful structure. Expressed as `procedure.operator`, `procedure.body`, and operator-specific fields.

Both layers are determined jointly from both input models. Neither is authoritative alone.

## Procedure operators

`body` = ordered list of named steps, each with `name`, optional `description`, non-empty `activity_refs`.

| Operator | When | `body` | `activity_refs` per step | `decomposition` |
|---|---|---|---|---|
| `SEQ` | Ordered non-repeating phases | One domain-operation step per phase | That step's range | Recursive child nodes |
| `WHILE` | Repeated until condition holds | Abstract per-pass template; annotate failed attempts | Spans ALL passes | Semantic sub-phases (distinct goal-focus areas), NOT pass enumeration; empty if none |
| `FOR` | Same ops applied to each named item | Abstract per-item template | Spans ALL items | One child per semantically distinct item; none for fungible items |
| `CHOICE` | Mutually exclusive; one observed | Observed branch | Observed range | Observed branch as child |

`WHILE` requires `condition` and `condition_grounding`. `FOR` requires exactly
`bindings.iteration_variable` and a non-empty `bindings.collection`.

{while_condition_grounding_guidance()}

## Grounded objective contract

Preserve `deliverables`, `success_criteria`, `observed_outcome`, and
`evidence_refs` recursively from the objective model. Every node needs at least
one concrete deliverable and one verifiable success criterion. Do not infer an
achieved outcome from an intended success criterion. If reconciliation changes
node boundaries, retain the relevant evidence-backed entries, and keep every
nested evidence ref within the node's `evidence_refs`.

## Reconciliation: joint reasoning at every level

At each level, decide simultaneously: what phases exist, which operator governs each, and what the objective goal of each is. These are one integrated decision.

**Both models draw the same boundary** → honor it.

**Procedure shows WHILE or FOR** → strong structural signal: that range is one iterative phase. The WHILE/FOR node is the parent; any semantic sub-phases become its decomposition children — not sequential siblings.

**Procedure shows only flat SEQ** → weak structural signal; no opinion on internal phase boundaries. Defer to the objective model's semantic clusters.

**Objective shows a clear semantic phase shift** → supports a new boundary even through a continuous SEQ.

**Neither model shows structure** → scan the source `activities` directly. Look for consecutive activities applying the same steps to named items. Use Change operator (SEQ → FOR) if a clear FOR pattern emerges.

**WHILE nodes**: `objective` and `summary` must express the **ultimate desired outcome**, not the iteration process. In `procedure.body`, annotate failed attempts and the successful resolution.

**Structural adjustments** — make one only when the evidence clearly warrants it:
- **Split**: a node covers two procedure-distinct phases
- **Merge**: siblings governed by one coherent procedure phase
- **Insert parent**: siblings all belong to the same procedure macro-phase
- **Further decompose**: a leaf is too coarse; procedure reveals internal structure
- **Re-parent / move**: a child belongs to a different phase than its siblings
- **Change operator**: named items processed identically → FOR; repetition-until-condition → WHILE

## Procedure body quality

Every body step describes the **domain operation**: what operation, what inputs, what transformation, what output — integrated into natural prose. For WHILE/FOR, describe the abstract per-pass/per-item template. Omit trace noise.

## Coverage and IDs

Return canonical version `0.2`. The root must cover every source activity exactly. A node with children equals the disjoint union of those children; a leaf equals the union of its procedure-body steps. A non-leaf body may cover a subset of its node but never outside it. Use compact ranges (`activity_NNNN` or `activity_NNNN-activity_MMMM`). Root id = task-thread id (e.g., `C1`); children: `C1.1`, `C1.2`, ...; grandchildren: `C1.1.1`, etc. Sequential, no gaps, no reuse.

## Workflow

1. Read `{input_source}`, `{input_objective}`, `{input_procedure}`, and `{schema_file}`.
2. Write the candidate JSON to `{output_file}`.
3. Run: `python {validator_file} {output_file} {input_source} --text`
4. Repair and rerun if validation fails.
5. Finish only after validation passes. The final answer can be a short note that validation passed."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    started_at: float = field(default_factory=time.monotonic)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0

    def elapsed_secs(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def record(self, response: Any, model: str) -> None:
        usage = normalize_litellm_usage(response)
        self.llm_requests += 1
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        cost = estimated_litellm_completion_cost_usd(response, model)
        if cost is not None:
            self.estimated_usd += cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_secs": self.elapsed_secs(),
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": round(self.estimated_usd, 6),
        }


@dataclass(frozen=True)
class ModelPair:
    key: str
    input_path: Path | None
    objective_path: Path
    procedure_path: Path
    activity_count: int | None


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class ReconciliationReporter(ConsoleProgressReporter):
    run_name = "bidirectional_alignment"
    success_title = "Bidirectional Alignment Complete"
    failure_title = "Bidirectional Alignment Failed"
    default_failure_stage = STAGE_PREFLIGHT

    def __init__(self, *, no_console: bool = False) -> None:
        super().__init__(stages=STAGES, no_console=no_console)
        self._root_lock = Lock()

    def render(self) -> Any:
        if not all((self._Panel, self._Table, self._Text, self._Group, self._box)):
            return "Bidirectional alignment (reconciliation)"
        return self._Panel(
            self._Group(
                self._Text("Bidirectional Alignment", style="bold cyan"),
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
        return (
            f"elapsed={format_duration(time.monotonic() - self.started_at)} "
            f"direct_model={metrics.get('direct_model', '')} "
            f"codex_model={metrics.get('codex_model', '')} "
            f"roots={counters.get('roots', 0)} "
            f"succeeded={counters.get('succeeded', 0)} "
            f"direct={counters.get('direct_llm', 0)} "
            f"codex={counters.get('codex_cli', 0)} "
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
            if isinstance(item, dict):
                table.add_row(str(item.get("status", "")), str(item.get("name", "")), str(item.get("detail", "")))
        return table

    def _paths_table(self) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        for label, path in self.state.paths.items():
            table.add_row(label, str(path))
        return table

    def initialize_root_statuses(self, pairs: list[ModelPair], mode_map: dict[str, str]) -> None:
        with self._root_lock:
            statuses = []
            for pair in pairs:
                mode = mode_map.get(pair.key, "?")
                n = str(pair.activity_count) if pair.activity_count is not None else "?"
                statuses.append({"name": pair.objective_path.stem, "status": "queued", "detail": f"n={n} mode={mode}"})
            self.set_metric("root_statuses", statuses)

    def update_root_status(self, pair: ModelPair, status: str, detail: str) -> None:
        with self._root_lock:
            current = self.state.metrics.get("root_statuses") or []
            name = pair.objective_path.stem
            updated = False
            for item in current:
                if isinstance(item, dict) and item.get("name") == name:
                    item["status"] = status
                    item["detail"] = detail
                    updated = True
                    break
            if not updated:
                current.append({"name": name, "status": status, "detail": detail})
            self.set_metric("root_statuses", current)


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile objective and procedure models into a unified task model.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Pipeline data directory.")
    parser.add_argument("--config", type=Path, default=None, help="Task model induction config path.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N task threads.")
    parser.add_argument("--preflight-only", "--preflight_only", dest="preflight_only", action="store_true")
    parser.add_argument("--no_console", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def activity_count(input_path: Path | None) -> int | None:
    if input_path is None:
        return None
    try:
        source = read_json(input_path)
        activities = source.get("activities")
        return len(activities) if isinstance(activities, list) else None
    except Exception:
        return None


def extract_json_from_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _expanded_objective_refs(refs: Any) -> set[int]:
    if not isinstance(refs, list):
        return set()
    expanded: set[int] = set()
    for ref in refs:
        if not isinstance(ref, str):
            continue
        match = OBJECTIVE_ACTIVITY_RE.fullmatch(ref.strip())
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if start <= end:
            expanded.update(range(start, end + 1))
    return expanded


def preserve_unchanged_objective_grounding(
    candidate: dict[str, Any],
    objective_model: dict[str, Any],
) -> dict[str, Any]:
    """Copy Step 4 grounding onto unified nodes whose id and coverage are unchanged."""

    result = deepcopy(candidate)
    root = result.get("root")
    if not isinstance(root, dict):
        return result

    source_by_id: dict[str, tuple[set[int], dict[str, Any]]] = {}

    def index_source(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id:
            source_by_id[node_id] = (
                _expanded_objective_refs(node.get("subgoal_segments")),
                node,
            )
        for child in node.get("decomposition") or []:
            index_source(child)

    index_source(objective_model)

    def apply(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_id = node.get("id")
        indexed = source_by_id.get(node_id) if isinstance(node_id, str) else None
        if indexed is not None:
            source_refs, source_node = indexed
            candidate_refs = _expanded_objective_refs(node.get("activity_refs"))
            if source_refs and source_refs == candidate_refs:
                for field in GROUNDING_FIELDS:
                    if field in source_node:
                        node[field] = deepcopy(source_node[field])
        for child in node.get("decomposition") or []:
            apply(child)

    apply(root)
    return result


# ---------------------------------------------------------------------------
# Model discovery (reads from merged outputs of step4 + step5)
# ---------------------------------------------------------------------------

def _load_merged_roots(path: Path, model_key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    roots = payload.get("roots")
    if not isinstance(roots, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in roots:
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        output_file = item.get("output_file")
        model = item.get(model_key)
        if not isinstance(output_file, str) or not isinstance(model, dict):
            continue
        key = Path(str(item.get("input_file") or output_file)).name
        result[key] = {
            "path": Path(output_file),
            "model": model,
            "input_file": item.get("input_file"),
        }
    return result


def discover_models(
    data_dir: Path,
    objective_output_dir: str,
    procedure_output_dir: str,
) -> list[ModelPair]:
    objective_roots = _load_merged_roots(data_dir / "hierarchy.json", "hierarchy")
    procedure_roots = _load_merged_roots(data_dir / "task_model_with_procedures.json", "procedure_task_model")

    # Fallback: scan directories directly
    if not objective_roots:
        obj_dir = data_dir / objective_output_dir
        for path in sorted(obj_dir.glob("*.json")) if obj_dir.exists() else []:
            if path.name.endswith(".meta.json") or path.name == "manifest.json":
                continue
            objective_roots[path.name] = {"path": path, "model": {}, "input_file": None}

    if not procedure_roots:
        proc_dir = data_dir / procedure_output_dir
        for path in sorted(proc_dir.glob("*.json")) if proc_dir.exists() else []:
            if path.name.endswith(".meta.json") or path.name == "manifest.json":
                continue
            procedure_roots[path.name] = {"path": path, "model": {}, "input_file": None}

    # Single-pair case with mismatched names
    if len(objective_roots) == 1 and len(procedure_roots) == 1 and not (set(objective_roots) & set(procedure_roots)):
        obj_key = next(iter(objective_roots))
        proc_key = next(iter(procedure_roots))
        objective_roots[proc_key] = objective_roots.pop(obj_key)

    pairs: list[ModelPair] = []
    for key in sorted(set(objective_roots) & set(procedure_roots)):
        obj = objective_roots[key]
        proc = procedure_roots[key]
        input_file = obj.get("input_file") or proc.get("input_file")
        input_path: Path | None = None
        if isinstance(input_file, str):
            p = Path(input_file)
            if not p.is_absolute():
                p = data_dir / p
            if p.exists():
                input_path = p.resolve()
        pairs.append(
            ModelPair(
                key=key,
                input_path=input_path,
                objective_path=Path(obj["path"]),
                procedure_path=Path(proc["path"]),
                activity_count=activity_count(input_path),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Direct LLM reconciliation (small traces)
# ---------------------------------------------------------------------------

def _call_llm_json(
    *,
    model: str,
    system_prompt: str,
    content: dict[str, Any],
    llm_timeout_secs: float,
    stats: RunStats,
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
    stats.record(response, model)
    return extract_json_from_response(completion_message_content(response))


def run_direct_reconciliation(
    *,
    pair: ModelPair,
    output_path: Path,
    model: str,
    llm_timeout_secs: float,
    max_retries: int,
) -> dict[str, Any]:
    stats = RunStats()
    objective = read_json(pair.objective_path)
    procedure = read_json(pair.procedure_path)
    source = read_json(pair.input_path) if pair.input_path else None

    content: dict[str, Any] = {
        "objective_model": objective,
        "procedure_model": procedure,
    }
    if source is not None:
        content["source"] = source

    candidate = preserve_unchanged_objective_grounding(
        _call_llm_json(
            model=model,
            system_prompt=reconciliation_generation_prompt(),
            content=content,
            llm_timeout_secs=llm_timeout_secs,
            stats=stats,
        ),
        objective,
    )

    feedback = validate_unified_output(candidate, source=source)
    retry_count = 0
    while not feedback.valid and retry_count < max_retries:
        retry_count += 1
        candidate = preserve_unchanged_objective_grounding(
            _call_llm_json(
                model=model,
                system_prompt=reconciliation_repair_prompt(),
                content={
                    **content,
                    "previous_result": candidate,
                    "validation_feedback": feedback.as_text(),
                },
                llm_timeout_secs=llm_timeout_secs,
                stats=stats,
            ),
            objective,
        )
        feedback = validate_unified_output(candidate, source=source)

    if not feedback.valid:
        _write_direct_meta(
            output_path,
            pair=pair,
            model=model,
            max_retries=max_retries,
            retry_count=retry_count,
            valid=False,
            feedback_text=feedback.as_text(),
            stats=stats,
        )
        raise RuntimeError(f"reconciliation validation failed for {pair.objective_path.name}: {feedback.as_text()}")

    payload = UnifiedTaskModel.model_validate(candidate).model_dump()
    write_json_atomic(output_path, payload)
    _write_direct_meta(
        output_path,
        pair=pair,
        model=model,
        max_retries=max_retries,
        retry_count=retry_count,
        valid=True,
        feedback_text=feedback.as_text(),
        stats=stats,
    )
    usage = stats.as_dict()
    return {
        "execution_mode": "direct_llm",
        "usage": {k: usage[k] for k in ("llm_requests", "input_tokens", "output_tokens", "total_tokens")},
        "estimated_usd": round(stats.estimated_usd, 6),
        "task_model": payload,
    }


def _write_direct_meta(
    output_path: Path,
    *,
    pair: ModelPair,
    model: str,
    max_retries: int,
    retry_count: int,
    valid: bool,
    feedback_text: str,
    stats: RunStats,
) -> None:
    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(pair.input_path) if pair.input_path else None,
        "objective_path": str(pair.objective_path),
        "procedure_path": str(pair.procedure_path),
        "output_path": str(output_path),
        "model": model,
        "max_retries": max_retries,
        "retry_count": retry_count,
        "valid": valid,
        "validation_feedback": feedback_text,
        "execution_mode": "direct_llm",
        "activity_count": pair.activity_count,
        "cost": stats.as_dict(),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)


# ---------------------------------------------------------------------------
# Codex sandbox reconciliation (large traces)
# ---------------------------------------------------------------------------

def run_codex_reconciliation(
    *,
    pair: ModelPair,
    output_path: Path,
    codex_config: dict[str, Any],
    rebuild_image: bool,
) -> dict[str, Any]:
    source_text = pair.input_path.read_text(encoding="utf-8") if pair.input_path else "null\n"
    result = CodexCliSandbox().run_file_task(
        prompt=codex_reconciliation_prompt(),
        files={
            "source.json": source_text,
            "objective_model.json": pair.objective_path.read_text(encoding="utf-8"),
            "procedure_model.json": pair.procedure_path.read_text(encoding="utf-8"),
            "unified_schema.json": unified_schema_text(),
            "validate_unified_model.py": validator_script_text(),
        },
        output_file="unified_model.json",
        codex_config=codex_config,
        rebuild_image=rebuild_image,
    )
    result["execution_mode"] = "codex_cli"
    result["activity_count"] = pair.activity_count

    codex_usage = extract_max_usage_from_json_events(
        str(result.get("stdout") or ""), str(result.get("stderr") or "")
    )
    estimated_usd = _estimate_usd(codex_usage, str(codex_config.get("model") or ""))
    if codex_usage:
        result["usage"] = codex_usage
    if estimated_usd is not None:
        result["estimated_usd"] = estimated_usd

    if not result.get("ok"):
        raise RuntimeError(_codex_failure_message(result))

    output_content = result.get("output_content")
    if not isinstance(output_content, str) or not output_content.strip():
        raise RuntimeError("Codex run succeeded but returned no unified_model.json")

    try:
        parsed = preserve_unchanged_objective_grounding(
            json.loads(output_content),
            read_json(pair.objective_path),
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"returned unified_model.json is invalid JSON: {exc}") from exc

    source = json.loads(source_text) if source_text.strip() and source_text.strip() != "null" else None
    feedback = validate_unified_output(parsed, source=source)
    if not feedback.valid:
        raise RuntimeError("returned unified_model.json failed local validation: " + feedback.as_text())

    payload = UnifiedTaskModel.model_validate(parsed).model_dump()
    write_json_atomic(output_path, payload)

    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(pair.input_path) if pair.input_path else None,
        "objective_path": str(pair.objective_path),
        "procedure_path": str(pair.procedure_path),
        "output_path": str(output_path),
        "run_id": result.get("run_id"),
        "session_id": result.get("session_id"),
        "model": (result.get("codex") or {}).get("model") or codex_config.get("model"),
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "usage": codex_usage,
        "estimated_usd": estimated_usd,
        "proxy_cost": result.get("proxy_cost"),
        "execution_mode": "codex_cli",
        "activity_count": pair.activity_count,
        "validation_feedback": feedback.as_text(),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)

    result["task_model"] = payload
    return result


def _codex_failure_message(result: dict[str, Any]) -> str:
    codex_config = result.get("codex") or {}
    details: list[str] = []
    for key in ("run_id", "returncode"):
        if result.get(key) not in (None, 0, ""):
            details.append(f"{key}={result[key]}")
    for key in ("model", "model_reasoning_effort"):
        if codex_config.get(key):
            details.append(f"{key}={codex_config[key]}")
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    if stdout:
        details.append(f"stdout={stdout[-500:]}")
    if stderr:
        details.append(f"stderr={stderr[-500:]}")
    message = str(result.get("error") or "Codex reconciliation run failed")
    return f"{message} ({'; '.join(details)})" if details else message


def _estimate_usd(usage: dict[str, int], model: str) -> float | None:
    estimated = estimate_litellm_usage_cost_usd(usage, model)
    return round(estimated, 6) if estimated is not None else None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cached_root_result(
    *,
    pair: ModelPair,
    output_path: Path,
    execution_mode: str,
) -> dict[str, Any] | None:
    if not output_path.exists():
        return None

    try:
        raw_payload = preserve_unchanged_objective_grounding(
            read_json(output_path),
            read_json(pair.objective_path),
        )
        source = read_json(pair.input_path) if pair.input_path else None
        feedback = validate_unified_output(raw_payload, source=source)
        if not feedback.valid:
            return None
        payload = UnifiedTaskModel.model_validate(raw_payload).model_dump()
    except Exception:
        return None

    if payload != read_json(output_path):
        write_json_atomic(output_path, payload)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded_meta = read_json(meta_path)
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
        except Exception:
            meta = {}

    return {
        "input_file": str(pair.input_path) if pair.input_path else None,
        "objective_file": str(pair.objective_path),
        "procedure_file": str(pair.procedure_path),
        "output_file": str(output_path),
        "ok": True,
        "run_id": meta.get("run_id"),
        "session_id": meta.get("session_id"),
        "execution_mode": str(meta.get("execution_mode") or execution_mode),
        "activity_count": pair.activity_count,
        "usage": meta.get("usage") or meta.get("cost"),
        "estimated_usd": meta.get("estimated_usd") or (meta.get("cost") or {}).get("estimated_usd"),
        "proxy_cost": meta.get("proxy_cost"),
        "task_model": payload,
    }


# ---------------------------------------------------------------------------
# Merged output
# ---------------------------------------------------------------------------

def _sum_root_usage(results: list[dict[str, Any]]) -> dict[str, int]:
    return sum_usage_dicts([item.get("usage") for item in results if isinstance(item, dict)])


def write_merged_output(
    output_path: Path,
    output_dir: Path,
    results: list[dict[str, Any]],
    *,
    model: str,
    preflight_only: bool,
) -> dict[str, Any]:
    usage = _sum_root_usage(results)
    estimated = round(sum(float(item.get("estimated_usd") or 0.0) for item in results), 6)
    cost = {"source": "estimated_from_usage", "total_usd": estimated, "usage": usage} if usage or estimated else None
    payload_model = UnifiedTaskModelOutput(
        meta=UnifiedTaskModelMergedMeta(
            created_at=utc_now_iso(),
            model=model,
            output_dir=str(output_dir),
            num_roots=len(results),
            num_succeeded=sum(1 for item in results if item.get("ok")),
            preflight_only=preflight_only,
            cost=cost,
        ),
        roots=[UnifiedTaskModelRootResult.model_validate(item) for item in results],
    )
    payload = payload_model.model_dump(mode="json")
    write_json_atomic(output_path, payload)
    if cost is not None:
        write_json_atomic(output_path.with_suffix(output_path.suffix + ".cost.json"), cost)
    return payload


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def bidirectional_alignment(
    *,
    data_dir: str | Path,
    objective_output_dir: str,
    procedure_output_dir: str,
    output_dir: str,
    output_file_name: str,
    direct_model: str,
    direct_litellm_params: dict[str, Any] | None,
    direct_llm_max_activities: int,
    codex_config: dict[str, Any],
    max_retries: int,
    workers: int,
    llm_timeout_secs: float,
    limit: int | None = None,
    reuse_cache: bool = False,
    preflight_only: bool = False,
    no_console: bool = False,
    rebuild_codex_sandbox: bool = False,
) -> dict[str, Any] | None:
    with ReconciliationReporter(no_console=no_console) as reporter:
        try:
            reporter.set_metric("direct_model", direct_model)
            reporter.set_metric("codex_model", codex_config.get("model") or "")
            reporter.start_stage(STAGE_LOAD_INPUTS, "discovering model pairs")

            resolved_data_dir = Path(data_dir).expanduser()
            resolved_output_dir = resolved_data_dir / output_dir
            resolved_output = resolved_data_dir / output_file_name

            pairs = discover_models(resolved_data_dir, objective_output_dir, procedure_output_dir)
            if limit is not None:
                pairs = pairs[:limit]

            reporter.add_path("data_dir", resolved_data_dir)
            reporter.add_path("output_dir", resolved_output_dir)
            reporter.add_path("output", resolved_output)
            reporter.set_counter("roots", len(pairs))
            reporter.finish_stage(STAGE_LOAD_INPUTS, f"found {len(pairs)} model pairs")

            reporter.start_stage(STAGE_PREFLIGHT, "validating planned work")
            if workers < 1:
                raise ValueError("workers must be >= 1")
            if not pairs:
                raise FileNotFoundError("No objective/procedure model pairs found.")
            reporter.finish_stage(STAGE_PREFLIGHT, "ready")

            # Assign execution mode per pair
            mode_map: dict[str, str] = {}
            for pair in pairs:
                count = pair.activity_count
                mode = (
                    "direct_llm"
                    if count is not None
                    and count < direct_llm_max_activities
                    and not direct_llm_input_too_large(
                        pair.objective_path, pair.procedure_path, pair.input_path
                    )
                    else "codex_cli"
                )
                if mode == "codex_cli" and count is not None and count < direct_llm_max_activities:
                    reporter.progress(
                        f"routing {pair.key} to codex_cli: inputs too large for direct LLM"
                    )
                mode_map[pair.key] = mode

            reporter.initialize_root_statuses(pairs, mode_map)
            reporter.set_counter("direct_llm", sum(1 for m in mode_map.values() if m == "direct_llm"))
            reporter.set_counter("codex_cli", sum(1 for m in mode_map.values() if m == "codex_cli"))

            if preflight_only:
                preflight_results = [
                    {
                        "input_file": str(pair.input_path) if pair.input_path else None,
                        "objective_file": str(pair.objective_path),
                        "procedure_file": str(pair.procedure_path),
                        "output_file": str(resolved_output_dir / pair.key),
                        "ok": True,
                        "execution_mode": mode_map[pair.key],
                        "activity_count": pair.activity_count,
                    }
                    for pair in pairs
                ]
                reporter.set_counter("succeeded", len(preflight_results))
                reporter.mark_stage_done(STAGE_RECONCILIATION, "skipped")
                reporter.start_stage(STAGE_MERGE, "writing preflight output")
                payload = write_merged_output(
                    resolved_output,
                    resolved_output_dir,
                    preflight_results,
                    model=direct_model if all(m == "direct_llm" for m in mode_map.values()) else str(codex_config.get("model") or ""),
                    preflight_only=True,
                )
                reporter.finish_stage(STAGE_MERGE, "wrote preflight output")
                reporter.final_success("preflight complete; no LLM or Codex calls were made")
                return payload

            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            reporter.start_stage(STAGE_RECONCILIATION, "reconciling model pairs")
            reporter.set_metric("parallelism", max(1, min(workers, len(pairs))))

            results: list[dict[str, Any]] = []
            errors: list[str] = []
            max_workers = max(1, min(workers, len(pairs)))

            def runner(pair: ModelPair, out_path: Path, execution_mode: str) -> dict[str, Any]:
                n = str(pair.activity_count) if pair.activity_count is not None else "?"
                if reuse_cache:
                    cached = load_cached_root_result(pair=pair, output_path=out_path, execution_mode=execution_mode)
                    if cached is not None:
                        reporter.update_root_status(pair, "done", f"n={n} cache")
                        reporter.progress(f"loaded cache for {pair.objective_path.stem}")
                        return cached
                reporter.update_root_status(pair, "running", f"n={n} mode={execution_mode}")
                reporter.progress(f"running {pair.objective_path.stem} (n={n}, {execution_mode})")
                if execution_mode == "direct_llm":
                    with litellm_model_config(model_alias=direct_model, litellm_params=direct_litellm_params):
                        run_result = run_direct_reconciliation(
                            pair=pair,
                            output_path=out_path,
                            model=direct_model,
                            llm_timeout_secs=llm_timeout_secs,
                            max_retries=max_retries,
                        )
                else:
                    run_result = run_with_retries(
                        lambda: run_codex_reconciliation(
                            pair=pair,
                            output_path=out_path,
                            codex_config=codex_config,
                            rebuild_image=rebuild_codex_sandbox,
                        ),
                        attempts=max_retries,
                        on_retry=lambda attempt, exc: reporter.progress(
                            f"codex retry {attempt}/{max_retries} for {pair.key}: {str(exc)[:200]}"
                        ),
                    )
                reporter.update_root_status(pair, "done", f"n={n} mode={execution_mode}")
                reporter.progress(f"finished {pair.objective_path.stem}")
                return run_result

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(
                        runner,
                        pair,
                        resolved_output_dir / pair.key,
                        mode_map[pair.key],
                    ): pair
                    for pair in pairs
                }
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    out_path = resolved_output_dir / pair.key
                    try:
                        run_result = future.result()
                        results.append({
                            "input_file": run_result.get("input_file", str(pair.input_path) if pair.input_path else None),
                            "objective_file": run_result.get("objective_file", str(pair.objective_path)),
                            "procedure_file": run_result.get("procedure_file", str(pair.procedure_path)),
                            "output_file": run_result.get("output_file", str(out_path)),
                            "ok": True,
                            "run_id": run_result.get("run_id"),
                            "session_id": run_result.get("session_id"),
                            "execution_mode": run_result.get("execution_mode", mode_map[pair.key]),
                            "activity_count": run_result.get("activity_count", pair.activity_count),
                            "usage": run_result.get("usage"),
                            "estimated_usd": run_result.get("estimated_usd"),
                            "proxy_cost": run_result.get("proxy_cost"),
                            "task_model": run_result.get("task_model"),
                        })
                        reporter.increment("succeeded")
                    except Exception as exc:
                        n = str(pair.activity_count) if pair.activity_count is not None else "?"
                        reporter.update_root_status(pair, "failed", f"n={n} {exc}")
                        errors.append(f"{pair.key}: {exc}")
                        results.append({
                            "input_file": str(pair.input_path) if pair.input_path else None,
                            "objective_file": str(pair.objective_path),
                            "procedure_file": str(pair.procedure_path),
                            "ok": False,
                            "error": str(exc),
                            "execution_mode": mode_map[pair.key],
                            "activity_count": pair.activity_count,
                        })
                    reporter.progress(f"completed {len(results)}/{len(pairs)} roots")

            results.sort(key=lambda item: str(item.get("objective_file") or ""))
            usage = _sum_root_usage(results)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                reporter.set_metric(key, usage.get(key, 0))
            reporter.set_metric(
                "llm_requests",
                sum(
                    int((item.get("usage") or {}).get("llm_requests") or 0)
                    for item in results
                    if isinstance(item, dict)
                ),
            )
            reporter.set_metric("estimated_usd", sum(float(item.get("estimated_usd") or 0.0) for item in results))
            reporter.finish_stage(STAGE_RECONCILIATION, "completed")

            reporter.start_stage(STAGE_MERGE, "writing merged output")
            model_label = (
                direct_model
                if all(m == "direct_llm" for m in mode_map.values())
                else str(codex_config.get("model") or "")
            )
            payload = write_merged_output(
                resolved_output,
                resolved_output_dir,
                results,
                model=model_label,
                preflight_only=False,
            )
            reporter.finish_stage(STAGE_MERGE, "saved output")

            if errors:
                raise RuntimeError("one or more reconciliation runs failed: " + "; ".join(errors))
            reporter.final_success(f"reconciled {len(results)} task-thread model pairs")
            return payload

        except Exception as exc:
            reporter.fail_active_stage(exc)
            reporter.final_failure()
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve() if args.config else resolve_config_path()
    config = load_config(config_path)
    if config.dotenv_path:
        try:
            from dotenv import load_dotenv
            load_dotenv(resolve_dotenv_path(config_path, config.dotenv_path), override=False)
        except ModuleNotFoundError:
            pass

    stage = config.bidirectional_alignment_stage
    direct_branch = stage.direct_llm_branch
    codex_branch = stage.codex_branch
    codex_config = {
        "model": codex_branch.model,
        "model_reasoning_effort": codex_branch.model_reasoning_effort,
        "personality": codex_branch.personality,
        "model_provider": codex_branch.model_provider,
        "provider_name": codex_branch.provider_name,
        "command_timeout_seconds": codex_branch.command_timeout_seconds,
        "litellm_params": codex_branch.litellm_params,
    }
    bidirectional_alignment(
        data_dir=args.data_dir,
        objective_output_dir=stage.objective_output_dir,
        procedure_output_dir=stage.procedure_output_dir,
        output_dir=stage.output_dir,
        output_file_name=stage.output_file_name,
        direct_model=direct_branch.model,
        direct_litellm_params=direct_branch.litellm_params,
        direct_llm_max_activities=direct_branch.direct_llm_max_activities,
        codex_config=codex_config,
        max_retries=stage.max_retries,
        workers=stage.workers,
        llm_timeout_secs=stage.llm_timeout_seconds,
        limit=args.limit,
        reuse_cache=stage.reuse_cache,
        preflight_only=args.preflight_only,
        no_console=args.no_console,
        rebuild_codex_sandbox=stage.rebuild_codex_sandbox,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
