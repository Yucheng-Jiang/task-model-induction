#!/usr/bin/env python3
"""Generate and validate a recursive hierarchical objective.

Input can be either:
- a flat task-thread objective file with ``task_thread_objective`` and
  ``activities`` entries; or
- an existing hierarchy-like JSON file, such as a file with ``root_objective``.

The output hierarchy uses the recursive schema:

{
  "id": "...",
  "objective": "...",
  "summary": "...",
  "deliverables": [{"kind": "...", "target": "...", "expected_state": "...", "evidence_refs": ["..."]}],
  "success_criteria": [{"predicate": "...", "verifier": "...", "evidence_refs": ["..."], "confidence": 0.0}],
  "observed_outcome": {"status": "unknown", "description": "...", "evidence_refs": ["..."]},
  "evidence_refs": ["..."],
  "subgoal_segments": ["subgoal_segment_0000", "subgoal_segment_0001-subgoal_segment_0004"],
  "decomposition": [ ... same schema recursively ... ]
}

If deterministic validation fails, the script composes constructive feedback and
asks the LLM to repair the previous result using the feedback and original input.
The repair loop runs up to ``--max-retries`` times.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import ValidationError

try:
    from task_model_induction.codex_cli_sandbox import CodexCliSandbox
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.reporting.model_readable_report import write_objective_collection_markdown, write_objective_markdown
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.schemas import HierarchicalObjectiveInductionOutput, HierarchicalObjectiveNode
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
except ModuleNotFoundError:
    from codex_cli_sandbox import CodexCliSandbox
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.model_readable_report import write_objective_collection_markdown, write_objective_markdown
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import HierarchicalObjectiveInductionOutput, HierarchicalObjectiveNode
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


DEFAULT_MAX_RETRIES = 3
DEFAULT_TASK_THREAD_OUTPUT_DIR = "task_thread_objective_model"

STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_GENERATION = "hierarchy generation"
STAGE_MERGE = "merge output"
STAGES = [STAGE_LOAD_INPUTS, STAGE_PREFLIGHT, STAGE_GENERATION, STAGE_MERGE]
SUBGOAL_SEGMENT_RE = re.compile(
    r"^(?P<prefix>subgoal_segment|activity)_(?P<start>\d{4})"
    r"(?:-(?P=prefix)_(?P<end>\d{4}))?$"
)
COMPACT_SEGMENT_RE = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")
SEMANTIC_ACTION_RE = re.compile(r"^semantic_action_(?P<idx>\d{4})$")
def _objective_style_guidance(
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> str:
    return f"""Objectives follow the computational thinking paradigm of recursive decomposition:
- Each node states a SUB-GOAL: what needs to be accomplished at this level to advance the parent goal. The hierarchy of sub-goals, taken together, covers the full task.
- Write as a goal to be achieved (e.g. "Compile X", "Gather Y for each Z", "Verify W is ready") — not as a low-level procedure (which tools, which steps) and not as a passive state predicate.
- Objectives must be TOOL-AGNOSTIC and USER-AGNOSTIC: two users who accomplish the same sub-task through completely different means share the same sub-goal statement. Do not reference specific applications, UI elements, file paths, or step-by-step procedures in an `objective` field.
- Save concrete and evidential details — which artifact/state is produced, how success can be checked, which tool was used, which file was opened, and how the state was reached — for the grounding fields and `summary`, NOT the `objective` field.
- Ground EVERY node with at least one concrete `deliverables` entry and one verifiable `success_criteria` entry. Use OCR text, content-of-interest entities, before/after state, and action evidence from the input when available.
- `observed_outcome` records what the trace actually establishes, independently of intended success. Never mark it `achieved` merely because a success criterion was inferred; use `unknown` or `partial` unless evidence verifies the outcome.
- Put the evidence ids used by the node in `evidence_refs`, and repeat the relevant subset on each deliverable, success criterion, and observed outcome.

Granularity rules:
- The ROOT node must decompose whenever the input holds more than one activity/subgoal segment. An empty root "decomposition" is valid only when the input holds exactly one activity. This rule wins over every size guideline below.
- A child node must represent a sub-outcome that is a necessary precondition or component of the parent outcome — ask "what smaller success state contributes to the parent outcome?".
- If a node already represents one coherent atomic success state, use "decomposition": [] instead of inventing procedural children.
- A node covering exactly one activity/subgoal segment must not have decomposition.
- Prefer not to create a decomposed child that covers exactly one activity/subgoal segment; merge it into the parent or group it with a sibling unless it represents a genuinely distinct sub-goal.
- Nodes covering more than {large_node_review_threshold} activities/subgoal segments should be reviewed for possible further decomposition.
- Non-root nodes covering fewer than {small_decomposition_review_threshold} activities/subgoal segments should usually remain undecomposed unless the child objectives are genuinely necessary.
"""

def generation_prompt_text(
    *,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> str:
    return f"""You induce a hierarchical objective model from task activity observations using computational thinking and recursive decomposition.

The input contains activity segments describing WHAT A USER DID. Your job is to abstract over those actions and recover the hierarchy of SUB-GOALS they were pursuing — the recursive decomposition of the task into what needs to be accomplished at each level.
Return only a valid JSON object.

{_objective_style_guidance(
    large_node_review_threshold=large_node_review_threshold,
    small_decomposition_review_threshold=small_decomposition_review_threshold,
)}

Required output schema, recursively:
{{
  "id": "<stable hierarchical id, e.g. C1 or C1.1>",
  "objective": "<sub-goal: what needs to be accomplished at this level, tool-agnostic and user-agnostic>",
  "summary": "<brief evidence-grounded summary; may reference specific tools, files, or actions observed>",
  "deliverables": [
    {{"kind": "<file|message|record|state|other>", "target": "<concrete artifact or state>", "expected_state": "<state the deliverable must reach>", "evidence_refs": ["<source evidence id>"]}}
  ],
  "success_criteria": [
    {{"predicate": "<independently checkable condition>", "verifier": "<after_state_ocr|state_delta|action_result|human|unknown>", "evidence_refs": ["<source evidence id>"], "confidence": <0.0-1.0 or null>}}
  ],
  "observed_outcome": {{"status": "<achieved|partial|failed|abandoned|unknown>", "description": "<what the observations establish>", "evidence_refs": ["<source evidence id>"]}},
  "evidence_refs": ["<source evidence id>"],
  "subgoal_segments": [
    "<single integer id such as 16 or closed integer range string such as 16-23>"
  ],
  "decomposition": [<child nodes with the same schema> or <empty list if no further decomposition is needed>]
}}

Use the integer local-objective ids shown in the input. For example, if the input
contains activity_id 16 through 23, write "subgoal_segments": ["16-23"].
"""


def repair_prompt_text(
    *,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> str:
    return f"""Your previous hierarchical objective output failed deterministic validation.
Return only a valid JSON object.

Repair the hierarchy using the original input, your previous result, and the validation feedback.
While repairing, also check that every `objective` field states a sub-goal (what needs to be accomplished at that level) rather than a low-level action narration or passive state predicate — rewrite any that do not.

{_objective_style_guidance(
    large_node_review_threshold=large_node_review_threshold,
    small_decomposition_review_threshold=small_decomposition_review_threshold,
)}

Required output schema, recursively:
{{
  "id": "<stable hierarchical id>",
  "objective": "<sub-goal: what needs to be accomplished at this level, tool-agnostic and user-agnostic>",
  "summary": "<brief evidence-grounded summary; may reference specific tools, files, or actions observed>",
  "deliverables": [
    {{"kind": "<artifact/state kind>", "target": "<concrete artifact or state>", "expected_state": "<required state>", "evidence_refs": ["<source evidence id>"]}}
  ],
  "success_criteria": [
    {{"predicate": "<checkable condition>", "verifier": "<verification source or unknown>", "evidence_refs": ["<source evidence id>"], "confidence": <0.0-1.0 or null>}}
  ],
  "observed_outcome": {{"status": "<achieved|partial|failed|abandoned|unknown>", "description": "<what was actually observed>", "evidence_refs": ["<source evidence id>"]}},
  "evidence_refs": ["<source evidence id>"],
  "subgoal_segments": [
    "<single integer id such as 16 or closed integer range string such as 16-23>"
  ],
  "decomposition": [<child nodes with the same schema>]
}}

Use the integer local-objective ids shown in the input. For example, if the input
contains activity_id 16 through 23, write "subgoal_segments": ["16-23"].
"""


def output_schema_text() -> str:
    return json.dumps(HierarchicalObjectiveNode.model_json_schema(), indent=2, ensure_ascii=True)


def induction_prompt(
    *,
    input_file: str = "input/task_thread_objectives.json",
    schema_file: str = "input/output_schema.json",
    validator_file: str = "input/validate_hierarchy.py",
    output_file: str = "output/hierarchy.json",
    large_node_review_threshold: int = 20,
    small_decomposition_review_threshold: int = 5,
) -> str:
    return f"""You induce a hierarchical objective model from task activity observations using computational thinking and recursive decomposition.

The input contains activity segments describing WHAT A USER DID. Your job is to abstract over those actions and recover the hierarchy of SUB-GOALS they were pursuing — the recursive decomposition of the task into what needs to be accomplished at each level.

Input files:
- `{input_file}`: the task-thread objective JSON to analyze.
- `{schema_file}`: the exact recursive JSON schema for the required output.
- `{validator_file}`: deterministic validator you must run before finishing.

{_objective_style_guidance(
    large_node_review_threshold=large_node_review_threshold,
    small_decomposition_review_threshold=small_decomposition_review_threshold,
)}

Required output schema, recursively:
{{
  "id": "<stable hierarchical id, e.g. C1. C1.1, and so on>",
  "objective": "<sub-goal: what needs to be accomplished at this level, tool-agnostic and user-agnostic>",
  "summary": "<brief evidence-grounded summary; may reference specific tools, files, or actions observed>",
  "deliverables": [
    {{"kind": "<artifact/state kind>", "target": "<concrete artifact or state>", "expected_state": "<required state>", "evidence_refs": ["<source evidence id>"]}}
  ],
  "success_criteria": [
    {{"predicate": "<checkable condition>", "verifier": "<verification source or unknown>", "evidence_refs": ["<source evidence id>"], "confidence": <0.0-1.0 or null>}}
  ],
  "observed_outcome": {{"status": "<achieved|partial|failed|abandoned|unknown>", "description": "<what was actually observed>", "evidence_refs": ["<source evidence id>"]}},
  "evidence_refs": ["<source evidence id>"],
  "subgoal_segments": [
    "<single source id such as activity_0016 or closed range such as activity_0016-activity_0023>"
  ],
  "decomposition": [<child nodes with the same schema> or <empty if further decomposition does not make meaningful abstraction of objective>]
}}

Workflow:
1. Read `{input_file}` and `{schema_file}`.
2. Write the candidate hierarchy JSON to `{output_file}`.
3. Run `python {validator_file} {output_file} {input_file} --text --large-node-review-threshold {large_node_review_threshold} --small-decomposition-review-threshold {small_decomposition_review_threshold}`.
4. If validation fails, use the feedback to repair `{output_file}` and rerun validation.
5. Review every leaf node before writing the final file. If a leaf covers multiple meaningfully distinct sub-outcomes, decompose it further. If changes are made, run validation again.
6. Finish only after validation passes.

Write only the final JSON file at `{output_file}`. The final answer can be a short note that validation passed."""


def validator_script_text() -> str:
    return (Path(__file__).resolve().parent / "validate" / "validate_hierarchy.py").read_text(encoding="utf-8")


@dataclass(frozen=True)
class ValidationFeedback:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def as_text(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) if lines else "No validation issues."


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
                "operation": operation,
                "model": model,
                "llm_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_usd": 0.0,
            },
        )
        bucket["llm_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[key] += int(usage.get(key, 0))
        if estimated_usd is not None:
            bucket["estimated_usd"] = round(float(bucket["estimated_usd"]) + estimated_usd, 6)

    def cost_breakdown(self) -> dict[str, Any]:
        return {
            "total": {
                "llm_requests": self.llm_requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "estimated_usd": round(self.estimated_usd, 6),
            },
            "by_operation": list(self.breakdown.values()),
        }


_ACTIVE_STATS: RunStats | None = None
_METRICS_ENABLED = True


class HierarchicalObjectiveReporter(ConsoleProgressReporter):
    run_name = "hierarchical_objective_induction"
    success_title = "Hierarchical Objective Induction Complete"
    failure_title = "Hierarchical Objective Induction Failed"
    default_failure_stage = STAGE_PREFLIGHT

    def __init__(self, *, no_console: bool = False) -> None:
        super().__init__(stages=STAGES, no_console=no_console)
        self._root_status_lock = Lock()

    def render(self) -> Any:
        if not all((self._Panel, self._Table, self._Text, self._Group, self._box)):
            return "Hierarchical objective induction"
        return self._Panel(
            self._Group(
                self._Text("Hierarchical Objective Induction", style="bold cyan"),
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
            f"direct={counters.get('direct_llm', 0)} "
            f"codex={counters.get('codex_cli', 0)} "
            f"succeeded={counters.get('succeeded', 0)} "
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
        for key, label in (
            ("direct_model", "direct model"),
            ("codex_model", "codex model"),
        ):
            table.add_row(label, str(self.state.metrics.get(key) or ""))
        for key, label in (
            ("roots", "roots"),
            ("succeeded", "succeeded"),
            ("direct_llm", "direct runs"),
            ("codex_cli", "codex runs"),
        ):
            table.add_row(label, str(self.state.counters.get(key, 0)))
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
                statuses.append(
                    {
                        "name": path.name,
                        "status": "queued",
                        "detail": detail,
                    }
                )
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


def usage_int(usage: Any, *keys: str) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def normalize_usage(response: Any) -> dict[str, int]:
    return normalize_litellm_usage(response)


def estimated_completion_cost_usd(response: Any, model: str) -> float | None:
    return estimated_litellm_completion_cost_usd(response, model)


def print_metrics(stats: RunStats) -> None:
    if not _METRICS_ENABLED:
        return
    print(
        "[metrics] "
        f"elapsed={stats.elapsed_secs():.2f}s "
        f"llm_requests={stats.llm_requests} "
        f"input_tokens={stats.input_tokens} "
        f"output_tokens={stats.output_tokens} "
        f"total_tokens={stats.total_tokens} "
        f"estimated_usd=${stats.estimated_usd:.6f}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a recursive hierarchical objective from task-thread objective JSON."
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing task_threads.json.")
    parser.add_argument("--config", type=Path, default=None, help="Task model induction config path.")
    parser.add_argument(
        "--output", "--output_path", dest="output",
        help=(
            "Output JSON path. Use '-' for stdout. "
            "Default: configured hierarchy.json in the data directory."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=f"Maximum LLM repair attempts after validation failure. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--preflight-only",
        "--preflight_only",
        dest="preflight_only",
        action="store_true",
        help="Normalize and validate without making LLM calls.",
    )
    parser.add_argument("--no_console", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Input JSON must be an object: {path}")
    return payload


def load_validation_source(input_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    source_file = source.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        return source
    referenced_path = Path(source_file)
    if not referenced_path.is_absolute():
        referenced_path = input_path.parent / referenced_path
    if not referenced_path.exists():
        return source
    referenced = read_json(referenced_path)
    return {
        "hierarchy_input": source,
        "source_observations": referenced,
    }


def semantic_action_id_to_int(value: str) -> int | None:
    match = SEMANTIC_ACTION_RE.match(value)
    return int(match.group("idx")) if match else None


def segment_id_to_int(value: str) -> int | None:
    match = SUBGOAL_SEGMENT_RE.match(value)
    return int(match.group("start")) if match and not match.group("end") else None


def int_to_semantic_action_id(value: int) -> str:
    return f"semantic_action_{value:04d}"


def preprocess_ids_for_llm(
    value: Any,
    key: str | None = None,
    *,
    segment_int_by_id: dict[str, int],
) -> Any:
    """Compact source ids to integers before prompting."""

    if isinstance(value, dict):
        return {
            child_key: preprocess_ids_for_llm(
                child_value,
                child_key,
                segment_int_by_id=segment_int_by_id,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            preprocess_ids_for_llm(item, key, segment_int_by_id=segment_int_by_id)
            for item in value
        ]
    if isinstance(value, str) and key in {
        "semantic_action_id",
        "start_semantic_action_id",
        "end_semantic_action_id",
        "semantic_action_ids",
    }:
        parsed = semantic_action_id_to_int(value)
        return parsed if parsed is not None else value
    if isinstance(value, str) and key in {
        "activity_id",
        "subgoal_segments",
        "evidence_subgoal_segments",
        "source_activity_ids",
    }:
        return compact_segment_ref_for_llm(value, segment_int_by_id)
    return value


def compact_segment_ref_for_llm(value: str, segment_int_by_id: dict[str, int]) -> int | str:
    match = SUBGOAL_SEGMENT_RE.match(value)
    if not match:
        return value
    prefix = match.group("prefix")
    start_id = f"{prefix}_{int(match.group('start')):04d}"
    start = segment_int_by_id.get(start_id, int(match.group("start")))
    end_text = match.group("end")
    if end_text is None:
        return start
    end_id = f"{prefix}_{int(end_text):04d}"
    end = segment_int_by_id.get(end_id, int(end_text))
    return f"{start}-{end}"


def collect_segment_id_maps(source: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            activities = value.get("activities")
            if isinstance(activities, list):
                for item in activities:
                    if not isinstance(item, dict):
                        continue
                    segment_id = item.get("activity_id")
                    if isinstance(segment_id, str) and segment_id not in seen:
                        parsed = segment_id_to_int(segment_id)
                        if parsed is not None:
                            seen.add(segment_id)
                            ordered_ids.append(segment_id)
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str):
                parsed = segment_id_to_int(segment_id)
                if parsed is not None and segment_id not in seen:
                    seen.add(segment_id)
                    ordered_ids.append(segment_id)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    if ordered_ids:
        by_int = {idx: segment_id for idx, segment_id in enumerate(ordered_ids)}
        by_id = {segment_id: idx for idx, segment_id in by_int.items()}
        return by_int, by_id

    fallback_by_int: dict[int, str] = {}
    fallback_by_id: dict[str, int] = {}
    return fallback_by_int, fallback_by_id


def collect_segment_id_by_int(source: dict[str, Any]) -> dict[int, str]:
    return collect_segment_id_maps(source)[0]


def postprocess_ids_from_llm(
    value: Any,
    *,
    segment_id_by_int: dict[int, str],
    key: str | None = None,
) -> Any:
    """Convert compact integer ids back to the source id strings before writing."""

    if isinstance(value, dict):
        return {
            child_key: postprocess_ids_from_llm(
                child_value,
                segment_id_by_int=segment_id_by_int,
                key=child_key,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        converted: list[Any] = []
        for item in value:
            processed = postprocess_ids_from_llm(item, segment_id_by_int=segment_id_by_int, key=key)
            if key in {"subgoal_segments", "evidence_subgoal_segments", "source_activity_ids"} and isinstance(processed, list):
                converted.extend(processed)
            else:
                converted.append(processed)
        return converted
    if isinstance(value, int) and key in {
        "semantic_action_id",
        "start_semantic_action_id",
        "end_semantic_action_id",
        "semantic_action_ids",
    }:
        return int_to_semantic_action_id(value)
    if key in {"subgoal_segments", "evidence_subgoal_segments", "source_activity_ids"}:
        return expand_compact_segment_ref_from_llm(value, segment_id_by_int)
    return value


def expand_compact_segment_ref_from_llm(value: Any, segment_id_by_int: dict[int, str]) -> str | list[str]:
    if isinstance(value, int):
        return segment_id_by_int.get(value, f"subgoal_segment_{value:04d}")
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    match = COMPACT_SEGMENT_RE.match(stripped)
    if not match:
        return stripped
    start = int(match.group("start"))
    end = int(match.group("end") or match.group("start"))
    if start > end:
        return stripped
    if start == end:
        return segment_id_by_int.get(start, f"subgoal_segment_{start:04d}")
    segment_ids = [
        segment_id_by_int.get(idx, f"subgoal_segment_{idx:04d}")
        for idx in range(start, end + 1)
    ]
    return compact_activity_ids(segment_ids) or segment_ids


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_hierarchy.json")


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


def call_llm(
    system_prompt: str,
    content: dict[str, Any],
    model: str,
    llm_timeout_secs: float,
    *,
    segment_id_by_int: dict[int, str],
    operation: str,
) -> dict[str, Any]:
    segment_int_by_id = {segment_id: idx for idx, segment_id in segment_id_by_int.items()}
    llm_content = preprocess_ids_for_llm(content, segment_int_by_id=segment_int_by_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(llm_content, ensure_ascii=False)},
    ]
    response = litellm_completion(
        model=model,
        messages=messages,
        temperature=0.0 if "gpt-5" not in model and "kimi" not in model else 1.0,
        timeout=llm_timeout_secs,
        request_timeout=llm_timeout_secs,
        response_format={"type": "json_object"},
    )
    if _ACTIVE_STATS is not None:
        usage = normalize_usage(response)
        call_usd = estimated_completion_cost_usd(response, model)
        _ACTIVE_STATS.record_call(operation=operation, model=model, usage=usage, estimated_usd=call_usd)
        print_metrics(_ACTIVE_STATS)
    return postprocess_ids_from_llm(
        extract_json_from_response(completion_message_content(response)),
        segment_id_by_int=segment_id_by_int,
    )


def is_new_schema_node(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return {"id", "objective", "summary", "subgoal_segments", "decomposition"}.issubset(value.keys())


def infer_root_id(source: dict[str, Any]) -> str:
    for key in ("canonical_root_id", "id"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    root = source.get("root_objective")
    if isinstance(root, dict):
        value = root.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "root"


def explicit_source_root_id(source: dict[str, Any]) -> str | None:
    """Return a source-declared root id without inventing a fallback."""

    candidates = [source]
    hierarchy_input = source.get("hierarchy_input")
    if isinstance(hierarchy_input, dict):
        candidates.insert(0, hierarchy_input)
    for candidate in candidates:
        for key in ("canonical_root_id", "id"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        root = candidate.get("root_objective")
        if isinstance(root, dict):
            value = root.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in result:
            result.append(text)
    return result


def _expand_legacy_evidence_refs(refs: list[str]) -> list[str]:
    expanded: list[str] = []
    for ref in refs:
        match = SUBGOAL_SEGMENT_RE.match(ref)
        if match and match.group("end") is not None:
            start = int(match.group("start"))
            end = int(match.group("end"))
            if start <= end:
                for idx in range(start, end + 1):
                    item = f"{match.group('prefix')}_{idx:04d}"
                    if item not in expanded:
                        expanded.append(item)
                continue
        if ref not in expanded:
            expanded.append(ref)
    return expanded


def normalize_objective_grounding(
    value: dict[str, Any],
    *,
    objective: str,
    fallback_evidence_refs: list[str],
) -> dict[str, Any]:
    """Translate legacy string grounding into the shared recursive contract."""

    grounding = value.get("objective_grounding")
    source = grounding if isinstance(grounding, dict) else value
    evidence_refs = _non_empty_string_list(source.get("evidence_refs"))
    if not evidence_refs:
        evidence_refs = _non_empty_string_list(value.get("evidence_refs"))
    if not evidence_refs:
        evidence_refs = _expand_legacy_evidence_refs(
            _non_empty_string_list(fallback_evidence_refs)
        )

    deliverables: list[dict[str, Any]] = []
    raw_deliverables = source.get("deliverables")
    if isinstance(raw_deliverables, list):
        for raw in raw_deliverables:
            if not isinstance(raw, dict):
                continue
            target = str(raw.get("target") or "").strip()
            if not target:
                continue
            expected_state = str(raw.get("expected_state") or "").strip()
            deliverables.append(
                {
                    "kind": str(raw.get("kind") or "artifact_or_state").strip() or "artifact_or_state",
                    "target": target,
                    "expected_state": expected_state or f"The {target} is complete.",
                    "evidence_refs": _non_empty_string_list(raw.get("evidence_refs")) or evidence_refs,
                }
            )

    legacy_deliverable = str(value.get("deliverable") or source.get("deliverable") or "").strip()
    legacy_success = value.get("success_criteria")
    if isinstance(legacy_success, str):
        legacy_success = legacy_success.strip()
    else:
        legacy_success = ""
    target = legacy_deliverable or objective or "Objective outcome"
    expected_state = legacy_success or f"The {target} is complete."
    if not deliverables:
        deliverables = [
            {
                "kind": "artifact_or_state",
                "target": target,
                "expected_state": expected_state,
                "evidence_refs": evidence_refs,
            }
        ]

    criteria: list[dict[str, Any]] = []
    raw_criteria = source.get("success_criteria")
    if isinstance(raw_criteria, list):
        for raw in raw_criteria:
            if not isinstance(raw, dict):
                continue
            predicate = str(raw.get("predicate") or "").strip()
            if not predicate:
                continue
            confidence = raw.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                confidence = None
            criteria.append(
                {
                    "predicate": predicate,
                    "verifier": str(raw.get("verifier") or "unknown").strip() or "unknown",
                    "evidence_refs": _non_empty_string_list(raw.get("evidence_refs")) or evidence_refs,
                    "confidence": confidence,
                }
            )
    if not criteria:
        criteria = [
            {
                "predicate": expected_state,
                "verifier": "unknown",
                "evidence_refs": evidence_refs,
                "confidence": None,
            }
        ]

    raw_outcome = source.get("observed_outcome")
    allowed_statuses = {"achieved", "partial", "failed", "abandoned", "unknown"}
    if isinstance(raw_outcome, dict):
        status = str(raw_outcome.get("status") or "unknown").strip()
        if status not in allowed_statuses:
            status = "unknown"
        description = str(raw_outcome.get("description") or "").strip()
        outcome = {
            "status": status,
            "description": description or "The available evidence does not independently verify completion.",
            "evidence_refs": _non_empty_string_list(raw_outcome.get("evidence_refs")),
        }
    else:
        outcome = {
            "status": "unknown",
            "description": "The available evidence does not independently verify completion.",
            "evidence_refs": [],
        }

    return {
        "deliverables": deliverables,
        "success_criteria": criteria,
        "observed_outcome": outcome,
        "evidence_refs": evidence_refs,
    }


def normalize_existing_hierarchy(source: dict[str, Any]) -> dict[str, Any] | None:
    """Project a hierarchy-like input into the strict recursive schema."""

    root = source.get("root_objective") if isinstance(source.get("root_objective"), dict) else source
    if not isinstance(root, dict):
        return None
    if not (
        is_new_schema_node(root)
        or "decomposition" in root
        or "evidence_subgoal_segments" in root
        or "source_activity_ids" in root
    ):
        return None

    root_id = infer_root_id(source)

    def unique_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                unique.append(value)
        return unique

    def normalize_node(node: dict[str, Any], fallback_id: str) -> dict[str, Any]:
        node_id = str(node.get("id") or fallback_id).strip()
        objective = str(node.get("objective") or node.get("label") or "").strip()
        summary = str(node.get("summary") or node.get("additional_context") or "").strip()
        subgoal_segments = node.get("subgoal_segments")
        if subgoal_segments is None:
            subgoal_segments = node.get("evidence_subgoal_segments")
        if subgoal_segments is None:
            subgoal_segments = node.get("source_activity_ids")
        if not isinstance(subgoal_segments, list):
            subgoal_segments = []

        children = node.get("decomposition", [])
        if not isinstance(children, list):
            children = []

        normalized_children = [
            normalize_node(child, f"{node_id}.{idx}")
            for idx, child in enumerate(children, start=1)
            if isinstance(child, dict)
        ]
        normalized_segments = [str(item).strip() for item in subgoal_segments if str(item).strip()]
        if not normalized_segments and normalized_children:
            normalized_segments = unique_preserve_order(
                [
                    segment
                    for child in normalized_children
                    for segment in child["subgoal_segments"]
                ]
            )
        if not summary:
            if normalized_children:
                child_objectives = "; ".join(child["objective"] for child in normalized_children[:3])
                summary = f"Decomposes into: {child_objectives}."
            else:
                summary = objective

        grounding = normalize_objective_grounding(
            node,
            objective=objective,
            fallback_evidence_refs=normalized_segments,
        )

        return {
            "id": node_id,
            "objective": objective,
            "summary": summary,
            **grounding,
            "subgoal_segments": normalized_segments,
            "decomposition": normalized_children,
        }

    return normalize_node(root, root_id)


def fill_root_segments_from_source(candidate: dict[str, Any], validation_source: dict[str, Any]) -> dict[str, Any]:
    segment_ids = sorted(
        collect_segment_action_index(validation_source),
        key=lambda segment_id: int(SUBGOAL_SEGMENT_RE.match(segment_id).group("start"))
        if SUBGOAL_SEGMENT_RE.match(segment_id)
        else -1,
    )
    if segment_ids:
        candidate = dict(candidate)
        candidate["subgoal_segments"] = compact_activity_ids(segment_ids)
    return candidate


def initial_candidate(
    source: dict[str, Any],
    validation_source: dict[str, Any],
    model: str,
    llm_timeout_secs: float,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
    preflight_only: bool,
    segment_id_by_int: dict[int, str],
) -> dict[str, Any]:
    normalized = normalize_existing_hierarchy(source)
    if normalized is not None:
        return fill_root_segments_from_source(normalized, validation_source)
    if preflight_only:
        root_id = infer_root_id(source)
        objective = str(source.get("task_thread_objective") or source.get("objective") or "").strip()
        grounding = normalize_objective_grounding(
            source,
            objective=objective,
            fallback_evidence_refs=list(collect_segment_action_index(validation_source)),
        )
        return fill_root_segments_from_source({
            "id": root_id,
            "objective": objective,
            "summary": "Preflight placeholder generated without LLM calls.",
            **grounding,
            "subgoal_segments": [],
            "decomposition": [],
        }, validation_source)
    return call_llm(
        generation_prompt_text(
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
        ),
        {"original_input": validation_source},
        model,
        llm_timeout_secs=llm_timeout_secs,
        segment_id_by_int=segment_id_by_int,
        operation="generation",
    )


def collect_known_activity_ids(source: dict[str, Any]) -> set[str]:
    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str):
                ids.add(segment_id)
            for key in ("subgoal_segments", "evidence_subgoal_segments", "source_activity_ids"):
                raw = value.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str):
                            match = SUBGOAL_SEGMENT_RE.match(item)
                            if match:
                                prefix = match.group("prefix")
                                start = int(match.group("start"))
                                end = int(match.group("end") or match.group("start"))
                                if start <= end:
                                    for segment_num in range(start, end + 1):
                                        ids.add(f"{prefix}_{segment_num:04d}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_source_activity_ids(source: dict[str, Any]) -> set[str]:
    """Collect concrete activity rows, excluding hierarchy evidence references."""

    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str) and SUBGOAL_SEGMENT_RE.match(segment_id):
                ids.add(segment_id)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_known_evidence_ids(source: dict[str, Any]) -> set[str]:
    """Collect source activity, semantic-action, and raw-action identifiers."""

    ids: set[str] = set()
    scalar_keys = {
        "activity_id",
        "semantic_action_id",
        "start_semantic_action_id",
        "end_semantic_action_id",
        "raw_action_id",
        "action_id",
        "start_action_id",
        "end_action_id",
    }
    list_keys = {
        "activity_ids",
        "semantic_action_ids",
        "raw_action_ids",
        "action_ids",
        "source_activity_ids",
        "subgoal_segments",
        "evidence_subgoal_segments",
    }

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        match = SUBGOAL_SEGMENT_RE.match(text)
        if match:
            start = int(match.group("start"))
            end = int(match.group("end") or match.group("start"))
            if start <= end:
                ids.update(
                    f"{match.group('prefix')}_{idx:04d}"
                    for idx in range(start, end + 1)
                )
                return
        ids.add(text)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in scalar_keys:
                    add(child)
                elif key in list_keys and isinstance(child, list):
                    for item in child:
                        add(item)
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return ids


def collect_evidence_activity_owners(source: dict[str, Any]) -> dict[str, set[str]]:
    """Map each source evidence id to the concrete activities that contain it."""

    scalar_keys = {
        "activity_id",
        "semantic_action_id",
        "start_semantic_action_id",
        "end_semantic_action_id",
        "raw_action_id",
        "action_id",
        "start_action_id",
        "end_action_id",
    }
    list_keys = {
        "activity_ids",
        "semantic_action_ids",
        "raw_action_ids",
        "action_ids",
    }
    owners: dict[str, set[str]] = {}

    def add(owner: str, value: Any) -> None:
        if isinstance(value, str) and value.strip():
            owners.setdefault(value.strip(), set()).add(owner)

    def collect_from_activity(value: Any, owner: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in scalar_keys:
                    add(owner, child)
                elif key in list_keys and isinstance(child, list):
                    for item in child:
                        add(owner, item)
                collect_from_activity(child, owner)
        elif isinstance(value, list):
            for item in value:
                collect_from_activity(item, owner)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            activity_id = value.get("activity_id")
            if isinstance(activity_id, str) and SUBGOAL_SEGMENT_RE.fullmatch(activity_id):
                collect_from_activity(value, activity_id)
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return owners


def compact_activity_ids(segment_ids: list[str]) -> list[str]:
    by_prefix: dict[str, list[int]] = {}
    for segment_id in segment_ids:
        match = SUBGOAL_SEGMENT_RE.match(segment_id)
        if match and not match.group("end"):
            by_prefix.setdefault(match.group("prefix"), []).append(int(match.group("start")))
    numeric_ids = sorted(set(by_prefix.get("activity") or by_prefix.get("subgoal_segment") or []))
    if not numeric_ids:
        return []
    prefix = "activity" if "activity" in by_prefix else "subgoal_segment"

    compacted: list[str] = []
    start = previous = numeric_ids[0]
    for current in numeric_ids[1:]:
        if current == previous + 1:
            previous = current
            continue
        compacted.append(format_subgoal_segment_ref(start, previous, prefix=prefix))
        start = previous = current
    compacted.append(format_subgoal_segment_ref(start, previous, prefix=prefix))
    return compacted


def count_subgoal_segment_refs(refs: list[str]) -> int:
    segment_ids: set[str] = set()
    for ref in refs:
        match = SUBGOAL_SEGMENT_RE.match(ref)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if start > end:
            continue
        prefix = match.group("prefix")
        for segment_num in range(start, end + 1):
            segment_ids.add(f"{prefix}_{segment_num:04d}")
    return len(segment_ids)


def format_subgoal_segment_ref(start: int, end: int, *, prefix: str) -> str:
    if start == end:
        return f"{prefix}_{start:04d}"
    return f"{prefix}_{start:04d}-{prefix}_{end:04d}"


def semantic_action_indices_from_segment(segment: dict[str, Any]) -> set[int]:
    raw_ids = segment.get("semantic_action_ids")
    parsed_ids: set[int] = set()
    if isinstance(raw_ids, list):
        for raw_id in raw_ids:
            if isinstance(raw_id, int):
                parsed_ids.add(raw_id)
            elif isinstance(raw_id, str):
                parsed = semantic_action_id_to_int(raw_id)
                if parsed is not None:
                    parsed_ids.add(parsed)
    if parsed_ids:
        return parsed_ids

    start_idx = segment.get("start_semantic_action_idx")
    end_idx = segment.get("end_semantic_action_idx")
    if isinstance(start_idx, int) and isinstance(end_idx, int) and start_idx <= end_idx:
        return set(range(start_idx, end_idx + 1))
    return set()


def collect_segment_action_index(source: dict[str, Any]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            segment_id = value.get("activity_id")
            if isinstance(segment_id, str):
                actions = semantic_action_indices_from_segment(value)
                if actions:
                    index[segment_id] = actions
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    return index


def expand_subgoal_segment_refs(
    refs: list[str],
    known_segment_ids: set[str],
    path: str,
) -> tuple[list[str], list[str]]:
    expanded: list[str] = []
    errors: list[str] = []
    for idx, ref in enumerate(refs):
        match = SUBGOAL_SEGMENT_RE.match(ref)
        if not match:
            errors.append(
                f"{path}.subgoal_segments[{idx}] contains invalid reference {ref!r}; "
                "use subgoal_segment_0000 or subgoal_segment_0000-subgoal_segment_0004."
            )
            continue

        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if start > end:
            errors.append(f"{path}.subgoal_segments[{idx}] range {ref!r} has start greater than end.")
            continue

        for segment_num in range(start, end + 1):
            segment_id = f"{match.group('prefix')}_{segment_num:04d}"
            expanded.append(segment_id)
            if known_segment_ids and segment_id not in known_segment_ids:
                errors.append(
                    f"{path}.subgoal_segments[{idx}] range {ref!r} includes unknown segment "
                    f"{segment_id!r}."
                )
    return expanded, errors


def actions_for_segment_refs(
    refs: list[str],
    segment_action_index: dict[str, set[int]],
    known_segment_ids: set[str],
    path: str,
) -> tuple[set[int], list[str]]:
    expanded_segments, errors = expand_subgoal_segment_refs(refs, known_segment_ids, path)
    actions: set[int] = set()
    for segment_id in expanded_segments:
        actions.update(segment_action_index.get(segment_id, set()))
    return actions, errors


def validate_segment_ref(ref: str, known_segment_ids: set[str], path: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    match = SUBGOAL_SEGMENT_RE.match(ref)
    if not match:
        errors.append(
            f"{path}.subgoal_segments contains invalid reference {ref!r}; "
            "use subgoal_segment_0000 or subgoal_segment_0000-subgoal_segment_0004."
        )
        return errors, warnings

    start = int(match.group("start"))
    end_text = match.group("end")
    if end_text is not None and start > int(end_text):
        errors.append(f"{path}.subgoal_segments range {ref!r} has start greater than end.")

    if known_segment_ids:
        prefix = match.group("prefix")
        start_id = f"{prefix}_{match.group('start')}"
        if start_id not in known_segment_ids:
            errors.append(f"{path}.subgoal_segments references unknown start segment {start_id!r}.")
        if end_text is not None:
            end_id = f"{prefix}_{end_text}"
            if end_id not in known_segment_ids:
                errors.append(f"{path}.subgoal_segments references unknown end segment {end_id!r}.")

    return errors, warnings


def validate_hierarchy(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    allow_empty_root_decomposition: bool = False,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> ValidationFeedback:
    errors: list[str] = []
    warnings: list[str] = []
    known_segment_ids = collect_known_activity_ids(source)
    source_segment_ids = collect_source_activity_ids(source)
    known_evidence_ids = collect_known_evidence_ids(source)
    evidence_activity_owners = collect_evidence_activity_owners(source)
    segment_action_index = collect_segment_action_index(source)
    all_source_actions: set[int] = set()
    for actions in segment_action_index.values():
        all_source_actions.update(actions)
    source_activities = source.get("activities")
    if not isinstance(source_activities, list):
        source_observations = source.get("source_observations")
        if isinstance(source_observations, dict):
            source_activities = source_observations.get("activities")
    source_root_objective = source.get("root_objective")
    if not isinstance(source_root_objective, dict):
        hierarchy_input = source.get("hierarchy_input")
        if isinstance(hierarchy_input, dict):
            source_root_objective = hierarchy_input.get("root_objective")
    source_requires_decomposition = (
        isinstance(source_activities, list)
        and len(source_activities) > 1
    ) or (
        isinstance(source_root_objective, dict)
        and isinstance(source_root_objective.get("decomposition"), list)
        and len(source_root_objective["decomposition"]) > 0
    )

    try:
        node = HierarchicalObjectiveNode.model_validate(candidate)
    except ValidationError as exc:
        return ValidationFeedback(valid=False, errors=[str(exc)], warnings=[])

    expected_root_id = explicit_source_root_id(source)
    if expected_root_id is not None and node.id != expected_root_id:
        errors.append(
            f"Root id must match source root id {expected_root_id!r}; got {node.id!r}."
        )

    if source_requires_decomposition and not node.decomposition and not allow_empty_root_decomposition:
        errors.append(
            "Root decomposition is empty, but the source contains multiple activities "
            "or an existing decomposition. Add meaningful child objective nodes."
        )

    seen_ids: set[str] = set()
    all_hierarchy_actions: set[int] = set()
    root_actions: set[int] = set()
    root_segments: set[str] = set()

    def format_action_ranges(actions: set[int]) -> str:
        if not actions:
            return "[]"
        values = sorted(actions)
        ranges: list[str] = []
        start = previous = values[0]
        for current in values[1:]:
            if current == previous + 1:
                previous = current
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = current
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return "[" + ", ".join(ranges) + "]"

    def format_segment_ids(segment_ids: set[str]) -> str:
        return "[" + ", ".join(sorted(segment_ids)) + "]"

    def walk(current: HierarchicalObjectiveNode, path: str, parent_id: str | None) -> set[str]:
        nonlocal root_actions, root_segments
        if current.id in seen_ids:
            errors.append(f"{path}.id {current.id!r} is duplicated elsewhere in the hierarchy.")
        seen_ids.add(current.id)

        if not current.id.strip():
            errors.append(f"{path}.id must be non-empty.")
        if not current.objective.strip():
            errors.append(f"{path}.objective must be non-empty.")
        if not current.summary.strip():
            errors.append(f"{path}.summary must be non-empty.")

        node_evidence_refs = set(current.evidence_refs)
        unknown_evidence_refs = node_evidence_refs - known_evidence_ids
        if unknown_evidence_refs:
            errors.append(
                f"{path}.evidence_refs contains identifiers not present in the source: "
                f"{format_segment_ids(unknown_evidence_refs)}."
            )
        for field_name, grounded_items in (
            ("deliverables", current.deliverables),
            ("success_criteria", current.success_criteria),
        ):
            for idx, grounded_item in enumerate(grounded_items):
                extra_refs = set(grounded_item.evidence_refs) - node_evidence_refs
                if extra_refs:
                    errors.append(
                        f"{path}.{field_name}[{idx}].evidence_refs must be a subset of "
                        f"{path}.evidence_refs; extra refs: {format_segment_ids(extra_refs)}."
                    )
        outcome_extra_refs = set(current.observed_outcome.evidence_refs) - node_evidence_refs
        if outcome_extra_refs:
            errors.append(
                f"{path}.observed_outcome.evidence_refs must be a subset of {path}.evidence_refs; "
                f"extra refs: {format_segment_ids(outcome_extra_refs)}."
            )
        if current.observed_outcome.status != "unknown" and not current.observed_outcome.evidence_refs:
            errors.append(
                f"{path}.observed_outcome with status {current.observed_outcome.status!r} "
                "must cite evidence_refs."
            )

        if parent_id is not None and not current.id.startswith(f"{parent_id}."):
            errors.append(
                f"{path}.id {current.id!r} must be hierarchically nested under parent id {parent_id!r}."
            )

        expected_child_ids = [f"{current.id}.{idx}" for idx in range(1, len(current.decomposition) + 1)]
        actual_child_ids = [child.id for child in current.decomposition]
        if actual_child_ids != expected_child_ids:
            errors.append(
                f"{path}.decomposition child ids must be sequential {expected_child_ids}; "
                f"got {actual_child_ids}."
            )

        expanded_segments, segment_errors = expand_subgoal_segment_refs(
            current.subgoal_segments,
            known_segment_ids,
            path,
        )
        errors.extend(segment_errors)
        if len(expanded_segments) != len(set(expanded_segments)):
            errors.append(f"{path}.subgoal_segments contains overlapping or duplicate references.")
        current_segments = set(expanded_segments)
        out_of_span_evidence = {
            evidence_ref
            for evidence_ref in node_evidence_refs
            if evidence_activity_owners.get(evidence_ref)
            and not (evidence_activity_owners[evidence_ref] & current_segments)
        }
        if out_of_span_evidence:
            errors.append(
                f"{path}.evidence_refs cites evidence owned by activities outside this "
                f"node's subgoal_segments: {format_segment_ids(out_of_span_evidence)}."
            )
        current_actions: set[int] = set()
        for segment_id in current_segments:
            current_actions.update(segment_action_index.get(segment_id, set()))
        all_hierarchy_actions.update(current_actions)
        if path == "$":
            root_actions = current_actions
            root_segments = current_segments

        if not current.subgoal_segments:
            errors.append(f"{path}.subgoal_segments must be non-empty.")
        segment_count = count_subgoal_segment_refs(current.subgoal_segments)
        if segment_count == 1 and current.decomposition:
            errors.append(
                f"{path} covers exactly one activity/subgoal segment and must not be decomposed."
            )
        if segment_count == 1 and parent_id is not None:
            warnings.append(
                f"{path} is a decomposed child covering exactly one activity/subgoal segment; "
                "consider merging it into the parent or grouping it with a broader objective."
            )
        if segment_count > large_node_review_threshold:
            warnings.append(
                f"{path} covers {segment_count} activities/subgoal segments; "
                "review whether it needs further decomposition."
            )
        if 0 < segment_count < small_decomposition_review_threshold and current.decomposition:
            warnings.append(
                f"{path} covers only {segment_count} activities/subgoal segments but has decomposition; "
                "review whether the decomposition is necessary or the parent objective is sufficient."
            )

        child_segment_sets: list[set[str]] = []
        for idx, child in enumerate(current.decomposition):
            child_path = f"{path}.decomposition[{idx}]"
            child_segments = walk(child, child_path, current.id)
            extra_child_segments = child_segments - current_segments
            if extra_child_segments:
                errors.append(
                    f"{child_path}.subgoal_segments must be a subset of its parent; "
                    f"extra segments: {format_segment_ids(extra_child_segments)}."
                )
            for previous_idx, previous_segments in enumerate(child_segment_sets):
                overlap = child_segments & previous_segments
                if overlap:
                    errors.append(
                        f"{child_path}.subgoal_segments overlaps sibling "
                        f"{path}.decomposition[{previous_idx}]: {format_segment_ids(overlap)}."
                    )
            child_segment_sets.append(child_segments)

        if child_segment_sets:
            child_union = set().union(*child_segment_sets)
            missing_from_children = current_segments - child_union
            extra_in_children = child_union - current_segments
            if missing_from_children or extra_in_children:
                detail: list[str] = []
                if missing_from_children:
                    detail.append(f"missing {format_segment_ids(missing_from_children)}")
                if extra_in_children:
                    detail.append(f"extra {format_segment_ids(extra_in_children)}")
                errors.append(
                    f"{path}.decomposition must partition the parent's subgoal_segments exactly; "
                    + "; ".join(detail)
                    + "."
                )
        return current_segments

    walk(node, "$", None)
    if source_segment_ids:
        missing_source_segments = source_segment_ids - root_segments
        extra_root_segments = root_segments - source_segment_ids
        if missing_source_segments:
            errors.append(
                "Root subgoal_segments do not cover all source activities. "
                f"Missing: {format_segment_ids(missing_source_segments)}."
            )
        if extra_root_segments:
            errors.append(
                "Root subgoal_segments contain activities outside the source. "
                f"Extra: {format_segment_ids(extra_root_segments)}."
            )
    if all_source_actions:
        missing_from_root = all_source_actions - root_actions
        extra_in_root = root_actions - all_source_actions
        if missing_from_root:
            errors.append(
                "Root subgoal_segments do not cover all source semantic actions. "
                f"Missing inclusive semantic action index ranges: {format_action_ranges(missing_from_root)}."
            )
        if extra_in_root:
            errors.append(
                "Root subgoal_segments cover semantic actions outside the source. "
                f"Extra semantic action index ranges: {format_action_ranges(extra_in_root)}."
            )

        missing_from_hierarchy = all_source_actions - all_hierarchy_actions
        if missing_from_hierarchy:
            errors.append(
                "Hierarchy subgoal_segments do not cover all source semantic actions. "
                f"Missing inclusive semantic action index ranges: {format_action_ranges(missing_from_hierarchy)}."
            )
    return ValidationFeedback(valid=not errors, errors=errors, warnings=warnings)


def compact_feedback_text_for_llm(text: str, segment_id_by_int: dict[int, str]) -> str:
    """Rewrite source ids in validation feedback into the compacted integer
    namespace the LLM sees, so repair instructions reference ids the model
    can actually locate in its input and previous result."""
    segment_int_by_id = {seg_id: seg_int for seg_int, seg_id in segment_id_by_int.items()}

    def replace(match: "re.Match[str]") -> str:
        seg_id = match.group(0)
        mapped = segment_int_by_id.get(seg_id)
        return str(mapped) if mapped is not None else seg_id

    return re.sub(r"(?:subgoal_segment|activity)_\d{4}", replace, text)


def repair_candidate(
    *,
    source: dict[str, Any],
    previous: dict[str, Any],
    feedback: ValidationFeedback,
    model: str,
    llm_timeout_secs: float,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
    segment_id_by_int: dict[int, str],
) -> dict[str, Any]:
    return call_llm(
        repair_prompt_text(
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
        ),
        {
            "original_input": source,
            "previous_result": previous,
            "validation_feedback": compact_feedback_text_for_llm(
                feedback.as_text(), segment_id_by_int
            ),
        },
        model,
        llm_timeout_secs=llm_timeout_secs,
        segment_id_by_int=segment_id_by_int,
        operation="repair",
    )


def write_output(path_text: str | None, input_path: Path, hierarchy: dict[str, Any]) -> Path | None:
    if path_text == "-":
        print(json.dumps(hierarchy, indent=2, ensure_ascii=False))
        return None
    output_path = Path(path_text).resolve() if path_text else default_output_path(input_path).resolve()
    write_json_atomic(output_path, hierarchy)
    write_objective_markdown(output_path, hierarchy)
    return output_path


def planned_output_path(path_text: str | None, input_path: Path) -> Path | None:
    if path_text == "-":
        return None
    return Path(path_text).resolve() if path_text else default_output_path(input_path).resolve()


def write_meta(
    output_path: Path | None,
    *,
    input_path: Path,
    model: str,
    max_retries: int,
    retry_count: int,
    preflight_only: bool,
    valid: bool,
    feedback: ValidationFeedback,
    stats: RunStats,
) -> Path | None:
    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path is not None else None,
        "model": model,
        "max_retries": max_retries,
        "retry_count": retry_count,
        "preflight_only": preflight_only,
        "valid": valid,
        "errors": feedback.errors,
        "warnings": feedback.warnings,
        "cost": stats.as_dict(),
    }
    if output_path is None:
        print("[meta] " + json.dumps(meta, ensure_ascii=False), file=sys.stderr)
        if _METRICS_ENABLED:
            print("[cost] hierarchical_objective_induction " + json.dumps(stats.cost_breakdown(), ensure_ascii=False), file=sys.stderr)
        return None
    meta_path = output_path.with_suffix(f"{output_path.suffix}.meta.json")
    write_json_atomic(meta_path, meta)
    if _METRICS_ENABLED:
        print("[cost] hierarchical_objective_induction " + json.dumps(stats.cost_breakdown(), ensure_ascii=False), file=sys.stderr)
    return meta_path


def activity_count(input_path: Path) -> int | None:
    try:
        source = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    activities = source.get("activities") if isinstance(source, dict) else None
    return len(activities) if isinstance(activities, list) else None


def derived_hierarchy_inputs(task_threads_path: Path) -> list[Path]:
    source = read_json(task_threads_path)
    derived_dir = source.get("meta", {}).get("derived_objectives_dir") if isinstance(source.get("meta"), dict) else None
    if not isinstance(derived_dir, str):
        manifest_path = task_threads_path.parent / "derived_task_thread_objectives" / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            roots = manifest.get("roots")
            if isinstance(roots, list):
                paths = [Path(item["file"]).expanduser() for item in roots if isinstance(item, dict) and isinstance(item.get("file"), str)]
                return [path if path.is_absolute() else task_threads_path.parent / path for path in paths]
        return [task_threads_path]
    manifest_path = Path(derived_dir).expanduser() / "manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = task_threads_path.parent / manifest_path
    if not manifest_path.exists():
        return [task_threads_path]
    manifest = read_json(manifest_path)
    roots = manifest.get("roots")
    if not isinstance(roots, list):
        return [task_threads_path]
    paths = [Path(item["file"]).expanduser() for item in roots if isinstance(item, dict) and isinstance(item.get("file"), str)]
    return [path if path.is_absolute() else manifest_path.parent / path for path in paths] or [task_threads_path]


def run_direct_hierarchy_stage(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    llm_timeout_secs: float,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
    max_retries: int,
    preflight_only: bool,
) -> dict[str, Any]:
    global _ACTIVE_STATS
    stats = RunStats()
    previous_stats = _ACTIVE_STATS
    _ACTIVE_STATS = stats
    try:
        source = read_json(input_path)
        validation_source = load_validation_source(input_path, source)
        segment_id_by_int = collect_segment_id_by_int(validation_source)
        candidate = initial_candidate(
            source,
            validation_source,
            model=model,
            llm_timeout_secs=llm_timeout_secs,
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
            preflight_only=preflight_only,
            segment_id_by_int=segment_id_by_int,
        )
        feedback = validate_hierarchy(
            candidate,
            validation_source,
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
        )
        retry_count = 0
        while not feedback.valid and retry_count < max_retries and not preflight_only:
            retry_count += 1
            candidate = repair_candidate(
                source=validation_source,
                previous=candidate,
                feedback=feedback,
                model=model,
                llm_timeout_secs=llm_timeout_secs,
                large_node_review_threshold=large_node_review_threshold,
                small_decomposition_review_threshold=small_decomposition_review_threshold,
                segment_id_by_int=segment_id_by_int,
            )
            feedback = validate_hierarchy(
                candidate,
                validation_source,
                large_node_review_threshold=large_node_review_threshold,
                small_decomposition_review_threshold=small_decomposition_review_threshold,
            )
        if not feedback.valid:
            write_meta(
                output_path,
                input_path=input_path,
                model=model,
                max_retries=max_retries,
                retry_count=retry_count,
                preflight_only=preflight_only,
                valid=False,
                feedback=feedback,
                stats=stats,
            )
            raise RuntimeError(f"hierarchy validation failed for {input_path}: {feedback.as_text()}")
        write_output(str(output_path), input_path, candidate)
        write_meta(
            output_path,
            input_path=input_path,
            model=model,
            max_retries=max_retries,
            retry_count=retry_count,
            preflight_only=preflight_only,
            valid=True,
            feedback=feedback,
            stats=stats,
        )
        return {
            "execution_mode": "direct_llm",
            "usage": stats.as_dict(),
            "estimated_usd": round(stats.estimated_usd, 6),
            "cost_breakdown": stats.cost_breakdown(),
            "activity_count": activity_count(input_path),
        }
    finally:
        _ACTIVE_STATS = previous_stats


def run_codex_hierarchy_stage(
    *,
    input_path: Path,
    output_path: Path,
    codex_config: dict[str, Any],
    rebuild_image: bool,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> dict[str, Any]:
    result = CodexCliSandbox().run_file_task(
        prompt=induction_prompt(
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
        ),
        files={
            "task_thread_objectives.json": input_path.read_text(encoding="utf-8"),
            "output_schema.json": output_schema_text(),
            "validate_hierarchy.py": validator_script_text(),
        },
        output_file="hierarchy.json",
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
        codex_config = result.get("codex") or {}
        details: list[str] = []
        if result.get("run_id"):
            details.append(f"run_id={result['run_id']}")
        if result.get("returncode") not in (None, 0):
            details.append(f"returncode={result['returncode']}")
        if codex_config.get("model"):
            details.append(f"model={codex_config['model']}")
        if codex_config.get("model_reasoning_effort"):
            details.append(f"reasoning_effort={codex_config['model_reasoning_effort']}")
        if codex_config.get("personality"):
            details.append(f"personality={codex_config['personality']}")
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        if stdout:
            details.append(f"stdout={stdout[-500:]}")
        if stderr:
            details.append(f"stderr={stderr[-500:]}")
        message = str(result.get("error") or "Codex run failed")
        if details:
            message = f"{message} ({'; '.join(details)})"
        raise RuntimeError(message)
    output_content = result.get("output_content")
    if not isinstance(output_content, str) or not output_content.strip():
        raise RuntimeError("Codex run succeeded but returned no hierarchy.json")
    parsed = json.loads(output_content)
    HierarchicalObjectiveNode.model_validate(parsed)
    source = read_json(input_path)
    feedback = validate_hierarchy(
        parsed,
        load_validation_source(input_path, source),
        large_node_review_threshold=large_node_review_threshold,
        small_decomposition_review_threshold=small_decomposition_review_threshold,
    )
    if not feedback.valid:
        raise RuntimeError(f"Codex hierarchy failed host validation: {feedback.as_text()}")
    write_json_atomic(output_path, parsed)
    write_objective_markdown(output_path, parsed)
    validation_text = (result.get("output_files") or {}).get("validation.txt")
    if isinstance(validation_text, str) and validation_text.strip():
        output_path.with_suffix(output_path.suffix + ".validation.txt").write_text(validation_text, encoding="utf-8")
    meta = {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "run_id": result.get("run_id"),
        "model": (result.get("codex") or {}).get("model") or codex_config.get("model"),
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "usage": usage,
        "estimated_usd": estimated_usd,
        "proxy_cost": result.get("proxy_cost"),
        "execution_mode": "codex_cli",
        "activity_count": result["activity_count"],
        "preflight_only": False,
        "valid": True,
        "errors": [],
        "warnings": feedback.warnings,
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)
    return result


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
        normalized = normalize_usage_dict(value)
        if normalized:
            found.append(normalized)
        for child in value.values():
            found.extend(find_usage_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_usage_dicts(child))
    return found


def normalize_usage_dict(value: dict[str, Any]) -> dict[str, int]:
    input_tokens = usage_int(value, "input_tokens", "prompt_tokens")
    output_tokens = usage_int(value, "output_tokens", "completion_tokens")
    total_tokens = usage_int(value, "total_tokens") or input_tokens + output_tokens
    if not any((input_tokens, output_tokens, total_tokens)):
        return {}
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


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


def write_merged_hierarchy_output(
    *,
    output_path: Path,
    output_dir: Path,
    root_results: list[dict[str, Any]],
    model: str,
    preflight_only: bool,
) -> dict[str, Any]:
    hierarchy_ids = [
        item["hierarchy"].get("id")
        for item in root_results
        if isinstance(item, dict)
        and item.get("ok")
        and isinstance(item.get("hierarchy"), dict)
    ]
    duplicate_ids = sorted(
        {root_id for root_id in hierarchy_ids if root_id and hierarchy_ids.count(root_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"merged hierarchy contains duplicate root ids: {duplicate_ids}")
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
    payload = HierarchicalObjectiveInductionOutput.model_validate(payload).model_dump()
    write_json_atomic(output_path, payload)
    write_objective_collection_markdown(output_path, payload)
    if cost is not None:
        write_json_atomic(output_path.with_suffix(output_path.suffix + ".cost.json"), cost)
    return payload


def validate_merged_hierarchy_cache(
    payload: dict[str, Any],
    *,
    hierarchy_inputs: list[Path],
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
) -> dict[str, Any]:
    """Validate every cached root against the source input it claims to cover."""

    validated = HierarchicalObjectiveInductionOutput.model_validate(payload).model_dump()
    roots = payload.get("roots")
    if not isinstance(roots, list) or len(roots) != len(hierarchy_inputs):
        raise ValueError("merged hierarchy cache root count does not match current inputs")

    inputs_by_resolved = {path.expanduser().resolve(): path for path in hierarchy_inputs}
    inputs_by_name: dict[str, list[Path]] = {}
    for path in hierarchy_inputs:
        inputs_by_name.setdefault(path.name, []).append(path)

    matched_sources: set[Path] = set()
    cached_root_ids: set[str] = set()
    for idx, root_result in enumerate(roots):
        if not isinstance(root_result, dict) or not root_result.get("ok"):
            raise ValueError(f"merged hierarchy cache root {idx} is missing or unsuccessful")
        hierarchy = root_result.get("hierarchy")
        if not isinstance(hierarchy, dict):
            raise ValueError(f"merged hierarchy cache root {idx} has no hierarchy")
        hierarchy_id = hierarchy.get("id")
        if isinstance(hierarchy_id, str) and hierarchy_id in cached_root_ids:
            raise ValueError(f"merged hierarchy cache has duplicate root id {hierarchy_id!r}")
        if isinstance(hierarchy_id, str):
            cached_root_ids.add(hierarchy_id)
        input_file = root_result.get("input_file")
        source_path: Path | None = None
        if isinstance(input_file, str) and input_file.strip():
            candidate = Path(input_file).expanduser()
            try:
                source_path = inputs_by_resolved.get(candidate.resolve())
            except OSError:
                source_path = None
            if source_path is None:
                matches = inputs_by_name.get(candidate.name, [])
                if len(matches) == 1:
                    source_path = matches[0]
        if source_path is None:
            raise ValueError(f"cannot match cached root {idx} to a current hierarchy input")
        resolved_source_path = source_path.expanduser().resolve()
        if resolved_source_path in matched_sources:
            raise ValueError(
                f"merged hierarchy cache matches current input {source_path.name} more than once"
            )
        matched_sources.add(resolved_source_path)
        source = read_json(source_path)
        feedback = validate_hierarchy(
            hierarchy,
            load_validation_source(source_path, source),
            large_node_review_threshold=large_node_review_threshold,
            small_decomposition_review_threshold=small_decomposition_review_threshold,
        )
        if not feedback.valid:
            raise ValueError(
                f"cached root {source_path.name} failed validation: {feedback.as_text()}"
            )
    missing_inputs = set(inputs_by_resolved) - matched_sources
    if missing_inputs:
        raise ValueError(
            "merged hierarchy cache does not contain every current input: "
            + ", ".join(sorted(path.name for path in missing_inputs))
        )
    return validated


def hierarchical_objective_induction(
    *,
    data_dir: str | Path,
    input_file_name: str,
    output_file_name: str,
    output_path: str | Path | None = None,
    direct_model: str,
    direct_litellm_params: dict[str, Any] | None,
    codex_config: dict[str, Any],
    max_retries: int,
    direct_llm_max_activities: int,
    workers: int,
    llm_timeout_secs: float,
    large_node_review_threshold: int,
    small_decomposition_review_threshold: int,
    force_per_root_outputs: bool = False,
    reuse_cache: bool = False,
    preflight_only: bool = False,
    no_console: bool = False,
    rebuild_codex_sandbox: bool = False,
) -> dict[str, Any] | None:
    global _METRICS_ENABLED
    previous_metrics_enabled = _METRICS_ENABLED
    _METRICS_ENABLED = False
    try:
        with HierarchicalObjectiveReporter(no_console=no_console) as reporter:
            try:
                reporter.set_metric("direct_model", direct_model)
                reporter.set_metric("codex_model", codex_config.get("model") or "")
                reporter.start_stage(STAGE_LOAD_INPUTS, "resolving paths")
                resolved_data_dir = Path(data_dir).expanduser()
                resolved_input = resolved_data_dir / input_file_name
                resolved_output = (
                    Path(output_path).expanduser()
                    if output_path
                    else resolved_data_dir / output_file_name
                )
                if not resolved_output.is_absolute():
                    resolved_output = resolved_data_dir / resolved_output
                if not resolved_input.exists():
                    raise FileNotFoundError(f"hierarchy input not found: {resolved_input}")
                hierarchy_inputs = derived_hierarchy_inputs(resolved_input)
                reporter.add_path("input", resolved_input)
                output_dir = resolved_output.parent / DEFAULT_TASK_THREAD_OUTPUT_DIR
                reporter.add_path("output", output_dir)
                reporter.set_counter("roots", len(hierarchy_inputs))
                reporter.finish_stage(STAGE_LOAD_INPUTS, f"loaded {len(hierarchy_inputs)} root inputs")

                reporter.start_stage(STAGE_PREFLIGHT, "validating planned work")
                if max_retries < 0:
                    raise ValueError("max_retries must be >= 0")
                if not hierarchy_inputs:
                    raise ValueError("No hierarchy inputs found.")
                reporter.finish_stage(STAGE_PREFLIGHT, "ready")
                if preflight_only:
                    reporter.mark_stage_done(STAGE_GENERATION, "skipped")
                    reporter.mark_stage_done(STAGE_MERGE, "skipped")
                    reporter.final_success("preflight complete; no LLM or Codex calls were made")
                    return None

                if reuse_cache:
                    if len(hierarchy_inputs) == 1 and not force_per_root_outputs:
                        cache_path = output_dir / hierarchy_inputs[0].name
                        if cache_path.exists():
                            try:
                                raw_cached = read_json(cache_path)
                                payload = HierarchicalObjectiveNode.model_validate(raw_cached).model_dump()
                                cached_source = read_json(hierarchy_inputs[0])
                                cached_feedback = validate_hierarchy(
                                    raw_cached,
                                    load_validation_source(hierarchy_inputs[0], cached_source),
                                    large_node_review_threshold=large_node_review_threshold,
                                    small_decomposition_review_threshold=small_decomposition_review_threshold,
                                )
                                if not cached_feedback.valid:
                                    raise ValueError(cached_feedback.as_text())
                            except (ValidationError, ValueError, TypeError) as exc:
                                reporter.progress(
                                    f"ignoring stale hierarchy cache {cache_path.name}: {exc}"
                                )
                            else:
                                reporter.start_stage(STAGE_GENERATION, "loading cached hierarchy")
                                reporter.finish_stage(STAGE_GENERATION, "loaded cached hierarchy")
                                reporter.start_stage(STAGE_MERGE, "using cached hierarchy output")
                                if not resolved_output.exists():
                                    write_merged_hierarchy_output(
                                        output_path=resolved_output,
                                        output_dir=output_dir,
                                        root_results=[
                                            build_root_result(
                                                hierarchy_inputs[0], cache_path, {"execution_mode": "cached"}
                                            )
                                        ],
                                        model=codex_config.get("model") or direct_model,
                                        preflight_only=False,
                                    )
                                reporter.finish_stage(STAGE_MERGE, "loaded cached output")
                                reporter.final_success("induced hierarchy from cache")
                                return payload
                    elif resolved_output.exists():
                        try:
                            payload = validate_merged_hierarchy_cache(
                                read_json(resolved_output),
                                hierarchy_inputs=hierarchy_inputs,
                                large_node_review_threshold=large_node_review_threshold,
                                small_decomposition_review_threshold=small_decomposition_review_threshold,
                            )
                        except (ValidationError, ValueError, TypeError) as exc:
                            reporter.progress(
                                f"ignoring stale merged hierarchy cache {resolved_output.name}: {exc}"
                            )
                        else:
                            reporter.start_stage(STAGE_GENERATION, "loading cached hierarchy")
                            reporter.finish_stage(STAGE_GENERATION, "loaded cached hierarchy")
                            reporter.start_stage(STAGE_MERGE, "using cached hierarchy output")
                            reporter.finish_stage(STAGE_MERGE, "loaded cached output")
                            reporter.final_success("induced hierarchy from cache")
                            return payload

                output_dir.mkdir(parents=True, exist_ok=True)

                root_plans: list[tuple[Path, int | None, str]] = []
                for path in hierarchy_inputs:
                    count = activity_count(path)
                    if (
                        count is not None
                        and count <= direct_llm_max_activities
                        and not direct_llm_input_too_large(path)
                    ):
                        execution_mode = "direct_llm"
                    else:
                        execution_mode = "codex_cli"
                        if count is not None and count <= direct_llm_max_activities:
                            reporter.progress(
                                f"routing {path.name} to codex_cli: input too large for direct LLM"
                            )
                    root_plans.append((path, count, execution_mode))
                mode_counts: dict[str, int] = {
                    "direct_llm": 0,
                    "codex_cli": 0,
                }
                root_plan_summary_parts: list[str] = []
                for path, count, mode in root_plans:
                    mode_counts[mode] += 1
                    activity_text = str(count) if count is not None else "unknown"
                    root_plan_summary_parts.append(f"{path.name}(n={activity_text},{mode})")
                reporter.initialize_root_statuses(root_plans)
                reporter.set_counter("direct_llm", mode_counts["direct_llm"])
                reporter.set_counter("codex_cli", mode_counts["codex_cli"])
                reporter.set_metric("direct_llm_max_activities", direct_llm_max_activities)
                reporter.progress(f"root plan: {', '.join(root_plan_summary_parts)}")

                reporter.start_stage(STAGE_GENERATION, "generating hierarchies")
                root_results: list[dict[str, Any]] = []
                errors: list[str] = []

                def runner(path: Path, out_path: Path, count: int | None, execution_mode: str) -> dict[str, Any]:
                    activity_text = str(count) if count is not None else "unknown"
                    detail = f"n={activity_text} mode={execution_mode}"
                    if reuse_cache and out_path.exists():
                        try:
                            raw_cached = read_json(out_path)
                            HierarchicalObjectiveNode.model_validate(raw_cached)
                            cached_feedback = validate_hierarchy(
                                raw_cached,
                                load_validation_source(path, read_json(path)),
                                large_node_review_threshold=large_node_review_threshold,
                                small_decomposition_review_threshold=small_decomposition_review_threshold,
                            )
                            if not cached_feedback.valid:
                                raise ValueError(cached_feedback.as_text())
                        except (ValidationError, ValueError, TypeError) as exc:
                            reporter.progress(f"ignoring stale root cache {out_path.name}: {exc}")
                        else:
                            reporter.update_root_status(path, "done", detail + " cached")
                            reporter.progress(f"root cached: {path.name} mode={execution_mode} status=reused")
                            return {"execution_mode": execution_mode, "cached": True}
                    reporter.update_root_status(path, "running", detail)
                    reporter.progress(
                        f"root start: {path.name} mode={execution_mode} activities={activity_text}"
                    )
                    if execution_mode == "direct_llm":
                        with litellm_model_config(model_alias=direct_model, litellm_params=direct_litellm_params):
                            result = run_direct_hierarchy_stage(
                                input_path=path,
                                output_path=out_path,
                                model=direct_model,
                                llm_timeout_secs=llm_timeout_secs,
                                large_node_review_threshold=large_node_review_threshold,
                                small_decomposition_review_threshold=small_decomposition_review_threshold,
                                max_retries=max_retries,
                                preflight_only=False,
                            )
                        reporter.update_root_status(path, "done", detail)
                        reporter.progress(
                            f"root done: {path.name} mode={execution_mode} status=success"
                        )
                        return result
                    result = run_with_retries(
                        lambda: run_codex_hierarchy_stage(
                            input_path=path,
                            output_path=out_path,
                            codex_config=codex_config,
                            rebuild_image=rebuild_codex_sandbox,
                            large_node_review_threshold=large_node_review_threshold,
                            small_decomposition_review_threshold=small_decomposition_review_threshold,
                        ),
                        attempts=max_retries,
                        on_retry=lambda attempt, exc: reporter.progress(
                            f"codex retry {attempt}/{max_retries} for {path.name}: {str(exc)[:200]}"
                        ),
                    )
                    reporter.update_root_status(path, "done", detail)
                    reporter.progress(
                        f"root done: {path.name} mode={execution_mode} status=success"
                    )
                    return result

                if len(hierarchy_inputs) == 1 and not force_per_root_outputs:
                    path = hierarchy_inputs[0]
                    count = activity_count(path)
                    execution_mode = (
                        "direct_llm"
                        if count is not None
                        and count <= direct_llm_max_activities
                        and not direct_llm_input_too_large(path)
                        else "codex_cli"
                    )
                    out_path = output_dir / path.name
                    run_result = runner(path, out_path, count, execution_mode)
                    root_results.append(build_root_result(path, out_path, run_result))
                else:
                    max_workers = max(1, min(workers, len(hierarchy_inputs)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_paths = {}
                        for path, count, execution_mode in root_plans:
                            out_path = output_dir / path.name
                            future_to_paths[executor.submit(runner, path, out_path, count, execution_mode)] = (
                                path,
                                out_path,
                                execution_mode,
                                count,
                            )
                        for future in as_completed(future_to_paths):
                            path, out_path, execution_mode, count = future_to_paths[future]
                            try:
                                root_results.append(build_root_result(path, out_path, future.result()))
                            except Exception as exc:
                                activity_text = str(count) if count is not None else "unknown"
                                reporter.update_root_status(
                                    path,
                                    "failed",
                                    f"n={activity_text} mode={execution_mode}",
                                )
                                errors.append(
                                    f"{path.name}: mode={execution_mode} activities={activity_text}: {exc}"
                                )
                                root_results.append({"input_file": str(path), "output_file": str(out_path), "ok": False, "error": str(exc)})
                root_results.sort(key=lambda item: str(item.get("input_file") or ""))
                reporter.set_counter("succeeded", sum(1 for item in root_results if item.get("ok")))
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
                # Every session owes the caller the merged artifacts (hierarchy.json,
                # its Markdown render, and the cost sidecar) — the cache-hit path above
                # writes them for a single root, so a fresh run must too.
                merged = write_merged_hierarchy_output(
                    output_path=resolved_output,
                    output_dir=output_dir,
                    root_results=root_results,
                    model=direct_model,
                    preflight_only=False,
                )
                if len(root_results) == 1 and not force_per_root_outputs:
                    raw_hierarchy = root_results[0].get("hierarchy")
                    if not isinstance(raw_hierarchy, dict):
                        raise RuntimeError("single-root hierarchy output was not written")
                    payload = HierarchicalObjectiveNode.model_validate(raw_hierarchy).model_dump()
                else:
                    payload = merged
                reporter.finish_stage(STAGE_MERGE, "saved output")
                if errors:
                    raise RuntimeError("one or more hierarchy root runs failed: " + "; ".join(errors))
                reporter.final_success(f"induced {len(root_results)} hierarchy roots")
                return payload
            except Exception as exc:
                reporter.fail_active_stage(exc)
                reporter.final_failure()
                raise
    finally:
        _METRICS_ENABLED = previous_metrics_enabled


def build_root_result(input_path: Path, output_path: Path, run_result: dict[str, Any]) -> dict[str, Any]:
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "ok": True,
        "run_id": run_result.get("run_id"),
        "session_id": run_result.get("session_id"),
        "execution_mode": run_result.get("execution_mode"),
        "activity_count": run_result.get("activity_count"),
        "usage": run_result.get("usage"),
        "estimated_usd": run_result.get("estimated_usd"),
        "cost_breakdown": run_result.get("cost_breakdown"),
        "proxy_cost": run_result.get("proxy_cost"),
    }
    if output_path.exists():
        result["hierarchy"] = json.loads(output_path.read_text(encoding="utf-8"))
    return result


def main() -> int:
    args = parse_args()
    if args.data_dir is None:
        raise ValueError("--data_dir is required for step4 objective model induction.")
    config_path = Path(args.config).expanduser().resolve() if args.config else resolve_config_path()
    config = load_config(config_path)
    if config.dotenv_path:
        try:
            from dotenv import load_dotenv

            dotenv_path = resolve_dotenv_path(config_path, config.dotenv_path)
            load_dotenv(dotenv_path, override=False)
        except ModuleNotFoundError:
            pass

    stage_config = config.hierarchical_objective_induction
    direct_branch = stage_config.direct_llm_branch
    codex_branch = stage_config.codex_branch
    output_path_arg = args.output or stage_config.output_file_name
    max_retries = args.max_retries if args.max_retries is not None else stage_config.max_retries
    direct_model = direct_branch.model
    codex_config = {
        "model": codex_branch.model,
        "model_reasoning_effort": codex_branch.model_reasoning_effort,
        "personality": codex_branch.personality,
        "model_provider": codex_branch.model_provider,
        "provider_name": codex_branch.provider_name,
        "command_timeout_seconds": codex_branch.command_timeout_seconds,
        "litellm_params": codex_branch.litellm_params,
    }
    hierarchical_objective_induction(
        data_dir=args.data_dir,
        input_file_name=stage_config.input_file_name,
        output_file_name=stage_config.output_file_name,
        output_path=output_path_arg,
        direct_model=direct_model,
        direct_litellm_params=direct_branch.litellm_params,
        codex_config=codex_config,
        max_retries=max_retries,
        direct_llm_max_activities=direct_branch.direct_llm_max_activities,
        workers=stage_config.workers,
        llm_timeout_secs=stage_config.llm_timeout_seconds,
        large_node_review_threshold=stage_config.large_node_review_threshold,
        small_decomposition_review_threshold=stage_config.small_decomposition_review_threshold,
        force_per_root_outputs=stage_config.force_per_root_outputs,
        reuse_cache=stage_config.reuse_cache,
        preflight_only=args.preflight_only,
        no_console=args.no_console,
        rebuild_codex_sandbox=stage_config.rebuild_codex_sandbox,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
