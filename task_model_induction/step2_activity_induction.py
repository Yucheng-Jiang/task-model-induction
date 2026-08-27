#!/usr/bin/env python3
"""Induce activities from atom semantic actions.

An activity is a maximal contiguous sequence of semantic actions that is
coherently explained by a single local user objective. It begins when the user
starts pursuing a new intermediate objective and ends when that objective is
achieved, abandoned, or superseded by another objective.

Input:
    atom_semantic_actions.jsonl

Output:
    activity.jsonl
    activity.meta.json

Example:
    uv run python -m task_model_induction.step2_activity_induction \
      --data_dir /path/to/trajectory_dir

Preflight without LLM calls:
    uv run python -m task_model_induction.step2_activity_induction \
      --data_dir /path/to/trajectory_dir \
      --preflight_only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.schemas import (
        AtomSemanticAction,
        Activity,
        ActivityInductionMeta,
        ActivityInductionOutput,
    )
    from task_model_induction.utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        litellm_completion,
        litellm_model_config,
        normalize_litellm_usage,
        read_jsonl_objects,
        utc_now_iso,
        write_json_atomic,
        write_jsonl_atomic,
    )
except ModuleNotFoundError:
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import (
        AtomSemanticAction,
        Activity,
        ActivityInductionMeta,
        ActivityInductionOutput,
    )
    from utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        litellm_completion,
        litellm_model_config,
        normalize_litellm_usage,
        read_jsonl_objects,
        utc_now_iso,
        write_json_atomic,
        write_jsonl_atomic,
    )


DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_INPUT_FILE_NAME = "atom_semantic_actions.jsonl"
DEFAULT_OUTPUT_FILE_NAME = "activity.jsonl"
DEFAULT_SEGMENTATION_BATCH_SIZE = 40
DEFAULT_MERGE_BATCH_SIZE = 16
DEFAULT_MERGE_BATCH_OVERLAP = 2
DEFAULT_MAX_PRIOR_SEGMENTS = 8
MAX_TEXT_FIELD_CHARS = 700

STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_SEGMENTATION = "segmentation"
STAGE_FORWARD_MERGE = "forward merge"
STAGE_WRITE_OUTPUT = "write output"
STAGES = [
    STAGE_LOAD_INPUTS,
    STAGE_PREFLIGHT,
    STAGE_SEGMENTATION,
    STAGE_FORWARD_MERGE,
    STAGE_WRITE_OUTPUT,
]


def _unique_nonempty(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def semantic_actions_fingerprint(actions: list[AtomSemanticAction]) -> str:
    payload = {
        "version": 1,
        "semantic_actions": [action.model_dump(mode="json") for action in actions],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


ACTIVITY_DEFINITION = """Activity definition:
An activity is a maximal contiguous sequence of semantic actions that is coherently explained by a single local user objective. It begins when the user starts pursuing a new intermediate objective and ends when that objective is achieved, abandoned, or superseded by another objective.

Granularity:
- An activity is larger than one atom semantic action, but smaller than a full task thread.
- It should be describable as one natural workflow step, such as "Retrieve the target artifact", "Configure the required component", "Start the local service", or "Submit the completed work item".
- Do not split just because the user clicks, scrolls, switches focus, corrects a typo, waits, or performs setup needed for the same activity.
- Do split when the activity changes, when the user starts inspecting/editing a different artifact for a different purpose, when a message/form submission completes a segment and the next action starts a new objective, or when the user abandons/supersedes the current objective.
- Prefer activity continuity over app continuity: the same app can contain many activities, and one activity can cross apps if the activity remains the same.

Field semantics:
- objective is a self-contained description of the intended local outcome or intermediate state. Include the relevant artifact/person/app/project when needed so it can be understood without reading neighboring segments.
- additional_context describes the observed mechanism/procedure and concrete evidence needed to understand the objective.
- Prefer outcome language such as "Inspect the contents of the project directory" over procedural language such as "Open a terminal, navigate to the directory, and list its contents".
- Keep tool/app names in objective only when they disambiguate the target or outcome, such as "Inspect the team chat thread about the deployment" or "Open the staging environment sign-in page".
"""


SEGMENTATION_PROMPT = """You segment chronological atom semantic actions into activities.

{definition}

=== WHAT HAPPENED BEFORE THIS BATCH ===
{prior_context}

=== SEMANTIC ACTIONS TO SEGMENT (chronological order, index 0 = earliest in this batch) ===
{actions_list}

=== TASK ===
Partition the current batch into contiguous activities.

For each segment output:
1. start_idx and end_idx, inclusive, using the batch-local indices.
2. objective: one concise, self-contained phrase/sentence naming the intended local outcome or intermediate state. Include the concrete target, artifact, person, project, channel, file, URL, or app needed to interpret the activity. Do not list the procedure.
3. additional_context: one to three concise sentences with the observed procedure and concrete evidence needed to understand that objective: relevant apps, people, files, URLs, commands, fields, channels, visible result, or state transition.

Coverage rules:
- Every index from 0 to {max_idx} must appear exactly once.
- Segments must be consecutive with no gaps, overlaps, or reordering.
- Keep scaffolding actions with the objective they enable when evidence supports it.
- If a activity appears to continue across a batch boundary, produce the best segment inside this batch; a later merge pass will join adjacent segments.

Output ONLY valid JSON:
{{
  "segments": [
    {{
      "start_idx": <int>,
      "end_idx": <int>,
      "objective": "<activity>",
      "additional_context": "<concise evidence-grounded context>"
    }}
  ]
}}"""


SEGMENTATION_RETRY_PROMPT = """Your previous activity segmentation output had validation errors.

{definition}

=== ORIGINAL SEMANTIC ACTIONS ===
{actions_list}

=== WHAT HAPPENED BEFORE THIS BATCH ===
{prior_context}

=== YOUR PREVIOUS OUTPUT ===
{previous_output}

=== ERRORS ===
{errors}

Produce corrected JSON with the same schema. Every index from 0 to {max_idx} must appear exactly once."""


MERGE_PROMPT = """You refine chronological candidate activities into final activities.

{definition}

Maintain a running understanding of the user's current activities.

=== WHAT HAPPENED BEFORE THESE CANDIDATES ===
{prior_context}

=== CURRENT CANDIDATE SUBGOAL SEGMENTS (chronological order, index 0 = earliest) ===
{segments_list}

=== TASK ===
Do NOT rebuild the full segmentation. Output ONLY adjacent candidate ranges you are confident should be merged.

Merge adjacent candidates when they are one activity, including cases where a batch boundary split one objective, or where one candidate is scaffolding/navigation/setup for the next candidate.

Keep candidates separate when they correspond to different activities, even if they use the same app, person, project, or artifact.

For each merge action output:
1. start_idx and end_idx, inclusive, using the candidate indices in this batch.
2. objective: a concise, self-contained outcome-oriented objective for the merged segment. Include the concrete target, artifact, person, project, channel, file, URL, or app needed to interpret the activity. Do not list the procedure.
3. additional_context: one to three concise sentences preserving the important procedure/evidence across the merged candidates.

Rules:
- Output only ranges that should be merged.
- Each merge range must contain at least 2 adjacent candidates.
- Merge ranges must not overlap.
- Return an empty list if no merges are justified.

Output ONLY valid JSON:
{{
  "merge_actions": [
    {{
      "start_idx": <int>,
      "end_idx": <int>,
      "objective": "<merged activity>",
      "additional_context": "<merged context>"
    }}
  ]
}}"""


MERGE_RETRY_PROMPT = """Your previous merge output had validation errors.

{definition}

=== ORIGINAL CANDIDATES ===
{segments_list}

=== WHAT HAPPENED BEFORE THESE CANDIDATES ===
{prior_context}

=== YOUR PREVIOUS OUTPUT ===
{previous_output}

=== ERRORS ===
{errors}

Produce corrected JSON with this schema:
{{
  "merge_actions": [
    {{
      "start_idx": <int>,
      "end_idx": <int>,
      "objective": "<merged activity>",
      "additional_context": "<merged context>"
    }}
  ]
}}

Output only ranges that should be merged. Return an empty list if no merge is justified."""


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.monotonic)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_usd: float = 0.0
    breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def elapsed_secs(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def as_meta(self) -> dict[str, Any]:
        return {
            "elapsed_secs": self.elapsed_secs(),
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "estimated_usd": round(self.estimated_usd, 6),
            "cost_breakdown": self.cost_breakdown(),
        }

    def record_call(self, *, operation: str, model: str, usage: dict[str, int], estimated_usd: float | None) -> None:
        self.llm_requests += 1
        for key, value in usage.items():
            setattr(self, key, getattr(self, key) + value)
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
                "cache_read_tokens": 0,
                "estimated_usd": 0.0,
            },
        )
        bucket["llm_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens"):
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
                "cache_read_tokens": self.cache_read_tokens,
                "estimated_usd": round(self.estimated_usd, 6),
            },
            "by_operation": list(self.breakdown.values()),
        }


class ActivityReporter(ConsoleProgressReporter):
    run_name = "activity_induction"
    success_title = "Activity Induction Complete"
    failure_title = "Activity Induction Failed"
    default_failure_stage = STAGE_PREFLIGHT
    refresh_per_second = 4

    def __init__(self, *, no_console: bool = False) -> None:
        self._last_plain_progress_at = 0.0
        super().__init__(stages=list(STAGES), no_console=no_console)

    def should_log_plain_progress(self, detail: str) -> bool:
        now = time.monotonic()
        if detail != self._last_plain_progress and now - self._last_plain_progress_at >= 2.0:
            self._last_plain_progress = detail
            self._last_plain_progress_at = now
            return True
        return False

    def render(self) -> Any:
        assert self._Panel is not None
        assert self._Group is not None
        assert self._Text is not None
        assert self._box is not None
        parts: list[Any] = [self._header(), self._metrics_table(), self._stage_table()]
        if self.state.error:
            parts.append(self._Text(f"error: {self.clip(self.state.error, 180)}", style="bold red"))
        return self._Panel(
            self._Group(*parts),
            title=self._Text("Activity Induction", style="bold cyan"),
            border_style="cyan",
            box=self._box.ROUNDED,
        )

    def _header(self) -> Any:
        assert self._Table is not None
        assert self._Text is not None
        table = self._Table.grid(expand=True)
        active = self._stage_label(self.state.active_stage) if self.state.active_stage else "Idle"
        table.add_row(self._Text("Task Model Induction", style="bold cyan"))
        table.add_row(self._Text(active, style="dim"))
        table.add_row(self._Text(f"Elapsed {format_duration(time.monotonic() - self.started_at)}", style="dim"))
        facts = self._facts_table()
        if facts.row_count:
            table.add_row("")
            table.add_row(facts)
        return table

    def _facts_table(self) -> Any:
        assert self._Table is not None
        table = self._Table.grid(padding=(0, 2))
        table.add_column(width=12, style="dim")
        table.add_column(ratio=1)
        for key in ("model", "limit", "cache"):
            value = self.state.metrics.get(key)
            if value not in (None, ""):
                table.add_row(key, self.clip(value, 110))
        for label in ("input", "output", "meta"):
            path = self.state.paths.get(label)
            if path is not None:
                table.add_row(label, self.clip(path, 110))
        return table

    def _metrics_table(self) -> Any:
        assert self._Table is not None
        table = self._Table.grid(expand=True, padding=(0, 2))
        table.add_column(width=18, style="dim")
        table.add_column(ratio=1)
        table.add_column(width=18, style="dim")
        table.add_column(ratio=1)
        counters = self.state.counters
        table.add_row(
            "semantic actions",
            str(counters.get("semantic_actions", 0)),
            "activities",
            str(counters.get("activities", 0)),
        )
        table.add_row(
            "candidate segments",
            str(counters.get("candidate_segments", 0)),
            "merge actions",
            str(counters.get("merge_actions", 0)),
        )
        table.add_row(
            "segmentation batches",
            self._progress(counters.get("segmentation_batches", 0), counters.get("expected_segmentation_batches", 0)),
            "merge batches",
            self._progress(counters.get("merge_batches", 0), counters.get("expected_merge_batches", 0)),
        )
        table.add_row(
            "tokens",
            str(counters.get("total_tokens", 0)),
            "in / out",
            f"{counters.get('input_tokens', 0)} / {counters.get('output_tokens', 0)}",
        )
        table.add_row(
            "llm requests",
            str(counters.get("llm_requests", 0)),
            "estimated cost",
            self._format_cost(),
        )
        errors = counters.get("segmentation_errors", 0) + counters.get("merge_errors", 0)
        table.add_row("errors", str(errors), "", "")
        return table

    def _stage_table(self) -> Any:
        assert self._Table is not None
        assert self._Text is not None
        table = self._Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=3)
        table.add_column(width=28)
        table.add_column(ratio=1)
        for index, stage in enumerate(self.state.stages, start=1):
            table.add_row(
                self._stage_indicator(stage.status),
                self._Text(f"{index}. {self._stage_label(stage.name)}", style=self._stage_style(stage.status)),
                self._Text(self.clip(stage.detail, 120), style="dim"),
            )
        return table

    def render_success(self, detail: str) -> Any:
        return self._summary_panel(title=self.success_title, border_style="green", detail=detail)

    def render_failure(self, message: str) -> Any:
        return self._summary_panel(title=self.failure_title, border_style="red", detail=message)

    def _summary_panel(self, *, title: str, border_style: str, detail: str) -> Any:
        assert self._Panel is not None
        assert self._Table is not None
        assert self._Text is not None
        assert self._box is not None
        counters = self.state.counters
        table = self._Table.grid(padding=(0, 2))
        table.add_column(width=18, style="dim")
        table.add_column(ratio=1)
        rows: list[tuple[str, object]] = [
            ("status", detail),
            ("elapsed", format_duration(time.monotonic() - self.started_at)),
            ("model", self.state.metrics.get("model", "")),
            ("semantic actions", counters.get("semantic_actions", 0)),
            ("candidate segments", counters.get("candidate_segments", 0)),
            ("activities", counters.get("activities", 0)),
            (
                "segmentation batches",
                f"{counters.get('segmentation_batches', 0)}/{counters.get('expected_segmentation_batches', 0)}",
            ),
            (
                "merge batches",
                f"{counters.get('merge_batches', 0)}/{counters.get('expected_merge_batches', 0)}",
            ),
            ("llm requests", counters.get("llm_requests", 0)),
            ("tokens", counters.get("total_tokens", 0)),
            ("estimated cost", self._format_cost()),
            ("errors", counters.get("segmentation_errors", 0) + counters.get("merge_errors", 0)),
        ]
        for label, path in self.state.paths.items():
            rows.append((label, path))
        if self.state.error:
            rows.append(("error", self.state.error))
        for label, value in rows:
            table.add_row(label, self.clip(value, 140))
        return self._Panel(table, title=self._Text(title, style=f"bold {border_style}"), border_style=border_style, box=self._box.ROUNDED)

    def _stage_indicator(self, status: str) -> Any:
        assert self._Spinner is not None
        assert self._Text is not None
        if status == "active":
            return self._Spinner("dots", style="cyan")
        if status == "done":
            return self._Text("ok", style="green")
        if status == "failed":
            return self._Text("x", style="bold red")
        return self._Text(".", style="dim")

    def _stage_style(self, status: str) -> str:
        if status == "active":
            return "bold cyan"
        if status == "done":
            return "green"
        if status == "failed":
            return "bold red"
        return "white"

    def _stage_label(self, name: str | None) -> str:
        labels = {
            STAGE_LOAD_INPUTS: "Load Inputs",
            STAGE_PREFLIGHT: "Preflight",
            STAGE_SEGMENTATION: "Segmentation",
            STAGE_FORWARD_MERGE: "Forward Merge",
            STAGE_WRITE_OUTPUT: "Write Output",
        }
        return labels.get(name or "", name or "")

    def _progress(self, completed: int, total: int) -> str:
        if total <= 0:
            return "0/0"
        pct = (completed / total) * 100
        return f"{completed}/{total} ({pct:.1f}%)"

    def _format_cost(self) -> str:
        value = self.state.metrics.get("estimated_usd", 0)
        try:
            return f"${float(value):.6f}"
        except Exception:
            return str(value)

    def plain_summary(self) -> str:
        counters = self.state.counters
        return (
            f"elapsed={format_duration(time.monotonic() - self.started_at)} "
            f"semantic_actions={counters.get('semantic_actions', 0)} "
            f"candidate_segments={counters.get('candidate_segments', 0)} "
            f"activities={counters.get('activities', 0)} "
            f"segmentation_batches={counters.get('segmentation_batches', 0)}/{counters.get('expected_segmentation_batches', 0)} "
            f"merge_batches={counters.get('merge_batches', 0)}/{counters.get('expected_merge_batches', 0)} "
            f"llm_requests={counters.get('llm_requests', 0)} "
            f"input_tokens={counters.get('input_tokens', 0)} "
            f"output_tokens={counters.get('output_tokens', 0)} "
            f"total_tokens={counters.get('total_tokens', 0)} "
            f"estimated_usd={self.state.metrics.get('estimated_usd', 0)} "
            f"errors={counters.get('segmentation_errors', 0) + counters.get('merge_errors', 0)} "
            f"output={self.state.paths.get('output') or ''}"
        )


@dataclass
class ActivityIR:
    start_semantic_action_idx: int
    end_semantic_action_idx: int
    objective: str = "Unclassified activity"
    additional_context: str = ""

    def semantic_action_count(self) -> int:
        return self.end_semantic_action_idx - self.start_semantic_action_idx + 1

    def to_model(self, index: int, actions: list[AtomSemanticAction]) -> Activity:
        segment_actions = actions[self.start_semantic_action_idx : self.end_semantic_action_idx + 1]
        source_actions = [source for action in segment_actions for source in action.actions]
        raw_action_ids = _unique_nonempty(
            source.action_id for source in source_actions
        ) or _unique_nonempty(
            action_id
            for action in segment_actions
            for action_id in (action.raw_action_ids or [action.start_action_id, action.end_action_id])
        )
        apps_used = _unique_nonempty(source.active_application for source in source_actions) or _unique_nonempty(
            app for action in segment_actions for app in action.apps_used
        )
        entities = _unique_nonempty(
            source.grounded_visual_content or source.visual_content for source in source_actions
        ) or _unique_nonempty(
            entity for action in segment_actions for entity in action.entities
        )
        ocr_texts = _unique_nonempty(source.md_results for source in source_actions) or _unique_nonempty(
            text for action in segment_actions for text in action.ocr_texts
        )
        return Activity(
            activity_id=f"activity_{index:04d}",
            start_semantic_action_idx=self.start_semantic_action_idx,
            end_semantic_action_idx=self.end_semantic_action_idx,
            start_semantic_action_id=segment_actions[0].semantic_action_id,
            end_semantic_action_id=segment_actions[-1].semantic_action_id,
            semantic_action_ids=[action.semantic_action_id for action in segment_actions],
            start_action_idx=segment_actions[0].start_action_idx,
            end_action_idx=segment_actions[-1].end_action_idx,
            start_action_id=segment_actions[0].start_action_id,
            end_action_id=segment_actions[-1].end_action_id,
            objective=(self.objective or "Unclassified activity").strip(),
            additional_context=(self.additional_context or "").strip(),
            semantic_actions=[action.semantic_action for action in segment_actions],
            source_actions=source_actions,
            raw_action_ids=raw_action_ids,
            apps_used=apps_used,
            entities=entities,
            pre_state=next(
                (
                    source.state_before
                    for source in source_actions
                    if source.state_before
                ),
                next((action.pre_state for action in segment_actions if action.pre_state), ""),
            ),
            post_state=next(
                (
                    source.state_after
                    for source in reversed(source_actions)
                    if source.state_after
                ),
                next((action.post_state for action in reversed(segment_actions) if action.post_state), ""),
            ),
            ocr_texts=ocr_texts,
            semantic_action_count=len(segment_actions),
            event_count=sum(action.event_count for action in segment_actions),
        )


@dataclass
class SegmentGroup:
    start_idx: int
    end_idx: int
    objective: str = "Unclassified activity"
    additional_context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentGroup":
        return cls(
            start_idx=int(data.get("start_idx", 0)),
            end_idx=int(data.get("end_idx", 0)),
            objective=_string(data.get("objective")) or "Unclassified activity",
            additional_context=_string(data.get("additional_context")) or _string(data.get("detail_brief")),
        )


@dataclass
class MergeAction:
    start_idx: int
    end_idx: int
    objective: str = ""
    additional_context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeAction":
        return cls(
            start_idx=int(data.get("start_idx", 0)),
            end_idx=int(data.get("end_idx", 0)),
            objective=_string(data.get("objective")),
            additional_context=_string(data.get("additional_context")) or _string(data.get("detail_brief")),
        )


def output_meta_path(output_path: Path) -> Path:
    if output_path.name.endswith(".jsonl"):
        return output_path.with_name(f"{output_path.name[:-6]}.meta.json")
    return output_path.with_suffix(".meta.json")


def resolve_stage_path(data_dir: Path, file_name: str | Path) -> Path:
    candidate = Path(file_name)
    return candidate if candidate.is_absolute() else data_dir / candidate


def read_semantic_actions(path: Path, *, limit: int | None = None) -> list[AtomSemanticAction]:
    rows = read_jsonl_objects(path)
    if limit is not None:
        rows = rows[:limit]
    return [AtomSemanticAction.model_validate(row) for row in rows]


def write_activity_output(output_path: Path, output: ActivityInductionOutput, *, stats: RunStats) -> None:
    rows = [segment.model_dump(mode="json") for segment in output.activities]
    write_jsonl_atomic(output_path, rows)
    meta = output.meta.model_dump(mode="json")
    meta.update(stats.as_meta())
    write_json_atomic(output_meta_path(output_path), meta)


def read_activity_output(output_path: Path) -> ActivityInductionOutput:
    rows = read_jsonl_objects(output_path)
    for row in rows:
        if "additional_context" not in row and "detail_brief" in row:
            row["additional_context"] = row.pop("detail_brief")
        if "activity_id" not in row:
            for legacy_key in ("subgoal_segment_id", "local_objective_id"):
                if legacy_key in row:
                    row["activity_id"] = row.pop(legacy_key)
                    break
        if "activity_id" in row:
            for prefix in ("local_objective_", "subgoal_segment_"):
                if row["activity_id"].startswith(prefix):
                    row["activity_id"] = "activity_" + row["activity_id"][len(prefix):]
                    break
    segments = [Activity.model_validate(row) for row in rows]
    meta_path = output_meta_path(output_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"activity metadata sidecar not found: {meta_path}")
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if "num_activities" not in meta_payload:
        for legacy_key in ("num_local_objectives", "num_subgoal_segments"):
            if legacy_key in meta_payload:
                meta_payload["num_activities"] = meta_payload.pop(legacy_key)
                break
    meta = ActivityInductionMeta.model_validate(meta_payload)
    return ActivityInductionOutput(meta=meta, activities=segments)


def rehydrate_activity_evidence(
    cached: ActivityInductionOutput,
    semantic_actions: list[AtomSemanticAction],
    *,
    input_fingerprint: str | None = None,
) -> ActivityInductionOutput | None:
    """Join cached activity boundaries to the current semantic-action file."""

    current_fingerprint = input_fingerprint or semantic_actions_fingerprint(semantic_actions)
    if (
        not semantic_actions
        or not cached.activities
        or cached.meta.num_semantic_actions != len(semantic_actions)
        or cached.meta.input_fingerprint != current_fingerprint
    ):
        return None

    expected_start = 0
    seen_ids: set[str] = set()
    refreshed: list[Activity] = []
    for index, activity in enumerate(cached.activities):
        start = activity.start_semantic_action_idx
        end = activity.end_semantic_action_idx
        if (
            start != expected_start
            or end < start
            or end >= len(semantic_actions)
            or activity.activity_id in seen_ids
        ):
            return None

        current_start_id = semantic_actions[start].semantic_action_id
        current_end_id = semantic_actions[end].semantic_action_id
        if activity.start_semantic_action_id and activity.start_semantic_action_id != current_start_id:
            return None
        if activity.end_semantic_action_id and activity.end_semantic_action_id != current_end_id:
            return None

        rebuilt = ActivityIR(
            start_semantic_action_idx=start,
            end_semantic_action_idx=end,
            objective=activity.objective,
            additional_context=activity.additional_context,
        ).to_model(index, semantic_actions)
        refreshed.append(rebuilt.model_copy(update={"activity_id": activity.activity_id}))
        seen_ids.add(activity.activity_id)
        expected_start = end + 1

    if expected_start != len(semantic_actions):
        return None

    cached.activities = refreshed
    cached.meta.num_semantic_actions = len(semantic_actions)
    cached.meta.num_activities = len(refreshed)
    cached.meta.input_fingerprint = current_fingerprint
    cached.meta.reused_cache = True
    return cached


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    if not response_text:
        return {}
    candidates: list[str] = []
    if "```json" in response_text:
        candidates.append(response_text.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in response_text:
        candidates.append(response_text.split("```", 1)[1].split("```", 1)[0].strip())
    candidates.append(response_text.strip())
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def call_llm_json(
    *,
    prompt: str,
    content: Any,
    model_name: str,
    stats: RunStats,
    operation: str,
    reporter: ActivityReporter | None = None,
) -> dict[str, Any]:
    user_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
    response = litellm_completion(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0 if "gpt-5" not in model_name and "kimi" not in model_name else 1.0,
        timeout=120.0,
        request_timeout=120.0,
        response_format={"type": "json_object"},
    )
    usage = _normalize_usage(response)
    call_usd = _estimated_completion_cost_usd(response, model_name)
    stats.record_call(operation=operation, model=model_name, usage=usage, estimated_usd=call_usd)
    if reporter is not None:
        reporter.increment("llm_requests")
        for key, value in usage.items():
            reporter.increment(key, value)
        if call_usd is not None:
            current_usd = float(reporter.state.metrics.get("estimated_usd", 0.0) or 0.0)
            reporter.set_metric("estimated_usd", round(current_usd + call_usd, 6))
        reporter.state.metrics["cost_breakdown"] = stats.cost_breakdown()
    text = completion_message_content(response)
    return extract_json_from_response(text)


def segment_semantic_actions(
    actions: list[AtomSemanticAction],
    *,
    model: str,
    batch_size: int,
    max_prior_segments: int,
    stats: RunStats,
    reporter: ActivityReporter | None = None,
    progress: bool = False,
    max_retries: int = 2,
) -> list[ActivityIR]:
    segments: list[ActivityIR] = []
    total_batches = (len(actions) + batch_size - 1) // batch_size
    for start in range(0, len(actions), batch_size):
        end = min(len(actions), start + batch_size)
        batch = actions[start:end]
        batch_number = (start // batch_size) + 1
        _log_progress(progress, stats, event="segmentation_batch_start", batch=batch_number, batches=total_batches, semantic_action_range=[start, end - 1])
        if reporter is not None:
            reporter.progress(f"batch {batch_number}/{total_batches} semantic_actions={start}-{end - 1}")
        previous_output: dict[str, Any] | None = None
        errors: list[str] | None = None
        payload: dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            actions_list = describe_semantic_action_batch(batch)
            prior_context = summarize_prior_segments(segments, actions, max_prior_segments=max_prior_segments)
            if attempt == 0:
                prompt = SEGMENTATION_PROMPT.format(
                    definition=ACTIVITY_DEFINITION,
                    prior_context=prior_context,
                    actions_list=actions_list,
                    max_idx=len(batch) - 1,
                )
            else:
                prompt = SEGMENTATION_RETRY_PROMPT.format(
                    definition=ACTIVITY_DEFINITION,
                    actions_list=actions_list,
                    prior_context=prior_context,
                    previous_output=json.dumps(previous_output or {}, ensure_ascii=False, indent=2),
                    errors="\n".join(f"- {error}" for error in errors or []),
                    max_idx=len(batch) - 1,
                )
            payload = call_llm_json(
                prompt=prompt,
                content=[],
                model_name=model,
                stats=stats,
                operation="segmentation",
                reporter=reporter,
            )
            groups = parse_groups(payload.get("segments"))
            errors = check_groups(groups, len(batch))
            if not errors:
                break
            if reporter is not None:
                reporter.increment("segmentation_errors")
            previous_output = payload
            _log_progress(progress, stats, event="segmentation_batch_retry", batch=batch_number, attempt=attempt + 1, errors=errors[:3])
        if errors:
            raise ValueError(f"invalid activity segmentation for batch {start}:{end - 1}: {'; '.join(errors)}")
        for group in groups:
            segments.append(
                ActivityIR(
                    start_semantic_action_idx=start + group.start_idx,
                    end_semantic_action_idx=start + group.end_idx,
                    objective=group.objective,
                    additional_context=group.additional_context,
                )
            )
        if reporter is not None:
            reporter.increment("segmentation_batches")
            reporter.set_counter("candidate_segments", len(segments))
        _log_progress(progress, stats, event="segmentation_batch_done", batch=batch_number, batches=total_batches, candidate_segments=len(segments))
    return segments


def merge_segments_forward(
    candidates: list[ActivityIR],
    actions: list[AtomSemanticAction],
    *,
    model: str,
    merge_batch_size: int,
    merge_batch_overlap: int,
    max_prior_segments: int,
    stats: RunStats,
    reporter: ActivityReporter | None = None,
    progress: bool = False,
    max_retries: int = 2,
) -> list[ActivityIR]:
    if len(candidates) <= 1:
        return candidates
    batch_step = max(1, merge_batch_size - merge_batch_overlap)
    merged_boundaries: set[int] = set()
    recorded_actions: list[MergeAction] = []
    total_batches = _expected_merge_batches(len(candidates), merge_batch_size, batch_step)
    batch_number = 0
    for batch_start in range(0, len(candidates), batch_step):
        batch_end = min(len(candidates), batch_start + merge_batch_size)
        batch = candidates[batch_start:batch_end]
        if len(batch) <= 1:
            break
        batch_number += 1
        _log_progress(progress, stats, event="merge_batch_start", batch=batch_number, batches=total_batches, candidate_range=[batch_start, batch_end - 1])
        if reporter is not None:
            reporter.progress(f"batch {batch_number}/{total_batches} candidates={batch_start}-{batch_end - 1}")
        previous_output: dict[str, Any] | None = None
        errors: list[str] | None = None
        payload: dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            segments_list = describe_candidate_segment_batch(batch, actions)
            prior_context = summarize_prior_segments(candidates[:batch_start], actions, max_prior_segments=max_prior_segments)
            if attempt == 0:
                prompt = MERGE_PROMPT.format(
                    definition=ACTIVITY_DEFINITION,
                    prior_context=prior_context,
                    segments_list=segments_list,
                )
            else:
                prompt = MERGE_RETRY_PROMPT.format(
                    definition=ACTIVITY_DEFINITION,
                    segments_list=segments_list,
                    prior_context=prior_context,
                    previous_output=json.dumps(previous_output or {}, ensure_ascii=False, indent=2),
                    errors="\n".join(f"- {error}" for error in errors or []),
                )
            payload = call_llm_json(
                prompt=prompt,
                content=[],
                model_name=model,
                stats=stats,
                operation="merge",
                reporter=reporter,
            )
            merge_actions = parse_merge_actions(payload.get("merge_actions"))
            errors = check_merge_actions(merge_actions, len(batch))
            if not errors:
                break
            if reporter is not None:
                reporter.increment("merge_errors")
            previous_output = payload
            _log_progress(progress, stats, event="merge_batch_retry", batch=batch_number, attempt=attempt + 1, errors=errors[:3])
        if errors:
            merge_actions = patch_merge_actions(merge_actions, len(batch))
            patched_errors = check_merge_actions(merge_actions, len(batch))
            if patched_errors:
                raise ValueError(f"invalid activity merge for candidates {batch_start}:{batch_end - 1}: {'; '.join(patched_errors)}")
            _log_progress(
                progress,
                stats,
                event="merge_batch_patched",
                batch=batch_number,
                discarded_errors=errors[:3],
                merge_actions=len(merge_actions),
            )
        for action in merge_actions:
            global_start = batch_start + action.start_idx
            global_end = batch_start + action.end_idx
            if global_start < 0 or global_end >= len(candidates) or global_start >= global_end:
                continue
            recorded_actions.append(
                MergeAction(
                    start_idx=global_start,
                    end_idx=global_end,
                    objective=action.objective,
                    additional_context=action.additional_context,
                )
            )
            for boundary_idx in range(global_start, global_end):
                merged_boundaries.add(boundary_idx)
        if reporter is not None:
            reporter.increment("merge_batches")
            reporter.increment("merge_actions", len(merge_actions))
        _log_progress(progress, stats, event="merge_batch_done", batch=batch_number, merge_actions=len(merge_actions))
        if batch_end >= len(candidates):
            break

    finalized: list[ActivityIR] = []
    cursor = 0
    while cursor < len(candidates):
        group_start = cursor
        while cursor < len(candidates) - 1 and cursor in merged_boundaries:
            cursor += 1
        group_end = cursor
        if group_start == group_end:
            finalized.append(candidates[group_start])
        else:
            finalized.append(collapse_segments(candidates[group_start : group_end + 1], pick_merge_label(group_start, group_end, recorded_actions)))
        cursor += 1
    return finalized


def induce_activities(
    *,
    data_dir: str | Path,
    input_file_name: str | Path = DEFAULT_INPUT_FILE_NAME,
    output_file_name: str | Path = DEFAULT_OUTPUT_FILE_NAME,
    model: str = DEFAULT_MODEL,
    segmentation_batch_size: int = DEFAULT_SEGMENTATION_BATCH_SIZE,
    merge_batch_size: int = DEFAULT_MERGE_BATCH_SIZE,
    merge_batch_overlap: int = DEFAULT_MERGE_BATCH_OVERLAP,
    max_prior_segments: int = DEFAULT_MAX_PRIOR_SEGMENTS,
    limit: int | None = None,
    reuse_cache: bool = False,
    preflight_only: bool = False,
    progress: bool = False,
    no_console: bool = False,
) -> ActivityInductionOutput | None:
    data_dir = Path(data_dir)
    input_path = resolve_stage_path(data_dir, input_file_name)
    output_path = resolve_stage_path(data_dir, output_file_name)
    stats = RunStats()
    reporter = ActivityReporter(no_console=no_console)
    try:
        with reporter:
            reporter.add_path("input", input_path)
            reporter.add_path("output", output_path)
            reporter.add_path("meta", output_meta_path(output_path))
            reporter.set_metric("model", model)
            if limit is not None:
                reporter.set_metric("limit", limit)

            reporter.start_stage(STAGE_LOAD_INPUTS, str(input_path))
            actions = read_semantic_actions(input_path, limit=limit)
            if not actions:
                raise ValueError(f"no semantic actions found in {input_path}")
            input_fingerprint = semantic_actions_fingerprint(actions)
            reporter.set_counter("semantic_actions", len(actions))
            reporter.finish_stage(STAGE_LOAD_INPUTS, f"{len(actions)} semantic actions loaded")

            reporter.start_stage(STAGE_PREFLIGHT, "validating expected work")
            effective_segmentation_batch_size = max(1, segmentation_batch_size)
            effective_merge_batch_size = max(2, merge_batch_size)
            effective_merge_batch_overlap = max(0, min(merge_batch_overlap, effective_merge_batch_size - 1))
            expected_segmentation_batches = (len(actions) + effective_segmentation_batch_size - 1) // effective_segmentation_batch_size
            reporter.set_counter("expected_segmentation_batches", expected_segmentation_batches)
            reporter.set_counter(
                "expected_merge_batches",
                _expected_merge_batches(
                    expected_segmentation_batches,
                    effective_merge_batch_size,
                    max(1, effective_merge_batch_size - effective_merge_batch_overlap),
                ),
            )
            if output_path.exists() and reuse_cache:
                reporter.set_metric("cache", "will reuse existing activities")
            elif output_path.exists():
                reporter.set_metric("cache", "will overwrite existing activities")
            else:
                reporter.set_metric("cache", "new local-objective output")
            reporter.finish_stage(STAGE_PREFLIGHT, "preflight complete")
            if preflight_only:
                reporter.mark_stage_done(STAGE_SEGMENTATION, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_FORWARD_MERGE, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_WRITE_OUTPUT, "skipped by --preflight_only")
                reporter.final_success("preflight complete; no LLM calls were made")
                return None

            if reuse_cache and output_path.exists():
                try:
                    cached = rehydrate_activity_evidence(
                        read_activity_output(output_path),
                        actions,
                        input_fingerprint=input_fingerprint,
                    )
                except Exception as exc:
                    cached = None
                    reporter.set_metric("cache", f"ignored unreadable cache: {exc}")
                if cached is not None:
                    reporter.start_stage(STAGE_WRITE_OUTPUT, "upgrading reusable activities")
                    reporter.set_counter("candidate_segments", cached.meta.num_candidate_segments)
                    reporter.set_counter("activities", len(cached.activities))
                    write_activity_output(output_path, cached, stats=stats)
                    reporter.finish_stage(STAGE_WRITE_OUTPUT, str(output_path))
                    reporter.mark_stage_done(STAGE_SEGMENTATION, "loaded from cache")
                    reporter.mark_stage_done(STAGE_FORWARD_MERGE, "loaded from cache")
                    reporter.final_success("loaded and evidence-refreshed cached activities")
                    return cached
                reporter.set_metric("cache", "ignored incompatible activity cache")

            reporter.start_stage(STAGE_SEGMENTATION, f"batch_size={effective_segmentation_batch_size}")
            candidates = segment_semantic_actions(
                actions,
                model=model,
                batch_size=effective_segmentation_batch_size,
                max_prior_segments=max(0, max_prior_segments),
                stats=stats,
                reporter=reporter,
                progress=progress,
            )
            reporter.set_counter("candidate_segments", len(candidates))
            reporter.finish_stage(STAGE_SEGMENTATION, f"{len(candidates)} candidate activities")

            reporter.set_counter(
                "expected_merge_batches",
                _expected_merge_batches(
                    len(candidates),
                    effective_merge_batch_size,
                    max(1, effective_merge_batch_size - effective_merge_batch_overlap),
                ),
            )
            reporter.start_stage(STAGE_FORWARD_MERGE, f"batch_size={effective_merge_batch_size}")
            merged = merge_segments_forward(
                candidates,
                actions,
                model=model,
                merge_batch_size=effective_merge_batch_size,
                merge_batch_overlap=effective_merge_batch_overlap,
                max_prior_segments=max(0, max_prior_segments),
                stats=stats,
                reporter=reporter,
                progress=progress,
            )
            reporter.set_counter("activities", len(merged))
            reporter.finish_stage(STAGE_FORWARD_MERGE, f"{len(merged)} activities")

            activities = [segment.to_model(index, actions) for index, segment in enumerate(merged)]
            output = ActivityInductionOutput(
                meta=ActivityInductionMeta(
                    created_at=utc_now_iso(),
                    model=model,
                    input_path=str(input_path),
                    input_fingerprint=input_fingerprint,
                    output_path=str(output_path),
                    num_semantic_actions=len(actions),
                    num_candidate_segments=len(candidates),
                    num_activities=len(activities),
                    segmentation_batch_size=effective_segmentation_batch_size,
                    merge_batch_size=effective_merge_batch_size,
                    merge_batch_overlap=effective_merge_batch_overlap,
                    max_prior_segments=max(0, max_prior_segments),
                ),
                activities=activities,
            )
            reporter.start_stage(STAGE_WRITE_OUTPUT, str(output_path))
            write_activity_output(output_path, output, stats=stats)
            reporter.finish_stage(STAGE_WRITE_OUTPUT, str(output_path))
            reporter.final_success("activity induction complete")
            return output
    except Exception as exc:
        reporter.fail_active_stage(exc)
        reporter.final_failure()
        setattr(exc, "_activity_reported", True)
        raise


def describe_semantic_action_batch(actions: list[AtomSemanticAction]) -> str:
    lines: list[str] = []
    for idx, action in enumerate(actions):
        payload = semantic_action_payload(action, idx)
        lines.append(f"[{idx}] {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def semantic_action_payload(action: AtomSemanticAction, idx: int | None = None) -> dict[str, Any]:
    apps: list[str] = []
    goals: list[str] = []
    for source in action.actions:
        if source.active_application and source.active_application not in apps:
            apps.append(source.active_application)
        if source.goal and source.goal not in goals:
            goals.append(source.goal)
    payload: dict[str, Any] = {
        "semantic_action_idx": idx,
        "semantic_action_id": action.semantic_action_id,
        "action_id_range": [action.start_action_id, action.end_action_id],
        "action_idx_range": [action.start_action_idx, action.end_action_idx],
        "semantic_action": action.semantic_action,
        "action_details": _clip(action.action_details, MAX_TEXT_FIELD_CHARS),
    }
    if apps:
        payload["applications"] = apps[:6]
    if goals:
        payload["source_action_goals"] = goals[:4]
    return payload


def describe_candidate_segment_batch(segments: list[ActivityIR], actions: list[AtomSemanticAction]) -> str:
    lines: list[str] = []
    for idx, segment in enumerate(segments):
        segment_actions = actions[segment.start_semantic_action_idx : segment.end_semantic_action_idx + 1]
        sample_actions = segment_actions[:2] + [action for action in segment_actions[-2:] if action not in segment_actions[:2]]
        payload = {
            "candidate_idx": idx,
            "semantic_action_idx_range": [segment.start_semantic_action_idx, segment.end_semantic_action_idx],
            "semantic_action_id_range": [segment_actions[0].semantic_action_id, segment_actions[-1].semantic_action_id],
            "action_idx_range": [segment_actions[0].start_action_idx, segment_actions[-1].end_action_idx],
            "objective": segment.objective,
            "additional_context": _clip(segment.additional_context, MAX_TEXT_FIELD_CHARS),
            "semantic_action_count": len(segment_actions),
            "sample_semantic_actions": [
                {
                    "semantic_action_id": action.semantic_action_id,
                    "semantic_action": action.semantic_action,
                    "action_details": _clip(action.action_details, 260),
                }
                for action in sample_actions
            ],
        }
        lines.append(f"[{idx}] {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def summarize_prior_segments(
    segments: list[ActivityIR],
    actions: list[AtomSemanticAction],
    *,
    max_prior_segments: int,
) -> str:
    if not segments or max_prior_segments <= 0:
        return "Start of session. No prior activities."
    relevant = segments[-max_prior_segments:]
    lines: list[str] = []
    for segment in relevant:
        first = actions[segment.start_semantic_action_idx]
        last = actions[segment.end_semantic_action_idx]
        lines.append(
            f"- semantic_actions={first.semantic_action_id}..{last.semantic_action_id} "
            f"actions={first.start_action_id}..{last.end_action_id}: "
            f"{_clip(segment.objective, 160)}"
        )
    return "Recent prior activities:\n" + "\n".join(lines)


def parse_groups(value: Any) -> list[SegmentGroup]:
    if not isinstance(value, list):
        return []
    groups: list[SegmentGroup] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            groups.append(SegmentGroup.from_dict(item))
        except (TypeError, ValueError):
            continue
    groups.sort(key=lambda group: group.start_idx)
    return groups


def parse_merge_actions(value: Any) -> list[MergeAction]:
    if not isinstance(value, list):
        return []
    actions: list[MergeAction] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            actions.append(MergeAction.from_dict(item))
        except (TypeError, ValueError):
            continue
    actions.sort(key=lambda action: action.start_idx)
    return actions


def check_groups(groups: list[SegmentGroup], batch_size: int) -> list[str]:
    if not groups:
        return ["missing segments list"]
    errors: list[str] = []
    covered: list[int] = []
    for group in groups:
        if group.start_idx < 0 or group.end_idx < 0 or group.start_idx >= batch_size or group.end_idx >= batch_size:
            errors.append(f"index out of range: start_idx={group.start_idx}, end_idx={group.end_idx}")
            continue
        if group.start_idx > group.end_idx:
            errors.append(f"start_idx greater than end_idx: {group.start_idx}>{group.end_idx}")
            continue
        if not group.objective.strip():
            errors.append(f"segment {group.start_idx}-{group.end_idx} missing objective")
        covered.extend(range(group.start_idx, group.end_idx + 1))
    expected = list(range(batch_size))
    if sorted(covered) != expected:
        missing = sorted(set(expected) - set(covered))
        extra = sorted(set(covered) - set(expected))
        duplicates = sorted({idx for idx in covered if covered.count(idx) > 1})
        if missing:
            errors.append(f"missing indices: {missing}")
        if extra:
            errors.append(f"unexpected indices: {extra}")
        if duplicates:
            errors.append(f"duplicate indices: {duplicates}")
    return errors


def check_merge_actions(actions: list[MergeAction], batch_size: int) -> list[str]:
    errors: list[str] = []
    covered: list[int] = []
    for action in actions:
        if action.start_idx < 0 or action.end_idx < 0 or action.start_idx >= batch_size or action.end_idx >= batch_size:
            errors.append(f"index out of range: start_idx={action.start_idx}, end_idx={action.end_idx}")
            continue
        if action.start_idx >= action.end_idx:
            errors.append(f"merge range must span at least two candidates: {action.start_idx}-{action.end_idx}")
            continue
        covered.extend(range(action.start_idx, action.end_idx + 1))
    duplicates = sorted({idx for idx in covered if covered.count(idx) > 1})
    if duplicates:
        errors.append(f"overlapping merge ranges at indices: {duplicates}")
    return errors


def patch_merge_actions(actions: list[MergeAction], batch_size: int) -> list[MergeAction]:
    valid: list[MergeAction] = []
    covered: set[int] = set()
    for action in sorted(actions, key=lambda item: item.start_idx):
        if action.start_idx < 0 or action.end_idx < 0:
            continue
        if action.start_idx >= batch_size or action.end_idx >= batch_size:
            continue
        if action.start_idx >= action.end_idx:
            continue
        indices = set(range(action.start_idx, action.end_idx + 1))
        if covered & indices:
            continue
        valid.append(action)
        covered.update(indices)
    return valid


def collapse_segments(segments: list[ActivityIR], merge_action: MergeAction | None) -> ActivityIR:
    if merge_action is not None:
        objective = merge_action.objective
        additional_context = merge_action.additional_context
    else:
        objective = segments[-1].objective or segments[0].objective
        additional_context = " ".join(segment.additional_context for segment in segments if segment.additional_context)
    return ActivityIR(
        start_semantic_action_idx=segments[0].start_semantic_action_idx,
        end_semantic_action_idx=segments[-1].end_semantic_action_idx,
        objective=(objective or segments[-1].objective or segments[0].objective).strip(),
        additional_context=_clip(additional_context, 1200),
    )


def pick_merge_label(group_start: int, group_end: int, actions: list[MergeAction]) -> MergeAction | None:
    exact: MergeAction | None = None
    best: MergeAction | None = None
    best_width = -1
    for action in actions:
        if action.start_idx == group_start and action.end_idx == group_end:
            exact = action
            break
        if action.start_idx >= group_start and action.end_idx <= group_end:
            width = action.end_idx - action.start_idx
            if width > best_width:
                best = action
                best_width = width
    return exact or best


def _expected_merge_batches(total: int, batch_size: int, batch_step: int) -> int:
    if total <= 1:
        return 0
    count = 0
    batch_start = 0
    while batch_start < total:
        batch_end = min(total, batch_start + batch_size)
        if batch_end - batch_start <= 1:
            break
        count += 1
        if batch_end >= total:
            break
        batch_start += batch_step
    return count


def _normalize_usage(response: Any) -> dict[str, int]:
    return normalize_litellm_usage(response)


def _estimated_completion_cost_usd(response: Any, model_name: str) -> float | None:
    return estimated_litellm_completion_cost_usd(response, model_name)


def _usage_int(usage: Any, *keys: str) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def _clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _log_progress(enabled: bool, stats: RunStats, *, event: str, **fields: Any) -> None:
    if not enabled:
        return
    payload = {
        "event": event,
        "elapsed_secs": stats.elapsed_secs(),
        "llm_requests": stats.llm_requests,
        "total_tokens": stats.total_tokens,
        "estimated_usd": round(stats.estimated_usd, 6),
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Induce activities from atom semantic actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing atom_semantic_actions.jsonl.")
    parser.add_argument("--config", type=Path, default=None, help="Task model induction config path.")
    parser.add_argument(
        "--input_file_name",
        "--input_path",
        dest="input_file_name",
        type=Path,
        default=None,
        help="Input JSONL filename relative to --data_dir, or an absolute path.",
    )
    parser.add_argument(
        "--output_file_name",
        "--output_path",
        "--output",
        dest="output_file_name",
        type=Path,
        default=None,
        help="Output JSONL path relative to --data_dir, or an absolute path.",
    )
    parser.add_argument("--model", type=str, default=None, help="LiteLLM model name.")
    parser.add_argument("--segmentation_batch_size", type=int, default=None, help="Semantic actions per first-pass segmentation batch.")
    parser.add_argument("--merge_batch_size", type=int, default=None, help="Candidate activities per forward merge batch.")
    parser.add_argument("--merge_batch_overlap", type=int, default=None, help="Overlap between forward merge batches.")
    parser.add_argument("--max_prior_segments", type=int, default=None, help="Recent prior local-objective summaries shown in prompts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of semantic actions to process.")
    parser.add_argument("--preflight_only", action="store_true", help="Validate input and print planned work without LLM calls.")
    parser.add_argument("--no_console", action="store_true", help="Suppress JSON progress logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve() if args.config is not None else resolve_config_path()
        config = load_config(config_path)
        if config.dotenv_path:
            try:
                from dotenv import load_dotenv

                dotenv_path = resolve_dotenv_path(config_path, config.dotenv_path)
                load_dotenv(dotenv_path, override=False)
            except ModuleNotFoundError:
                pass

        stage_config = config.activity_induction
        model = args.model or stage_config.model
        input_file_name = args.input_file_name or stage_config.input_file_name
        output_file_name = args.output_file_name or stage_config.output_file_name
        segmentation_batch_size = args.segmentation_batch_size or stage_config.segmentation_batch_size
        merge_batch_size = args.merge_batch_size or stage_config.merge_batch_size
        merge_batch_overlap = (
            args.merge_batch_overlap if args.merge_batch_overlap is not None else stage_config.merge_batch_overlap
        )
        max_prior_segments = args.max_prior_segments or stage_config.max_prior_segments
        limit = args.limit if args.limit is not None else stage_config.limit
        with litellm_model_config(model_alias=stage_config.model, litellm_params=stage_config.litellm_params):
            induce_activities(
                data_dir=Path(args.data_dir),
                input_file_name=input_file_name,
                output_file_name=output_file_name,
                model=model,
                segmentation_batch_size=segmentation_batch_size,
                merge_batch_size=merge_batch_size,
                merge_batch_overlap=merge_batch_overlap,
                max_prior_segments=max_prior_segments,
                limit=limit,
                reuse_cache=stage_config.reuse_cache,
                preflight_only=args.preflight_only,
                progress=args.no_console,
                no_console=args.no_console,
            )
        return 0
    except KeyboardInterrupt:
        print("Local objective induction interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if not getattr(exc, "_activity_reported", False):
            print(f"Local objective induction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
