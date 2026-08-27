#!/usr/bin/env python3
"""Induce atom semantic actions from grounded UI trajectory rows.

This is stage 1 of task modeling. It reads a trajectory JSONL, preferably
`processed_trajectory_with_goals.jsonl`, enriches each action's visual content
from OCR markdown, and uses backward segmentation followed by a forward merge
pass to compress grounded UI rows into atom semantic actions.

Input:
    processed_trajectory_with_goals.jsonl, falling back to
    processed_trajectory.jsonl.

Output:
    atom_semantic_actions.jsonl
    atom_semantic_actions.meta.json

Example:
    uv run python task_model_induction/step1_semantic_action_induction.py \
      --data_dir /path/to/trajectory_dir

Preflight without LLM calls:
    uv run python task_model_induction/step1_semantic_action_induction.py \
      --data_dir /path/to/trajectory_dir \
      --preflight_only

Console behavior:
    Interactive terminals show a live Rich status panel. Use --no_console for
    plain structured stderr logs in scripts or CI.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.schemas import (
        AtomSemanticAction,
        SemanticActionInductionMeta,
        SemanticActionInductionOutput,
        SemanticActionSourceAction,
    )
    from task_model_induction.utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        is_action_row,
        litellm_completion,
        litellm_model_configs,
        normalize_litellm_usage,
        read_jsonl_objects,
        row_id,
        safe_action_text,
        string_field,
        utc_now_iso,
        write_json_atomic,
        write_jsonl_atomic,
    )
except ModuleNotFoundError:
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import (
        AtomSemanticAction,
        SemanticActionInductionMeta,
        SemanticActionInductionOutput,
        SemanticActionSourceAction,
    )
    from utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        is_action_row,
        litellm_completion,
        litellm_model_configs,
        normalize_litellm_usage,
        read_jsonl_objects,
        row_id,
        safe_action_text,
        string_field,
        utc_now_iso,
        write_json_atomic,
        write_jsonl_atomic,
    )


DEFAULT_RAW_INPUT_FILE_NAME = "processed_trajectory.jsonl"
DEFAULT_GROUNDED_INPUT_FILE_NAME = "processed_trajectory_with_goals.jsonl"
DEFAULT_SEMANTIC_ACTION_OUTPUT_FILE_NAME = "atom_semantic_actions.jsonl"
DEFAULT_VISUAL_ENRICHMENT_CACHE_FILE_NAME = "visual_content_enrichment_cache.jsonl"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_ENRICHMENT_MODEL = "openai/gpt-5-nano"
DEFAULT_ENRICHMENT_WORKERS = 128
DEFAULT_BACKWARD_BATCH_SIZE = 40
DEFAULT_BACKWARD_BATCH_OVERLAP = 4
DEFAULT_BACKWARD_WORKERS = 32

STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_VISUAL_ENRICHMENT = "visual enrichment"
STAGE_SEMANTIC_ACTIONS = "semantic actions"
STAGE_FORWARD_MERGE = "forward merge"
STAGE_ACTION_DETAILS = "action details"
STAGE_WRITE_OUTPUT = "write output"
STAGES = [
    STAGE_LOAD_INPUTS,
    STAGE_PREFLIGHT,
    STAGE_VISUAL_ENRICHMENT,
    STAGE_SEMANTIC_ACTIONS,
    STAGE_FORWARD_MERGE,
    STAGE_ACTION_DETAILS,
    STAGE_WRITE_OUTPUT,
]


def _unique_nonempty(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


ATOM_SEMANTIC_ACTION_DEFINITION = """Atom semantic action definition:
An atom semantic action is the smallest contiguous unit of observed activity that expresses one intentional user operation at semantic, tool-agnostic granularity.

It converts low-level UI/control operations into one meaningful semantic action.

It must satisfy all of these:
- Single intentional operation: the range answers what the user intentionally did here.
- Observable state transition: an observer can describe what changed in the interface, document, browser, terminal, file, or application state.
- Mechanics collapsed: clicks, focus changes, cursor movement, scrolling, selection, waits, and retries stay together when they are only needed to perform the same semantic action.
- Semantic action granularity: the output should describe one meaningful user-facing operation, not raw UI mechanics.
- Local evidence grounded: the action must be supported by events, screenshots, or UI text in the range, possibly clarified by later context.
- Task-neutral when possible: describe the immediate operation, not the full broader task unless the local operation itself reveals it.

Do not create separate semantic actions for repeated navigation, scrolling, cursor movement, selection expansion, transient app switching, or correction/retry loops when the intentional operation is unchanged.
Some single-event actions are legitimate when they accomplish a meaningful state transition, such as submitting a form.
Do not merge distinct intentional operations just because they occur in the same app or contribute to the same broader task.
Do not encode incidental UI mechanics as the semantic action unless the mechanic is itself the meaningful operation."""


BACKWARD_SEMANTIC_ACTION_SEGMENT_PROMPT = """You are analyzing a user's computer workflow by looking at actions in REVERSE order from the end of the session backward.

KEY INSIGHT: later outcomes help explain earlier low-level actions.

Each segment should be one candidate atom semantic action.

{semantic_action_definition}

=== WHAT HAPPENS AFTER THESE ACTIONS ===
{future_context}

=== ACTIONS TO ANALYZE (chronological order, index 0 = earliest) ===
{actions_list}

=== TASK ===
Segment these low-level actions into atom semantic actions.

For each group output:
1. semantic_action: one concise sentence describing the intentional operation

Semantic-action rules:
- Use semantic, operation-level language.
- Prefer the immediate operation over the broader task objective.
- Keep concrete apps, clicks, typing, commands, URLs, files, and navigation out of semantic_action unless essential.
- Avoid semantic actions that start with purely mechanical verbs like click, scroll, focus, move, hover, drag, or wait unless that operation is itself the meaningful user action.
- Do not skip failed attempts or corrections; include them with the operation they are trying to complete when intent is unchanged.

Coverage rules:
- Groups must be consecutive indices with no gaps, overlaps, or reorder.
- Every index from 0 to {max_idx} must appear exactly once.
- Split when the range contains multiple intentional operations.
- Keep together repeated low-level interaction needed to finish the same operation.
- Avoid broad summaries that combine setup, navigation, inspection, editing, submission, and follow-up into one group.
- Long repeated scrolling, paging, or selection within the same open resource can remain one group when the intent stays unchanged.

Output ONLY valid JSON:
{{
  "groups": [
    {{
      "start_idx": <int>,
      "end_idx": <int>,
      "semantic_action": "<atom semantic action>"
    }}
  ]
}}

List groups in REVERSE chronological order, latest group first. start_idx and end_idx are inclusive."""


BACKWARD_SEMANTIC_ACTION_RETRY_PROMPT = """You previously segmented this batch, but your output had coverage errors.

{semantic_action_definition}

=== ORIGINAL ACTIONS (chronological order, index 0 = earliest) ===
{actions_list}

=== WHAT HAPPENS AFTER THESE ACTIONS ===
{future_context}

=== YOUR PREVIOUS OUTPUT ===
{previous_output}

=== ERRORS ===
{errors}

Produce corrected JSON with the same schema. Every index from 0 to {max_idx} must appear exactly once."""


FORWARD_SEMANTIC_ACTION_MERGE_PROMPT = """You are refining chronological candidate atom semantic actions into final atom semantic actions.

{semantic_action_definition}

Maintain a running understanding of what the user is working on right now from prior context.
Some backward-segmented candidates are objective-bearing operations. Others are scaffolding that only makes sense when attached to a neighboring operation.

=== WHAT HAPPENED BEFORE THESE CANDIDATES ===
{prior_context}

=== CURRENT CANDIDATE SEMANTIC ACTIONS (chronological order, index 0 = earliest) ===
{segments_list}

=== TASK ===
Do NOT rebuild the full segmentation.
Output ONLY merge actions you are confident about.
Default behavior is to keep candidates separate.

Merge adjacent candidates when they are one intentional operation or when scaffolding candidates should attach to a neighboring objective-bearing candidate.
Typical scaffolding to merge includes:
- invoking a keyboard shortcut to prepare the next action;
- moving focus within a draft input;
- resizing or rearranging a pane as setup for the next operation;
- switching apps without accomplishing a user-facing state transition;
- placing the cursor in a field before the actual edit;
- pressing a modifier key;
- selection, copy, paste, typing corrections, repeated submit attempts, or retries that serve one edit/message/form operation.

Keep candidates separate when each candidate accomplishes its own user-facing state transition, even if it is a single event, such as submitting a form, opening citation details, selecting a score, or opening a different review stage.
When merged labels disagree with representative action evidence, write the merged_semantic_action from the observed state transition instead of copying earlier candidate wording.

For each merge action output:
1. start_idx
2. end_idx
3. merged_semantic_action: one concise sentence for the merged intentional operation

Rules:
- Output only ranges that should be merged.
- Each merge range must contain at least 2 adjacent candidates.
- Merge ranges must not overlap.
- Use the candidate indices from this batch.
- Return an empty list if no merges are justified.

Output ONLY valid JSON:
{{
  "merge_actions": [
    {{
      "start_idx": <int>,
      "end_idx": <int>,
      "merged_semantic_action": "<atom semantic action>"
    }}
  ]
}}"""


FORWARD_SEMANTIC_ACTION_MERGE_RETRY_PROMPT = """You previously produced merge actions, but the output had validation errors.

{semantic_action_definition}

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
      "merged_semantic_action": "<atom semantic action>"
    }}
  ]
}}

Output only ranges that should be merged. Return an empty list if no merge is justified."""


VISUAL_CONTENT_ENRICHMENT_PROMPT = """You enrich focused UI visual content for workflow understanding.

Use the objective, existing focused visual content, active application, and OCR/markdown text to produce a compact replacement for visual_content.

Rules:
- Preserve the original focused visual content when it is useful.
- Add only details from the markdown that clarify the objective or visible target.
- Include concrete visible labels, selected items, field names, document titles, filenames, commands, values, or UI state when relevant.
- Do not summarize the entire screen. Ignore irrelevant markdown.
- Keep it concise, do not be verbose, only include necessary information.

Output ONLY valid JSON:
{
  "visual_content": "<enriched visual content>"
}"""


SEMANTIC_ACTION_DETAILS_PROMPT = """You add evidence-grounded details to one semantic action.

Use the semantic action label and the enriched visual content from its source actions to write additional details that clarify what the action operated on and what visible context matters.

Rules:
- Do not rename the semantic action.
- Include concrete visible objects, fields, filenames, selected items, content snippets, state changes, or app context when they help interpret the action.
- Keep details factual and grounded in the provided visual content.
- Keep it concise, one to three sentences.
- Return an empty string if there are no useful additional details.

Output ONLY valid JSON:
{
  "action_details": "<additional details>"
}"""


_REPORTER_LOCK = threading.Lock()


class SemanticActionReporter(ConsoleProgressReporter):
    run_name = "semantic_action_induction"
    success_title = "Semantic Action Induction Complete"
    failure_title = "Semantic Action Induction Failed"
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
        parts: list[Any] = [
            self._header(),
            self._metrics_table(),
            self._stage_table(),
        ]
        if self.state.error:
            parts.append(self._Text(f"error: {self.clip(self.state.error, 180)}", style="bold red"))
        return self._Panel(
            self._Group(*parts),
            title=self._Text("Semantic Action Induction", style="bold cyan"),
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
        for key in ("model", "enrichment_model", "limits", "cache"):
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
        enrichment_errors = counters.get("visual_enrichment_errors", 0) + counters.get("action_detail_errors", 0)
        semantic_errors = counters.get("semantic_action_errors", 0)
        merge_errors = counters.get("merge_errors", 0)
        table.add_row(
            "actions",
            str(counters.get("actions", 0)),
            "semantic actions",
            str(counters.get("semantic_actions", 0)),
        )
        table.add_row(
            "backward batches",
            self._progress(
                counters.get("backward_batches", 0),
                counters.get("expected_backward_batches", 0),
            ),
            "merge batches",
            self._progress(
                counters.get("merge_batches", 0),
                counters.get("expected_merge_batches", 0),
            ),
        )
        table.add_row(
            "tokens",
            str(counters.get("total_tokens", 0)),
            "in / out",
            f"{counters.get('input_tokens', 0)} / {counters.get('output_tokens', 0)}",
        )
        table.add_row(
            "estimated cost",
            self._format_cost(),
            "errors",
            str(enrichment_errors + semantic_errors + merge_errors),
        )
        table.add_row(
            "llm requests",
            str(counters.get("llm_requests", 0)),
            "merge actions",
            str(counters.get("merge_actions", 0)),
        )
        table.add_row(
            "visual enrichments",
            self._progress(counters.get("visual_enrichments", 0), counters.get("expected_visual_enrichments", 0)),
            "action details",
            self._progress(counters.get("action_details", 0), counters.get("expected_action_details", 0)),
        )
        table.add_row(
            "visual cache hits",
            str(counters.get("visual_enrichment_cache_hits", 0)),
            "",
            "",
        )
        if enrichment_errors:
            table.add_row("enrichment errors", str(enrichment_errors), "", "")
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
        errors = (
            counters.get("semantic_action_errors", 0)
            + counters.get("merge_errors", 0)
            + counters.get("visual_enrichment_errors", 0)
            + counters.get("action_detail_errors", 0)
        )
        table = self._Table.grid(padding=(0, 2))
        table.add_column(width=18, style="dim")
        table.add_column(ratio=1)
        rows: list[tuple[str, object]] = [
            ("status", detail),
            ("elapsed", format_duration(time.monotonic() - self.started_at)),
            ("model", self.state.metrics.get("model", "")),
            ("actions", counters.get("actions", 0)),
            ("backward candidates", counters.get("backward_semantic_actions", 0)),
            ("semantic actions", counters.get("semantic_actions", 0)),
            (
                "backward batches",
                f"{counters.get('backward_batches', 0)}/{counters.get('expected_backward_batches', 0)}",
            ),
            (
                "merge batches",
                f"{counters.get('merge_batches', 0)}/{counters.get('expected_merge_batches', 0)}",
            ),
            (
                "visual enrichments",
                f"{counters.get('visual_enrichments', 0)}/{counters.get('expected_visual_enrichments', 0)}",
            ),
            ("visual cache hits", counters.get("visual_enrichment_cache_hits", 0)),
            (
                "action details",
                f"{counters.get('action_details', 0)}/{counters.get('expected_action_details', 0)}",
            ),
            ("llm requests", counters.get("llm_requests", 0)),
            ("tokens", counters.get("total_tokens", 0)),
            ("estimated cost", self._format_cost()),
            ("errors", errors),
        ]
        for label, path in self.state.paths.items():
            rows.append((label, path))
        if self.state.error:
            rows.append(("error", self.state.error))
        for label, value in rows:
            table.add_row(label, self.clip(value, 140))
        return self._Panel(
            table,
            title=self._Text(title, style=f"bold {border_style}"),
            border_style=border_style,
            box=self._box.ROUNDED,
        )

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
            STAGE_VISUAL_ENRICHMENT: "Visual Enrichment",
            STAGE_SEMANTIC_ACTIONS: "Semantic Actions",
            STAGE_FORWARD_MERGE: "Forward Merge",
            STAGE_ACTION_DETAILS: "Action Details",
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
        errors = (
            counters.get("semantic_action_errors", 0)
            + counters.get("merge_errors", 0)
            + counters.get("visual_enrichment_errors", 0)
            + counters.get("action_detail_errors", 0)
        )
        return (
            f"elapsed={format_duration(time.monotonic() - self.started_at)} "
            f"actions={counters.get('actions', 0)} "
            f"backward_semantic_actions={counters.get('backward_semantic_actions', 0)} "
            f"semantic_actions={counters.get('semantic_actions', 0)} "
            f"backward_batches={counters.get('backward_batches', 0)}/{counters.get('expected_backward_batches', 0)} "
            f"merge_batches={counters.get('merge_batches', 0)}/{counters.get('expected_merge_batches', 0)} "
            f"visual_enrichments={counters.get('visual_enrichments', 0)}/{counters.get('expected_visual_enrichments', 0)} "
            f"action_details={counters.get('action_details', 0)}/{counters.get('expected_action_details', 0)} "
            f"llm_requests={counters.get('llm_requests', 0)} "
            f"input_tokens={counters.get('input_tokens', 0)} "
            f"output_tokens={counters.get('output_tokens', 0)} "
            f"total_tokens={counters.get('total_tokens', 0)} "
            f"estimated_usd={self.state.metrics.get('estimated_usd', 0)} "
            f"errors={errors} "
            f"output={self.state.paths.get('output') or ''}"
        )


@dataclass
class ActionTraceEntry:
    action_id: str | None
    action: str
    original_index: int | None = None
    status: str | None = None
    goal: str | None = None
    active_application: str | None = None
    grounded_visual_content: str | None = None
    visual_content: str | None = None
    ocr_results: dict[str, Any] | None = None
    md_results: str | None = None
    state_before: str | None = None
    state_after: str | None = None
    time_before: float | str | None = None
    time_after: float | str | None = None
    time_range: float | None = None


@dataclass
class SemanticActionGroup:
    start_idx: int
    end_idx: int
    semantic_action: str = "Unclassified semantic action"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticActionGroup":
        return cls(
            start_idx=int(data.get("start_idx", 0)),
            end_idx=int(data.get("end_idx", 0)),
            semantic_action=(
                data.get("semantic_action")
                or data.get("objective")
                or data.get("goal")
                or "Unclassified semantic action"
            ).strip(),
        )


@dataclass
class SemanticActionIR:
    actions: list[ActionTraceEntry]
    semantic_action: str = "Unclassified semantic action"
    action_details: str = ""
    start_action_idx: int = -1
    end_action_idx: int = -1

    def event_count(self) -> int:
        return len(self.actions)

    def action_metadata(self) -> list[SemanticActionSourceAction]:
        return [
            SemanticActionSourceAction(
                action_idx=self.start_action_idx + offset,
                original_index=action.original_index,
                action_id=action.action_id,
                action=action.action,
                status=action.status,
                goal=action.goal,
                active_application=action.active_application,
                grounded_visual_content=action.grounded_visual_content,
                visual_content=action.visual_content,
                ocr_results=action.ocr_results,
                md_results=action.md_results,
                state_before=action.state_before,
                state_after=action.state_after,
                time_before=action.time_before,
                time_after=action.time_after,
                time_range=action.time_range,
            )
            for offset, action in enumerate(self.actions)
        ]

    def to_model(self, index: int) -> AtomSemanticAction:
        start_id = self.actions[0].action_id if self.actions else None
        end_id = self.actions[-1].action_id if self.actions else None
        source_actions = self.action_metadata()
        return AtomSemanticAction(
            semantic_action_id=f"semantic_action_{index:04d}",
            start_action_idx=int(self.start_action_idx),
            end_action_idx=int(self.end_action_idx),
            start_action_id=start_id,
            end_action_id=end_id,
            semantic_action=(self.semantic_action or "Unclassified semantic action").strip(),
            action_details=(self.action_details or "").strip(),
            actions=source_actions,
            raw_action_ids=_unique_nonempty(action.action_id for action in source_actions),
            apps_used=_unique_nonempty(action.active_application for action in source_actions),
            entities=_unique_nonempty(
                action.grounded_visual_content or action.visual_content for action in source_actions
            ),
            pre_state=next((action.state_before for action in source_actions if action.state_before), ""),
            post_state=next(
                (action.state_after for action in reversed(source_actions) if action.state_after),
                "",
            ),
            ocr_texts=_unique_nonempty(action.md_results for action in source_actions),
            event_count=self.event_count(),
        )


@dataclass
class SemanticActionMergeAction:
    start_idx: int
    end_idx: int
    merged_semantic_action: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticActionMergeAction":
        return cls(
            start_idx=int(data.get("start_idx", 0)),
            end_idx=int(data.get("end_idx", 0)),
            merged_semantic_action=(
                data.get("merged_semantic_action")
                or data.get("semantic_action")
                or data.get("merged_action")
                or ""
            ).strip(),
        )


@dataclass(frozen=True)
class BackwardSegmentationWindow:
    window_start: int
    window_end: int
    core_start: int
    core_end: int


def induction_meta_path(output_path: Path) -> Path:
    if output_path.name.endswith(".jsonl"):
        return output_path.with_name(f"{output_path.name[:-6]}.meta.json")
    return output_path.with_suffix(".meta.json")


def visual_enrichment_cache_path(output_path: Path) -> Path:
    if output_path.name.endswith(".jsonl"):
        return output_path.with_name(f"{output_path.name[:-6]}.visual_enrichment_cache.jsonl")
    return output_path.with_name(DEFAULT_VISUAL_ENRICHMENT_CACHE_FILE_NAME)


def resolve_stage_path(data_dir: Path, file_name: str | Path) -> Path:
    candidate = Path(file_name)
    return candidate if candidate.is_absolute() else data_dir / candidate


def resolve_action_input_path(
    data_dir: Path,
    input_file_name: str | Path = DEFAULT_GROUNDED_INPUT_FILE_NAME,
) -> Path:
    input_path = resolve_stage_path(data_dir, input_file_name)
    if input_path.exists():
        return input_path
    fallback = data_dir / DEFAULT_RAW_INPUT_FILE_NAME
    if fallback.exists():
        return fallback
    return input_path


def build_backward_segmentation_windows(
    total: int,
    *,
    batch_size: int,
    overlap: int,
) -> list[BackwardSegmentationWindow]:
    if total <= 0:
        return []
    batch_size = max(1, batch_size)
    overlap = max(0, min(overlap, batch_size - 1))
    if total <= batch_size:
        return [BackwardSegmentationWindow(0, total, 0, total)]

    step = max(1, batch_size - overlap)
    left_trim = overlap // 2
    right_trim = overlap - left_trim
    windows: list[BackwardSegmentationWindow] = []
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        core_start = start if start == 0 else min(end, start + left_trim)
        core_end = end if end >= total else max(core_start, end - right_trim)
        if core_start < core_end:
            windows.append(
                BackwardSegmentationWindow(
                    window_start=start,
                    window_end=end,
                    core_start=core_start,
                    core_end=core_end,
                )
            )
        if end >= total:
            break
        start += step
    return windows


def write_semantic_action_output(output_path: Path, output: SemanticActionInductionOutput) -> None:
    write_jsonl_atomic(
        output_path,
        [action.model_dump(mode="json") for action in output.semantic_actions],
    )
    write_json_atomic(induction_meta_path(output_path), output.meta.model_dump(mode="json"))


def reporter_cost_meta(reporter: SemanticActionReporter) -> dict[str, Any]:
    counters = reporter.state.counters
    raw_breakdown = reporter.state.metrics.get("cost_breakdown")
    by_operation = []
    if isinstance(raw_breakdown, dict) and isinstance(raw_breakdown.get("by_operation"), dict):
        by_operation = list(raw_breakdown["by_operation"].values())
    return {
        "elapsed_secs": round(time.monotonic() - reporter.started_at, 2),
        "llm_requests": counters.get("llm_requests", 0),
        "input_tokens": counters.get("input_tokens", 0),
        "output_tokens": counters.get("output_tokens", 0),
        "total_tokens": counters.get("total_tokens", 0),
        "estimated_usd": float(reporter.state.metrics.get("estimated_usd", 0.0) or 0.0),
        "cost_breakdown": {
            "total": {
                "llm_requests": counters.get("llm_requests", 0),
                "input_tokens": counters.get("input_tokens", 0),
                "output_tokens": counters.get("output_tokens", 0),
                "total_tokens": counters.get("total_tokens", 0),
                "estimated_usd": float(reporter.state.metrics.get("estimated_usd", 0.0) or 0.0),
            },
            "by_operation": by_operation,
        },
    }


def read_semantic_action_output(output_path: Path) -> SemanticActionInductionOutput:
    semantic_actions = [
        AtomSemanticAction.model_validate(row)
        for row in read_jsonl_objects(output_path)
    ]
    meta_path = induction_meta_path(output_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"semantic-action metadata sidecar not found: {meta_path}")
    meta = SemanticActionInductionMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
    return SemanticActionInductionOutput(meta=meta, semantic_actions=semantic_actions)


def rehydrate_semantic_action_evidence(
    cached: SemanticActionInductionOutput,
    actions: list[ActionTraceEntry],
    *,
    input_fingerprint: str | None = None,
) -> SemanticActionInductionOutput | None:
    """Join a cached segmentation to the current grounded rows.

    Semantic labels are reusable only when their ranges still form an exact,
    ordered partition of the current trajectory and any persisted boundary IDs
    agree.  Successful joins rewrite the source evidence, which upgrades old
    caches without another model call.  ``None`` marks an incompatible cache.
    """

    current_fingerprint = input_fingerprint or action_trace_fingerprint(actions)
    if (
        not actions
        or cached.meta.num_actions != len(actions)
        or not cached.semantic_actions
        or cached.meta.input_fingerprint != current_fingerprint
    ):
        return None

    expected_start = 0
    seen_ids: set[str] = set()
    refreshed: list[AtomSemanticAction] = []
    for index, semantic_action in enumerate(cached.semantic_actions):
        start = semantic_action.start_action_idx
        end = semantic_action.end_action_idx
        if (
            start != expected_start
            or end < start
            or end >= len(actions)
            or semantic_action.semantic_action_id in seen_ids
        ):
            return None

        current_start_id = actions[start].action_id
        current_end_id = actions[end].action_id
        if semantic_action.start_action_id and current_start_id and semantic_action.start_action_id != current_start_id:
            return None
        if semantic_action.end_action_id and current_end_id and semantic_action.end_action_id != current_end_id:
            return None

        rebuilt = SemanticActionIR(
            actions=actions[start : end + 1],
            semantic_action=semantic_action.semantic_action,
            action_details=semantic_action.action_details,
            start_action_idx=start,
            end_action_idx=end,
        ).to_model(index)
        if len(semantic_action.actions) == len(rebuilt.actions):
            for offset, current_source in enumerate(rebuilt.actions):
                cached_source = semantic_action.actions[offset]
                if (
                    cached_source.action_id == current_source.action_id
                    and cached_source.visual_content
                ):
                    rebuilt.actions[offset] = current_source.model_copy(
                        update={"visual_content": cached_source.visual_content}
                    )
        refreshed.append(
            rebuilt.model_copy(update={"semantic_action_id": semantic_action.semantic_action_id})
        )
        seen_ids.add(semantic_action.semantic_action_id)
        expected_start = end + 1

    if expected_start != len(actions):
        return None

    cached.semantic_actions = refreshed
    cached.meta.num_actions = len(actions)
    cached.meta.num_semantic_actions = len(refreshed)
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
    reporter: SemanticActionReporter | None,
    operation: str,
) -> dict[str, Any]:
    response = litellm_completion(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.0 if "gpt-5" not in model_name and "kimi" not in model_name else 1.0,
        timeout=120.0,
        request_timeout=120.0,
        response_format={"type": "json_object"},
    )
    if reporter is not None:
        with _REPORTER_LOCK:
            reporter.increment("llm_requests")
            for key, value in _normalize_usage(response).items():
                reporter.increment(key, value)
            call_usd = _estimated_completion_cost_usd(response, model_name)
            if call_usd is not None:
                current_usd = float(reporter.state.metrics.get("estimated_usd", 0.0) or 0.0)
                reporter.set_metric("estimated_usd", round(current_usd + call_usd, 6))
            _record_reporter_cost(
                reporter,
                operation=operation,
                model=model_name,
                usage=_normalize_usage(response),
                estimated_usd=call_usd,
            )
    text = completion_message_content(response)
    return extract_json_from_response(text)


def _record_reporter_cost(
    reporter: SemanticActionReporter,
    *,
    operation: str,
    model: str,
    usage: dict[str, int],
    estimated_usd: float | None,
) -> None:
    raw = reporter.state.metrics.get("cost_breakdown")
    breakdown = raw if isinstance(raw, dict) else {"by_operation": {}}
    by_operation = breakdown.setdefault("by_operation", {})
    bucket = by_operation.setdefault(
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
    reporter.state.metrics["cost_breakdown"] = breakdown


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


def _normalize_usage(response: Any) -> dict[str, int]:
    return normalize_litellm_usage(response)


def _estimated_completion_cost_usd(response: Any, model_name: str) -> float | None:
    return estimated_litellm_completion_cost_usd(response, model_name)


def _markdown_results(row: dict[str, Any]) -> str | None:
    direct = string_field(row, "md_results")
    if direct:
        return direct
    ocr_results = row.get("ocr_results")
    if isinstance(ocr_results, dict):
        value = ocr_results.get("md_results")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def action_trace_fingerprint(actions: list[ActionTraceEntry]) -> str:
    """Fingerprint every grounded field that can affect semantic induction."""

    return _stable_hash(
        {
            "version": 1,
            "actions": [
                {
                    "action_id": action.action_id,
                    "original_index": action.original_index,
                    "action": action.action,
                    "status": action.status,
                    "goal": action.goal,
                    "active_application": action.active_application,
                    "grounded_visual_content": action.grounded_visual_content or action.visual_content,
                    "ocr_results": action.ocr_results,
                    "md_results": action.md_results,
                    "state_before": action.state_before,
                    "state_after": action.state_after,
                    "time_before": action.time_before,
                    "time_after": action.time_after,
                    "time_range": action.time_range,
                }
                for action in actions
            ],
        }
    )


def visual_enrichment_cache_key(action: ActionTraceEntry, *, model_name: str) -> str:
    return _stable_hash(
        {
            "version": 1,
            "model": model_name,
            "action_id": action.action_id or "",
            "action": action.action or "",
            "goal": action.goal or "",
            "active_application": action.active_application or "",
            "visual_content": action.visual_content or "",
            "md_results_sha256": hashlib.sha256((action.md_results or "").encode("utf-8")).hexdigest(),
        }
    )


def load_visual_enrichment_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    cache: dict[str, str] = {}
    for row in read_jsonl_objects(cache_path):
        key = row.get("cache_key")
        value = row.get("enriched_visual_content")
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            cache[key] = value.strip()
    return cache


def append_visual_enrichment_cache_row(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_action_trace(input_path: Path) -> list[ActionTraceEntry]:
    rows = [row for row in read_jsonl_objects(input_path) if is_action_row(row)]
    actions: list[ActionTraceEntry] = []
    for idx, row in enumerate(rows):
        action_id = row_id(row, idx)
        original_index = row.get("original_index")
        if not isinstance(original_index, int) or isinstance(original_index, bool) or original_index < 0:
            original_index = idx
        ocr_results = row.get("ocr_results")
        actions.append(
            ActionTraceEntry(
                action_id=action_id,
                action=safe_action_text(row, idx),
                original_index=original_index,
                status=string_field(row, "status"),
                goal=string_field(row, "goal"),
                active_application=string_field(row, "active_application"),
                grounded_visual_content=string_field(row, "visual_content"),
                visual_content=string_field(row, "visual_content"),
                ocr_results=ocr_results if isinstance(ocr_results, dict) else None,
                md_results=_markdown_results(row),
                state_before=string_field(row, "state_before"),
                state_after=string_field(row, "state_after"),
                time_before=row.get("time_before"),
                time_after=row.get("time_after"),
                time_range=_optional_float(row.get("time_range")),
            )
        )
    return actions


class SemanticActionBuilder:
    """Backward-looking UI adapter that produces atom semantic actions."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        enrichment_model_name: str | None = None,
        enrichment_workers: int = DEFAULT_ENRICHMENT_WORKERS,
        backward_batch_size: int = DEFAULT_BACKWARD_BATCH_SIZE,
        backward_batch_overlap: int = DEFAULT_BACKWARD_BATCH_OVERLAP,
        backward_workers: int = DEFAULT_BACKWARD_WORKERS,
        max_future_semantic_actions: int = 8,
        merge_batch_size: int = 16,
        merge_batch_overlap: int = 2,
        max_prior_semantic_actions: int = 8,
        reporter: SemanticActionReporter | None = None,
    ) -> None:
        self.model_name = model_name
        self.enrichment_model_name = enrichment_model_name or DEFAULT_ENRICHMENT_MODEL
        self.enrichment_workers = max(1, enrichment_workers)
        self.backward_batch_size = max(1, backward_batch_size)
        self.backward_batch_overlap = max(0, min(backward_batch_overlap, self.backward_batch_size - 1))
        self.backward_workers = max(1, backward_workers)
        self.max_future_semantic_actions = max(1, max_future_semantic_actions)
        self.merge_batch_size = max(2, merge_batch_size)
        self.merge_batch_overlap = max(0, min(merge_batch_overlap, self.merge_batch_size - 1))
        self.max_prior_semantic_actions = max(1, max_prior_semantic_actions)
        self.reporter = reporter

    def process(
        self,
        actions: list[ActionTraceEntry],
        *,
        input_path: Path,
        output_path: Path,
        reuse_cache: bool = False,
        save_cache: bool = True,
    ) -> SemanticActionInductionOutput:
        input_fingerprint = action_trace_fingerprint(actions)
        if reuse_cache and output_path.exists():
            try:
                cached = rehydrate_semantic_action_evidence(
                    read_semantic_action_output(output_path),
                    actions,
                    input_fingerprint=input_fingerprint,
                )
            except Exception:
                cached = None
            if cached is not None:
                if save_cache:
                    write_semantic_action_output(output_path, cached)
                if self.reporter is not None:
                    self.reporter.set_counter("semantic_actions", len(cached.semantic_actions))
                return cached

        enriched_actions = self.enrich_action_visual_content(
            actions,
            cache_path=visual_enrichment_cache_path(output_path),
        )
        backward_segments = self._backward_segment(enriched_actions)
        segments = self._merge_segments_forward(backward_segments)
        self.enrich_semantic_action_details(segments)
        semantic_actions = [segment.to_model(index) for index, segment in enumerate(segments)]
        if self.reporter is not None:
            self.reporter.set_counter("backward_semantic_actions", len(backward_segments))
            self.reporter.set_counter("semantic_actions", len(semantic_actions))

        output = SemanticActionInductionOutput(
            meta=SemanticActionInductionMeta(
                created_at=utc_now_iso(),
                model=self.model_name,
                enrichment_model=self.enrichment_model_name,
                input_path=str(input_path),
                input_fingerprint=input_fingerprint,
                num_actions=len(actions),
                limits=None,
                num_semantic_actions=len(semantic_actions),
                num_backward_semantic_actions=len(backward_segments),
                visual_enrichment_workers=self.enrichment_workers,
                action_detail_workers=self.enrichment_workers,
                backward_batch_size=self.backward_batch_size,
                backward_batch_overlap=self.backward_batch_overlap,
                backward_workers=self.backward_workers,
                max_future_semantic_actions=self.max_future_semantic_actions,
                merge_batch_size=self.merge_batch_size,
                merge_batch_overlap=self.merge_batch_overlap,
                max_prior_semantic_actions=self.max_prior_semantic_actions,
                **(reporter_cost_meta(self.reporter) if self.reporter is not None else {}),
            ),
            semantic_actions=semantic_actions,
        )
        if save_cache:
            write_semantic_action_output(output_path, output)
            if output.meta.cost_breakdown:
                print(
                    "[cost] semantic_action_induction "
                    + json.dumps(output.meta.cost_breakdown, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
        return output

    def enrich_action_visual_content(
        self,
        actions: list[ActionTraceEntry],
        *,
        cache_path: Path | None = None,
    ) -> list[ActionTraceEntry]:
        if not actions:
            return actions
        work_items = [
            (idx, action)
            for idx, action in enumerate(actions)
            if (action.md_results or "").strip()
        ]
        if self.reporter is not None:
            self.reporter.set_counter("expected_visual_enrichments", len(work_items))
        if not work_items:
            return actions

        cached_enrichments = load_visual_enrichment_cache(cache_path) if cache_path is not None else {}
        missing_work_items: list[tuple[int, ActionTraceEntry, str]] = []
        for idx, action in work_items:
            cache_key = visual_enrichment_cache_key(action, model_name=self.enrichment_model_name)
            cached = cached_enrichments.get(cache_key)
            if cached:
                action.visual_content = cached
                if self.reporter is not None:
                    with _REPORTER_LOCK:
                        self.reporter.increment("visual_enrichments")
                        self.reporter.increment("visual_enrichment_cache_hits")
                continue
            missing_work_items.append((idx, action, cache_key))
        if not missing_work_items:
            if self.reporter is not None:
                with _REPORTER_LOCK:
                    self.reporter.progress(f"visual_enrichment_cache_hits={len(work_items)}/{len(work_items)}")
            return actions

        cache_write_lock = threading.Lock()
        max_workers = min(self.enrichment_workers, len(missing_work_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._enrich_one_action_visual_content, action): (idx, action, cache_key)
                for idx, action, cache_key in missing_work_items
            }
            for future in as_completed(futures):
                idx, action, cache_key = futures[future]
                try:
                    enriched = future.result()
                except Exception as exc:
                    if self.reporter is not None:
                        with _REPORTER_LOCK:
                            self.reporter.increment("visual_enrichment_errors")
                            self.reporter.progress(f"visual enrichment fallback: {exc}")
                    continue
                if enriched:
                    original_visual_content = action.visual_content
                    actions[idx].visual_content = enriched
                    if cache_path is not None:
                        cache_row = {
                            "cache_key": cache_key,
                            "created_at": utc_now_iso(),
                            "enrichment_model": self.enrichment_model_name,
                            "action_idx": idx,
                            "action_id": action.action_id,
                            "goal": action.goal,
                            "active_application": action.active_application,
                            "original_visual_content": original_visual_content,
                            "enriched_visual_content": enriched,
                            "md_results_sha256": hashlib.sha256((action.md_results or "").encode("utf-8")).hexdigest(),
                        }
                        with cache_write_lock:
                            append_visual_enrichment_cache_row(cache_path, cache_row)
                if self.reporter is not None:
                    with _REPORTER_LOCK:
                        self.reporter.increment("visual_enrichments")
                        self.reporter.progress(f"visual_enrichment={self.reporter.state.counters.get('visual_enrichments', 0)}/{len(work_items)}")
        return actions

    def _enrich_one_action_visual_content(self, action: ActionTraceEntry) -> str:
        payload = {
            "objective": action.goal or "",
            "active_application": action.active_application or "",
            "visual_content": action.visual_content or "",
            "md_results": action.md_results or "",
        }
        data = call_llm_json(
            prompt=VISUAL_CONTENT_ENRICHMENT_PROMPT,
            content=json.dumps(payload, ensure_ascii=False),
            model_name=self.enrichment_model_name,
            reporter=self.reporter,
            operation="visual_enrichment",
        )
        enriched = data.get("visual_content")
        if isinstance(enriched, str) and enriched.strip():
            return enriched.strip()
        return action.visual_content or ""

    def enrich_semantic_action_details(self, segments: list[SemanticActionIR]) -> list[SemanticActionIR]:
        if not segments:
            return segments
        if self.reporter is not None:
            self.reporter.set_counter("expected_action_details", len(segments))
        max_workers = min(self.enrichment_workers, len(segments))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._detail_one_semantic_action, segment): idx
                for idx, segment in enumerate(segments)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    details = future.result()
                except Exception as exc:
                    if self.reporter is not None:
                        with _REPORTER_LOCK:
                            self.reporter.increment("action_detail_errors")
                            self.reporter.progress(f"action detail fallback: {exc}")
                    details = ""
                segments[idx].action_details = details
                if self.reporter is not None:
                    with _REPORTER_LOCK:
                        self.reporter.increment("action_details")
                        self.reporter.progress(f"action_details={self.reporter.state.counters.get('action_details', 0)}/{len(segments)}")
        return segments

    def _detail_one_semantic_action(self, segment: SemanticActionIR) -> str:
        visual_contents = [
            action.visual_content.strip()
            for action in segment.actions
            if action.visual_content and action.visual_content.strip()
        ]
        payload = {
            "semantic_action": segment.semantic_action,
            "action_range": [segment.start_action_idx, segment.end_action_idx],
            "event_count": segment.event_count(),
            "visual_contents": _uniq_truncated(visual_contents, max_items=12, max_chars=320),
        }
        data = call_llm_json(
            prompt=SEMANTIC_ACTION_DETAILS_PROMPT,
            content=json.dumps(payload, ensure_ascii=False),
            model_name=self.enrichment_model_name,
            reporter=self.reporter,
            operation="action_detail",
        )
        details = data.get("action_details")
        return details.strip() if isinstance(details, str) else ""

    def _backward_segment(self, actions: list[ActionTraceEntry]) -> list[SemanticActionIR]:
        if not actions:
            return []
        if self.backward_workers <= 1:
            return self._backward_segment_sequential(actions)

        windows = build_backward_segmentation_windows(
            len(actions),
            batch_size=self.backward_batch_size,
            overlap=self.backward_batch_overlap,
        )
        if self.reporter is not None:
            self.reporter.set_counter("expected_backward_batches", len(windows))
        if not windows:
            return []

        future_context = (
            "Parallel local segmentation mode. This window may include overlap with neighboring "
            "windows; use the local action evidence and ignore duplicated edge context."
        )
        max_workers = min(self.backward_workers, len(windows))
        results: dict[int, list[SemanticActionIR]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._segment_backward_window, actions, window, future_context): window
                for window in windows
            }
            for future in as_completed(futures):
                window = futures[future]
                try:
                    segments = future.result()
                except Exception as exc:
                    if self.reporter is not None:
                        with _REPORTER_LOCK:
                            self.reporter.increment("semantic_action_errors")
                            self.reporter.progress(f"semantic action fallback: {exc}")
                    segments = [
                        SemanticActionIR(
                            actions=actions[window.core_start : window.core_end],
                            start_action_idx=window.core_start,
                            end_action_idx=window.core_end - 1,
                        )
                    ]
                results[window.core_start] = segments
                if self.reporter is not None:
                    with _REPORTER_LOCK:
                        self.reporter.increment("backward_batches")
                        completed = self.reporter.state.counters.get("backward_batches", 0)
                        self.reporter.progress(
                            f"batch={completed}/{len(windows)} "
                            f"actions={window.window_start}-{window.window_end - 1} "
                            f"core={window.core_start}-{window.core_end - 1}"
                        )

        stitched: list[SemanticActionIR] = []
        for window in sorted(windows, key=lambda item: item.core_start):
            stitched.extend(results.get(window.core_start, []))
        return stitched

    def _backward_segment_sequential(self, actions: list[ActionTraceEntry]) -> list[SemanticActionIR]:
        if self.reporter is not None:
            expected_batches = (len(actions) + self.backward_batch_size - 1) // self.backward_batch_size
            self.reporter.set_counter("expected_backward_batches", expected_batches)

        all_segments: list[SemanticActionIR] = []
        future_context = "End of session. No subsequent actions."
        end = len(actions)
        batch_num = 0
        while end > 0:
            start = max(0, end - self.backward_batch_size)
            batch = actions[start:end]
            batch_num += 1
            if self.reporter is not None:
                self.reporter.increment("backward_batches")
                self.reporter.progress(f"batch={batch_num} actions={start}-{end - 1}")

            batch_groups = self._segment_batch(batch, future_context)
            new_segments: list[SemanticActionIR] = []
            for group in batch_groups:
                segment_actions = batch[group.start_idx : group.end_idx + 1]
                new_segments.append(
                    SemanticActionIR(
                        actions=segment_actions,
                        semantic_action=group.semantic_action,
                        start_action_idx=start + group.start_idx,
                        end_action_idx=start + group.end_idx,
                    )
                )

            all_segments = new_segments + all_segments
            future_context = self._summarize_future_semantic_actions(all_segments)
            end = start

        return all_segments

    def _segment_backward_window(
        self,
        actions: list[ActionTraceEntry],
        window: BackwardSegmentationWindow,
        future_context: str,
    ) -> list[SemanticActionIR]:
        batch = actions[window.window_start : window.window_end]
        batch_groups = self._segment_batch(batch, future_context)
        segments: list[SemanticActionIR] = []
        for group in batch_groups:
            global_start = window.window_start + group.start_idx
            global_end = window.window_start + group.end_idx
            clipped_start = max(global_start, window.core_start)
            clipped_end = min(global_end, window.core_end - 1)
            if clipped_start > clipped_end:
                continue
            segments.append(
                SemanticActionIR(
                    actions=actions[clipped_start : clipped_end + 1],
                    semantic_action=group.semantic_action,
                    start_action_idx=clipped_start,
                    end_action_idx=clipped_end,
                )
            )
        return segments

    def _segment_batch(
        self,
        batch: list[ActionTraceEntry],
        future_context: str,
    ) -> list[SemanticActionGroup]:
        batch_size = len(batch)
        actions_desc = self._describe_action_batch(batch)
        prompt = BACKWARD_SEMANTIC_ACTION_SEGMENT_PROMPT.format(
            semantic_action_definition=ATOM_SEMANTIC_ACTION_DEFINITION,
            future_context=future_context,
            actions_list=actions_desc,
            max_idx=batch_size - 1,
        )
        try:
            data = call_llm_json(
                prompt=prompt,
                content=[],
                model_name=self.model_name,
                reporter=self.reporter,
                operation="backward_segmentation",
            )
            groups = data.get("groups", [])
            errors = self._check_groups(groups, batch_size)
            if not errors:
                return self._normalize_groups(groups)

            retry_prompt = BACKWARD_SEMANTIC_ACTION_RETRY_PROMPT.format(
                semantic_action_definition=ATOM_SEMANTIC_ACTION_DEFINITION,
                actions_list=actions_desc,
                future_context=future_context,
                previous_output=json.dumps({"groups": groups}, indent=2, ensure_ascii=False),
                errors="\n".join(f"- {error}" for error in errors),
                max_idx=batch_size - 1,
            )
            retry_data = call_llm_json(
                prompt=retry_prompt,
                content=[],
                model_name=self.model_name,
                reporter=self.reporter,
                operation="backward_segmentation_retry",
            )
            retry_groups = retry_data.get("groups", [])
            retry_errors = self._check_groups(retry_groups, batch_size)
            if not retry_errors:
                return self._normalize_groups(retry_groups)
            return self._patch_groups(retry_groups, batch_size)
        except Exception as exc:
            if self.reporter is not None:
                with _REPORTER_LOCK:
                    self.reporter.increment("semantic_action_errors")
                    self.reporter.progress(f"semantic action fallback: {exc}")
            return [SemanticActionGroup(start_idx=0, end_idx=batch_size - 1)]

    def _check_groups(self, groups: Any, batch_size: int) -> list[str]:
        errors: list[str] = []
        if not isinstance(groups, list) or not groups:
            return ["No groups returned."]
        covered: set[int] = set()
        for group in groups:
            if not isinstance(group, dict):
                errors.append(f"Invalid group payload: {group}")
                continue
            try:
                start_idx = int(group.get("start_idx", -1))
                end_idx = int(group.get("end_idx", -1))
            except (TypeError, ValueError):
                errors.append(f"Non-integer index in group: {group}")
                continue
            if start_idx < 0 or end_idx < 0 or start_idx >= batch_size or end_idx >= batch_size:
                errors.append(f"Index out of range: start_idx={start_idx}, end_idx={end_idx}")
                continue
            if start_idx > end_idx:
                errors.append(f"start_idx ({start_idx}) > end_idx ({end_idx})")
                continue
            indices = set(range(start_idx, end_idx + 1))
            overlap = covered & indices
            if overlap:
                errors.append(f"Overlapping indices: {sorted(overlap)}")
            covered.update(indices)
        missing = set(range(batch_size)) - covered
        if missing:
            errors.append(f"Missing indices: {sorted(missing)}")
        return errors

    def _normalize_groups(self, groups: list[dict[str, Any]]) -> list[SemanticActionGroup]:
        normalized: list[SemanticActionGroup] = []
        for group in groups:
            try:
                normalized.append(SemanticActionGroup.from_dict(group))
            except (TypeError, ValueError):
                continue
        normalized.sort(key=lambda item: item.start_idx)
        return normalized

    def _patch_groups(self, groups: Any, batch_size: int) -> list[SemanticActionGroup]:
        if not isinstance(groups, list):
            return [SemanticActionGroup(start_idx=0, end_idx=batch_size - 1)]
        normalized = self._normalize_groups([group for group in groups if isinstance(group, dict)])
        valid: list[SemanticActionGroup] = []
        covered: set[int] = set()
        for group in normalized:
            start_idx = max(0, min(batch_size - 1, group.start_idx))
            end_idx = max(start_idx, min(batch_size - 1, group.end_idx))
            indices = set(range(start_idx, end_idx + 1))
            if covered & indices:
                continue
            group.start_idx = start_idx
            group.end_idx = end_idx
            valid.append(group)
            covered.update(indices)
        for missing_idx in sorted(set(range(batch_size)) - covered):
            valid.append(SemanticActionGroup(start_idx=missing_idx, end_idx=missing_idx))
        valid.sort(key=lambda item: item.start_idx)
        return valid or [SemanticActionGroup(start_idx=0, end_idx=batch_size - 1)]

    def _merge_segments_forward(self, segments: list[SemanticActionIR]) -> list[SemanticActionIR]:
        if len(segments) <= 1:
            if self.reporter is not None:
                self.reporter.set_counter("expected_merge_batches", 0)
            return segments

        total = len(segments)
        batch_step = max(1, self.merge_batch_size - self.merge_batch_overlap)
        if self.reporter is not None:
            self.reporter.set_counter("expected_merge_batches", self._expected_merge_batches(total, batch_step))
        batch_start = 0
        merged_boundaries: set[int] = set()
        recorded_actions: list[SemanticActionMergeAction] = []
        batch_num = 0

        while batch_start < total:
            batch_end = min(total, batch_start + self.merge_batch_size)
            batch = segments[batch_start:batch_end]
            batch_num += 1
            if self.reporter is not None:
                self.reporter.increment("merge_batches")
                self.reporter.progress(f"merge_batch={batch_num} candidates={batch_start}-{batch_end - 1}")

            merge_actions = self._merge_batch(
                batch_desc=self._describe_segment_batch_for_merge(batch),
                prior_context=self._summarize_prior_semantic_actions(segments[:batch_start]),
                batch_size=len(batch),
            )
            for action in merge_actions:
                global_start = batch_start + action.start_idx
                global_end = batch_start + action.end_idx
                if global_start < 0 or global_end >= total or global_start >= global_end:
                    continue
                recorded_actions.append(
                    SemanticActionMergeAction(
                        start_idx=global_start,
                        end_idx=global_end,
                        merged_semantic_action=action.merged_semantic_action,
                    )
                )
                for boundary_idx in range(global_start, global_end):
                    merged_boundaries.add(boundary_idx)
                if self.reporter is not None:
                    self.reporter.increment("merge_actions")

            if batch_end >= total:
                break
            batch_start += batch_step

        finalized: list[SemanticActionIR] = []
        cursor = 0
        while cursor < total:
            group_start = cursor
            while cursor < total - 1 and cursor in merged_boundaries:
                cursor += 1
            group_end = cursor
            if group_start == group_end:
                finalized.append(segments[group_start])
            else:
                semantic_action = self._pick_group_label(group_start, group_end, recorded_actions)
                finalized.append(
                    self._collapse_segments(
                        segments[group_start : group_end + 1],
                        semantic_action=semantic_action,
                    )
                )
            cursor += 1
        return finalized

    def _expected_merge_batches(self, total: int, batch_step: int) -> int:
        if total <= 1:
            return 0
        count = 0
        batch_start = 0
        while batch_start < total:
            batch_end = min(total, batch_start + self.merge_batch_size)
            count += 1
            if batch_end >= total:
                break
            batch_start += batch_step
        return count

    def _merge_batch(
        self,
        *,
        batch_desc: str,
        prior_context: str,
        batch_size: int,
    ) -> list[SemanticActionMergeAction]:
        prompt = FORWARD_SEMANTIC_ACTION_MERGE_PROMPT.format(
            semantic_action_definition=ATOM_SEMANTIC_ACTION_DEFINITION,
            prior_context=prior_context,
            segments_list=batch_desc,
        )
        try:
            data = call_llm_json(
                prompt=prompt,
                content=[],
                model_name=self.model_name,
                reporter=self.reporter,
                operation="forward_merge",
            )
            merge_actions = self._parse_merge_actions(data)
            errors = self._check_merge_actions(merge_actions, batch_size)
            if not errors:
                return merge_actions
            retry_prompt = FORWARD_SEMANTIC_ACTION_MERGE_RETRY_PROMPT.format(
                semantic_action_definition=ATOM_SEMANTIC_ACTION_DEFINITION,
                segments_list=batch_desc,
                prior_context=prior_context,
                previous_output=json.dumps(data, indent=2, ensure_ascii=False),
                errors="\n".join(f"- {error}" for error in errors),
            )
            retry_data = call_llm_json(
                prompt=retry_prompt,
                content=[],
                model_name=self.model_name,
                reporter=self.reporter,
                operation="forward_merge_retry",
            )
            retry_actions = self._parse_merge_actions(retry_data)
            retry_errors = self._check_merge_actions(retry_actions, batch_size)
            if not retry_errors:
                return retry_actions
            return self._patch_merge_actions(retry_actions, batch_size)
        except Exception as exc:
            if self.reporter is not None:
                self.reporter.increment("merge_errors")
                self.reporter.progress(f"forward merge fallback: {exc}")
            return []

    def _parse_merge_actions(self, data: dict[str, Any]) -> list[SemanticActionMergeAction]:
        raw = data.get("merge_actions")
        if not isinstance(raw, list):
            return []
        actions: list[SemanticActionMergeAction] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                actions.append(SemanticActionMergeAction.from_dict(item))
            except (TypeError, ValueError):
                continue
        actions.sort(key=lambda action: action.start_idx)
        return actions

    def _check_merge_actions(self, actions: list[SemanticActionMergeAction], batch_size: int) -> list[str]:
        errors: list[str] = []
        covered: set[int] = set()
        for action in actions:
            if action.start_idx < 0 or action.end_idx < 0 or action.start_idx >= batch_size or action.end_idx >= batch_size:
                errors.append(f"Index out of range: start_idx={action.start_idx}, end_idx={action.end_idx}")
                continue
            if action.start_idx >= action.end_idx:
                errors.append(f"Merge range must span at least two candidates: {action.start_idx}-{action.end_idx}")
                continue
            indices = set(range(action.start_idx, action.end_idx + 1))
            overlap = covered & indices
            if overlap:
                errors.append(f"Overlapping merge ranges: {sorted(overlap)}")
            covered.update(indices)
        return errors

    def _patch_merge_actions(
        self,
        actions: list[SemanticActionMergeAction],
        batch_size: int,
    ) -> list[SemanticActionMergeAction]:
        valid: list[SemanticActionMergeAction] = []
        covered: set[int] = set()
        for action in sorted(actions, key=lambda item: item.start_idx):
            if action.start_idx < 0 or action.end_idx >= batch_size or action.start_idx >= action.end_idx:
                continue
            indices = set(range(action.start_idx, action.end_idx + 1))
            if covered & indices:
                continue
            valid.append(action)
            covered.update(indices)
        return valid

    def _collapse_segments(
        self,
        segments: list[SemanticActionIR],
        *,
        semantic_action: str = "",
    ) -> SemanticActionIR:
        merged_actions: list[ActionTraceEntry] = []
        for segment in segments:
            merged_actions.extend(segment.actions)
        return SemanticActionIR(
            actions=merged_actions,
            semantic_action=(semantic_action or segments[-1].semantic_action or segments[0].semantic_action or "Unclassified semantic action").strip(),
            action_details="",
            start_action_idx=segments[0].start_action_idx,
            end_action_idx=segments[-1].end_action_idx,
        )

    def _pick_group_label(
        self,
        group_start: int,
        group_end: int,
        actions: list[SemanticActionMergeAction],
    ) -> str:
        exact: SemanticActionMergeAction | None = None
        best: SemanticActionMergeAction | None = None
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
        chosen = exact or best
        if chosen is None:
            return ""
        return chosen.merged_semantic_action

    def _describe_action_batch(self, actions: list[ActionTraceEntry]) -> str:
        return "\n".join(
            f"[{idx}] {json.dumps(self._action_payload(action), ensure_ascii=False)}"
            for idx, action in enumerate(actions)
        )

    def _action_payload(self, action: ActionTraceEntry) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action.action}
        if action.action_id:
            payload["id"] = action.action_id
        if action.status and action.status != "success":
            payload["status"] = action.status
        if action.goal:
            payload["grounded_immediate_intent"] = action.goal
        if action.active_application:
            payload["active_application"] = action.active_application
        if action.visual_content:
            payload["important_content_on_screen"] = _truncate_text(action.visual_content, 220)
        return payload

    def _summarize_future_semantic_actions(self, segments: list[SemanticActionIR]) -> str:
        if not segments:
            return "End of session. No subsequent actions."
        parts: list[str] = []
        for segment in segments[: self.max_future_semantic_actions]:
            parts.append(f"- {segment.semantic_action} ({segment.event_count()} events)")
        if len(segments) > self.max_future_semantic_actions:
            parts.append(f"- ... ({len(segments) - self.max_future_semantic_actions} more semantic actions)")
        return "The user then goes on to:\n" + "\n".join(parts)

    def _summarize_prior_semantic_actions(self, segments: list[SemanticActionIR]) -> str:
        if not segments:
            return "Start of session. No prior semantic actions."
        relevant = segments[-self.max_prior_semantic_actions :]
        parts = [
            f"- {_truncate_text(segment.semantic_action, 140)} ({segment.event_count()} events)"
            for segment in relevant
        ]
        return "The user has already done:\n" + "\n".join(parts)

    def _describe_segment_batch_for_merge(self, segments: list[SemanticActionIR]) -> str:
        lines: list[str] = []
        for idx, segment in enumerate(segments):
            representative_actions = [
                self._representative_action_payload(action)
                for action in (segment.actions[:2] + segment.actions[-2:])
            ]
            payload = {
                "action_range": [segment.start_action_idx, segment.end_action_idx],
                "semantic_action": segment.semantic_action,
                "event_count": segment.event_count(),
                "representative_actions": representative_actions,
            }
            lines.append(f"[{idx}] {json.dumps(payload, ensure_ascii=False)}")
        return "\n".join(lines)

    def _representative_action_payload(self, action: ActionTraceEntry) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": _truncate_text(action.action, 140)}
        if action.goal:
            payload["grounded_immediate_intent"] = _truncate_text(action.goal, 160)
        if action.active_application:
            payload["active_application"] = action.active_application
        return payload


def semantic_action_induction(
    *,
    data_dir: str | Path,
    input_file_name: str | Path = DEFAULT_GROUNDED_INPUT_FILE_NAME,
    output_file_name: str | Path = DEFAULT_SEMANTIC_ACTION_OUTPUT_FILE_NAME,
    model: str = DEFAULT_MODEL,
    enrichment_model: str | None = None,
    enrichment_workers: int = DEFAULT_ENRICHMENT_WORKERS,
    backward_batch_size: int = DEFAULT_BACKWARD_BATCH_SIZE,
    backward_batch_overlap: int = DEFAULT_BACKWARD_BATCH_OVERLAP,
    backward_workers: int = DEFAULT_BACKWARD_WORKERS,
    max_future_semantic_actions: int = 8,
    merge_batch_size: int = 16,
    merge_batch_overlap: int = 2,
    max_prior_semantic_actions: int = 8,
    limits: int | None = None,
    reuse_cache: bool = False,
    preflight_only: bool = False,
    no_console: bool = False,
) -> SemanticActionInductionOutput | None:
    data_dir = Path(data_dir)
    enrichment_model_name = enrichment_model or DEFAULT_ENRICHMENT_MODEL
    input_path = resolve_action_input_path(data_dir, input_file_name)
    output_path = resolve_stage_path(data_dir, output_file_name)
    meta_path = induction_meta_path(output_path)
    visual_cache_path = visual_enrichment_cache_path(output_path)

    reporter = SemanticActionReporter(no_console=no_console)
    try:
        with reporter:
            reporter.add_path("input", input_path)
            reporter.add_path("output", output_path)
            reporter.add_path("meta", meta_path)
            reporter.add_path("visual_cache", visual_cache_path)
            reporter.set_metric("model", model)
            reporter.set_metric("enrichment_model", enrichment_model_name)

            reporter.start_stage(STAGE_LOAD_INPUTS, str(input_path))
            if not input_path.exists():
                raise FileNotFoundError(f"input trajectory not found: {input_path}")
            actions = load_action_trace(input_path)
            if limits is not None:
                if limits < 1:
                    raise ValueError(f"limits must be a positive integer when provided: {limits}")
                actions = actions[:limits]
                reporter.set_metric("limits", limits)
            if not actions:
                raise ValueError(f"no action rows found in {input_path}")
            input_fingerprint = action_trace_fingerprint(actions)
            reporter.set_counter("actions", len(actions))
            reporter.finish_stage(STAGE_LOAD_INPUTS, f"{len(actions)} actions loaded")

            reporter.start_stage(STAGE_PREFLIGHT, "validating expected work")
            reporter.set_counter(
                "expected_backward_batches",
                len(
                    build_backward_segmentation_windows(
                        len(actions),
                        batch_size=backward_batch_size,
                        overlap=backward_batch_overlap if backward_workers > 1 else 0,
                    )
                ),
            )
            reporter.set_counter(
                "expected_visual_enrichments",
                sum(1 for action in actions if (action.md_results or "").strip()),
            )
            cache_exists = output_path.exists()
            if cache_exists and reuse_cache:
                reporter.set_metric("cache", "will reuse existing semantic actions")
            elif cache_exists:
                reporter.set_metric("cache", "will overwrite existing semantic actions")
            else:
                reporter.set_metric("cache", "new semantic-action output")
            reporter.finish_stage(STAGE_PREFLIGHT, "preflight complete")
            if preflight_only:
                reporter.mark_stage_done(STAGE_VISUAL_ENRICHMENT, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_SEMANTIC_ACTIONS, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_FORWARD_MERGE, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_ACTION_DETAILS, "skipped by --preflight_only")
                reporter.mark_stage_done(STAGE_WRITE_OUTPUT, "skipped by --preflight_only")
                reporter.final_success("preflight complete; no LLM calls were made")
                return None

            if reuse_cache and cache_exists:
                try:
                    output = rehydrate_semantic_action_evidence(
                        read_semantic_action_output(output_path),
                        actions,
                        input_fingerprint=input_fingerprint,
                    )
                except Exception as exc:
                    output = None
                    reporter.set_metric("cache", f"ignored unreadable cache: {exc}")
                if output is not None:
                    reporter.start_stage(STAGE_WRITE_OUTPUT, "upgrading reusable semantic actions")
                    write_semantic_action_output(output_path, output)
                    reporter.set_counter("semantic_actions", len(output.semantic_actions))
                    reporter.finish_stage(STAGE_WRITE_OUTPUT, str(output_path))
                    reporter.mark_stage_done(STAGE_VISUAL_ENRICHMENT, "loaded from cache")
                    reporter.mark_stage_done(STAGE_SEMANTIC_ACTIONS, "loaded from cache")
                    reporter.mark_stage_done(STAGE_FORWARD_MERGE, "loaded from cache")
                    reporter.mark_stage_done(STAGE_ACTION_DETAILS, "loaded from cache")
                    reporter.final_success("loaded and evidence-refreshed cached semantic actions")
                    return output
                reporter.set_metric("cache", "ignored incompatible semantic-action cache")

            builder = SemanticActionBuilder(
                model_name=model,
                enrichment_model_name=enrichment_model_name,
                enrichment_workers=enrichment_workers,
                backward_batch_size=backward_batch_size,
                backward_batch_overlap=backward_batch_overlap,
                backward_workers=backward_workers,
                max_future_semantic_actions=max_future_semantic_actions,
                merge_batch_size=merge_batch_size,
                merge_batch_overlap=merge_batch_overlap,
                max_prior_semantic_actions=max_prior_semantic_actions,
                reporter=reporter,
            )

            reporter.start_stage(STAGE_VISUAL_ENRICHMENT, f"workers={enrichment_workers}")
            actions = builder.enrich_action_visual_content(actions, cache_path=visual_cache_path)
            reporter.finish_stage(
                STAGE_VISUAL_ENRICHMENT,
                f"{reporter.state.counters.get('visual_enrichments', 0)} actions enriched",
            )

            reporter.start_stage(
                STAGE_SEMANTIC_ACTIONS,
                f"batch_size={backward_batch_size} overlap={backward_batch_overlap} workers={backward_workers}",
            )
            backward_segments = builder._backward_segment(actions)
            reporter.set_counter("backward_semantic_actions", len(backward_segments))
            reporter.finish_stage(STAGE_SEMANTIC_ACTIONS, f"{len(backward_segments)} candidate semantic actions")

            reporter.start_stage(STAGE_FORWARD_MERGE, f"batch_size={merge_batch_size}")
            merged_segments = builder._merge_segments_forward(backward_segments)
            reporter.set_counter("semantic_actions", len(merged_segments))
            reporter.finish_stage(STAGE_FORWARD_MERGE, f"{len(merged_segments)} semantic actions")

            reporter.start_stage(STAGE_ACTION_DETAILS, f"workers={enrichment_workers}")
            builder.enrich_semantic_action_details(merged_segments)
            reporter.finish_stage(
                STAGE_ACTION_DETAILS,
                f"{reporter.state.counters.get('action_details', 0)} semantic actions enriched",
            )
            semantic_actions = [segment.to_model(index) for index, segment in enumerate(merged_segments)]

            reporter.start_stage(STAGE_WRITE_OUTPUT, str(output_path))
            output = SemanticActionInductionOutput(
                meta=SemanticActionInductionMeta(
                    created_at=utc_now_iso(),
                    model=model,
                    enrichment_model=enrichment_model_name,
                    input_path=str(input_path),
                    input_fingerprint=input_fingerprint,
                    num_actions=len(actions),
                    limits=limits,
                    num_semantic_actions=len(semantic_actions),
                    num_backward_semantic_actions=len(backward_segments),
                    visual_enrichment_workers=enrichment_workers,
                    action_detail_workers=enrichment_workers,
                    backward_batch_size=backward_batch_size,
                    backward_batch_overlap=max(
                        0,
                        min(backward_batch_overlap, max(1, backward_batch_size) - 1),
                    ),
                    backward_workers=max(1, backward_workers),
                    max_future_semantic_actions=max_future_semantic_actions,
                    merge_batch_size=merge_batch_size,
                    merge_batch_overlap=merge_batch_overlap,
                    max_prior_semantic_actions=max_prior_semantic_actions,
                    **reporter_cost_meta(reporter),
                ),
                semantic_actions=semantic_actions,
            )
            write_semantic_action_output(output_path, output)
            print(
                "[cost] semantic_action_induction "
                + json.dumps(output.meta.cost_breakdown or {}, ensure_ascii=False),
                file=sys.stderr,
                flush=True,
            )
            reporter.finish_stage(STAGE_WRITE_OUTPUT, str(output_path))
            reporter.final_success("semantic action induction complete")
            return output
    except Exception as exc:
        reporter.fail_active_stage(exc)
        reporter.final_failure()
        setattr(exc, "_semantic_action_reported", True)
        raise


def _truncate_text(text: str, max_chars: int = 180) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _uniq_truncated(values: list[str], *, max_items: int, max_chars: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join((value or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(_truncate_text(normalized, max_chars))
        if len(result) >= max_items:
            break
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build atom semantic actions by backward segmenting low-level UI/action rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing the trajectory JSONL.")
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
    parser.add_argument("--model", type=str, default=None, help="LLM model.")
    parser.add_argument("--enrichment_model", type=str, default=None, help=f"LLM model for visual/action-detail enrichment. Defaults to {DEFAULT_ENRICHMENT_MODEL}.")
    parser.add_argument("--enrichment_workers", type=int, default=None, help="Parallel enrichment workers.")
    parser.add_argument("--backward_batch_size", type=int, default=None, help="Actions per backward segmentation batch.")
    parser.add_argument("--backward_batch_overlap", type=int, default=None, help="Overlapping actions between parallel backward segmentation batches.")
    parser.add_argument("--backward_workers", type=int, default=None, help="Parallel backward segmentation workers. Use 1 for the legacy sequential pass.")
    parser.add_argument(
        "--max_future_semantic_actions",
        "--max_future_segments",
        dest="max_future_semantic_actions",
        type=int,
        default=None,
        help="Future semantic actions to summarize for backward context.",
    )
    parser.add_argument("--merge_batch_size", type=int, default=None, help="Candidate semantic actions per forward merge batch.")
    parser.add_argument("--merge_batch_overlap", type=int, default=None, help="Overlapping candidates between forward merge batches.")
    parser.add_argument(
        "--limits",
        dest="limits",
        type=int,
        default=None,
        help="Only process the first N action rows after loading the input trajectory.",
    )
    parser.add_argument(
        "--max_prior_semantic_actions",
        "--max_prior_segments",
        dest="max_prior_semantic_actions",
        type=int,
        default=None,
        help="Prior semantic actions to summarize for forward merge context.",
    )
    parser.add_argument("--reuse_cache", action="store_true", help="Load the existing output JSONL when present.")
    parser.add_argument("--preflight_only", action="store_true", help="Validate inputs and expected work without LLM calls.")
    parser.add_argument("--no_console", action="store_true", help="Disable the live Rich console and emit plain logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

        stage_config = config.semantic_action_induction
        model = args.model or stage_config.model
        enrichment_model = args.enrichment_model or stage_config.enrichment_model
        with litellm_model_configs(
            [
                (stage_config.model, stage_config.induction_litellm_config),
                (stage_config.enrichment_model, stage_config.enrichment_litellm_config),
            ]
        ):
            semantic_action_induction(
                data_dir=args.data_dir,
                input_file_name=args.input_file_name or stage_config.input_file_name,
                output_file_name=args.output_file_name or stage_config.output_file_name,
                model=model,
                enrichment_model=enrichment_model,
                enrichment_workers=args.enrichment_workers or stage_config.enrichment_workers,
                backward_batch_size=args.backward_batch_size or stage_config.backward_batch_size,
                backward_batch_overlap=(
                    args.backward_batch_overlap
                    if args.backward_batch_overlap is not None
                    else stage_config.backward_batch_overlap
                ),
                backward_workers=args.backward_workers or stage_config.backward_workers,
                max_future_semantic_actions=(
                    args.max_future_semantic_actions or stage_config.max_future_semantic_actions
                ),
                merge_batch_size=args.merge_batch_size or stage_config.merge_batch_size,
                merge_batch_overlap=(
                    args.merge_batch_overlap
                    if args.merge_batch_overlap is not None
                    else stage_config.merge_batch_overlap
                ),
                max_prior_semantic_actions=(
                    args.max_prior_semantic_actions or stage_config.max_prior_semantic_actions
                ),
                limits=args.limits if args.limits is not None else stage_config.limits,
                reuse_cache=args.reuse_cache or stage_config.reuse_cache,
                preflight_only=args.preflight_only,
                no_console=args.no_console,
            )
        return 0
    except KeyboardInterrupt:
        print("Semantic action induction interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if not getattr(exc, "_semantic_action_reported", False):
            print(f"Semantic action induction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
