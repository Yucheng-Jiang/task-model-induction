import argparse
import contextlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from task_model_induction.config import load_config, resolve_config_path, resolve_dotenv_path
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from task_model_induction.schemas import TaskThreadsInductionOutput
    from task_model_induction.utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        litellm_completion,
        litellm_model_config,
        normalize_litellm_usage,
        utc_now_iso,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from config import load_config, resolve_config_path, resolve_dotenv_path
    from reporting.progress_reporter import ConsoleProgressReporter, format_duration
    from schemas import TaskThreadsInductionOutput
    from utils import (
        completion_message_content,
        estimated_litellm_completion_cost_usd,
        litellm_completion,
        litellm_model_config,
        normalize_litellm_usage,
        utc_now_iso,
        write_json_atomic,
    )


DEFAULT_INPUT_FILE_NAME = "activity.jsonl"
DEFAULT_OUTPUT_FILE_NAME = "task_threads.json"
DEFAULT_DERIVED_OBJECTIVES_DIR_NAME = "derived_task_thread_objectives"

STAGE_LOAD_INPUTS = "load inputs"
STAGE_PREFLIGHT = "preflight"
STAGE_ROOT_DISCOVERY = "root discovery"
STAGE_CONSOLIDATION = "root consolidation"
STAGE_WRITE_OUTPUT = "write output"
STAGES = [
    STAGE_LOAD_INPUTS,
    STAGE_PREFLIGHT,
    STAGE_ROOT_DISCOVERY,
    STAGE_CONSOLIDATION,
    STAGE_WRITE_OUTPUT,
]


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.monotonic)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0
    breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def elapsed_secs(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def as_meta(self) -> Dict[str, Any]:
        return {
            "elapsed_secs": self.elapsed_secs(),
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": round(self.estimated_usd, 6),
            "cost_breakdown": self.cost_breakdown(),
        }

    def record_call(self, *, operation: str, model: str, usage: Dict[str, int], estimated_usd: float | None) -> None:
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
                "estimated_usd": 0.0,
            },
        )
        bucket["llm_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[key] += int(usage.get(key, 0))
        if estimated_usd is not None:
            bucket["estimated_usd"] = round(float(bucket["estimated_usd"]) + estimated_usd, 6)

    def cost_breakdown(self) -> Dict[str, Any]:
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


_ACTIVE_STATS: Optional[RunStats] = None
_PROGRESS_ENABLED = True
_ALGORITHM_LOGS_ENABLED = False


def _algorithm_log(*args: Any, **kwargs: Any) -> None:
    if _ALGORITHM_LOGS_ENABLED:
        print(*args, **kwargs)


class TaskThreadsReporter(ConsoleProgressReporter):
    run_name = "task_threads_induction"
    success_title = "Task Threads Induction Complete"
    failure_title = "Task Threads Induction Failed"
    default_failure_stage = STAGE_PREFLIGHT

    def __init__(self, *, no_console: bool = False) -> None:
        super().__init__(stages=STAGES, no_console=no_console)

    def render(self) -> Any:
        if not all((self._Panel, self._Table, self._Text, self._Group, self._box)):
            return "Task threads induction"
        title = self._Text("Task Threads Induction", style="bold cyan")
        stage_table = self._stage_table()
        metrics_table = self._metrics_table()
        return self._Panel(
            self._Group(title, stage_table, metrics_table),
            title="Running",
            border_style="cyan",
            box=self._box.ROUNDED,
        )

    def render_success(self, detail: str) -> Any:
        if not all((self._Panel, self._Table, self._Group, self._box)):
            return detail
        return self._Panel(
            self._Group(self._summary_table(detail), self._metrics_table(), self._paths_table()),
            title=self.success_title,
            border_style="green",
            box=self._box.ROUNDED,
        )

    def render_failure(self, message: str) -> Any:
        if not all((self._Panel, self._Table, self._Group, self._box)):
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
            f"model={metrics.get('model', '')} "
            f"activities={counters.get('activities', 0)} "
            f"provisional_roots={counters.get('provisional_roots', 0)} "
            f"task_threads={counters.get('task_threads', 0)} "
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
            ("model", "model"),
            ("activities", "activities"),
            ("provisional_roots", "provisional roots"),
            ("task_threads", "task threads"),
            ("semantic_assignments", "semantic assignments"),
        ):
            value = self.state.metrics.get(key) if key == "model" else self.state.counters.get(key)
            table.add_row(label, str(value or 0))
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

    def _paths_table(self) -> Any:
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(overflow="fold")
        for label, path in self.state.paths.items():
            table.add_row(label, str(path))
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


def _normalize_usage(response: Any) -> Dict[str, int]:
    return normalize_litellm_usage(response)


def _estimated_completion_cost_usd(response: Any, model_name: str) -> float | None:
    return estimated_litellm_completion_cost_usd(response, model_name)


def _print_metrics(stats: RunStats) -> None:
    if not _PROGRESS_ENABLED:
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


def call_llm(
    prompt: str,
    content: Any = None,
    *,
    model_name: str,
    timeout_secs: float,
    json_mode: bool = False,
    operation: str = "task_thread",
) -> str:
    user_content = content if isinstance(content, str) else json.dumps(content if content is not None else [], ensure_ascii=False)
    response = litellm_completion(
        model=model_name,
        messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
        ],
        temperature=0.0 if "gpt-5" not in model_name and "kimi" not in model_name else 1.0,
        timeout=timeout_secs,
        request_timeout=timeout_secs,
        **({"response_format": {"type": "json_object"}} if json_mode else {}),
    )
    if _ACTIVE_STATS is not None:
        usage = _normalize_usage(response)
        call_usd = _estimated_completion_cost_usd(response, model_name)
        _ACTIVE_STATS.record_call(operation=operation, model=model_name, usage=usage, estimated_usd=call_usd)
        _print_metrics(_ACTIVE_STATS)
    return completion_message_content(response)


def extract_json_from_response(response_str: str) -> dict:
    try:
        if "```json" in response_str:
            json_str = response_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response_str:
            json_str = response_str.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            json_str = response_str.strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting JSON: {e}", file=sys.stderr)
        try:
            match = re.search(r"\{.*\}", response_str, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}


ROOT_THREAD_DISCOVERY_PROMPT = """You are building a task thread forest from chronologically ordered LEAF latent tasks.

Each leaf is already a local task. Your job is to attach each leaf to a durable ROOT thread.

Core mental model:
- Ask: "Which long-running top-level objective is this leaf advancing right now?"
- Roots represent durable objectives / deliverables, not contiguous time blocks.
- Leaves may interleave across roots.
- A root can pause and later resume.

Critical rules:
- Objective continuity beats adjacency.
- Two adjacent leaves can belong to different roots.
- Two distant leaves can belong to the same root.
- Different apps do NOT imply different roots.
- Interruptions do NOT imply different roots.
- Create a NEW root only when a genuinely new durable objective appears.
- Do NOT create a new root for a tiny opportunistic detour unless it clearly becomes its own durable objective.
- Prefer a small number of strong roots over many near-duplicates.
- Communication leaves belong to the root defined by the SUBJECT of the message, not to a generic communication root.
- Setup, debugging, repo inspection, and environment preparation should stay under the same root as the later deliverable if they are clearly in service of that deliverable.
- Keep side research or analysis work separate from product-building roots when the underlying deliverable is different, even if both mention similar topics or happen in the same apps.

Available existing roots:
{existing_roots}

Most recent assigned leaves before this batch:
{recent_context}

Current leaves to assign:
{leaf_batch}

Task:
1. Reuse an existing root whenever the leaf advances the same durable objective / deliverable.
2. Create a new root only when needed.
3. New roots created inside this batch can be referenced by later leaves in the same batch.
4. After assigning leaves, update each touched root's label/objective/deliverable/success_criteria/summary/last_update/anchor so future batches can judge fit.

Root update rules:
- summary is at most two sentences describing the durable thread so far.
- last_update is exactly one concise sentence describing the latest assigned leaf or leaves.
- anchor is a minimal concise list of stable identifiers for matching future work: project names, repos, datasets, products, people, files, or systems. Normalize aliases when they clearly refer to the same project; for example, two different names for the same codebase or product should share one anchor entry.
- Do not let anchor grow into a keyword dump. Prefer 1-5 meaningful identifiers.
- Existing roots may get improved label/objective values when the new assignment clarifies the durable objective.
- Keep `objective` abstract and tool-agnostic. Put the concrete artifact or state in `deliverable`, and put the independently checkable completion condition in `success_criteria`.

Output ONLY valid JSON:
{{
  "new_roots": [
    {{
      "temp_root_id": "new_1",
      "label": "<short human-readable root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state this root advances>",
      "success_criteria": "<how completion would be observable>",
      "summary": "<optional initial root summary, max two sentences>",
      "last_update": "<optional latest event, one sentence>",
      "anchor": ["<stable identifier>"]
    }}
  ],
  "assignments": [
    {{
      "leaf_idx": <global leaf index>,
      "assigned_root_id": "<existing root id or temp_root_id from new_roots>"
    }}
  ],
  "root_updates": [
    {{
      "root_id": "<existing root id or temp_root_id touched by this batch>",
      "label": "<updated short label>",
      "objective": "<updated durable objective>",
      "deliverable": "<updated concrete artifact/state>",
      "success_criteria": "<updated observable completion criteria>",
      "summary": "<updated summary, max two sentences>",
      "last_update": "<what happened last, one sentence>",
      "anchor": ["<minimal stable identifiers>"]
    }}
  ]
}}"""


ROOT_THREAD_DISCOVERY_RETRY_PROMPT = """You previously assigned a leaf batch to durable root threads, but the output had validation errors.

Rules that must hold:
- Every current leaf index must appear exactly once in assignments.
- assigned_root_id must reference either an existing root id or a temp_root_id declared in new_roots.
- New roots should be created sparingly, only for genuinely new durable objectives.
- Objective continuity beats adjacency and app changes.
- Include root_updates for roots touched by the assignments when possible.
- Each root_update summary is at most two sentences, last_update is one sentence, and anchor is a minimal list of stable identifiers.

Available existing roots:
{existing_roots}

Most recent assigned leaves before this batch:
{recent_context}

Current leaves to assign:
{leaf_batch}

Previous output:
{previous_output}

Errors:
{errors}

Produce a corrected JSON object with the same schema:
{{
  "new_roots": [
    {{
      "temp_root_id": "new_1",
      "label": "<short root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state>",
      "success_criteria": "<observable completion criteria>",
      "summary": "<optional initial root summary, max two sentences>",
      "last_update": "<optional latest event, one sentence>",
      "anchor": ["<stable identifier>"]
    }}
  ],
  "assignments": [
    {{
      "leaf_idx": <global leaf index>,
      "assigned_root_id": "<existing root id or temp_root_id>"
    }}
  ],
  "root_updates": [
    {{
      "root_id": "<existing root id or temp_root_id touched by this batch>",
      "label": "<updated short label>",
      "objective": "<updated durable objective>",
      "deliverable": "<updated concrete artifact/state>",
      "success_criteria": "<updated observable completion criteria>",
      "summary": "<updated summary, max two sentences>",
      "last_update": "<what happened last, one sentence>",
      "anchor": ["<minimal stable identifiers>"]
    }}
  ]
}}"""


ROOT_THREAD_CONSOLIDATION_PROMPT = """You are consolidating provisional durable root threads into the final task thread forest.

Each provisional root was discovered from chronological leaves. Some provisional roots may actually belong to the SAME durable objective and should be merged.

Core rules:
- Merge provisional roots if they advance the same long-running deliverable / objective, even if they are far apart in time, use different apps, or are interrupted.
- Keep roots separate if they represent genuinely different durable objectives.
- Objective continuity beats adjacency.
- Tiny opportunistic one-off roots may be absorbed into a nearby substantive root if they do not establish an independent durable objective.
- Prefer a compact set of strong canonical roots.
- Early setup/debugging/investigation roots should be merged into the later product root when they clearly enable that same deliverable.
- Communication-heavy provisional roots should be merged based on what the messages are ABOUT, not merely because they happen in the same messaging tool.

Provisional roots:
{provisional_roots}

Output ONLY valid JSON:
{{
  "canonical_roots": [
    {{
      "canonical_root_id": "C1",
      "label": "<short human-readable root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state this root advances>",
      "success_criteria": "<observable completion criteria>",
      "member_root_ids": ["R001", "R004"]
    }}
  ]
}}"""


ROOT_THREAD_CONSOLIDATION_RETRY_PROMPT = """You previously consolidated provisional roots into canonical roots, but the output had validation errors.

Rules that must hold:
- Every provisional root id must appear in exactly one canonical root's member_root_ids.
- Do not omit or duplicate provisional roots.
- Keep distinct durable objectives separate.
- Merge only when the same top-level deliverable / objective is being advanced.

Provisional roots:
{provisional_roots}

Previous output:
{previous_output}

Errors:
{errors}

Produce a corrected JSON object with the same schema:
{{
  "canonical_roots": [
    {{
      "canonical_root_id": "C1",
      "label": "<short root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state>",
      "success_criteria": "<observable completion criteria>",
      "member_root_ids": ["R001", "R004"]
    }}
  ]
}}"""


CANONICAL_ROOT_MERGE_REVIEW_PROMPT = """You are reviewing the first-pass canonical roots in a task thread forest for accidental duplicate splitting.

Different canonical roots should be MERGED if they are still advancing the same durable objective / deliverable, even when:
- the work spans different phases such as setup, deployment, testing, feedback, and iteration,
- the product/repo/site name drifted over time,
- the early phase is framed as investigation/tutorial work and the later phase is framed as deployment/bug fixing for the same frontend.

Keep roots separate only when the underlying durable deliverable is genuinely different.

Canonical roots to review:
{canonical_roots}

Output ONLY valid JSON:
{{
  "canonical_roots": [
    {{
      "canonical_root_id": "C1",
      "label": "<short human-readable root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state this merged root advances>",
      "success_criteria": "<observable completion criteria>",
      "member_canonical_root_ids": ["C1", "C4"]
    }}
  ]
}}"""


CANONICAL_ROOT_MERGE_RETRY_PROMPT = """You previously reviewed canonical roots for duplicate splitting, but the output had validation errors.

Rules that must hold:
- Every provided canonical_root_id must appear in exactly one merged canonical root's member_canonical_root_ids.
- Do not omit or duplicate canonical roots.
- Merge only when the same durable objective / deliverable is being advanced.

Canonical roots to review:
{canonical_roots}

Previous output:
{previous_output}

Errors:
{errors}

Produce a corrected JSON object with the same schema:
{{
  "canonical_roots": [
    {{
      "canonical_root_id": "C1",
      "label": "<short root label>",
      "objective": "<durable top-level objective>",
      "deliverable": "<artifact/state>",
      "success_criteria": "<observable completion criteria>",
      "member_canonical_root_ids": ["C1", "C4"]
    }}
  ]
}}"""


ROOT_REASSIGNMENT_REVIEW_PROMPT = """You are reviewing a small set of potentially ambiguous leaf-to-root assignments in a task thread forest.

Each candidate leaf already has a current canonical root assignment. Reassign it ONLY if another canonical root is clearly a better match.

Core rules:
- Keep the current assignment unless another canonical root is clearly better.
- Communication leaves belong to the root defined by what the message is ABOUT.
- Setup/debugging/runtime leaves belong to the deliverable or side project they are enabling.
- Objective continuity beats adjacency and app usage.
- Small side-project roots should remain separate when the underlying deliverable is different from the main product root.

Canonical roots:
{canonical_roots}

Candidate leaves:
{candidate_leaves}

Output ONLY valid JSON:
{{
  "assignments": [
    {{
      "leaf_idx": <leaf index>,
      "assigned_canonical_root_id": "<one canonical root id from the list above>"
    }}
  ]
}}"""


ROOT_REASSIGNMENT_RETRY_PROMPT = """You previously reviewed candidate leaf assignments, but the output had validation errors.

Rules that must hold:
- Every candidate leaf index must appear exactly once.
- assigned_canonical_root_id must be one of the provided canonical root ids.
- Keep the current assignment unless another canonical root is clearly better.

Canonical roots:
{canonical_roots}

Candidate leaves:
{candidate_leaves}

Previous output:
{previous_output}

Errors:
{errors}

Produce a corrected JSON object with the same schema:
{{
  "assignments": [
    {{
      "leaf_idx": <leaf index>,
      "assigned_canonical_root_id": "<one canonical root id from the list above>"
    }}
  ]
}}"""


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _coerce_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _truncate_text(text: str, max_chars: int = 180) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _limit_sentences(text: Any, max_sentences: int, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    limited = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
    return _truncate_text(limited or cleaned, max_chars)


def _concise_anchor_items(value: Any, max_items: int = 5) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for item in _normalize_text_list(value):
        cleaned = re.sub(r"\s+", " ", item).strip(" ,;")
        if not cleaned or len(cleaned) > 80:
            continue
        norm = _normalize_text(cleaned)
        if norm and norm not in seen:
            items.append(cleaned)
            seen.add(norm)
        if len(items) >= max_items:
            break
    return items


def _slugify_filename(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:80] or fallback


def _infer_anchor_from_leaf(leaf: "LeafLatentTaskRecord", max_items: int = 3) -> List[str]:
    text_parts = [leaf.objective, leaf.procedure_summary, *leaf.entities, *leaf.ocr_texts[:2]]
    text = " ".join(part for part in text_parts if part)
    candidates: List[str] = []
    for pattern in (
        r"\b[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+){1,}\b",
        r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b",
    ):
        for match in re.findall(pattern, text):
            value = match.strip(".,;:()[]{}")
            if value and value not in candidates:
                candidates.append(value)
    for entity in leaf.entities:
        if len(candidates) >= max_items:
            break
        entity = _truncate_text(entity, 60)
        if entity and entity not in candidates:
            candidates.append(entity)
    return _concise_anchor_items(candidates, max_items=max_items)


def _grounded_leaf_deliverable(leaf: "LeafLatentTaskRecord") -> str:
    control_terms = {
        "button",
        "checkbox",
        "dialog",
        "dropdown",
        "field",
        "icon",
        "link",
        "menu",
        "tab",
        "toolbar",
    }
    for entity in reversed(leaf.entities):
        tokens = set(re.findall(r"[a-z]+", entity.lower()))
        if not (tokens & control_terms):
            return _truncate_text(entity, 220)
    return _truncate_text(leaf.objective, 220)


def _grounded_leaf_success_criterion(leaf: "LeafLatentTaskRecord") -> str:
    deliverable = _grounded_leaf_deliverable(leaf)
    criterion = (
        f'Verify in the post-action UI that "{deliverable}" reflects the intended result '
        f'"{_truncate_text(leaf.objective, 140)}".'
    )
    if leaf.ocr_texts:
        criterion += f' Use visible text such as "{_truncate_text(leaf.ocr_texts[-1], 100)}" as evidence.'
    return _truncate_text(criterion, 260)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _normalized_item_set(value: Any) -> set[str]:
    return {item for item in (_normalize_text(text) for text in _normalize_text_list(value)) if item}


def _text_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _representative_text_samples(items: List[str], max_items: int) -> List[str]:
    unique_items = [text for text in _normalize_text_list(items) if text]
    if len(unique_items) <= max_items:
        return unique_items
    if max_items <= 1:
        return unique_items[:1]

    selected: List[str] = []
    seen: set[str] = set()
    last_idx = len(unique_items) - 1
    for sample_idx in range(max_items):
        pos = round(sample_idx * last_idx / (max_items - 1))
        value = unique_items[pos]
        norm = _normalize_text(value)
        if norm and norm not in seen:
            selected.append(value)
            seen.add(norm)
    if len(selected) == max_items:
        return selected
    for value in unique_items:
        norm = _normalize_text(value)
        if norm and norm not in seen:
            selected.append(value)
            seen.add(norm)
        if len(selected) == max_items:
            break
    return selected


def _available_boundary_ids(start_id: Optional[str], end_id: Optional[str]) -> List[str]:
    ids: List[str] = []
    for value in (start_id, end_id):
        text = str(value or "").strip()
        if text and text not in ids:
            ids.append(text)
    return ids


def _activity_source_actions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_actions = data.get("source_actions")
    if not isinstance(source_actions, list):
        return []
    return [item for item in source_actions if isinstance(item, dict)]


def _source_ocr_text(source: Dict[str, Any]) -> str:
    direct = str(source.get("md_results") or "").strip()
    if direct:
        return direct
    ocr_results = source.get("ocr_results")
    if isinstance(ocr_results, dict):
        return str(ocr_results.get("md_results") or "").strip()
    return ""


def _activity_evidence(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read aggregate evidence, filling gaps from lossless source actions."""

    source_actions = _activity_source_actions(data)
    apps = _normalize_text_list(data.get("apps_used"))
    apps.extend(
        app
        for app in (str(source.get("active_application") or "").strip() for source in source_actions)
        if app and app not in apps
    )
    entities = _normalize_text_list(data.get("entities"))
    entities.extend(
        entity
        for entity in (
            str(source.get("grounded_visual_content") or source.get("visual_content") or "").strip()
            for source in source_actions
        )
        if entity and entity not in entities
    )
    raw_action_ids = _normalize_text_list(data.get("raw_action_ids") or data.get("action_ids"))
    raw_action_ids.extend(
        action_id
        for action_id in (str(source.get("action_id") or "").strip() for source in source_actions)
        if action_id and action_id not in raw_action_ids
    )
    ocr_texts = _normalize_text_list(data.get("ocr_texts"))
    ocr_texts.extend(
        text
        for text in (_source_ocr_text(source) for source in source_actions)
        if text and text not in ocr_texts
    )
    pre_state = str(data.get("pre_state") or "").strip() or next(
        (str(source.get("state_before") or "").strip() for source in source_actions if source.get("state_before")),
        "",
    )
    post_state = str(data.get("post_state") or "").strip() or next(
        (
            str(source.get("state_after") or "").strip()
            for source in reversed(source_actions)
            if source.get("state_after")
        ),
        "",
    )
    return {
        "apps_used": apps,
        "entities": entities,
        "raw_action_ids": raw_action_ids,
        "ocr_texts": ocr_texts,
        "pre_state": pre_state,
        "post_state": post_state,
        "source_action_count": len(source_actions),
    }


@dataclass
class LeafLatentTaskRecord:
    leaf_idx: int
    activity_id: str
    start_node_idx: int
    end_node_idx: int
    start_semantic_action_idx: int = -1
    end_semantic_action_idx: int = -1
    start_semantic_action_id: Optional[str] = None
    end_semantic_action_id: Optional[str] = None
    start_action_idx: int = -1
    end_action_idx: int = -1
    start_action_id: Optional[str] = None
    end_action_id: Optional[str] = None
    objective: str = "Unclassified step"
    procedure_summary: str = ""
    apps_used: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)
    pre_state: str = ""
    post_state: str = ""
    num_actions: int = 0
    semantic_action_ids: List[str] = field(default_factory=list)
    raw_action_ids: List[str] = field(default_factory=list)
    semantic_actions: List[str] = field(default_factory=list)
    semantic_action_count: int = 0
    event_count: int = 0

    @classmethod
    def from_dict(cls, leaf_idx: int, data: Dict[str, Any]) -> "LeafLatentTaskRecord":
        evidence = _activity_evidence(data)
        start_semantic_action_idx = int(data.get("start_semantic_action_idx", data.get("start_node_idx", -1)))
        end_semantic_action_idx = int(data.get("end_semantic_action_idx", data.get("end_node_idx", -1)))
        semantic_action_count = _coerce_nonnegative_int(data.get("semantic_action_count"), default=1)
        event_count = _coerce_nonnegative_int(data.get("event_count"))
        return cls(
            leaf_idx=int(leaf_idx),
            activity_id=str(
                data.get("activity_id")
                or data.get("latent_leaf_objective_id")
                or f"subgoal_segment_{int(leaf_idx):04d}"
            ),
            start_node_idx=int(data.get("start_node_idx", start_semantic_action_idx)),
            end_node_idx=int(data.get("end_node_idx", end_semantic_action_idx)),
            start_semantic_action_idx=start_semantic_action_idx,
            end_semantic_action_idx=end_semantic_action_idx,
            start_semantic_action_id=data.get("start_semantic_action_id"),
            end_semantic_action_id=data.get("end_semantic_action_id"),
            start_action_idx=int(data.get("start_action_idx", -1)),
            end_action_idx=int(data.get("end_action_idx", -1)),
            start_action_id=data.get("start_action_id"),
            end_action_id=data.get("end_action_id"),
            objective=(
                data.get("objective")
                or data.get("leaf_latent_task")
                or data.get("subgoal")
                or "Unclassified step"
            ).strip(),
            procedure_summary=(
                data.get("procedure_summary")
                or data.get("additional_context")
                or data.get("detail_brief")
                or ""
            ).strip(),
            apps_used=evidence["apps_used"],
            entities=evidence["entities"],
            ocr_texts=evidence["ocr_texts"],
            pre_state=evidence["pre_state"],
            post_state=evidence["post_state"],
            num_actions=_coerce_nonnegative_int(
                data.get(
                    "num_actions",
                    data.get("event_count", evidence["source_action_count"] or data.get("semantic_action_count")),
                )
            ),
            semantic_action_ids=_normalize_text_list(data.get("semantic_action_ids")),
            raw_action_ids=evidence["raw_action_ids"],
            semantic_actions=_normalize_text_list(data.get("semantic_actions")),
            semantic_action_count=semantic_action_count,
            event_count=event_count,
        )

    def semantic_profile(self) -> str:
        parts = [
            f"objective: {self.objective}",
            f"procedure_summary: {self.procedure_summary}",
            f"apps_used: {', '.join(self.apps_used)}",
            f"entities: {', '.join(self.entities)}",
            f"ocr_text: {' | '.join(_truncate_text(text, 240) for text in self.ocr_texts[:3])}",
            f"pre_state: {self.pre_state}",
            f"post_state: {self.post_state}",
        ]
        return "\n".join(part for part in parts if _normalize_text(part))


@dataclass
class ProvisionalRootIR:
    root_id: str
    label: str = ""
    objective: str = ""
    deliverable: str = ""
    success_criteria: str = ""
    summary: str = ""
    last_update: str = ""
    anchor: List[str] = field(default_factory=list)
    leaf_indices: List[int] = field(default_factory=list)
    apps_used: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)
    sample_leaf_objectives: List[str] = field(default_factory=list)

    def add_leaf(self, leaf: LeafLatentTaskRecord) -> None:
        if leaf.leaf_idx not in self.leaf_indices:
            self.leaf_indices.append(leaf.leaf_idx)
            self.leaf_indices.sort()
        for app in leaf.apps_used:
            if app and app not in self.apps_used:
                self.apps_used.append(app)
        for entity in leaf.entities:
            if entity and entity not in self.entities:
                self.entities.append(entity)
        for text in leaf.ocr_texts:
            if text and text not in self.ocr_texts and len(self.ocr_texts) < 6:
                self.ocr_texts.append(text)
        objective = leaf.objective.strip()
        if objective and objective not in self.sample_leaf_objectives:
            self.sample_leaf_objectives.append(objective)
        for item in _infer_anchor_from_leaf(leaf):
            if item and item not in self.anchor and len(self.anchor) < 5:
                self.anchor.append(item)
        self.last_update = _limit_sentences(
            leaf.procedure_summary or leaf.objective,
            max_sentences=1,
            max_chars=180,
        )
        if not self.summary:
            self.summary = _limit_sentences(
                f"{self.objective or self.label}. {leaf.procedure_summary or leaf.objective}",
                max_sentences=2,
                max_chars=320,
            )

    def apply_update(self, update: Dict[str, Any]) -> None:
        label = _truncate_text(str(update.get("label") or "").strip(), 90)
        objective = _limit_sentences(update.get("objective"), max_sentences=1, max_chars=180)
        deliverable = _limit_sentences(update.get("deliverable"), max_sentences=1, max_chars=220)
        success_criteria = _limit_sentences(update.get("success_criteria"), max_sentences=1, max_chars=260)
        summary = _limit_sentences(update.get("summary"), max_sentences=2, max_chars=320)
        last_update = _limit_sentences(update.get("last_update"), max_sentences=1, max_chars=180)
        anchor = _concise_anchor_items(update.get("anchor"), max_items=5)
        if label:
            self.label = label
        if objective:
            self.objective = objective
        if deliverable:
            self.deliverable = deliverable
        if success_criteria:
            self.success_criteria = success_criteria
        if summary:
            self.summary = summary
        if last_update:
            self.last_update = last_update
        if anchor:
            self.anchor = anchor

    @property
    def first_leaf_idx(self) -> int:
        return self.leaf_indices[0] if self.leaf_indices else -1

    @property
    def last_leaf_idx(self) -> int:
        return self.leaf_indices[-1] if self.leaf_indices else -1

    def semantic_profile(self) -> str:
        parts = [
            f"label: {self.label}",
            f"objective: {self.objective}",
            f"deliverable: {self.deliverable}",
            f"success_criteria: {self.success_criteria}",
            f"summary: {self.summary}",
            f"last_update: {self.last_update}",
            f"anchor: {', '.join(self.anchor)}",
            f"apps_used: {', '.join(self.apps_used)}",
            f"entities: {', '.join(self.entities)}",
            f"ocr_text: {' | '.join(_truncate_text(text, 240) for text in self.ocr_texts[:3])}",
            f"sample_leaf_objectives: {' | '.join(_representative_text_samples(self.sample_leaf_objectives, 6))}",
        ]
        return "\n".join(part for part in parts if _normalize_text(part))


@dataclass
class CanonicalRootIR:
    canonical_root_id: str
    label: str = ""
    objective: str = ""
    deliverable: str = ""
    success_criteria: str = ""
    member_root_ids: List[str] = field(default_factory=list)


class TaskThreadInductionBuilder:
    DISCOVERY_BATCH_SIZE = 15
    MAX_RECENT_ASSIGNMENTS = 10
    MAX_ROOT_SAMPLES = 5
    CACHE_PIPELINE_VERSION = 3
    ENABLE_REASSIGNMENT_REVIEW = False

    def __init__(
        self,
        model_name: str,
        *,
        llm_timeout_secs: float,
        stats: Optional[RunStats] = None,
    ):
        self.model_name = model_name
        self.llm_timeout_secs = llm_timeout_secs
        self.stats = stats or RunStats()
        self._progress_fn: Optional[Callable[[str, str], None]] = None

    def _report(self, milestone: str, detail: str) -> None:
        if self._progress_fn is not None:
            try:
                self._progress_fn(milestone, detail)
            except Exception:
                pass

    @staticmethod
    def _save_cache(path: str, payload: Dict[str, Any]) -> None:
        write_json_atomic(Path(path), payload)

    @staticmethod
    def _load_cache(path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _resolve_leaf_tasks_path(data_dir: str, input_file_name: str = DEFAULT_INPUT_FILE_NAME) -> str:
        subgoal_path = os.path.join(data_dir, input_file_name)
        if os.path.exists(subgoal_path):
            return subgoal_path
        legacy_path = os.path.join(data_dir, "latent_leaf_objectives.jsonl")
        if os.path.exists(legacy_path):
            return legacy_path
        return subgoal_path

    @staticmethod
    def _episodes_from_leaf_indices(leaf_indices: List[int]) -> List[Tuple[int, int, List[int]]]:
        if not leaf_indices:
            return []
        ordered = sorted(set(int(idx) for idx in leaf_indices))
        episodes: List[Tuple[int, int, List[int]]] = []
        start = ordered[0]
        current = [ordered[0]]
        prev = ordered[0]
        for idx in ordered[1:]:
            if idx == prev + 1:
                current.append(idx)
            else:
                episodes.append((start, prev, list(current)))
                start = idx
                current = [idx]
            prev = idx
        episodes.append((start, prev, list(current)))
        return episodes

    def _load_leaf_tasks(self, path: str) -> List[LeafLatentTaskRecord]:
        input_path = Path(path)
        if input_path.suffix == ".jsonl":
            records = []
            with input_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
                    if isinstance(row, dict):
                        records.append(row)
        else:
            payload = self._load_cache(path)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = (
                    payload.get("activities")
                    or payload.get("subgoal_segments")
                    or payload.get("leaf_latent_tasks")
                    or payload.get("segments")
                    or []
                )
            else:
                records = []

        out: List[LeafLatentTaskRecord] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            out.append(LeafLatentTaskRecord.from_dict(idx, record))
        return out

    def _describe_leaf(self, leaf: LeafLatentTaskRecord) -> str:
        payload = {
            "leaf_idx": int(leaf.leaf_idx),
            "node_range": [int(leaf.start_node_idx), int(leaf.end_node_idx)],
            "objective": leaf.objective,
            "procedure_summary": leaf.procedure_summary,
            "apps_used": leaf.apps_used[:4],
            "entities": leaf.entities[:6],
            "ocr_text": [_truncate_text(text, 320) for text in leaf.ocr_texts[:3]],
            "pre_state": _truncate_text(leaf.pre_state, 120),
            "post_state": _truncate_text(leaf.post_state, 120),
            "num_actions": int(leaf.num_actions),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _describe_leaf_batch(self, leaves: List[LeafLatentTaskRecord]) -> str:
        return "\n".join(self._describe_leaf(leaf) for leaf in leaves)

    def _summarize_recent_assignments(
        self,
        assignments: Dict[int, str],
        roots_by_id: Dict[str, ProvisionalRootIR],
        leaves: List[LeafLatentTaskRecord],
        k: int,
    ) -> str:
        if not assignments:
            return "Start of session. No prior leaves."

        recent_leaf_indices = sorted(assignments.keys())[-k:]
        lines: List[str] = []
        for leaf_idx in recent_leaf_indices:
            root_id = assignments.get(leaf_idx, "")
            leaf = leaves[leaf_idx]
            root = roots_by_id.get(root_id)
            label = root.label if root and root.label else (root.objective if root else "")
            lines.append(
                f'- leaf {leaf_idx} -> {root_id} "{_truncate_text(label or root_id, 90)}" '
                f'objective="{_truncate_text(leaf.objective, 120)}"'
            )
        return "\n".join(lines)

    def _root_summary_payload(self, root: ProvisionalRootIR) -> Dict[str, Any]:
        sample_objs = _representative_text_samples(root.sample_leaf_objectives, self.MAX_ROOT_SAMPLES)
        payload = {
            "root_id": root.root_id,
            "label": root.label or root.objective,
            "objective": root.objective,
            "deliverable": root.deliverable,
            "success_criteria": root.success_criteria,
            "summary": root.summary,
            "last_update": root.last_update,
            "anchor": root.anchor[:5],
            "leaf_span": [root.first_leaf_idx, root.last_leaf_idx],
            "leaf_count": len(root.leaf_indices),
            "episode_ranges": [[s, e] for s, e, _ in self._episodes_from_leaf_indices(root.leaf_indices)],
            "apps_used": root.apps_used[:4],
            "entities": root.entities[:6],
            "ocr_text": [_truncate_text(text, 240) for text in root.ocr_texts[:3]],
            "sample_leaf_objectives": sample_objs[: self.MAX_ROOT_SAMPLES],
        }
        return payload

    def _summarize_existing_roots(self, roots_by_id: Dict[str, ProvisionalRootIR]) -> str:
        if not roots_by_id:
            return "No existing roots yet."
        ordered = sorted(
            roots_by_id.values(),
            key=lambda root: (root.last_leaf_idx, root.first_leaf_idx),
            reverse=True,
        )
        return "\n".join(
            json.dumps(self._root_summary_payload(root), ensure_ascii=False)
            for root in ordered
        )

    def _validate_discovery_output(
        self,
        data: Dict[str, Any],
        batch_leaf_indices: List[int],
        existing_root_ids: List[str],
    ) -> List[str]:
        errors: List[str] = []
        new_roots_raw = data.get("new_roots")
        assignments_raw = data.get("assignments")

        if not isinstance(new_roots_raw, list):
            errors.append("new_roots must be a list.")
            new_roots_raw = []
        if not isinstance(assignments_raw, list):
            errors.append("assignments must be a list.")
            assignments_raw = []

        temp_root_ids: set[str] = set()
        for root in new_roots_raw:
            if not isinstance(root, dict):
                errors.append(f"Invalid new root entry: {root}")
                continue
            temp_root_id = str(root.get("temp_root_id") or root.get("root_id") or "").strip()
            if not temp_root_id:
                errors.append(f"Missing temp_root_id in new root entry: {root}")
                continue
            if temp_root_id in temp_root_ids:
                errors.append(f"Duplicate temp_root_id: {temp_root_id}")
            temp_root_ids.add(temp_root_id)
            if not str(root.get("deliverable") or "").strip():
                errors.append(f"Missing deliverable for new root {temp_root_id}")
            if not str(root.get("success_criteria") or "").strip():
                errors.append(f"Missing success_criteria for new root {temp_root_id}")

        expected = set(batch_leaf_indices)
        seen: set[int] = set()
        valid_root_ids = set(existing_root_ids) | temp_root_ids
        for assignment in assignments_raw:
            if not isinstance(assignment, dict):
                errors.append(f"Invalid assignment entry: {assignment}")
                continue
            try:
                leaf_idx = int(assignment.get("leaf_idx", -1))
            except (TypeError, ValueError):
                errors.append(f"Non-integer leaf_idx in assignment: {assignment}")
                continue
            root_id = str(assignment.get("assigned_root_id") or assignment.get("root_id") or "").strip()
            if leaf_idx not in expected:
                errors.append(f"Assignment leaf_idx out of current batch: {leaf_idx}")
            if leaf_idx in seen:
                errors.append(f"Duplicate assignment for leaf_idx={leaf_idx}")
            seen.add(leaf_idx)
            if not root_id:
                errors.append(f"Missing assigned_root_id for leaf_idx={leaf_idx}")
            elif root_id not in valid_root_ids:
                errors.append(f"Unknown assigned_root_id={root_id} for leaf_idx={leaf_idx}")

        missing = sorted(expected - seen)
        if missing:
            errors.append(f"Missing assignments for leaf indices: {missing}")

        root_updates_raw = data.get("root_updates")
        if root_updates_raw is not None:
            if not isinstance(root_updates_raw, list):
                errors.append("root_updates must be a list when provided.")
            else:
                for update in root_updates_raw:
                    if not isinstance(update, dict):
                        errors.append(f"Invalid root_update entry: {update}")
                        continue
                    root_id = str(update.get("root_id") or update.get("assigned_root_id") or "").strip()
                    if root_id and root_id not in valid_root_ids:
                        errors.append(f"Unknown root_update root_id={root_id}")
                    if not str(update.get("deliverable") or "").strip():
                        errors.append(f"Missing deliverable for root_update {root_id or '<unknown>'}")
                    if not str(update.get("success_criteria") or "").strip():
                        errors.append(f"Missing success_criteria for root_update {root_id or '<unknown>'}")
        return errors

    def _heuristic_pick_existing_root(
        self,
        leaf: LeafLatentTaskRecord,
        roots_by_id: Dict[str, ProvisionalRootIR],
        recent_root_id: Optional[str],
    ) -> Optional[str]:
        leaf_profile = leaf.semantic_profile()
        leaf_apps = _normalized_item_set(leaf.apps_used)
        leaf_entities = _normalized_item_set(leaf.entities)
        leaf_anchor = _normalized_item_set(_infer_anchor_from_leaf(leaf, max_items=5))
        best_root_id: Optional[str] = None
        best_score = 0.0

        for root_id, root in roots_by_id.items():
            score = 0.0
            score += 3.0 * _text_similarity(leaf_profile, root.semantic_profile())
            score += 0.8 * len(leaf_apps & _normalized_item_set(root.apps_used))
            score += 0.8 * len(leaf_entities & _normalized_item_set(root.entities))
            score += 1.2 * len(leaf_anchor & _normalized_item_set(root.anchor))
            if recent_root_id and root_id == recent_root_id:
                score += 0.35
            if score > best_score:
                best_score = score
                best_root_id = root_id

        if best_root_id and best_score >= 1.9:
            return best_root_id
        if recent_root_id and leaf.num_actions <= 3:
            return recent_root_id
        return None

    def _heuristic_root_update(
        self,
        root_id: str,
        leaf: LeafLatentTaskRecord,
        root: Optional[ProvisionalRootIR],
    ) -> Dict[str, Any]:
        label = (root.label if root else "") or leaf.objective or root_id
        objective = (root.objective if root else "") or leaf.objective or label
        prior_summary = root.summary if root else ""
        latest = _limit_sentences(leaf.procedure_summary or leaf.objective, max_sentences=1, max_chars=180)
        summary = prior_summary or _limit_sentences(
            f"{objective}. {leaf.procedure_summary or leaf.objective}",
            max_sentences=2,
            max_chars=320,
        )
        anchors = list(root.anchor if root else [])
        for item in _infer_anchor_from_leaf(leaf):
            if item not in anchors and len(anchors) < 5:
                anchors.append(item)
        return {
            "root_id": root_id,
            "label": _truncate_text(label, 90),
            "objective": _limit_sentences(objective, max_sentences=1, max_chars=180),
            "deliverable": (root.deliverable if root else "") or _grounded_leaf_deliverable(leaf),
            "success_criteria": (root.success_criteria if root else "")
            or _grounded_leaf_success_criterion(leaf),
            "summary": summary,
            "last_update": latest,
            "anchor": anchors,
        }

    def _heuristic_discovery_output(
        self,
        batch: List[LeafLatentTaskRecord],
        roots_by_id: Dict[str, ProvisionalRootIR],
        next_root_num: int,
        recent_root_id: Optional[str],
    ) -> Dict[str, Any]:
        new_roots: List[Dict[str, Any]] = []
        assignments: List[Dict[str, Any]] = []
        root_updates: List[Dict[str, Any]] = []
        temp_counter = 1

        for leaf in batch:
            existing_root_id = self._heuristic_pick_existing_root(leaf, roots_by_id, recent_root_id)
            if existing_root_id:
                assignments.append(
                    {
                        "leaf_idx": leaf.leaf_idx,
                        "assigned_root_id": existing_root_id,
                    }
                )
                root_updates.append(self._heuristic_root_update(existing_root_id, leaf, roots_by_id.get(existing_root_id)))
                recent_root_id = existing_root_id
                continue

            temp_root_id = f"new_{temp_counter}"
            temp_counter += 1
            label = leaf.objective or f"Root {next_root_num}"
            last_update = _limit_sentences(leaf.procedure_summary or leaf.objective, max_sentences=1, max_chars=180)
            summary = _limit_sentences(
                f"{label}. {leaf.procedure_summary or leaf.objective}",
                max_sentences=2,
                max_chars=320,
            )
            anchor = _infer_anchor_from_leaf(leaf)
            new_roots.append(
                {
                    "temp_root_id": temp_root_id,
                    "label": label,
                    "objective": label,
                    "deliverable": _grounded_leaf_deliverable(leaf),
                    "success_criteria": _grounded_leaf_success_criterion(leaf),
                    "summary": summary,
                    "last_update": last_update,
                    "anchor": anchor,
                }
            )
            assignments.append(
                {
                    "leaf_idx": leaf.leaf_idx,
                    "assigned_root_id": temp_root_id,
                }
            )
            root_updates.append(
                {
                    "root_id": temp_root_id,
                    "label": _truncate_text(label, 90),
                    "objective": _limit_sentences(label, max_sentences=1, max_chars=180),
                    "deliverable": _grounded_leaf_deliverable(leaf),
                    "success_criteria": _grounded_leaf_success_criterion(leaf),
                    "summary": summary,
                    "last_update": last_update,
                    "anchor": anchor,
                }
            )
            recent_root_id = temp_root_id
        return {"new_roots": new_roots, "assignments": assignments, "root_updates": root_updates}

    def _run_discovery_batch(
        self,
        batch: List[LeafLatentTaskRecord],
        roots_by_id: Dict[str, ProvisionalRootIR],
        recent_context: str,
        next_root_num: int,
        recent_root_id: Optional[str],
    ) -> Dict[str, Any]:
        existing_roots = self._summarize_existing_roots(roots_by_id)
        leaf_batch = self._describe_leaf_batch(batch)
        prompt = ROOT_THREAD_DISCOVERY_PROMPT.format(
            existing_roots=existing_roots,
            recent_context=recent_context,
            leaf_batch=leaf_batch,
        )
        batch_leaf_indices = [leaf.leaf_idx for leaf in batch]
        existing_root_ids = list(roots_by_id.keys())

        try:
            response = call_llm(
                prompt=prompt,
                content=[],
                model_name=self.model_name,
                timeout_secs=self.llm_timeout_secs,
                operation="root_discovery",
            )
            data = extract_json_from_response(response)
            errors = self._validate_discovery_output(data, batch_leaf_indices, existing_root_ids)
            if not errors:
                return data

            print(f"    [Retry] Root discovery batch had errors: {'; '.join(errors)}")
            retry_prompt = ROOT_THREAD_DISCOVERY_RETRY_PROMPT.format(
                existing_roots=existing_roots,
                recent_context=recent_context,
                leaf_batch=leaf_batch,
                previous_output=json.dumps(data, indent=2, ensure_ascii=False),
                errors="\n".join(f"- {error}" for error in errors),
            )
            retry_response = call_llm(
                prompt=retry_prompt,
                content=[],
                model_name=self.model_name,
                timeout_secs=self.llm_timeout_secs,
                operation="root_discovery_retry",
            )
            retry_data = extract_json_from_response(retry_response)
            retry_errors = self._validate_discovery_output(retry_data, batch_leaf_indices, existing_root_ids)
            if not retry_errors:
                print("    [Retry] Root discovery output accepted.")
                return retry_data

            print(f"    [Retry] Root discovery still invalid: {'; '.join(retry_errors)}. Falling back to heuristic.")
        except Exception as e:
            print(f"    [Error] Root discovery batch failed: {e}. Falling back to heuristic.")

        return self._heuristic_discovery_output(batch, roots_by_id, next_root_num, recent_root_id)

    def _discover_provisional_roots(
        self,
        leaves: List[LeafLatentTaskRecord],
    ) -> Tuple[Dict[str, ProvisionalRootIR], Dict[int, str]]:
        roots_by_id: Dict[str, ProvisionalRootIR] = {}
        assignments: Dict[int, str] = {}
        next_root_num = 1
        recent_root_id: Optional[str] = None

        print("\n" + "=" * 60)
        print("ROOT DISCOVERY: Forward Thread Tracking")
        print("=" * 60)

        total_batches = (len(leaves) + self.DISCOVERY_BATCH_SIZE - 1) // self.DISCOVERY_BATCH_SIZE
        batch_num = 0
        for start in range(0, len(leaves), self.DISCOVERY_BATCH_SIZE):
            end = min(len(leaves), start + self.DISCOVERY_BATCH_SIZE)
            batch = leaves[start:end]
            batch_num += 1

            print(f"\n  [Root Batch {batch_num}] Leaves [{start}..{end - 1}]")
            recent_context = self._summarize_recent_assignments(
                assignments,
                roots_by_id,
                leaves,
                k=self.MAX_RECENT_ASSIGNMENTS,
            )
            data = self._run_discovery_batch(
                batch,
                roots_by_id,
                recent_context,
                next_root_num,
                recent_root_id,
            )

            temp_to_actual: Dict[str, str] = {}
            for root_data in data.get("new_roots", []):
                if not isinstance(root_data, dict):
                    continue
                actual_root_id = f"R{next_root_num:03d}"
                next_root_num += 1
                temp_root_id = str(root_data.get("temp_root_id") or root_data.get("root_id") or "").strip()
                if not temp_root_id:
                    continue
                temp_to_actual[temp_root_id] = actual_root_id
                roots_by_id[actual_root_id] = ProvisionalRootIR(
                    root_id=actual_root_id,
                    label=(root_data.get("label") or root_data.get("objective") or actual_root_id).strip(),
                    objective=(root_data.get("objective") or root_data.get("label") or "").strip(),
                    deliverable=(root_data.get("deliverable") or "").strip(),
                    success_criteria=(root_data.get("success_criteria") or "").strip(),
                    summary=_limit_sentences(root_data.get("summary"), max_sentences=2, max_chars=320),
                    last_update=_limit_sentences(root_data.get("last_update"), max_sentences=1, max_chars=180),
                    anchor=_concise_anchor_items(root_data.get("anchor"), max_items=5),
                )
                print(
                    f'    >> New provisional root {actual_root_id}: '
                    f'"{_truncate_text(roots_by_id[actual_root_id].label, 100)}"'
                )

            root_updates_by_id: Dict[str, Dict[str, Any]] = {}
            for update in data.get("root_updates", []):
                if not isinstance(update, dict):
                    continue
                update_root_id = str(update.get("root_id") or update.get("assigned_root_id") or "").strip()
                update_root_id = temp_to_actual.get(update_root_id, update_root_id)
                if not update_root_id:
                    continue
                normalized_update = dict(update)
                normalized_update["root_id"] = update_root_id
                root_updates_by_id[update_root_id] = normalized_update

            ordered_assignments = sorted(
                [
                    assignment for assignment in data.get("assignments", [])
                    if isinstance(assignment, dict)
                ],
                key=lambda item: int(item.get("leaf_idx", -1)),
            )
            touched_last_leaf_by_root: Dict[str, LeafLatentTaskRecord] = {}
            for assignment in ordered_assignments:
                leaf_idx = int(assignment.get("leaf_idx", -1))
                assigned_root_id = str(assignment.get("assigned_root_id") or assignment.get("root_id") or "").strip()
                assigned_root_id = temp_to_actual.get(assigned_root_id, assigned_root_id)
                if leaf_idx < 0 or leaf_idx >= len(leaves) or not assigned_root_id:
                    continue
                if assigned_root_id not in roots_by_id:
                    roots_by_id[assigned_root_id] = ProvisionalRootIR(
                        root_id=assigned_root_id,
                        label=assigned_root_id,
                        objective=assigned_root_id,
                    )
                assignments[leaf_idx] = assigned_root_id
                roots_by_id[assigned_root_id].add_leaf(leaves[leaf_idx])
                touched_last_leaf_by_root[assigned_root_id] = leaves[leaf_idx]
                recent_root_id = assigned_root_id
                print(
                    f'    leaf {leaf_idx:>3} -> {assigned_root_id} '
                    f'"{_truncate_text(roots_by_id[assigned_root_id].label, 70)}"'
                )
            for root_id, leaf in touched_last_leaf_by_root.items():
                root = roots_by_id.get(root_id)
                if not root:
                    continue
                update = root_updates_by_id.get(root_id) or self._heuristic_root_update(root_id, leaf, root)
                root.apply_update(update)
            self._report(
                "discovery_batch",
                f"batch {batch_num}/{total_batches} — {len(roots_by_id)} provisional roots",
            )

        print(f"\n  >> Provisional roots discovered: {len(roots_by_id)}")
        return roots_by_id, assignments

    def _describe_provisional_roots(self, roots_by_id: Dict[str, ProvisionalRootIR]) -> str:
        ordered = sorted(roots_by_id.values(), key=lambda root: root.first_leaf_idx)
        return "\n".join(
            json.dumps(self._root_summary_payload(root), ensure_ascii=False)
            for root in ordered
        )

    def _validate_consolidation_output(
        self,
        data: Dict[str, Any],
        provisional_root_ids: List[str],
    ) -> List[str]:
        errors: List[str] = []
        canonical_roots_raw = data.get("canonical_roots")
        if not isinstance(canonical_roots_raw, list):
            return ["canonical_roots must be a list."]
        if not canonical_roots_raw:
            return ["canonical_roots is empty."]

        expected = set(provisional_root_ids)
        seen: set[str] = set()
        canonical_ids: set[str] = set()
        for root in canonical_roots_raw:
            if not isinstance(root, dict):
                errors.append(f"Invalid canonical root entry: {root}")
                continue
            canonical_root_id = str(root.get("canonical_root_id") or root.get("root_id") or "").strip()
            if not canonical_root_id:
                errors.append(f"Missing canonical_root_id: {root}")
            elif canonical_root_id in canonical_ids:
                errors.append(f"Duplicate canonical_root_id: {canonical_root_id}")
            canonical_ids.add(canonical_root_id)
            if not str(root.get("deliverable") or "").strip():
                errors.append(f"Missing deliverable for canonical root {canonical_root_id}")
            if not str(root.get("success_criteria") or "").strip():
                errors.append(f"Missing success_criteria for canonical root {canonical_root_id}")

            member_root_ids = root.get("member_root_ids")
            if not isinstance(member_root_ids, list) or not member_root_ids:
                errors.append(f"member_root_ids missing/empty for canonical root {canonical_root_id}")
                continue
            for member_root_id in member_root_ids:
                member_root_id = str(member_root_id).strip()
                if member_root_id not in expected:
                    errors.append(f"Unknown provisional root id in member_root_ids: {member_root_id}")
                if member_root_id in seen:
                    errors.append(f"Duplicate provisional root assignment: {member_root_id}")
                seen.add(member_root_id)

        missing = sorted(expected - seen)
        if missing:
            errors.append(f"Missing provisional roots from consolidation: {missing}")
        return errors

    def _canonical_merge_review_summary_payload(self, root_payload: Dict[str, Any]) -> Dict[str, Any]:
        provisional_roots = [
            root for root in root_payload.get("provisional_roots", [])
            if isinstance(root, dict)
        ]
        return {
            "canonical_root_id": root_payload.get("canonical_root_id"),
            "label": root_payload.get("label"),
            "objective": root_payload.get("objective"),
            "deliverable": root_payload.get("deliverable"),
            "success_criteria": root_payload.get("success_criteria"),
            "summary": root_payload.get("summary"),
            "last_update": root_payload.get("last_update"),
            "anchor": root_payload.get("anchor", [])[:5],
            "leaf_count": root_payload.get("leaf_count"),
            "leaf_span": [root_payload.get("first_leaf_idx"), root_payload.get("last_leaf_idx")],
            "episode_ranges": [
                [episode.get("start_leaf_idx"), episode.get("end_leaf_idx")]
                for episode in root_payload.get("episodes", [])
                if isinstance(episode, dict)
            ],
            "apps_used": root_payload.get("apps_used", [])[:6],
            "entities": root_payload.get("entities", [])[:10],
            "sample_leaf_objectives": _representative_text_samples(
                _normalize_text_list(root_payload.get("sample_leaf_objectives")),
                6,
            ),
            "provisional_roots": [
                {
                    "root_id": item.get("root_id"),
                    "label": item.get("label"),
                    "objective": item.get("objective"),
                    "deliverable": item.get("deliverable"),
                    "summary": item.get("summary"),
                    "last_update": item.get("last_update"),
                    "anchor": item.get("anchor", [])[:5],
                    "episode_ranges": item.get("episode_ranges", []),
                    "sample_leaf_objectives": _representative_text_samples(
                        _normalize_text_list(item.get("sample_leaf_objectives")),
                        4,
                    ),
                }
                for item in provisional_roots[:4]
            ],
        }

    def _validate_canonical_merge_review_output(
        self,
        data: Dict[str, Any],
        canonical_root_ids: List[str],
    ) -> List[str]:
        errors: List[str] = []
        canonical_roots_raw = data.get("canonical_roots")
        if not isinstance(canonical_roots_raw, list):
            return ["canonical_roots must be a list."]
        if not canonical_roots_raw:
            return ["canonical_roots is empty."]

        expected = set(canonical_root_ids)
        seen: set[str] = set()
        merged_ids: set[str] = set()
        for root in canonical_roots_raw:
            if not isinstance(root, dict):
                errors.append(f"Invalid canonical root entry: {root}")
                continue
            merged_root_id = str(root.get("canonical_root_id") or root.get("root_id") or "").strip()
            if not merged_root_id:
                errors.append(f"Missing canonical_root_id: {root}")
            elif merged_root_id in merged_ids:
                errors.append(f"Duplicate canonical_root_id: {merged_root_id}")
            merged_ids.add(merged_root_id)
            if not str(root.get("deliverable") or "").strip():
                errors.append(f"Missing deliverable for merged canonical root {merged_root_id}")
            if not str(root.get("success_criteria") or "").strip():
                errors.append(f"Missing success_criteria for merged canonical root {merged_root_id}")

            member_root_ids = root.get("member_canonical_root_ids")
            if not isinstance(member_root_ids, list) or not member_root_ids:
                errors.append(f"member_canonical_root_ids missing/empty for canonical root {merged_root_id}")
                continue
            for member_root_id in member_root_ids:
                member_root_id = str(member_root_id).strip()
                if member_root_id not in expected:
                    errors.append(f"Unknown canonical root id in member_canonical_root_ids: {member_root_id}")
                if member_root_id in seen:
                    errors.append(f"Duplicate canonical root assignment: {member_root_id}")
                seen.add(member_root_id)

        missing = sorted(expected - seen)
        if missing:
            errors.append(f"Missing canonical roots from merge review: {missing}")
        return errors

    def _heuristic_consolidation(
        self,
        roots_by_id: Dict[str, ProvisionalRootIR],
    ) -> Dict[str, Any]:
        ordered = sorted(roots_by_id.values(), key=lambda root: root.first_leaf_idx)
        canonical_roots: List[Dict[str, Any]] = []
        for idx, root in enumerate(ordered, start=1):
            canonical_roots.append(
                {
                    "canonical_root_id": f"C{idx}",
                    "label": root.label or root.objective or root.root_id,
                    "objective": root.objective or root.label or root.root_id,
                    "deliverable": root.deliverable,
                    "success_criteria": root.success_criteria,
                    "member_root_ids": [root.root_id],
                }
            )
        return {"canonical_roots": canonical_roots}

    def _review_canonical_root_merges(
        self,
        canonical_roots: List[CanonicalRootIR],
        roots_by_id: Dict[str, ProvisionalRootIR],
        leaves: List[LeafLatentTaskRecord],
        provisional_assignments: Dict[int, str],
    ) -> List[CanonicalRootIR]:
        if len(canonical_roots) <= 1:
            return canonical_roots

        provisional_to_canonical: Dict[str, str] = {}
        canonical_root_by_id: Dict[str, CanonicalRootIR] = {}
        for root in canonical_roots:
            canonical_root_by_id[root.canonical_root_id] = root
            for provisional_root_id in root.member_root_ids:
                provisional_to_canonical[provisional_root_id] = root.canonical_root_id

        canonical_leaf_indices: Dict[str, List[int]] = {root.canonical_root_id: [] for root in canonical_roots}
        for leaf_idx, provisional_root_id in sorted(provisional_assignments.items()):
            canonical_root_id = provisional_to_canonical.get(provisional_root_id, "")
            if canonical_root_id:
                canonical_leaf_indices.setdefault(canonical_root_id, []).append(leaf_idx)

        root_payloads: List[Dict[str, Any]] = []
        for root in canonical_roots:
            leaf_indices = sorted(canonical_leaf_indices.get(root.canonical_root_id, []))
            if not leaf_indices:
                continue
            root_payloads.append(self._build_root_payload(root, roots_by_id, leaf_indices, leaves))
        if len(root_payloads) <= 1:
            return canonical_roots

        print("\n" + "=" * 60)
        print("ROOT CONSOLIDATION: Canonical Merge Review")
        print("=" * 60)

        review_text = "\n".join(
            json.dumps(self._canonical_merge_review_summary_payload(root_payload), ensure_ascii=False)
            for root_payload in root_payloads
        )
        canonical_root_ids = [str(root_payload.get("canonical_root_id") or "").strip() for root_payload in root_payloads]

        try:
            response = call_llm(
                prompt=CANONICAL_ROOT_MERGE_REVIEW_PROMPT.format(canonical_roots=review_text),
                content=[],
                model_name=self.model_name,
                timeout_secs=self.llm_timeout_secs,
                operation="canonical_merge_review",
            )
            data = extract_json_from_response(response)
            errors = self._validate_canonical_merge_review_output(data, canonical_root_ids)
            if errors:
                print(f"  [Retry] Canonical merge review had errors: {'; '.join(errors)}")
                retry_response = call_llm(
                    prompt=CANONICAL_ROOT_MERGE_RETRY_PROMPT.format(
                        canonical_roots=review_text,
                        previous_output=json.dumps(data, indent=2, ensure_ascii=False),
                        errors="\n".join(f"- {error}" for error in errors),
                    ),
                    content=[],
                    model_name=self.model_name,
                    timeout_secs=self.llm_timeout_secs,
                    operation="canonical_merge_review_retry",
                )
                retry_data = extract_json_from_response(retry_response)
                retry_errors = self._validate_canonical_merge_review_output(retry_data, canonical_root_ids)
                if retry_errors:
                    print(f"  [Retry] Canonical merge review still invalid: {'; '.join(retry_errors)}. Keeping first-pass roots.")
                    return canonical_roots
                print("  [Retry] Canonical merge review output accepted.")
                data = retry_data
        except Exception as e:
            print(f"  [Error] Canonical merge review failed: {e}. Keeping first-pass roots.")
            return canonical_roots

        merged_roots: List[CanonicalRootIR] = []
        for root in data.get("canonical_roots", []):
            if not isinstance(root, dict):
                continue
            member_canonical_root_ids = [
                str(item).strip()
                for item in root.get("member_canonical_root_ids", [])
                if str(item).strip()
            ]
            member_provisional_root_ids: List[str] = []
            for member_canonical_root_id in member_canonical_root_ids:
                member_root = canonical_root_by_id.get(member_canonical_root_id)
                if not member_root:
                    continue
                for provisional_root_id in member_root.member_root_ids:
                    if provisional_root_id and provisional_root_id not in member_provisional_root_ids:
                        member_provisional_root_ids.append(provisional_root_id)
            merged_roots.append(
                CanonicalRootIR(
                    canonical_root_id=str(root.get("canonical_root_id") or root.get("root_id") or "").strip(),
                    label=(root.get("label") or root.get("objective") or "").strip(),
                    objective=(root.get("objective") or root.get("label") or "").strip(),
                    deliverable=(root.get("deliverable") or "").strip(),
                    success_criteria=(root.get("success_criteria") or "").strip(),
                    member_root_ids=member_provisional_root_ids,
                )
            )

        merged_roots.sort(
            key=lambda root: min(
                (roots_by_id[root_id].first_leaf_idx for root_id in root.member_root_ids if root_id in roots_by_id),
                default=10**9,
            )
        )
        print(f"  >> Canonical roots after merge review: {len(merged_roots)}")
        return merged_roots

    def _consolidate_roots(
        self,
        roots_by_id: Dict[str, ProvisionalRootIR],
        leaves: List[LeafLatentTaskRecord],
        provisional_assignments: Dict[int, str],
    ) -> List[CanonicalRootIR]:
        if not roots_by_id:
            return []
        self._report("consolidation", f"consolidating {len(roots_by_id)} provisional roots")
        if len(roots_by_id) == 1:
            only_root = next(iter(roots_by_id.values()))
            return [
                CanonicalRootIR(
                    canonical_root_id="C1",
                    label=only_root.label or only_root.objective or only_root.root_id,
                    objective=only_root.objective or only_root.label or only_root.root_id,
                    deliverable=only_root.deliverable,
                    success_criteria=only_root.success_criteria,
                    member_root_ids=[only_root.root_id],
                )
            ]

        print("\n" + "=" * 60)
        print("ROOT CONSOLIDATION: Merge Duplicate Threads")
        print("=" * 60)

        provisional_roots = self._describe_provisional_roots(roots_by_id)
        provisional_root_ids = list(roots_by_id.keys())
        prompt = ROOT_THREAD_CONSOLIDATION_PROMPT.format(
            provisional_roots=provisional_roots,
        )

        data: Dict[str, Any]
        try:
            response = call_llm(
                prompt=prompt,
                content=[],
                model_name=self.model_name,
                timeout_secs=self.llm_timeout_secs,
                operation="root_consolidation",
            )
            data = extract_json_from_response(response)
            errors = self._validate_consolidation_output(data, provisional_root_ids)
            if errors:
                print(f"  [Retry] Consolidation had errors: {'; '.join(errors)}")
                retry_prompt = ROOT_THREAD_CONSOLIDATION_RETRY_PROMPT.format(
                    provisional_roots=provisional_roots,
                    previous_output=json.dumps(data, indent=2, ensure_ascii=False),
                    errors="\n".join(f"- {error}" for error in errors),
                )
                retry_response = call_llm(
                    prompt=retry_prompt,
                    content=[],
                    model_name=self.model_name,
                    timeout_secs=self.llm_timeout_secs,
                    operation="root_consolidation_retry",
                )
                retry_data = extract_json_from_response(retry_response)
                retry_errors = self._validate_consolidation_output(retry_data, provisional_root_ids)
                if retry_errors:
                    print(f"  [Retry] Consolidation still invalid: {'; '.join(retry_errors)}. Falling back to identity.")
                    data = self._heuristic_consolidation(roots_by_id)
                else:
                    print("  [Retry] Consolidation output accepted.")
                    data = retry_data
        except Exception as e:
            print(f"  [Error] Consolidation failed: {e}. Falling back to identity.")
            data = self._heuristic_consolidation(roots_by_id)

        canonical_roots: List[CanonicalRootIR] = []
        for root in data.get("canonical_roots", []):
            if not isinstance(root, dict):
                continue
            canonical_roots.append(
                CanonicalRootIR(
                    canonical_root_id=str(root.get("canonical_root_id") or root.get("root_id") or "").strip(),
                    label=(root.get("label") or root.get("objective") or "").strip(),
                    objective=(root.get("objective") or root.get("label") or "").strip(),
                    deliverable=(root.get("deliverable") or "").strip(),
                    success_criteria=(root.get("success_criteria") or "").strip(),
                    member_root_ids=[str(item).strip() for item in root.get("member_root_ids", []) if str(item).strip()],
                )
            )
        canonical_roots.sort(
            key=lambda root: min(
                (roots_by_id[root_id].first_leaf_idx for root_id in root.member_root_ids if root_id in roots_by_id),
                default=10**9,
            )
        )
        print(f"  >> Canonical roots before merge review: {len(canonical_roots)}")
        self._report("consolidation", f"merge review — {len(canonical_roots)} first-pass canonical roots")
        merged = self._review_canonical_root_merges(canonical_roots, roots_by_id, leaves, provisional_assignments)
        self._report("consolidation_done", f"{len(merged)} canonical roots")
        return merged

    def _build_root_payload(
        self,
        canonical_root: CanonicalRootIR,
        roots_by_id: Dict[str, ProvisionalRootIR],
        canonical_leaf_indices: List[int],
        leaves: List[LeafLatentTaskRecord],
    ) -> Dict[str, Any]:
        episodes = []
        for episode_idx, (start_leaf_idx, end_leaf_idx, leaf_indices) in enumerate(
            self._episodes_from_leaf_indices(canonical_leaf_indices)
        ):
            episode_leaves = [leaves[idx] for idx in leaf_indices]
            episode_payload = {
                "episode_id": f"{canonical_root.canonical_root_id}.E{episode_idx + 1}",
                "start_leaf_idx": start_leaf_idx,
                "end_leaf_idx": end_leaf_idx,
                "start_node_idx": episode_leaves[0].start_node_idx,
                "end_node_idx": episode_leaves[-1].end_node_idx,
                "leaf_indices": leaf_indices,
                "sample_leaf_objectives": _representative_text_samples(
                    [leaf.objective for leaf in episode_leaves],
                    2,
                ),
            }
            episodes.append(episode_payload)

        apps_used: List[str] = []
        entities: List[str] = []
        sample_leaf_objectives: List[str] = []
        anchor: List[str] = []
        member_summaries: List[str] = []
        member_last_updates: List[str] = []
        for root_id in canonical_root.member_root_ids:
            provisional_root = roots_by_id.get(root_id)
            if not provisional_root:
                continue
            member_summaries.append(provisional_root.summary)
            member_last_updates.append(provisional_root.last_update)
            for item in provisional_root.anchor:
                if item and item not in anchor and len(anchor) < 8:
                    anchor.append(item)
        for leaf_idx in canonical_leaf_indices:
            leaf = leaves[leaf_idx]
            for app in leaf.apps_used:
                if app and app not in apps_used:
                    apps_used.append(app)
            for entity in leaf.entities:
                if entity and entity not in entities:
                    entities.append(entity)
            if leaf.objective and leaf.objective not in sample_leaf_objectives:
                sample_leaf_objectives.append(leaf.objective)

        start_leaf = leaves[canonical_leaf_indices[0]]
        end_leaf = leaves[canonical_leaf_indices[-1]]
        representative_leaf_objectives = _representative_text_samples(sample_leaf_objectives, 8)
        return {
            "canonical_root_id": canonical_root.canonical_root_id,
            "label": canonical_root.label or canonical_root.objective or canonical_root.canonical_root_id,
            "objective": canonical_root.objective or canonical_root.label or canonical_root.canonical_root_id,
            "deliverable": canonical_root.deliverable,
            "success_criteria": canonical_root.success_criteria,
            "summary": _limit_sentences(" ".join(member_summaries), max_sentences=2, max_chars=320),
            "last_update": _limit_sentences(member_last_updates[-1] if member_last_updates else "", max_sentences=1, max_chars=180),
            "anchor": anchor[:5],
            "member_provisional_root_ids": canonical_root.member_root_ids,
            "leaf_indices": canonical_leaf_indices,
            "leaf_count": len(canonical_leaf_indices),
            "first_leaf_idx": canonical_leaf_indices[0],
            "last_leaf_idx": canonical_leaf_indices[-1],
            "start_node_idx": start_leaf.start_node_idx,
            "end_node_idx": end_leaf.end_node_idx,
            "apps_used": apps_used[:8],
            "entities": entities[:12],
            "sample_leaf_objectives": representative_leaf_objectives,
            "episodes": episodes,
            "provisional_roots": [
                self._root_summary_payload(roots_by_id[root_id])
                for root_id in canonical_root.member_root_ids
                if root_id in roots_by_id
            ],
        }

    def _build_task_thread_payload(
        self,
        leaves: List[LeafLatentTaskRecord],
        roots_by_id: Dict[str, ProvisionalRootIR],
        provisional_assignments: Dict[int, str],
        canonical_roots: List[CanonicalRootIR],
        canonical_assignment_overrides: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        provisional_to_canonical: Dict[str, str] = {}
        canonical_root_by_id: Dict[str, CanonicalRootIR] = {}
        for root in canonical_roots:
            canonical_root_by_id[root.canonical_root_id] = root
            for provisional_root_id in root.member_root_ids:
                provisional_to_canonical[provisional_root_id] = root.canonical_root_id

        canonical_leaf_indices: Dict[str, List[int]] = {root.canonical_root_id: [] for root in canonical_roots}
        for leaf_idx, provisional_root_id in sorted(provisional_assignments.items()):
            canonical_root_id = ""
            if canonical_assignment_overrides and leaf_idx in canonical_assignment_overrides:
                canonical_root_id = canonical_assignment_overrides[leaf_idx]
            else:
                canonical_root_id = provisional_to_canonical.get(provisional_root_id, "")
            if canonical_root_id is None:
                continue
            canonical_leaf_indices.setdefault(canonical_root_id, []).append(leaf_idx)

        roots_payload: List[Dict[str, Any]] = []
        episode_lookup: Dict[Tuple[str, int], str] = {}
        for root in canonical_roots:
            leaf_indices = sorted(canonical_leaf_indices.get(root.canonical_root_id, []))
            if not leaf_indices:
                continue
            root_payload = self._build_root_payload(root, roots_by_id, leaf_indices, leaves)
            roots_payload.append(root_payload)
            for episode in root_payload["episodes"]:
                for leaf_idx in episode["leaf_indices"]:
                    episode_lookup[(root.canonical_root_id, leaf_idx)] = episode["episode_id"]

        leaf_assignments_payload: List[Dict[str, Any]] = []
        for leaf in leaves:
            provisional_root_id = provisional_assignments.get(leaf.leaf_idx, "")
            if canonical_assignment_overrides and leaf.leaf_idx in canonical_assignment_overrides:
                canonical_root_id = canonical_assignment_overrides[leaf.leaf_idx]
            else:
                canonical_root_id = provisional_to_canonical.get(provisional_root_id, "")
            canonical_root = canonical_root_by_id.get(canonical_root_id)
            leaf_assignments_payload.append(
                {
                    "leaf_idx": leaf.leaf_idx,
                    "start_node_idx": leaf.start_node_idx,
                    "end_node_idx": leaf.end_node_idx,
                    "objective": leaf.objective,
                    "procedure_summary": leaf.procedure_summary,
                    "provisional_root_id": provisional_root_id,
                    "canonical_root_id": canonical_root_id,
                    "canonical_root_label": (
                        canonical_root.label or canonical_root.objective or canonical_root_id
                        if canonical_root
                        else ""
                    ),
                    "episode_id": episode_lookup.get((canonical_root_id, leaf.leaf_idx), ""),
                }
            )

        return {
            "meta": {
                "created_at_unix": time.time(),
                "model": self.model_name,
                "pipeline_version": self.CACHE_PIPELINE_VERSION,
                "num_leaf_latent_tasks": len(leaves),
                "num_provisional_roots": len(roots_by_id),
                "num_canonical_roots": len(roots_payload),
                "discovery_batch_size": self.DISCOVERY_BATCH_SIZE,
                "max_recent_assignments": self.MAX_RECENT_ASSIGNMENTS,
            },
            "roots": roots_payload,
            "leaf_assignments": leaf_assignments_payload,
        }

    def _root_profile_from_payload(self, root_payload: Dict[str, Any]) -> Dict[str, Any]:
        parts = [
            f'label: {root_payload.get("label", "")}',
            f'objective: {root_payload.get("objective", "")}',
            f'deliverable: {root_payload.get("deliverable", "")}',
            f'success_criteria: {root_payload.get("success_criteria", "")}',
            f'summary: {root_payload.get("summary", "")}',
            f'last_update: {root_payload.get("last_update", "")}',
            "anchor: " + ", ".join(_normalize_text_list(root_payload.get("anchor"))),
            "apps_used: " + ", ".join(_normalize_text_list(root_payload.get("apps_used"))),
            "entities: " + ", ".join(_normalize_text_list(root_payload.get("entities"))),
            "sample_leaf_objectives: "
            + " | ".join(_normalize_text_list(root_payload.get("sample_leaf_objectives"))),
        ]
        return {
            "text": "\n".join(part for part in parts if _normalize_text(part)),
            "apps": _normalized_item_set(root_payload.get("apps_used")),
            "entities": _normalized_item_set(root_payload.get("entities")),
            "anchor": _normalized_item_set(root_payload.get("anchor")),
        }

    def _select_reassignment_candidates(
        self,
        leaves: List[LeafLatentTaskRecord],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        roots = [root for root in payload.get("roots", []) if isinstance(root, dict)]
        if len(roots) <= 1:
            return []

        root_by_id = {root.get("canonical_root_id", ""): root for root in roots}
        root_profiles = {
            root_id: self._root_profile_from_payload(root)
            for root_id, root in root_by_id.items()
            if root_id
        }
        small_root_ids = {
            root_id
            for root_id, root in root_by_id.items()
            if _coerce_nonnegative_int(root.get("leaf_count")) <= 12
        }
        if not small_root_ids:
            return []

        candidates: List[Dict[str, Any]] = []
        for assignment in payload.get("leaf_assignments", []):
            if not isinstance(assignment, dict):
                continue
            leaf_idx = _coerce_nonnegative_int(assignment.get("leaf_idx"), default=-1)
            current_root_id = str(assignment.get("canonical_root_id") or "").strip()
            if leaf_idx < 0 or leaf_idx >= len(leaves) or current_root_id not in root_by_id:
                continue
            current_root = root_by_id[current_root_id]
            if _coerce_nonnegative_int(current_root.get("leaf_count")) <= 12:
                continue

            leaf = leaves[leaf_idx]
            leaf_profile = leaf.semantic_profile()
            leaf_apps = _normalized_item_set(leaf.apps_used)
            leaf_entities = _normalized_item_set(leaf.entities)
            leaf_anchor = _normalized_item_set(_infer_anchor_from_leaf(leaf, max_items=5))
            current_profile = root_profiles.get(current_root_id, {})
            current_score = 3.0 * _text_similarity(leaf_profile, current_profile.get("text", ""))
            current_score += 0.8 * len(leaf_apps & current_profile.get("apps", set()))
            current_score += 0.8 * len(leaf_entities & current_profile.get("entities", set()))
            current_score += 1.2 * len(leaf_anchor & current_profile.get("anchor", set()))
            alternatives: List[Tuple[str, float]] = []
            for alt_root_id in sorted(small_root_ids):
                if alt_root_id == current_root_id:
                    continue
                alt_profile = root_profiles.get(alt_root_id, {})
                alt_apps_overlap = len(leaf_apps & alt_profile.get("apps", set()))
                alt_entities_overlap = len(leaf_entities & alt_profile.get("entities", set()))
                alt_anchor_overlap = len(leaf_anchor & alt_profile.get("anchor", set()))
                alt_similarity = _text_similarity(leaf_profile, alt_profile.get("text", ""))
                alt_score = 3.0 * alt_similarity
                alt_score += 0.8 * alt_apps_overlap
                alt_score += 0.8 * alt_entities_overlap
                alt_score += 1.2 * alt_anchor_overlap
                if alt_score < current_score + 0.35:
                    continue
                if alt_similarity < 0.33 and alt_apps_overlap == 0 and alt_entities_overlap == 0 and alt_anchor_overlap == 0:
                    continue
                alternatives.append((alt_root_id, alt_score))
            if not alternatives:
                continue
            alternatives.sort(key=lambda item: item[1], reverse=True)
            candidates.append(
                {
                    "leaf_idx": leaf.leaf_idx,
                    "current_canonical_root_id": current_root_id,
                    "current_canonical_root_label": current_root.get("label", current_root_id),
                    "possible_alternatives": [
                        {
                            "canonical_root_id": alt_root_id,
                            "label": root_by_id[alt_root_id].get("label", alt_root_id),
                        }
                        for alt_root_id, _ in alternatives[:2]
                    ],
                    "leaf_summary": {
                        "leaf_idx": leaf.leaf_idx,
                        "objective": leaf.objective,
                        "procedure_summary": leaf.procedure_summary,
                        "apps_used": leaf.apps_used[:4],
                        "entities": leaf.entities[:6],
                    },
                }
            )
        return candidates[:20]

    def _validate_reassignment_output(
        self,
        data: Dict[str, Any],
        candidate_leaf_indices: List[int],
        canonical_root_ids: List[str],
    ) -> List[str]:
        assignments = data.get("assignments")
        if not isinstance(assignments, list):
            return ["assignments must be a list."]
        errors: List[str] = []
        expected = set(candidate_leaf_indices)
        seen: set[int] = set()
        valid_roots = set(canonical_root_ids)
        for assignment in assignments:
            if not isinstance(assignment, dict):
                errors.append(f"Invalid reassignment entry: {assignment}")
                continue
            leaf_idx = _coerce_nonnegative_int(assignment.get("leaf_idx"), default=-1)
            root_id = str(
                assignment.get("assigned_canonical_root_id")
                or assignment.get("canonical_root_id")
                or assignment.get("root_id")
                or ""
            ).strip()
            if leaf_idx not in expected:
                errors.append(f"Assignment leaf_idx out of candidate set: {leaf_idx}")
            if leaf_idx in seen:
                errors.append(f"Duplicate reassignment for leaf_idx={leaf_idx}")
            seen.add(leaf_idx)
            if root_id not in valid_roots:
                errors.append(f"Unknown canonical root id: {root_id}")
        missing = sorted(expected - seen)
        if missing:
            errors.append(f"Missing reassignment decisions for leaf indices: {missing}")
        return errors

    def _review_candidate_reassignments(
        self,
        leaves: List[LeafLatentTaskRecord],
        payload: Dict[str, Any],
    ) -> Dict[int, str]:
        candidates = self._select_reassignment_candidates(leaves, payload)
        if not candidates:
            return {}

        print("\n" + "=" * 60)
        print("ROOT REVIEW: Candidate Leaf Reassignment")
        print("=" * 60)
        print(f"  >> Reviewing {len(candidates)} ambiguous leaves against canonical roots.")

        canonical_roots = [root for root in payload.get("roots", []) if isinstance(root, dict)]
        canonical_root_ids = [str(root.get("canonical_root_id") or "").strip() for root in canonical_roots]
        canonical_roots_text = "\n".join(
            json.dumps(
                {
                    "canonical_root_id": root.get("canonical_root_id"),
                    "label": root.get("label"),
                    "objective": root.get("objective"),
                    "deliverable": root.get("deliverable"),
                    "sample_leaf_objectives": root.get("sample_leaf_objectives", [])[:6],
                },
                ensure_ascii=False,
            )
            for root in canonical_roots
        )
        candidate_text = "\n".join(json.dumps(candidate, ensure_ascii=False) for candidate in candidates)

        prompt = ROOT_REASSIGNMENT_REVIEW_PROMPT.format(
            canonical_roots=canonical_roots_text,
            candidate_leaves=candidate_text,
        )
        candidate_leaf_indices = [int(candidate["leaf_idx"]) for candidate in candidates]

        try:
            response = call_llm(
                prompt=prompt,
                content=[],
                model_name=self.model_name,
                timeout_secs=self.llm_timeout_secs,
                operation="reassignment_review",
            )
            data = extract_json_from_response(response)
            errors = self._validate_reassignment_output(data, candidate_leaf_indices, canonical_root_ids)
            if errors:
                print(f"  [Retry] Reassignment review had errors: {'; '.join(errors)}")
                retry_prompt = ROOT_REASSIGNMENT_RETRY_PROMPT.format(
                    canonical_roots=canonical_roots_text,
                    candidate_leaves=candidate_text,
                    previous_output=json.dumps(data, indent=2, ensure_ascii=False),
                    errors="\n".join(f"- {error}" for error in errors),
                )
                retry_response = call_llm(
                    prompt=retry_prompt,
                    content=[],
                    model_name=self.model_name,
                    timeout_secs=self.llm_timeout_secs,
                    operation="reassignment_review_retry",
                )
                retry_data = extract_json_from_response(retry_response)
                retry_errors = self._validate_reassignment_output(
                    retry_data,
                    candidate_leaf_indices,
                    canonical_root_ids,
                )
                if retry_errors:
                    print(f"  [Retry] Reassignment review still invalid: {'; '.join(retry_errors)}. Skipping.")
                    return {}
                data = retry_data
        except Exception as e:
            print(f"  [Error] Reassignment review failed: {e}. Skipping.")
            return {}

        overrides: Dict[int, str] = {}
        for assignment in data.get("assignments", []):
            if not isinstance(assignment, dict):
                continue
            leaf_idx = _coerce_nonnegative_int(assignment.get("leaf_idx"), default=-1)
            root_id = str(
                assignment.get("assigned_canonical_root_id")
                or assignment.get("canonical_root_id")
                or assignment.get("root_id")
                or ""
            ).strip()
            if leaf_idx >= 0 and root_id:
                overrides[leaf_idx] = root_id
                print(f"  >> leaf {leaf_idx} reassigned review target: {root_id}")
        return overrides

    def _compute_micro_root_absorptions(
        self,
        leaves: List[LeafLatentTaskRecord],
        payload: Dict[str, Any],
    ) -> Dict[int, str]:
        roots = [
            root for root in sorted(
                payload.get("roots", []),
                key=lambda item: _coerce_nonnegative_int(item.get("first_leaf_idx"), default=10**9),
            )
            if isinstance(root, dict)
        ]
        overrides: Dict[int, str] = {}
        if len(roots) <= 1:
            return overrides

        print("\n" + "=" * 60)
        print("ROOT CLEANUP: Micro-Root Absorption")
        print("=" * 60)

        for idx, root in enumerate(roots):
            if _coerce_nonnegative_int(root.get("leaf_count")) != 1:
                continue
            leaf_indices = root.get("leaf_indices") or []
            if len(leaf_indices) != 1:
                continue
            leaf_idx = _coerce_nonnegative_int(leaf_indices[0], default=-1)
            if leaf_idx < 0 or leaf_idx >= len(leaves):
                continue
            leaf = leaves[leaf_idx]
            if leaf.num_actions > 3:
                continue

            candidate_targets: List[Tuple[int, int, int, Dict[str, Any]]] = []
            if idx > 0:
                prev_root = roots[idx - 1]
                dist = leaf_idx - _coerce_nonnegative_int(prev_root.get("last_leaf_idx"), default=-10**9)
                if 0 <= dist <= 2:
                    candidate_targets.append(
                        (
                            dist,
                            _coerce_nonnegative_int(prev_root.get("leaf_count"), default=10**9),
                            0,
                            prev_root,
                        )
                    )
            if idx + 1 < len(roots):
                next_root = roots[idx + 1]
                dist = _coerce_nonnegative_int(next_root.get("first_leaf_idx"), default=10**9) - leaf_idx
                if 0 <= dist <= 2:
                    candidate_targets.append(
                        (
                            dist,
                            _coerce_nonnegative_int(next_root.get("leaf_count"), default=10**9),
                            1,
                            next_root,
                        )
                    )
            if not candidate_targets:
                continue

            candidate_targets.sort(key=lambda item: (item[0], item[1], item[2]))
            chosen_root = candidate_targets[0][3]
            chosen_root_id = str(chosen_root.get("canonical_root_id") or "").strip()
            current_root_id = str(root.get("canonical_root_id") or "").strip()
            if not chosen_root_id or chosen_root_id == current_root_id:
                continue
            overrides[leaf_idx] = chosen_root_id
            print(
                f"  >> Absorbing micro-root {current_root_id} leaf {leaf_idx} into {chosen_root_id} "
                f'("{chosen_root.get("label", chosen_root_id)}")'
            )
        return overrides

    def _print_root_summary(self, payload: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("FINAL ROOT SUMMARY")
        print("=" * 60)
        for root in payload.get("roots", []):
            if not isinstance(root, dict):
                continue
            episodes = root.get("episodes") or []
            episode_ranges = ", ".join(
                f'{episode.get("start_leaf_idx")}-{episode.get("end_leaf_idx")}'
                for episode in episodes
                if isinstance(episode, dict)
            )
            print(
                f'  {root.get("canonical_root_id", "")}: '
                f'{root.get("label", "")} | leaves={root.get("leaf_count", 0)} | '
                f'episodes=[{episode_ranges}]'
            )
            sample_objectives = root.get("sample_leaf_objectives") or []
            if sample_objectives:
                print(
                    f'    samples: {", ".join(_truncate_text(str(item), 100) for item in sample_objectives[:4])}'
                )

    def _to_output_payload(
        self,
        *,
        payload: Dict[str, Any],
        leaves: List[LeafLatentTaskRecord],
        input_path: str | Path,
        output_path: str | Path,
        reused_cache: bool = False,
        preflight_only: bool = False,
    ) -> Dict[str, Any]:
        leaf_by_idx = {leaf.leaf_idx: leaf for leaf in leaves}
        roots: List[Dict[str, Any]] = []
        for root in payload.get("roots", []):
            if not isinstance(root, dict):
                continue
            leaf_indices = [
                int(idx)
                for idx in root.get("leaf_indices", [])
                if isinstance(idx, int) or str(idx).isdigit()
            ]
            root_leaves = [leaf_by_idx[idx] for idx in leaf_indices if idx in leaf_by_idx]
            semantic_action_indices: List[int] = []
            raw_action_ids: List[str] = []
            activity_ids: List[str] = []
            semantic_action_ids: List[str] = []
            observed_applications: List[str] = []
            event_count = 0
            for leaf in root_leaves:
                activity_ids.append(leaf.activity_id)
                semantic_action_indices.extend(range(leaf.start_semantic_action_idx, leaf.end_semantic_action_idx + 1))
                raw_action_ids.extend(leaf.raw_action_ids or _available_boundary_ids(leaf.start_action_id, leaf.end_action_id))
                semantic_action_ids.extend(leaf.semantic_action_ids or _normalize_text_list(leaf.start_semantic_action_id))
                event_count += leaf.event_count
                for app in leaf.apps_used:
                    if app and app not in observed_applications:
                        observed_applications.append(app)
            semantic_action_indices = sorted(set(semantic_action_indices))

            roots.append(
                {
                    "canonical_root_id": root.get("canonical_root_id") or "",
                    "label": root.get("label") or root.get("objective") or "",
                    "objective": root.get("objective") or root.get("label") or "",
                    "deliverable": root.get("deliverable") or "",
                    "success_criteria": root.get("success_criteria") or "",
                    "summary": root.get("summary") or "",
                    "last_update": root.get("last_update") or "",
                    "anchor": _concise_anchor_items(root.get("anchor"), max_items=5),
                    "member_provisional_root_ids": _normalize_text_list(root.get("member_provisional_root_ids")),
                    "activity_id": _normalize_text_list(activity_ids),
                    "semantic_action_id": _normalize_text_list(semantic_action_ids),
                    "raw_action_id": _normalize_text_list(raw_action_ids),
                    "semantic_action_count": len(semantic_action_indices),
                    "observed_applications": observed_applications,
                    "provisional_roots": root.get("provisional_roots", []),
                    "event_count": event_count,
                }
            )

        semantic_action_assignments: List[Dict[str, Any]] = []
        for assignment in payload.get("leaf_assignments", []):
            if not isinstance(assignment, dict):
                continue
            leaf_idx = _coerce_nonnegative_int(assignment.get("leaf_idx"), default=-1)
            leaf = leaf_by_idx.get(leaf_idx)
            if leaf is None:
                continue
            indices = list(range(leaf.start_semantic_action_idx, leaf.end_semantic_action_idx + 1))
            ids = leaf.semantic_action_ids or _normalize_text_list(leaf.start_semantic_action_id)
            texts = leaf.semantic_actions or [leaf.objective]
            for offset, _semantic_action_idx in enumerate(indices):
                semantic_action_assignments.append(
                    {
                        "semantic_action_id": ids[offset] if offset < len(ids) else (ids[0] if ids else ""),
                        "raw_action_id": leaf.raw_action_ids or _available_boundary_ids(leaf.start_action_id, leaf.end_action_id),
                        "semantic_action": texts[offset] if offset < len(texts) else leaf.objective,
                        "provisional_root_id": assignment.get("provisional_root_id") or "",
                        "canonical_root_id": assignment.get("canonical_root_id") or "",
                        "canonical_root_label": assignment.get("canonical_root_label") or "",
                    }
                )

        return {
            "meta": {
                "created_at": utc_now_iso(),
                "model": self.model_name,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "num_semantic_actions": sum(leaf.semantic_action_count for leaf in leaves),
                "num_provisional_roots": int(payload.get("meta", {}).get("num_provisional_roots", 0)),
                "num_canonical_roots": len(roots),
                "discovery_batch_size": self.DISCOVERY_BATCH_SIZE,
                "max_recent_assignments": self.MAX_RECENT_ASSIGNMENTS,
                "pipeline_version": self.CACHE_PIPELINE_VERSION,
                "reused_cache": reused_cache,
                "preflight_only": preflight_only,
            },
            "roots": roots,
            "semantic_action_assignments": semantic_action_assignments,
        }

    def _load_activity_rows(self, input_path: str | Path) -> Dict[str, Dict[str, Any]]:
        rows_by_segment: Dict[str, Dict[str, Any]] = {}
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {input_path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    continue
                segment_id = str(row.get("activity_id") or "").strip()
                if segment_id:
                    rows_by_segment[segment_id] = row
        return rows_by_segment

    def _write_derived_task_thread_objectives(
        self,
        *,
        output_payload: Dict[str, Any],
        activity_path: str | Path,
        task_threads_path: str | Path,
        derived_objectives_dir: str | Path,
    ) -> Dict[str, Any]:
        rows_by_segment = self._load_activity_rows(activity_path)
        output_dir = Path(derived_objectives_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        roots_manifest: List[Dict[str, Any]] = []

        for root in output_payload.get("roots", []):
            if not isinstance(root, dict):
                continue
            canonical_root_id = str(root.get("canonical_root_id") or "").strip()
            if not canonical_root_id:
                continue
            label = str(root.get("label") or root.get("objective") or canonical_root_id).strip()
            segment_ids = _normalize_text_list(root.get("activity_id"))
            activities = [rows_by_segment[segment_id] for segment_id in segment_ids if segment_id in rows_by_segment]
            missing_ids = [segment_id for segment_id in segment_ids if segment_id not in rows_by_segment]
            objective = str(root.get("objective") or label).strip()
            deliverable = str(root.get("deliverable") or "").strip()
            success_criterion = str(root.get("success_criteria") or "").strip()
            evidence_refs = _normalize_text_list(
                segment_ids
                + _normalize_text_list(root.get("semantic_action_id"))
                + _normalize_text_list(root.get("raw_action_id"))
            )
            grounding_target = deliverable or objective
            expected_state = success_criterion or f"The {grounding_target} is complete."
            observed_outcome = root.get("observed_outcome")
            if not isinstance(observed_outcome, dict):
                observed_outcome = {
                    "status": "unknown",
                    "description": "The trace advances this objective, but completion was not independently verified.",
                    "evidence_refs": [],
                }
            filename = f"{canonical_root_id}_{_slugify_filename(label, fallback='task_thread')}.json"
            root_path = (output_dir / filename).resolve()
            root_payload = {
                "canonical_root_id": canonical_root_id,
                "label": label,
                "task_thread_objective": objective,
                # Keep the legacy strings so older Step 4 readers continue to work.
                "deliverable": deliverable,
                "success_criteria": success_criterion,
                "summary": root.get("summary") or "",
                # This is the lossless bridge to the recursive grounding contract.
                "objective_grounding": {
                    "deliverables": [
                        {
                            "kind": "artifact_or_state",
                            "target": grounding_target,
                            "expected_state": expected_state,
                            "evidence_refs": evidence_refs,
                        }
                    ],
                    "success_criteria": [
                        {
                            "predicate": expected_state,
                            "verifier": "unknown",
                            "evidence_refs": evidence_refs,
                            "confidence": None,
                        }
                    ],
                    "observed_outcome": observed_outcome,
                    "evidence_refs": evidence_refs,
                },
                "activities": activities,
            }
            write_json_atomic(root_path, root_payload)
            roots_manifest.append(
                {
                    "canonical_root_id": canonical_root_id,
                    "label": label,
                    "file": str(root_path),
                    "activity_count": len(activities),
                    "missing_count": len(missing_ids),
                    "missing_activity_id": missing_ids,
                }
            )

        manifest = {
            "source_task_model": str(Path(task_threads_path).resolve()),
            "source_activity": str(Path(activity_path).resolve()),
            "roots": roots_manifest,
        }
        write_json_atomic(output_dir / "manifest.json", manifest)
        print(f"  >> Derived task-thread objectives saved: {output_dir}")
        return manifest

    def process(
        self,
        *,
        leaf_tasks_path: str,
        output_path: Optional[str] = None,
        derived_objectives_dir: Optional[str] = None,
        reuse_cache: bool = False,
        save_cache: bool = True,
    ) -> Dict[str, Any]:
        global _ACTIVE_STATS
        previous_stats = _ACTIVE_STATS
        _ACTIVE_STATS = self.stats
        print("\n" + "=" * 60)
        print("TASK THREAD INDUCTION: Multi-Root Thread Tracking")
        print("=" * 60)

        try:
            if reuse_cache and output_path and os.path.exists(output_path):
                print(f"  >> Loading cached task threads from: {output_path}")
                try:
                    cached_output = TaskThreadsInductionOutput.model_validate(
                        self._load_cache(output_path)
                    )
                    if cached_output.meta.pipeline_version != self.CACHE_PIPELINE_VERSION:
                        raise ValueError(
                            f"pipeline version {cached_output.meta.pipeline_version} does not include "
                            f"the current evidence contract v{self.CACHE_PIPELINE_VERSION}"
                        )
                    payload = cached_output.model_dump()
                except Exception as exc:
                    print(f"  >> Ignoring stale task-thread cache without grounded roots: {exc}")
                else:
                    self._print_root_summary(payload)
                    derived_dir = derived_objectives_dir or str(Path(output_path).parent / DEFAULT_DERIVED_OBJECTIVES_DIR_NAME)
                    self._write_derived_task_thread_objectives(
                        output_payload=payload,
                        activity_path=leaf_tasks_path,
                        task_threads_path=output_path,
                        derived_objectives_dir=derived_dir,
                    )
                    return payload

            leaves = self._load_leaf_tasks(leaf_tasks_path)
            if not leaves:
                raise ValueError(f"No activities found in {leaf_tasks_path}")
            print(f"  >> Loaded {len(leaves)} activities from {leaf_tasks_path}")

            roots_by_id, provisional_assignments = self._discover_provisional_roots(leaves)
            canonical_roots = self._consolidate_roots(roots_by_id, leaves, provisional_assignments)
            payload = self._build_task_thread_payload(
                leaves,
                roots_by_id,
                provisional_assignments,
                canonical_roots,
            )
            micro_root_overrides = self._compute_micro_root_absorptions(leaves, payload)
            if micro_root_overrides:
                payload = self._build_task_thread_payload(
                    leaves,
                    roots_by_id,
                    provisional_assignments,
                    canonical_roots,
                    canonical_assignment_overrides=micro_root_overrides,
                )
            if self.ENABLE_REASSIGNMENT_REVIEW:
                reassignment_overrides = self._review_candidate_reassignments(leaves, payload)
                if reassignment_overrides:
                    merged_overrides = dict(micro_root_overrides)
                    merged_overrides.update(reassignment_overrides)
                    payload = self._build_task_thread_payload(
                        leaves,
                        roots_by_id,
                        provisional_assignments,
                        canonical_roots,
                        canonical_assignment_overrides=merged_overrides,
                    )

            self._print_root_summary(payload)
            output_payload = self._to_output_payload(
                payload=payload,
                leaves=leaves,
                input_path=leaf_tasks_path,
                output_path=output_path or "",
            )
            output_payload.setdefault("meta", {}).update(self.stats.as_meta())
            output_payload = TaskThreadsInductionOutput.model_validate(output_payload).model_dump()

            if output_path and save_cache:
                self._save_cache(output_path, output_payload)
                print(f"  >> Task thread cache saved: {output_path}")
                _algorithm_log(
                    "[cost] task_thread_induction "
                    + json.dumps(output_payload.get("meta", {}).get("cost_breakdown", {}), ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
                derived_dir = derived_objectives_dir or str(Path(output_path).parent / DEFAULT_DERIVED_OBJECTIVES_DIR_NAME)
                self._write_derived_task_thread_objectives(
                    output_payload=output_payload,
                    activity_path=leaf_tasks_path,
                    task_threads_path=output_path,
                    derived_objectives_dir=derived_dir,
                )

            _print_metrics(self.stats)
            return output_payload
        finally:
            _ACTIVE_STATS = previous_stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Induce interleaved user task threads from activity.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing activity.jsonl",
    )
    parser.add_argument("--config", type=str, default=None, help="Task model induction config path.")
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate paths and print planned work without LLM calls.",
    )
    parser.add_argument(
        "--no_console",
        action="store_true",
        help="Suppress elapsed-time and cost progress lines.",
    )
    return parser.parse_args(argv)


def task_thread_induction(
    *,
    data_dir: str | Path,
    input_file_name: str,
    output_file_name: str,
    derived_objectives_dir: str | Path | None = None,
    model: str,
    discovery_batch_size: int = 100,
    max_recent_assignments: int = 10,
    llm_timeout_secs: float = 120.0,
    reuse_cache: bool = False,
    enable_reassignment_review: bool = False,
    preflight_only: bool = False,
    no_console: bool = False,
) -> Optional[Dict[str, Any]]:
    global _PROGRESS_ENABLED
    previous_progress = _PROGRESS_ENABLED
    _PROGRESS_ENABLED = False
    try:
        with TaskThreadsReporter(no_console=no_console) as reporter:
            try:
                reporter.set_metric("model", model)
                reporter.start_stage(STAGE_LOAD_INPUTS, "resolving paths")
                resolved_data_dir = Path(data_dir).expanduser()
                resolved_input = Path(TaskThreadInductionBuilder._resolve_leaf_tasks_path(str(resolved_data_dir), input_file_name))
                resolved_output = Path(output_file_name).expanduser() if output_file_name else resolved_data_dir / DEFAULT_OUTPUT_FILE_NAME
                if not resolved_output.is_absolute():
                    resolved_output = resolved_data_dir / resolved_output

                resolved_derived_objectives_dir = (
                    Path(derived_objectives_dir).expanduser()
                    if derived_objectives_dir
                    else resolved_output.parent / DEFAULT_DERIVED_OBJECTIVES_DIR_NAME
                )
                if not resolved_derived_objectives_dir.is_absolute():
                    resolved_derived_objectives_dir = resolved_output.parent / resolved_derived_objectives_dir

                reporter.add_path("input", resolved_input)
                reporter.add_path("output", resolved_output)
                reporter.add_path("derived", resolved_derived_objectives_dir)
                if not resolved_input.exists():
                    raise FileNotFoundError(f"activity input not found: {resolved_input}")

                stats = RunStats()
                leaves = TaskThreadInductionBuilder(
                    model_name=model,
                    llm_timeout_secs=llm_timeout_secs,
                    stats=stats,
                )._load_leaf_tasks(str(resolved_input))
                reporter.set_counter("activities", len(leaves))
                reporter.set_metric("discovery_batch_size", max(1, discovery_batch_size))
                reporter.set_metric("max_recent_assignments", max(0, max_recent_assignments))
                reporter.finish_stage(STAGE_LOAD_INPUTS, f"loaded {len(leaves)} activities")

                reporter.start_stage(STAGE_PREFLIGHT, "validating planned work")
                if not leaves:
                    raise ValueError(f"No activities found in {resolved_input}")
                reporter.finish_stage(STAGE_PREFLIGHT, "ready")
                if preflight_only:
                    reporter.mark_stage_done(STAGE_ROOT_DISCOVERY, "skipped")
                    reporter.mark_stage_done(STAGE_CONSOLIDATION, "skipped")
                    reporter.mark_stage_done(STAGE_WRITE_OUTPUT, "skipped")
                    reporter.final_success("preflight complete; no LLM calls were made")
                    return None

                builder = TaskThreadInductionBuilder(model_name=model, llm_timeout_secs=llm_timeout_secs, stats=stats)
                builder.DISCOVERY_BATCH_SIZE = max(1, discovery_batch_size)
                builder.MAX_RECENT_ASSIGNMENTS = max(0, max_recent_assignments)
                builder.ENABLE_REASSIGNMENT_REVIEW = bool(enable_reassignment_review)

                total_batches = (len(leaves) + max(1, discovery_batch_size) - 1) // max(1, discovery_batch_size)
                reporter.start_stage(STAGE_ROOT_DISCOVERY, f"0/{total_batches} batches")

                def on_builder_progress(milestone: str, detail: str) -> None:
                    if milestone in ("discovery_batch",):
                        reporter.progress(detail)
                    elif milestone == "consolidation":
                        if reporter.state.active_stage == STAGE_ROOT_DISCOVERY:
                            reporter.finish_stage(STAGE_ROOT_DISCOVERY, detail)
                            reporter.start_stage(STAGE_CONSOLIDATION, detail)
                        else:
                            reporter.progress(detail)
                    elif milestone == "consolidation_done":
                        if reporter.state.active_stage == STAGE_CONSOLIDATION:
                            reporter.finish_stage(STAGE_CONSOLIDATION, detail)
                            reporter.start_stage(STAGE_WRITE_OUTPUT, "saving outputs")
                        else:
                            reporter.progress(detail)

                builder._progress_fn = on_builder_progress

                with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                    payload = builder.process(
                        leaf_tasks_path=str(resolved_input),
                        output_path=str(resolved_output),
                        derived_objectives_dir=str(resolved_derived_objectives_dir),
                        reuse_cache=reuse_cache,
                        save_cache=True,
                    )
                meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                num_threads = len(payload.get("roots", [])) if isinstance(payload, dict) else 0
                reporter.set_counter("provisional_roots", int(meta.get("num_provisional_roots") or 0))
                reporter.set_counter("task_threads", num_threads)
                reporter.set_counter(
                    "semantic_assignments",
                    len(payload.get("semantic_action_assignments", [])) if isinstance(payload, dict) else 0,
                )
                for key in ("llm_requests", "input_tokens", "output_tokens", "total_tokens", "estimated_usd"):
                    reporter.set_metric(key, meta.get(key, 0) or 0)
                # Close any stages still active if callbacks didn't fire (e.g. cache reuse)
                active = reporter.state.active_stage
                provisional = int(meta.get("num_provisional_roots") or 0)
                if active == STAGE_ROOT_DISCOVERY:
                    reporter.finish_stage(STAGE_ROOT_DISCOVERY, f"{provisional} provisional roots")
                    reporter.mark_stage_done(STAGE_CONSOLIDATION, f"{num_threads} canonical roots")
                    reporter.mark_stage_done(STAGE_WRITE_OUTPUT, "saved outputs")
                elif active == STAGE_CONSOLIDATION:
                    reporter.finish_stage(STAGE_CONSOLIDATION, f"{num_threads} canonical roots")
                    reporter.mark_stage_done(STAGE_WRITE_OUTPUT, "saved outputs")
                elif active == STAGE_WRITE_OUTPUT:
                    reporter.finish_stage(STAGE_WRITE_OUTPUT, "saved outputs")
                reporter.final_success(f"induced {num_threads} task threads")
                return payload
            except Exception as exc:
                reporter.fail_active_stage(exc)
                reporter.final_failure()
                setattr(exc, "_task_threads_reported", True)
                raise
    finally:
        _PROGRESS_ENABLED = previous_progress


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve() if args.config else resolve_config_path()
        config = load_config(config_path)
        if config.dotenv_path:
            try:
                from dotenv import load_dotenv

                dotenv_path = resolve_dotenv_path(config_path, config.dotenv_path)
                load_dotenv(dotenv_path, override=False)
            except ModuleNotFoundError:
                pass

        stage_config = config.task_threads_induction
        if args.data_dir is None:
            raise ValueError("--data_dir is required for step3 task thread induction.")
        with litellm_model_config(
            model_alias=stage_config.model,
            litellm_params=stage_config.litellm_params,
        ):
            task_thread_induction(
                data_dir=args.data_dir,
                input_file_name=stage_config.input_file_name,
                output_file_name=stage_config.output_file_name,
                derived_objectives_dir=stage_config.derived_objectives_dir,
                model=stage_config.model,
                discovery_batch_size=stage_config.discovery_batch_size,
                max_recent_assignments=stage_config.max_recent_assignments,
                llm_timeout_secs=stage_config.llm_timeout_seconds,
                reuse_cache=stage_config.reuse_cache,
                enable_reassignment_review=stage_config.enable_reassignment_review,
                preflight_only=args.preflight_only,
                no_console=args.no_console,
            )
        return 0
    except KeyboardInterrupt:
        print("Task threads induction interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if not getattr(exc, "_task_threads_reported", False):
            print(f"Task threads induction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
