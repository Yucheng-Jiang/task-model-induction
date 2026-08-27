#!/usr/bin/env python3
"""Run action grounding for computer-use activity JSONL entries.

Created: May 6, 2026.

This script reads activity entries, sends each action and its before/after
screenshots to an action-grounding service, and writes grounded goals,
visual/OCR context, cost metadata, and per-entry status to a JSONL output file.
By default, it reads processed_trajectory.jsonl from --data_dir and writes
processed_trajectory_with_goals.jsonl to the same directory.

Example:
    uv run python step0_action_grounding.py \
        --data_dir /path/to/data

By default, entries already present in the output JSONL are skipped. Use
--override_existing to reprocess them.
Completed entries are appended to a .progress.jsonl sidecar during the run for
resumability, then merged into the final output JSONL after completion.
The grounding service URL is read from config.yaml under
action_grounding_stage.grounding_url. Client-side parallelism is read from
action_grounding_stage.max_concurrent_requests.

Preflight without sending grounding requests:
    uv run python step0_action_grounding.py \
        --data_dir data/session_001 \
        --preflight_only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

DEFAULT_INPUT_FILE_NAME = "processed_trajectory.jsonl"
DEFAULT_OUTPUT_FILE_NAME = "processed_trajectory_with_goals.jsonl"
STAGE_HEALTH = "health"
STAGE_LOAD_INPUTS = "load inputs"
STAGE_GROUND_ACTIONS = "ground actions"
STAGE_WRITE_OUTPUT = "write output"
STAGE_PREFLIGHT = "preflight"
STAGES = [STAGE_HEALTH, STAGE_LOAD_INPUTS, STAGE_GROUND_ACTIONS, STAGE_WRITE_OUTPUT]

try:
    from action_grounding_service.app.schemas import (
        ActionGroundingRequest,
        ActionGroundingResponse,
        ScreenSize,
    )
except ModuleNotFoundError:
    from task_model_induction.action_grounding_service.app.schemas import (
        ActionGroundingRequest,
        ActionGroundingResponse,
        ScreenSize,
    )

try:
    from task_model_induction.schemas import (
        ActionGroundingOutput,
        ComputerUseActivityEntry,
    )
    from task_model_induction.config import load_action_grounding_stage_config
    from task_model_induction.utils import (
        image_data_url,
        infer_screen_size,
        iter_activity_jsonl,
        iter_jsonl_objects,
        read_jsonl_objects,
        resolve_path,
        utc_now_iso,
        write_action_grounding_jsonl_iter,
    )
    from task_model_induction.reporting.progress_reporter import ConsoleProgressReporter
except ModuleNotFoundError:
    from schemas import (
        ActionGroundingOutput,
        ComputerUseActivityEntry,
    )
    from config import load_action_grounding_stage_config
    from utils import (
        image_data_url,
        infer_screen_size,
        iter_activity_jsonl,
        iter_jsonl_objects,
        read_jsonl_objects,
        resolve_path,
        utc_now_iso,
        write_action_grounding_jsonl_iter,
    )
    from reporting.progress_reporter import ConsoleProgressReporter


class ActionGroundingReporter(ConsoleProgressReporter):
    run_name = "action_grounding"
    title = "Action grounding"
    success_title = "Action grounding complete"
    failure_title = "Action grounding failed"
    default_failure_stage = STAGE_PREFLIGHT

    def __init__(self, *, no_console: bool = False) -> None:
        super().__init__(stages=STAGES, no_console=no_console)

    def render_success(self, detail: str) -> str:
        completed = self.state.counters.get("completed", 0)
        total = self.state.counters.get("total", self.state.counters.get("loaded", 0))
        success = self.state.counters.get("success", 0)
        error = self.state.counters.get("error", 0)
        skipped = self.state.counters.get("skipped_existing", 0)
        service_warnings = self.state.counters.get("service_warnings", 0)
        estimated_usd = self.state.metrics.get("estimated_usd")
        output = self.state.paths.get("output")
        parts = [
            detail,
            f"{completed}/{total} processed",
            f"success={success}",
            f"error={error}",
            f"skipped={skipped}",
            f"service_warnings={service_warnings}",
        ]
        if estimated_usd is not None:
            parts.append(f"estimated_usd={estimated_usd}")
        if output is not None:
            parts.append(f"output={self.short_path(output)}")
        return " | ".join(parts)

    def plain_summary(self) -> str:
        output = self.state.paths.get("output")
        return (
            f"completed={self.state.counters.get('completed', 0)} "
            f"total={self.state.counters.get('total', self.state.counters.get('loaded', 0))} "
            f"success={self.state.counters.get('success', 0)} "
            f"error={self.state.counters.get('error', 0)} "
            f"skipped_existing={self.state.counters.get('skipped_existing', 0)} "
            f"service_warnings={self.state.counters.get('service_warnings', 0)} "
            f"estimated_usd={self.state.metrics.get('estimated_usd', 0)} "
            f"output={output or ''}"
        )

    def render(self) -> Any:
        if not self._Panel:
            return ""
        stage = self.state.active_stage or self.last_stage_name()
        completed = self.state.counters.get("completed", 0)
        total = self.state.counters.get("total", 0)
        loaded = self.state.counters.get("loaded", 0)
        success = self.state.counters.get("success", 0)
        error = self.state.counters.get("error", 0)
        skipped = self.state.counters.get("skipped_existing", 0)
        service_warnings = self.state.counters.get("service_warnings", 0)
        estimated_usd = self.state.metrics.get("estimated_usd", 0)
        eta = self.state.metrics.get("eta", "estimating")
        detail = self.active_detail()
        output = self.state.paths.get("output")

        lines = [f"stage: {stage}"]
        if stage == STAGE_LOAD_INPUTS and total == 0 and loaded > 0:
            lines.append(f"inputs: loaded={loaded} queued=calculating skipped=calculating")
        else:
            lines.append(f"progress: {self.progress_bar(completed, total)} {completed}/{total}")
            lines.append(
                f"results: success={success} error={error} skipped={skipped} service_warnings={service_warnings} estimated_usd={estimated_usd} eta={eta}"
            )
        service_warning = self.state.metrics.get("service_warning")
        if service_warning:
            lines.append(f"warning: {service_warning}")
        if detail:
            lines.append(f"current: {detail}")
        if output:
            lines.append(f"output: {self.short_path(output)}")
        return self._Panel("\n".join(lines), title="Action grounding", border_style="cyan")

    def should_log_plain_progress(self, detail: str) -> bool:
        completed = self.state.counters.get("completed", 0)
        total = self.state.counters.get("total", 0)
        if completed == 0:
            return False
        interval = max(1, min(100, total // 20 if total else 1))
        should_log = completed == total or completed % interval == 0
        if should_log and completed != self._last_plain_progress:
            self._last_plain_progress = completed
            return True
        return False


def build_grounding_request(entry: ComputerUseActivityEntry, data_dir: Path) -> ActionGroundingRequest:
    if not entry.action:
        raise ValueError("activity entry is missing action")

    before_path = resolve_path(entry.state_before, data_dir)
    if before_path is None or not before_path.exists():
        raise FileNotFoundError("before screenshot not found")

    after_path = resolve_path(entry.state_after, data_dir)
    after_image = image_data_url(after_path) if after_path is not None and after_path.exists() else None

    return ActionGroundingRequest(
        before_image=image_data_url(before_path),
        after_image=after_image,
        action=entry.action,
        screen_size=ScreenSize.model_validate(infer_screen_size(before_path)),
    )


def output_from_response(
    entry: ComputerUseActivityEntry,
    response: ActionGroundingResponse,
    *,
    original_index: int | None = None,
) -> ActionGroundingOutput:
    return ActionGroundingOutput(
        status="success",
        goal=response.goal,
        active_application=response.active_application,
        visual_content=response.visual_content,
        ocr_results=response.ocr_results.model_dump(mode="json"),
        md_results=response.md_results,
        cost=response.cost.model_dump(mode="json"),
        warnings=response.warnings or response.ocr_results.warnings,
        completed_at=utc_now_iso(),
        provenounce_id=entry.id,
        original_index=original_index,
        **entry.model_dump(mode="json"),
    )


def output_from_error(
    entry: ComputerUseActivityEntry,
    error: str,
    *,
    original_index: int | None = None,
) -> ActionGroundingOutput:
    return ActionGroundingOutput(
        status="error",
        error=error,
        completed_at=utc_now_iso(),
        provenounce_id=entry.id,
        original_index=original_index,
        **entry.model_dump(mode="json"),
    )


def scan_activity_input(entry: ComputerUseActivityEntry, data_dir: Path) -> dict[str, int]:
    before_path = resolve_path(entry.state_before, data_dir)
    after_path = resolve_path(entry.state_after, data_dir)
    return {
        "missing_before": int(before_path is None or not before_path.exists()),
        "missing_after": int(bool(entry.state_after) and (after_path is None or not after_path.exists())),
    }


def read_existing_action_grounding_outputs(path: Path) -> dict[str, ActionGroundingOutput]:
    if not path.exists():
        return {}
    outputs: dict[str, ActionGroundingOutput] = {}
    for row in read_jsonl_objects(path):
        output = ActionGroundingOutput.model_validate(row)
        outputs[output.provenounce_id] = output
    return outputs


def read_successful_action_grounding_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    success_ids: set[str] = set()
    for row in iter_jsonl_objects(path):
        if row.get("status") == "success":
            output_id = row.get("provenounce_id")
            if isinstance(output_id, str) and output_id:
                success_ids.add(output_id)
    return success_ids


def read_resumable_successful_action_grounding_ids(output_path: Path, progress_path: Path) -> set[str]:
    return read_successful_action_grounding_ids(output_path) | read_successful_action_grounding_ids(progress_path)


def progress_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".progress.jsonl")


def read_resumable_action_grounding_outputs(
    output_path: Path,
    progress_path: Path,
) -> dict[str, ActionGroundingOutput]:
    return {
        **read_existing_action_grounding_outputs(output_path),
        **read_existing_action_grounding_outputs(progress_path),
    }


def cached_output_matches_entry(
    output: ActionGroundingOutput | None,
    entry: ComputerUseActivityEntry,
) -> bool:
    """Return whether a successful cache row was grounded from this raw event."""

    if output is None or output.status != "success":
        return False
    # Legacy rows did not retain their source event. Keep them reusable once;
    # the merge below upgrades them so subsequent runs can compare exactly.
    if output.id is None:
        return True
    source = entry.model_dump(mode="json")
    return all(getattr(output, field) == value for field, value in source.items())


class ActionGroundingProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "ActionGroundingProgressWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def append(self, output: ActionGroundingOutput) -> None:
        if self._handle is None:
            raise RuntimeError("progress writer is not open")
        self._handle.write(json.dumps(output.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self._handle.flush()


def remove_progress_output(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def iter_merged_action_grounding_outputs(
    input_path: Path,
    output_by_id: dict[str, ActionGroundingOutput],
    *,
    limits: int | None = None,
) -> Iterator[ActionGroundingOutput]:
    """Yield selected outputs in raw-input order, enriched with source fields.

    The input trajectory is authoritative.  Cached rows that no longer occur
    in it, including rows outside ``limits``, must not leak into the merged
    output.  Rejoining here also upgrades legacy cached outputs that predate
    preservation of the raw event fields.
    """

    seen_input_ids: set[str] = set()
    for original_index, entry in enumerate(iter_activity_jsonl(input_path)):
        if limits is not None and original_index >= limits:
            break
        if entry.id in seen_input_ids:
            raise ValueError(f"duplicate raw activity id {entry.id!r} at index {original_index}")
        seen_input_ids.add(entry.id)
        output = output_by_id.get(entry.id)
        if output is None:
            continue
        yield output.model_copy(
            update={
                **entry.model_dump(mode="json"),
                "original_index": original_index,
                "provenounce_id": entry.id,
            }
        )


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def update_eta_metric(reporter: ActionGroundingReporter, started_at: float) -> None:
    completed = reporter.state.counters.get("completed", 0)
    total = reporter.state.counters.get("total", 0)
    if completed <= 0:
        reporter.set_metric("eta", "estimating")
        return
    remaining = max(0, total - completed)
    if remaining == 0:
        reporter.set_metric("eta", "0s")
        return
    elapsed = time.monotonic() - started_at
    reporter.set_metric("eta", format_duration((elapsed / completed) * remaining))


def update_cost_metrics(reporter: ActionGroundingReporter, output: ActionGroundingOutput) -> None:
    if not output.cost:
        return
    current_usd = float(reporter.state.metrics.get("estimated_usd", 0.0))
    reporter.set_metric("estimated_usd", round(current_usd + float(output.cost.get("total_usd", 0.0)), 6))

    for section_name in ("ocr", "grounding", "redaction"):
        section = output.cost.get(section_name) or {}
        for item in section.get("items", []):
            reporter.increment("requests", int(item.get("requests", 0)))
            reporter.increment("input_tokens", int(item.get("input_tokens", 0)))
            reporter.increment("output_tokens", int(item.get("output_tokens", 0)))
            reporter.increment("total_tokens", int(item.get("total_tokens", 0)))


def _is_transient_error(exc: Exception) -> bool:
    """Return True for errors that are likely transient and worth retrying."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (502, 503, 429):
        return True
    return False


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


async def ground_activity_entry(
    entry: ComputerUseActivityEntry,
    data_dir: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    original_index: int | None = None,
) -> ActionGroundingOutput:
    async with semaphore:
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                request = build_grounding_request(entry, data_dir)
                response = await client.post("/ground", json=request.model_dump(mode="json"))
                response.raise_for_status()
                grounded = ActionGroundingResponse.model_validate(response.json())
                return output_from_response(entry, grounded, original_index=original_index)
            except Exception as exc:
                last_exc = exc
                if _is_transient_error(exc) and attempt < max_retries:
                    delay = retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue
                return output_from_error(entry, str(exc), original_index=original_index)
        # Should not reach here, but just in case:
        return output_from_error(entry, str(last_exc), original_index=original_index)


async def check_grounding_service_health(client: httpx.AsyncClient) -> None:
    health_url = f"{str(client.base_url).rstrip('/')}/health"
    try:
        response = await client.get("/health", timeout=10.0)
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Action grounding service is not reachable at {health_url}. "
            "Start it with: uv run action-grounding-service init"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Action grounding service health check timed out at {health_url}") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Action grounding service health check returned HTTP {exc.response.status_code} at {health_url}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Action grounding service health check failed for {health_url}: {exc}"
        ) from exc


async def monitor_grounding_service_health(
    client: httpx.AsyncClient,
    reporter: ActionGroundingReporter,
    stop_event: asyncio.Event,
    interval_secs: float = 10.0,
) -> None:
    last_warning = ""
    while not stop_event.is_set():
        try:
            response = await client.get("/health/details", timeout=5.0)
            response.raise_for_status()
            details = response.json()
            omniparser = details.get("omniparser") if isinstance(details, dict) else None
            if isinstance(omniparser, dict) and not omniparser.get("ok", False):
                warning = str(omniparser.get("message") or "OmniParser is unhealthy")
                if warning != last_warning:
                    reporter.increment("service_warnings")
                    reporter.set_metric("service_warning", warning)
                    reporter.progress(f"service warning: {warning}")
                    last_warning = warning
        except Exception as exc:
            warning = f"health details unavailable: {exc}"
            if warning != last_warning:
                reporter.increment("service_warnings")
                reporter.set_metric("service_warning", warning)
                reporter.progress(f"service warning: {warning}")
                last_warning = warning
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_secs)
        except asyncio.TimeoutError:
            pass


async def action_grounding_async(
    data_dir: str | Path,
    limits: int | None = None,
    no_console: bool = False,
    preflight_only: bool = False,
    override_existing: bool = False,
) -> list[ActionGroundingOutput]:
    data_dir = Path(data_dir)
    stage_config = load_action_grounding_stage_config()
    grounding_url = stage_config.grounding_url
    max_concurrent = stage_config.max_concurrent_requests
    input_path = data_dir / DEFAULT_INPUT_FILE_NAME
    output_path = data_dir / DEFAULT_OUTPUT_FILE_NAME
    progress_path = progress_output_path(output_path)
    if max_concurrent < 1:
        raise ValueError("action_grounding_stage.max_concurrent_requests must be at least 1")
    if limits is not None and limits < 1:
        raise ValueError("--limits must be at least 1")

    timeout = httpx.Timeout(300.0)
    pool_limits = httpx.Limits(
        max_connections=max_concurrent + 8,
        max_keepalive_connections=max_concurrent + 4,
    )
    reporter = ActionGroundingReporter(no_console=no_console)

    try:
        with reporter:
            reporter.add_path("input", input_path)
            reporter.add_path("output", output_path)
            reporter.add_path("progress", progress_path)
            reporter.set_metric("grounding_url", grounding_url)
            reporter.set_metric("max_concurrent_requests", max_concurrent)

            reporter.start_stage(STAGE_LOAD_INPUTS, str(input_path))
            resumable_outputs = read_resumable_action_grounding_outputs(output_path, progress_path)
            output_ids_to_skip: set[str] = set()

            loaded_count = 0
            skipped_existing_count = 0
            entries_ready_count = 0
            missing_before = 0
            missing_after = 0
            seen_input_ids: set[str] = set()
            for original_index, entry in enumerate(iter_activity_jsonl(input_path)):
                if limits is not None and original_index >= limits:
                    break
                if entry.id in seen_input_ids:
                    raise ValueError(
                        f"duplicate raw activity id {entry.id!r} at index {original_index}"
                    )
                seen_input_ids.add(entry.id)
                loaded_count += 1
                if not override_existing and cached_output_matches_entry(
                    resumable_outputs.get(entry.id), entry
                ):
                    output_ids_to_skip.add(entry.id)
                    skipped_existing_count += 1
                    continue
                entries_ready_count += 1
                scan = scan_activity_input(entry, data_dir)
                missing_before += scan["missing_before"]
                missing_after += scan["missing_after"]

            reporter.set_counter("loaded", loaded_count if limits is None else min(loaded_count, limits))
            if limits is not None:
                reporter.set_counter("limited_to", reporter.state.counters["loaded"])
            reporter.set_counter("skipped_existing", skipped_existing_count)
            reporter.set_counter("missing_before", missing_before)
            reporter.set_counter("missing_after", missing_after)
            reporter.set_counter("total", entries_ready_count)
            reporter.set_metric("eta", "estimating" if entries_ready_count else "0s")
            reporter.finish_stage(
                STAGE_LOAD_INPUTS,
                f"{entries_ready_count} entries ready; {skipped_existing_count} skipped",
            )

            if preflight_only or entries_ready_count:
                reporter.start_stage(STAGE_HEALTH, f"GET {grounding_url.rstrip('/')}/health")
                async with httpx.AsyncClient(base_url=grounding_url, timeout=timeout, limits=pool_limits) as client:
                    await check_grounding_service_health(client)
                    reporter.finish_stage(STAGE_HEALTH, "service healthy")

                    if preflight_only:
                        for stage_name in (STAGE_GROUND_ACTIONS, STAGE_WRITE_OUTPUT):
                            reporter.mark_stage_done(stage_name, "skipped by --preflight_only")
                        reporter.final_success("preflight complete; no grounding requests were sent")
                        return []

                    reporter.start_stage(STAGE_GROUND_ACTIONS, f"{entries_ready_count} requests queued")
                    semaphore = asyncio.Semaphore(max_concurrent)
                    total_entries = entries_ready_count
                    health_monitor_stop = asyncio.Event()
                    health_monitor = asyncio.create_task(
                        monitor_grounding_service_health(client, reporter, health_monitor_stop)
                    )
                    grounding_started_at = time.monotonic()
                    success_count = 0
                    error_count = 0
                    try:
                        with ActionGroundingProgressWriter(progress_path) as progress_writer:
                            # Process in batches to avoid creating all tasks at once,
                            # which would load all images into memory and exhaust the
                            # connection pool.
                            batch_size = max_concurrent * 2
                            completed_count = 0
                            batch: list[tuple[int, ComputerUseActivityEntry]] = []
                            for original_index, entry in enumerate(iter_activity_jsonl(input_path)):
                                if limits is not None and original_index >= limits:
                                    break
                                if entry.id in output_ids_to_skip:
                                    continue
                                batch.append((original_index, entry))
                                if len(batch) < batch_size:
                                    continue
                                tasks = [
                                    asyncio.create_task(
                                        ground_activity_entry(
                                            batch_entry,
                                            data_dir,
                                            client,
                                            semaphore,
                                            original_index=batch_original_index,
                                        )
                                    )
                                    for batch_original_index, batch_entry in batch
                                ]
                                for completed in asyncio.as_completed(tasks):
                                    output = await completed
                                    progress_writer.append(output)
                                    completed_count += 1
                                    if output.status == "success":
                                        success_count += 1
                                    else:
                                        error_count += 1
                                    reporter.increment("completed")
                                    reporter.increment(output.status)
                                    if output.warnings:
                                        reporter.increment("service_warnings", len(output.warnings))
                                        reporter.set_metric("service_warning", output.warnings[-1])
                                    update_cost_metrics(reporter, output)
                                    update_eta_metric(reporter, grounding_started_at)
                                    reporter.progress(
                                        f"{completed_count}/{total_entries} provenounce_id={output.provenounce_id} status={output.status}"
                                    )
                                batch = []
                            if batch:
                                tasks = [
                                    asyncio.create_task(
                                        ground_activity_entry(
                                            batch_entry,
                                            data_dir,
                                            client,
                                            semaphore,
                                            original_index=batch_original_index,
                                        )
                                    )
                                    for batch_original_index, batch_entry in batch
                                ]
                                for completed in asyncio.as_completed(tasks):
                                    output = await completed
                                    progress_writer.append(output)
                                    completed_count += 1
                                    if output.status == "success":
                                        success_count += 1
                                    else:
                                        error_count += 1
                                    reporter.increment("completed")
                                    reporter.increment(output.status)
                                    if output.warnings:
                                        reporter.increment("service_warnings", len(output.warnings))
                                        reporter.set_metric("service_warning", output.warnings[-1])
                                    update_cost_metrics(reporter, output)
                                    update_eta_metric(reporter, grounding_started_at)
                                    reporter.progress(
                                        f"{completed_count}/{total_entries} provenounce_id={output.provenounce_id} status={output.status}"
                                    )
                    finally:
                        health_monitor_stop.set()
                        health_monitor.cancel()
                        try:
                            await health_monitor
                        except asyncio.CancelledError:
                            pass
                    reporter.finish_stage(STAGE_GROUND_ACTIONS, f"{success_count + error_count} entries grounded")
            else:
                success_count = 0
                error_count = 0
                reporter.mark_stage_done(STAGE_HEALTH, "skipped; no new entries")
                reporter.mark_stage_done(STAGE_GROUND_ACTIONS, "no new entries to ground")

            reporter.start_stage(STAGE_WRITE_OUTPUT, str(output_path))
            merged_outputs = read_resumable_action_grounding_outputs(output_path, progress_path)
            write_action_grounding_jsonl_iter(
                output_path,
                iter_merged_action_grounding_outputs(input_path, merged_outputs, limits=limits),
            )
            remove_progress_output(progress_path)
            reporter.finish_stage(STAGE_WRITE_OUTPUT, str(output_path))

            reporter.set_counter("success", success_count)
            reporter.set_counter("error", error_count)
            reporter.final_success(
                f"processed={success_count + error_count} skipped_existing={reporter.state.counters.get('skipped_existing', 0)}"
            )
            return list(iter_merged_action_grounding_outputs(input_path, merged_outputs, limits=limits))
    except Exception as exc:
        reporter.fail_active_stage(exc)
        reporter.final_failure()
        setattr(exc, "_action_grounding_reported", True)
        raise


def action_grounding(
    data_dir: str | Path,
    limits: int | None = None,
    no_console: bool = False,
    preflight_only: bool = False,
    override_existing: bool = False,
) -> list[ActionGroundingOutput]:
    return asyncio.run(
        action_grounding_async(
            data_dir=data_dir,
            limits=limits,
            no_console=no_console,
            preflight_only=preflight_only,
            override_existing=override_existing,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run action grounding for every computer-use activity JSONL entry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing the input JSONL.")
    parser.add_argument(
        "--limits",
        "--limit",
        dest="limits",
        type=int,
        default=None,
        help="Only process the first N activity entries.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Check service health and input readiness without sending grounding requests.",
    )
    parser.add_argument(
        "--override_existing",
        action="store_true",
        help="Reprocess entries even when they already exist in the output JSONL.",
    )
    parser.add_argument(
        "--no_console",
        action="store_true",
        help="Disable rich/live console UI and emit plain structured logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        action_grounding(
            data_dir=args.data_dir,
            limits=args.limits,
            no_console=args.no_console,
            preflight_only=args.preflight_only,
            override_existing=args.override_existing,
        )
        return 0
    except KeyboardInterrupt:
        print("Action grounding interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if not getattr(exc, "_action_grounding_reported", False):
            print(f"Action grounding failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
