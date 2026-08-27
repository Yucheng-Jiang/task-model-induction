"""Shared rich/plain progress reporting for task-model scripts."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["pending", "active", "done", "failed"]


@dataclass
class Stage:
    name: str
    status: StageStatus = "pending"
    detail: str = ""


@dataclass
class ProgressState:
    stages: list[Stage]
    counters: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, str | int | float] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    active_stage: str | None = None
    error: str | None = None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


class ConsoleProgressReporter:
    run_name = "progress"
    title = "Progress"
    success_title = "Complete"
    failure_title = "Failed"
    default_failure_stage = "preflight"
    refresh_per_second = 4

    def __init__(self, *, stages: list[str], no_console: bool = False) -> None:
        self.state = ProgressState(stages=[Stage(name) for name in stages])
        self.no_console = no_console
        self.started_at = time.monotonic()
        self._rich_console: Any | None = None
        self._live: Any | None = None
        self._Live: Any | None = None
        self._Panel: Any | None = None
        self._Spinner: Any | None = None
        self._Table: Any | None = None
        self._Text: Any | None = None
        self._Group: Any | None = None
        self._box: Any | None = None
        self._last_plain_progress: str | int | None = None

        if not no_console:
            try:
                from rich import box
                from rich.console import Console, Group
                from rich.live import Live
                from rich.panel import Panel
                from rich.spinner import Spinner
                from rich.table import Table
                from rich.text import Text

                self._box = box
                self._Group = Group
                self._Live = Live
                self._Panel = Panel
                self._Spinner = Spinner
                self._Table = Table
                self._Text = Text
                self._rich_console = Console(stderr=True, highlight=False)
            except Exception:
                self._rich_console = None

    @property
    def rich_enabled(self) -> bool:
        return bool(
            self._rich_console
            and getattr(self._rich_console, "is_terminal", False)
            and not self.no_console
        )

    def __enter__(self) -> "ConsoleProgressReporter":
        if self.rich_enabled:
            self._live = self._Live(
                self._render(),
                console=self._rich_console,
                refresh_per_second=self.refresh_per_second,
                transient=True,
            )
            self._live.start()
        else:
            self._plain(f"run=start name={self.run_name}")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.stop_live()

    def start_stage(self, name: str, detail: str = "") -> None:
        stage = self._stage(name)
        stage.status = "active"
        stage.detail = detail
        self.state.active_stage = name
        self._plain_or_refresh(f'stage=start name="{name}" detail="{detail}"')

    def finish_stage(self, name: str, detail: str = "") -> None:
        stage = self._stage(name)
        stage.status = "done"
        stage.detail = detail
        if self.state.active_stage == name:
            self.state.active_stage = None
        self._plain_or_refresh(f'stage=done name="{name}" detail="{detail}"')

    def mark_stage_done(self, name: str, detail: str = "") -> None:
        stage = self._stage(name)
        stage.status = "done"
        stage.detail = detail
        self.refresh()

    def fail_active_stage(self, exc: BaseException) -> None:
        name = self.state.active_stage or self.default_failure_stage
        if name not in {stage.name for stage in self.state.stages}:
            self.state.stages.append(Stage(name))
        stage = self._stage(name)
        stage.status = "failed"
        stage.detail = str(exc)
        self.state.error = str(exc)
        self._plain_or_refresh(f'stage=failed name="{name}" error="{exc}"')

    def set_counter(self, key: str, value: int) -> None:
        self.state.counters[key] = int(value)
        self.refresh()

    def increment(self, key: str, amount: int = 1) -> None:
        self.state.counters[key] = self.state.counters.get(key, 0) + int(amount)
        self.refresh()

    def set_metric(self, key: str, value: str | int | float) -> None:
        self.state.metrics[key] = value
        self.refresh()

    def add_path(self, label: str, path: Path) -> None:
        self.state.paths[label] = path
        self.refresh()

    def progress(self, detail: str) -> None:
        if self.state.active_stage:
            self._stage(self.state.active_stage).detail = detail
        if self.rich_enabled:
            self.refresh()
        elif self.should_log_plain_progress(detail):
            self._plain(f'progress="{detail}"')

    def should_log_plain_progress(self, detail: str) -> bool:
        if detail == self._last_plain_progress:
            return False
        self._last_plain_progress = detail
        return True

    def final_success(self, detail: str) -> None:
        self.stop_live()
        if self.rich_enabled and self._rich_console and self._Panel:
            content = self.render_success(detail)
            if isinstance(content, str):
                content = self._Panel(content, title=self.success_title, border_style="green")
            self._rich_console.print(content)
        else:
            self._plain(f'run=success detail="{detail}" {self.plain_summary()}')

    def final_failure(self) -> None:
        self.stop_live()
        message = self.state.error or "unknown error"
        if self.rich_enabled and self._rich_console and self._Panel:
            content = self.render_failure(message)
            if isinstance(content, str):
                content = self._Panel(content, title=self.failure_title, border_style="red")
            self._rich_console.print(content)
        else:
            self._plain(f'run=failed error="{message}" {self.plain_summary()}')

    def stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def stop(self) -> None:
        self.stop_live()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _stage(self, name: str) -> Stage:
        for stage in self.state.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def _plain_or_refresh(self, line: str) -> None:
        if self.rich_enabled:
            self.refresh()
        else:
            self._plain(line)

    def _plain(self, line: str) -> None:
        print(line, file=sys.stderr)

    def _render(self) -> Any:
        return self.render()

    def render(self) -> Any:
        return ""

    def render_success(self, detail: str) -> Any:
        return detail

    def render_failure(self, message: str) -> Any:
        return message

    def plain_summary(self) -> str:
        return ""

    def last_stage_name(self) -> str:
        for stage in reversed(self.state.stages):
            if stage.status != "pending":
                return stage.name
        return "starting"

    def active_detail(self) -> str:
        if self.state.active_stage:
            return self._stage(self.state.active_stage).detail
        for stage in reversed(self.state.stages):
            if stage.detail:
                return stage.detail
        return ""

    def progress_bar(self, completed: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return "[" + "." * width + "]"
        filled = min(width, max(0, round(width * completed / total)))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def short_path(self, path: Path, max_chars: int = 72) -> str:
        text = str(path)
        if len(text) <= max_chars:
            return text
        return "..." + text[-(max_chars - 3) :]

    def clip(self, value: object, limit: int) -> str:
        normalized = str(value or "").replace("\n", " ").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "."
