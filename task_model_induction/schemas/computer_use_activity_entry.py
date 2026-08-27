"""Schema for one computer-use activity input entry."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ComputerUseActivityEntry(BaseModel):
    """One computer-use activity row from the input JSONL."""

    model_config = ConfigDict(extra="forbid")

    id: str
    action: str | None = None
    state_before: str | None = None
    state_after: str | None = None
    time_before: float | str | None = None
    time_after: float | str | None = None
    time_range: float | None = None
