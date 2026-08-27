"""Schema for one action-grounding output entry."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionGroundingOutput(BaseModel):
    """One raw trajectory row enriched with action-grounding results.

    The source fields are optional so that progress/output files produced by
    older versions can still be read.  New outputs always populate them, and
    the final merge rehydrates legacy cached outputs from the current raw
    trajectory before writing them.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    goal: str | None = None
    active_application: str | None = None
    visual_content: str | None = None
    ocr_results: dict[str, Any] | None = None
    md_results: str | None = None
    cost: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    completed_at: str
    provenounce_id: str
    original_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based position of the event in the raw input trajectory.",
    )
    id: str | None = None
    action: str | None = None
    state_before: str | None = None
    state_after: str | None = None
    time_before: float | str | None = None
    time_after: float | str | None = None
    time_range: float | None = None
