"""Schemas for task-thread induction output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskThreadRoot(BaseModel):
    """One canonical task thread root induced from activities."""

    model_config = ConfigDict(extra="forbid")

    canonical_root_id: str
    label: str
    objective: str
    deliverable: str = Field(min_length=1)
    success_criteria: str = Field(min_length=1)
    summary: str = ""
    last_update: str = ""
    anchor: list[str] = Field(default_factory=list)
    member_provisional_root_ids: list[str] = Field(default_factory=list)
    activity_id: list[str] = Field(default_factory=list)
    semantic_action_id: list[str] = Field(default_factory=list)
    raw_action_id: list[str] = Field(default_factory=list)
    semantic_action_count: int
    observed_applications: list[str] = Field(default_factory=list)
    provisional_roots: list[dict[str, Any]] = Field(default_factory=list)
    event_count: int

    @field_validator(
        "canonical_root_id",
        "label",
        "objective",
        "deliverable",
        "success_criteria",
    )
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value


class SemanticActionTaskThreadAssignment(BaseModel):
    """Task-thread assignment for one semantic action."""

    model_config = ConfigDict(extra="forbid")

    semantic_action_id: str = ""
    raw_action_id: list[str] = Field(default_factory=list)
    semantic_action: str
    provisional_root_id: str = ""
    canonical_root_id: str = ""
    canonical_root_label: str = ""


class TaskThreadsInductionMeta(BaseModel):
    """Metadata for a task-thread induction run."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str
    input_path: str
    output_path: str
    num_semantic_actions: int
    num_provisional_roots: int
    num_canonical_roots: int
    discovery_batch_size: int
    max_recent_assignments: int
    pipeline_version: int
    reused_cache: bool = False
    preflight_only: bool = False
    elapsed_secs: float | None = None
    llm_requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_usd: float | None = None
    cost_breakdown: dict[str, Any] | None = None


class TaskThreadsInductionOutput(BaseModel):
    """Complete task-thread induction output."""

    model_config = ConfigDict(extra="forbid")

    meta: TaskThreadsInductionMeta
    roots: list[TaskThreadRoot]
    semantic_action_assignments: list[SemanticActionTaskThreadAssignment]
