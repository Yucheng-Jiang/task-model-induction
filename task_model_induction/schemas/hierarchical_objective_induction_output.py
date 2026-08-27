"""Schemas for hierarchical objective induction output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .objective_grounding import (
    ObjectiveDeliverable,
    ObjectiveObservedOutcome,
    ObjectiveSuccessCriterion,
)


class HierarchicalObjectiveNode(BaseModel):
    """Recursive objective decomposition node."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    deliverables: list[ObjectiveDeliverable] = Field(min_length=1)
    success_criteria: list[ObjectiveSuccessCriterion] = Field(min_length=1)
    observed_outcome: ObjectiveObservedOutcome
    evidence_refs: list[str] = Field(min_length=1)
    subgoal_segments: list[str] = Field(min_length=1)
    decomposition: list["HierarchicalObjectiveNode"] = Field(default_factory=list)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        normalized = [ref.strip() for ref in value]
        if any(not ref for ref in normalized):
            raise ValueError("evidence_refs entries must be non-empty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_refs must not contain duplicates")
        return normalized


class HierarchicalObjectiveInductionMeta(BaseModel):
    """Metadata written next to hierarchy outputs."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    input_path: str
    output_path: str | None = None
    model: str | None = None
    max_retries: int | None = None
    retry_count: int | None = None
    preflight_only: bool = False
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cost: dict[str, Any] | None = None
    execution_mode: str | None = None
    activity_count: int | None = None
    run_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    usage: dict[str, Any] | None = None
    estimated_usd: float | None = None
    proxy_cost: dict[str, Any] | None = None


class HierarchicalObjectiveInductionMergedMeta(BaseModel):
    """Metadata for a merged multi-root hierarchy output."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str | None = None
    output_dir: str
    num_roots: int
    num_succeeded: int
    preflight_only: bool = False
    cost: dict[str, Any] | None = None


class HierarchicalObjectiveRootResult(BaseModel):
    """One per-task-thread hierarchy generation result."""

    model_config = ConfigDict(extra="allow")

    input_file: str
    output_file: str
    ok: bool
    hierarchy: HierarchicalObjectiveNode | None = None
    run_id: str | None = None
    session_id: str | None = None
    execution_mode: str | None = None
    activity_count: int | None = None
    usage: dict[str, Any] | None = None
    estimated_usd: float | None = None
    cost_breakdown: dict[str, Any] | None = None
    proxy_cost: dict[str, Any] | None = None
    error: str | None = None


class HierarchicalObjectiveInductionOutput(BaseModel):
    """Merged output for hierarchical objective induction."""

    model_config = ConfigDict(extra="forbid")

    meta: HierarchicalObjectiveInductionMergedMeta
    roots: list[HierarchicalObjectiveRootResult]
