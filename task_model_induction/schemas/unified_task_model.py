"""Schemas for the unified task model produced by bidirectional alignment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .condition_grounding import WhileConditionGrounding
from .objective_grounding import (
    ObjectiveDeliverable,
    ObjectiveObservedOutcome,
    ObjectiveSuccessCriterion,
)


PrimitiveProcedureOperator = Literal["SEQ", "FOR", "WHILE", "CHOICE"]


class UnifiedProcedureBodyStep(BaseModel):
    """One step in a procedure body, valid for all operators.

    - SEQ body: one step per sequential phase, in order, each with its activity range.
    - WHILE body: abstract per-pass template steps; ``activity_refs`` spans ALL passes
      for that step (not a single pass).
    - FOR body: abstract per-item template steps; ``activity_refs`` spans ALL iterations
      for that step across every item in the collection.
    - CHOICE body: the one observed branch.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    activity_refs: list[str] = Field(min_length=1)


class UnifiedProcedureAnnotation(BaseModel):
    """Control-flow annotation attached to each unified node."""

    model_config = ConfigDict(extra="forbid")

    operator: PrimitiveProcedureOperator
    name: str = Field(min_length=1)
    description: str | None = None
    condition: str | None = None
    condition_grounding: WhileConditionGrounding | None = None
    bindings: dict[str, Any] | None = None
    body: list[UnifiedProcedureBodyStep] = Field(min_length=1)
    evidence_summary: str | None = None

    @model_validator(mode="after")
    def validate_condition_grounding(self) -> "UnifiedProcedureAnnotation":
        if self.operator == "WHILE":
            if self.condition_grounding is None:
                raise ValueError("WHILE requires condition_grounding")
            if not self.condition or self.condition.strip() != self.condition_grounding.predicate:
                raise ValueError(
                    "WHILE condition must exactly match condition_grounding.predicate"
                )
        elif self.condition_grounding is not None:
            raise ValueError("condition_grounding is only valid for WHILE")
        return self


class UnifiedTaskModelNode(BaseModel):
    """One node in the unified task model tree.

    Each node simultaneously carries the objective layer (goal + decomposition)
    and the procedure layer (operator + how the work was executed).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    summary: str | None = None
    deliverables: list[ObjectiveDeliverable] = Field(min_length=1)
    success_criteria: list[ObjectiveSuccessCriterion] = Field(min_length=1)
    observed_outcome: ObjectiveObservedOutcome
    evidence_refs: list[str] = Field(min_length=1)
    activity_refs: list[str] = Field(min_length=1)
    procedure: UnifiedProcedureAnnotation
    decomposition: list["UnifiedTaskModelNode"] = Field(default_factory=list)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        normalized = [ref.strip() for ref in value]
        if any(not ref for ref in normalized):
            raise ValueError("evidence_refs entries must be non-empty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_refs must not contain duplicates")
        return normalized


UnifiedTaskModelNode.model_rebuild()


class UnifiedTaskModel(BaseModel):
    """Unified task model for one task-thread root."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["0.2"] = "0.2"
    root: UnifiedTaskModelNode


class UnifiedTaskModelRootResult(BaseModel):
    """Per-task-thread reconciliation result written into the merged output."""

    model_config = ConfigDict(extra="allow")

    input_file: str | None = None
    objective_file: str
    procedure_file: str
    output_file: str | None = None
    ok: bool
    execution_mode: str = "direct_llm"
    activity_count: int | None = None
    run_id: str | None = None
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    estimated_usd: float | None = None
    proxy_cost: dict[str, Any] | None = None
    task_model: UnifiedTaskModel | None = None
    error: str | None = None


class UnifiedTaskModelMergedMeta(BaseModel):
    """Metadata for the merged unified task model output."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str | None = None
    output_dir: str
    num_roots: int
    num_succeeded: int
    preflight_only: bool = False
    cost: dict[str, Any] | None = None


class UnifiedTaskModelOutput(BaseModel):
    """Merged output produced by step6 bidirectional alignment."""

    model_config = ConfigDict(extra="forbid")

    meta: UnifiedTaskModelMergedMeta
    roots: list[UnifiedTaskModelRootResult]
