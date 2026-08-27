"""Schemas for procedure model induction output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .condition_grounding import WhileConditionGrounding


PrimitiveProcedureOperator = Literal[
    "SEQ",
    "FOR",
    "WHILE",
    "CHOICE",
]


class ProcedureNode(BaseModel):
    """One inferred procedure node."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    operator: PrimitiveProcedureOperator
    description: str
    bindings: dict[str, Any] | None = None
    body: dict[str, Any] | list[Any] | None = None
    condition: str | None = None
    condition_grounding: WhileConditionGrounding | None = None
    dataflow: list[Any] | None = None
    effects: list[Any] | None = None
    activity_refs: list[str] = Field(default_factory=list)
    evidence_summary: str

    @model_validator(mode="after")
    def validate_condition_grounding(self) -> "ProcedureNode":
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


class ProcedureTaskModel(BaseModel):
    """Procedure model for one task-thread objective root."""

    model_config = ConfigDict(extra="forbid")

    version: str
    root_procedure_id: str
    procedure_nodes: list[ProcedureNode] = Field(default_factory=list)


class ProcedureInductionMergedMeta(BaseModel):
    """Metadata for a merged multi-root procedure output."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str | None = None
    output_dir: str
    num_roots: int
    num_succeeded: int
    preflight_only: bool = False
    cost: dict[str, Any] | None = None


class ProcedureRootResult(BaseModel):
    """One per-task-thread procedure generation result."""

    model_config = ConfigDict(extra="allow")

    input_file: str
    output_file: str
    ok: bool
    procedure_task_model: ProcedureTaskModel | None = None
    run_id: str | None = None
    session_id: str | None = None
    execution_mode: str | None = None
    usage: dict[str, Any] | None = None
    estimated_usd: float | None = None
    proxy_cost: dict[str, Any] | None = None
    error: str | None = None


class ProcedureModelInductionOutput(BaseModel):
    """Merged output for procedure model induction."""

    model_config = ConfigDict(extra="forbid")

    meta: ProcedureInductionMergedMeta
    roots: list[ProcedureRootResult]
