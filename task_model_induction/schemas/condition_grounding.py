"""Grounding contract for condition-driven procedure loops."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


WhileConditionStatus = Literal["satisfied", "unsatisfied", "unknown"]


class WhileConditionGrounding(BaseModel):
    """Observable exit predicate and the trace evidence used to assess it."""

    model_config = ConfigDict(extra="forbid")

    predicate: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    observed_status: WhileConditionStatus

    @field_validator("predicate", "verifier")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        normalized = [ref.strip() for ref in value]
        if any(not ref for ref in normalized):
            raise ValueError("evidence_refs entries must be non-empty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_refs must not contain duplicates")
        return normalized
