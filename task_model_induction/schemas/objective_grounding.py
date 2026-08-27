"""Shared grounding contract for objective-model nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _non_empty_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must be a non-empty string")
    return value


class ObjectiveDeliverable(BaseModel):
    """A concrete artifact or state produced by accomplishing an objective."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    target: str
    expected_state: str
    evidence_refs: list[str] = Field(min_length=1)

    _validate_kind = field_validator("kind")(_non_empty_text)
    _validate_target = field_validator("target")(_non_empty_text)
    _validate_expected_state = field_validator("expected_state")(_non_empty_text)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        return _validate_evidence_refs(value)


class ObjectiveSuccessCriterion(BaseModel):
    """A verifiable predicate that defines success for an objective."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    verifier: str
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float | None = Field(ge=0.0, le=1.0)

    _validate_predicate = field_validator("predicate")(_non_empty_text)
    _validate_verifier = field_validator("verifier")(_non_empty_text)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        return _validate_evidence_refs(value)


class ObjectiveObservedOutcome(BaseModel):
    """What the trace actually establishes about the objective's outcome."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["achieved", "partial", "failed", "abandoned", "unknown"]
    description: str
    evidence_refs: list[str]

    _validate_description = field_validator("description")(_non_empty_text)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        return _validate_evidence_refs(value)


def _validate_evidence_refs(value: list[str]) -> list[str]:
    """Strip refs and reject empty or duplicate evidence identifiers."""

    normalized: list[str] = []
    seen: set[str] = set()
    for ref in value:
        stripped = ref.strip()
        if not stripped:
            raise ValueError("evidence_refs entries must be non-empty strings")
        if stripped in seen:
            raise ValueError(f"duplicate evidence ref: {stripped!r}")
        seen.add(stripped)
        normalized.append(stripped)
    return normalized
