"""Schemas for semantic-action induction output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SemanticActionSourceAction(BaseModel):
    """One source action row covered by an induced semantic action.

    The optional evidence fields make this a lossless bridge from action
    grounding while keeping semantic-action files written before the evidence
    contract readable.
    """

    model_config = ConfigDict(extra="forbid")

    action_idx: int
    original_index: int | None = Field(default=None, ge=0)
    action_id: str | None = None
    action: str | None = None
    status: str | None = None
    goal: str | None = None
    active_application: str | None = None
    grounded_visual_content: str | None = None
    visual_content: str | None = None
    ocr_results: dict[str, Any] | None = None
    md_results: str | None = None
    state_before: str | None = None
    state_after: str | None = None
    time_before: float | str | None = None
    time_after: float | str | None = None
    time_range: float | None = None


class AtomSemanticAction(BaseModel):
    """One semantic action induced from low-level UI evidence."""

    model_config = ConfigDict(extra="forbid")

    semantic_action_id: str
    start_action_idx: int
    end_action_idx: int
    start_action_id: str | None = None
    end_action_id: str | None = None
    semantic_action: str
    action_details: str = ""
    actions: list[SemanticActionSourceAction] = Field(default_factory=list)
    raw_action_ids: list[str] = Field(default_factory=list)
    apps_used: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    pre_state: str = ""
    post_state: str = ""
    ocr_texts: list[str] = Field(default_factory=list)
    event_count: int


class SemanticActionInductionMeta(BaseModel):
    """Metadata for an atom semantic-action induction run."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str
    enrichment_model: str | None = None
    input_path: str
    input_fingerprint: str | None = None
    num_actions: int
    limits: int | None = None
    num_semantic_actions: int
    num_backward_semantic_actions: int | None = None
    visual_enrichment_workers: int | None = None
    action_detail_workers: int | None = None
    backward_batch_size: int
    backward_batch_overlap: int | None = None
    backward_workers: int | None = None
    max_future_semantic_actions: int
    merge_batch_size: int | None = None
    merge_batch_overlap: int | None = None
    max_prior_semantic_actions: int | None = None
    reused_cache: bool = False
    preflight_only: bool = False
    elapsed_secs: float | None = None
    llm_requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_usd: float | None = None
    cost_breakdown: dict[str, Any] | None = None


class SemanticActionInductionOutput(BaseModel):
    """In-memory output for the atom semantic-action induction stage."""

    model_config = ConfigDict(extra="forbid")

    meta: SemanticActionInductionMeta
    semantic_actions: list[AtomSemanticAction]
