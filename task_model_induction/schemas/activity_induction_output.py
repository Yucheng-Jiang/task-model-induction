"""Schemas for activity induction output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .semantic_action_induction_output import SemanticActionSourceAction


class Activity(BaseModel):
    """One activity: a contiguous segment of semantic actions explained by a single objective."""

    model_config = ConfigDict(extra="forbid")

    activity_id: str
    start_semantic_action_idx: int
    end_semantic_action_idx: int
    start_semantic_action_id: str | None = None
    end_semantic_action_id: str | None = None
    semantic_action_ids: list[str] = Field(default_factory=list)
    start_action_idx: int
    end_action_idx: int
    start_action_id: str | None = None
    end_action_id: str | None = None
    objective: str
    additional_context: str = ""
    semantic_actions: list[str] = Field(default_factory=list)
    source_actions: list[SemanticActionSourceAction] = Field(default_factory=list)
    raw_action_ids: list[str] = Field(default_factory=list)
    apps_used: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    pre_state: str = ""
    post_state: str = ""
    ocr_texts: list[str] = Field(default_factory=list)
    semantic_action_count: int
    event_count: int


class ActivityInductionMeta(BaseModel):
    """Metadata for an activity induction run."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    model: str
    input_path: str
    input_fingerprint: str | None = None
    output_path: str
    num_semantic_actions: int
    num_candidate_segments: int
    num_activities: int
    segmentation_batch_size: int
    merge_batch_size: int
    merge_batch_overlap: int
    max_prior_segments: int
    reused_cache: bool = False
    preflight_only: bool = False
    elapsed_secs: float | None = None
    llm_requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    estimated_usd: float | None = None
    cost_breakdown: dict[str, Any] | None = None


class ActivityInductionOutput(BaseModel):
    """In-memory output for the activity induction stage."""

    model_config = ConfigDict(extra="forbid")

    meta: ActivityInductionMeta
    activities: list[Activity]
