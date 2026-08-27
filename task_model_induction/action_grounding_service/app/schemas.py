from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScreenSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class CursorLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class ActionGroundingRequest(BaseModel):
    """Formal input schema for the action grounding service."""

    model_config = ConfigDict(extra="forbid")

    before_image: str = Field(..., description="Data URL containing a base64-encoded JPEG/PNG screenshot.")
    after_image: str | None = Field(
        default=None,
        description="Optional data URL containing a base64-encoded screenshot after the action.",
    )
    action: str
    screen_size: ScreenSize
    highlight_action: bool = Field(
        default=False,
        description="Draw a temporary action-location overlay on the before screenshot sent to grounding VLMs.",
    )
    cursor_location: CursorLocation | None = Field(
        default=None,
        description="Optional cursor point to highlight. If omitted and highlight_action is true, coordinates are parsed from action.",
    )

    @field_validator("before_image")
    @classmethod
    def validate_before_image(cls, value: str) -> str:
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError("before_image must be a data:image/...;base64,... URL")
        return value

    @field_validator("after_image")
    @classmethod
    def validate_after_image(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError("after_image must be null or a data:image/...;base64,... URL")
        return value


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., description="Data URL containing a base64-encoded JPEG/PNG screenshot.")

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError("image must be a data:image/...;base64,... URL")
        return value


class OcrResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    screen_size: ScreenSize
    md_results: str | None = Field(default=None, description="OCR/markdown text extracted from before_image.")
    layout_details: list[Any] = Field(default_factory=list)
    data_info: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CostItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    usd: float = Field(default=0.0, ge=0.0)
    model: str | None = None
    provider: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    source: str = Field(default="not_reported")


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_usd: float = Field(default=0.0, ge=0.0)
    items: list[CostItem] = Field(default_factory=list)


class CostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Literal["USD"] = "USD"
    total_usd: float = Field(default=0.0, ge=0.0)
    ocr: CostBreakdown = Field(default_factory=CostBreakdown)
    grounding: CostBreakdown = Field(default_factory=CostBreakdown)
    redaction: CostBreakdown = Field(default_factory=CostBreakdown)


class ActionGroundingResponse(BaseModel):
    """Formal response schema for action grounding output."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(default="", description="Concise immediate user intent.")
    active_application: str = Field(default="", description="Application/window/tab name.")
    visual_content: str = Field(default="", description="Focused UI/content artifact.")
    ocr_results: OcrResults
    md_results: str = Field(default="", description="Convenience copy of ocr_results.md_results.")
    warnings: list[str] = Field(default_factory=list)
    cost: CostSummary = Field(default_factory=CostSummary)


class OcrResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr_results: OcrResults
    md_results: str = Field(default="", description="Convenience copy of ocr_results.md_results.")
    warnings: list[str] = Field(default_factory=list)
    cost: CostSummary = Field(default_factory=CostSummary)
