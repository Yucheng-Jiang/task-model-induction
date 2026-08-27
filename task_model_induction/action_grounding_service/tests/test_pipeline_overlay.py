from __future__ import annotations

import asyncio
import io
from typing import Any

from PIL import Image
import pytest

from action_grounding_service.app import pipeline
from action_grounding_service.app.costs import breakdown
from action_grounding_service.app.image_utils import pil_to_data_url
from action_grounding_service.app.omniparser import OmniParserResult
from action_grounding_service.app.schemas import ActionGroundingRequest
from action_grounding_service.app.vlm import ContextOutput, GoalOutput, VlmRunResult


def _has_red_pixel_near(image: Image.Image, x: int, y: int, radius: int = 4) -> bool:
    for px in range(max(0, x - radius), min(image.width, x + radius + 1)):
        for py in range(max(0, y - radius), min(image.height, y + radius + 1)):
            red, green, blue = image.getpixel((px, py))
            if red >= 160 and green <= 100 and blue <= 100:
                return True
    return False


def _decode_image_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def test_ground_action_sends_highlighted_before_image_to_vlms(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_goal_image = None
    seen_context_image = None

    async def fake_infer_goal(**kwargs: Any) -> VlmRunResult[GoalOutput]:
        nonlocal seen_goal_image
        seen_goal_image = _decode_image_bytes(kwargs["before_image_bytes"])
        return VlmRunResult(output=GoalOutput(goal="click target"), cost=breakdown())

    async def fake_infer_context(**kwargs: Any) -> VlmRunResult[ContextOutput]:
        nonlocal seen_context_image
        seen_context_image = _decode_image_bytes(kwargs["before_image_bytes"])
        return VlmRunResult(
            output=ContextOutput(active_application="app", visual_content="target"),
            cost=breakdown(),
        )

    async def fake_extract_markdown(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"md_results": "", "layout_details": [], "data_info": {}}

    async def fake_parse_layout(*args: Any, **kwargs: Any) -> OmniParserResult:
        return OmniParserResult(layout_details=[], warnings=[])

    monkeypatch.setattr(pipeline, "infer_goal", fake_infer_goal)
    monkeypatch.setattr(pipeline, "infer_context", fake_infer_context)
    monkeypatch.setattr(pipeline, "_extract_markdown", fake_extract_markdown)
    monkeypatch.setattr(pipeline, "parse_layout_with_status", fake_parse_layout)

    request = ActionGroundingRequest.model_validate(
        {
            "before_image": pil_to_data_url(Image.new("RGB", (200, 160), "white")),
            "after_image": None,
            "action": "click(50, 60)",
            "screen_size": {"width": 200, "height": 160},
            "highlight_action": True,
        }
    )

    asyncio.run(pipeline.ground_action(request))

    assert seen_goal_image is not None
    assert seen_context_image is not None
    assert _has_red_pixel_near(seen_goal_image, 32, 42)
    assert _has_red_pixel_near(seen_context_image, 32, 42)
