from __future__ import annotations

import asyncio
from typing import Any

from .action_overlay import annotate_action_overlay
from .backends import get_backend
from .costs import breakdown, combine_breakdowns, summarize_costs
from .image_utils import data_url_to_image, pil_to_jpeg_bytes
from .omniparser import parse_layout_with_status
from .schemas import ActionGroundingRequest, CostBreakdown, OcrResults, ScreenSize
from .vlm import infer_context, infer_goal
from .zoom import build_zoom_views


async def process_ocr(
    image_data_url: str,
    screen_size: ScreenSize | None = None,
) -> OcrResults:
    ocr_results, _ = await process_ocr_with_cost(image_data_url, screen_size)
    return ocr_results


async def process_ocr_with_cost(
    image_data_url: str,
    screen_size: ScreenSize | None = None,
) -> tuple[OcrResults, CostBreakdown]:
    image = data_url_to_image(image_data_url)
    width, height = image.size
    image_bytes = pil_to_jpeg_bytes(image)
    effective_screen_size = screen_size or ScreenSize(width=width, height=height)

    request = ActionGroundingRequest(
        before_image=image_data_url,
        after_image=None,
        action="",
        screen_size=effective_screen_size,
    )

    markdown_task = asyncio.create_task(_extract_markdown(request))
    layout_task = asyncio.create_task(parse_layout_with_status(image_bytes, width, height))
    markdown_data, omniparser_result = await asyncio.gather(markdown_task, layout_task)

    ocr_cost = _cost_from_markdown_data(markdown_data)
    md_results = str(markdown_data.get("md_results") or "")
    fallback_layout = markdown_data.get("layout_details")
    layout_details = omniparser_result.layout_details or (fallback_layout if isinstance(fallback_layout, list) else [])
    data_info = markdown_data.get("data_info")
    if not isinstance(data_info, dict):
        data_info = {"num_pages": 1, "pages": [{"width": width, "height": height}]}

    return (
        OcrResults.model_validate(
            {
                "screen_size": effective_screen_size.model_dump(),
                "md_results": md_results,
                "layout_details": layout_details,
                "data_info": data_info,
                "warnings": omniparser_result.warnings,
            }
        ),
        ocr_cost,
    )


async def ground_action(
    request: ActionGroundingRequest,
) -> dict[str, Any]:
    before_image = data_url_to_image(request.before_image)
    before_image_bytes = pil_to_jpeg_bytes(before_image)
    grounding_before_image = before_image
    if request.highlight_action or request.cursor_location is not None:
        grounding_before_image = annotate_action_overlay(
            before_image,
            request.action,
            screen_size=request.screen_size.model_dump(),
            cursor_location=request.cursor_location,
        )
    grounding_before_image_bytes = pil_to_jpeg_bytes(grounding_before_image)
    after_image_bytes = _optional_image_bytes(request.after_image)

    goal_task = asyncio.create_task(
        infer_goal(
            action=request.action,
            before_image_bytes=grounding_before_image_bytes,
            after_image_bytes=after_image_bytes,
        )
    )
    markdown_task = asyncio.create_task(_extract_markdown(request))
    layout_task = asyncio.create_task(parse_layout_with_status(before_image_bytes, before_image.width, before_image.height))

    omniparser_result = await layout_task
    layout_details = omniparser_result.layout_details
    zoom_views = build_zoom_views(
        before_image,
        request.action,
        layout_details,
        screen_size=request.screen_size.model_dump(),
    )
    context_task = asyncio.create_task(
        infer_context(
            action=request.action,
            before_image_bytes=grounding_before_image_bytes,
            zoom_views=zoom_views,
            after_image_bytes=after_image_bytes,
        )
    )

    goal_result, context_result, markdown_data = await asyncio.gather(
        goal_task,
        context_task,
        markdown_task,
    )

    ocr_cost = _cost_from_markdown_data(markdown_data)
    grounding_cost = combine_breakdowns(goal_result.cost, context_result.cost)
    md_results = str(markdown_data.get("md_results") or "")
    fallback_layout = markdown_data.get("layout_details")
    ocr_results = OcrResults.model_validate(
        {
            "screen_size": request.screen_size.model_dump(),
            "md_results": md_results,
            "layout_details": layout_details or (fallback_layout if isinstance(fallback_layout, list) else []),
            "data_info": _data_info(markdown_data, before_image.width, before_image.height),
            "warnings": omniparser_result.warnings,
        }
    )
    return {
        "goal": goal_result.output.goal.strip(),
        "active_application": context_result.output.active_application.strip(),
        "visual_content": context_result.output.visual_content.strip(),
        "ocr_results": ocr_results,
        "md_results": md_results,
        "warnings": omniparser_result.warnings,
        "cost": summarize_costs(
            ocr=ocr_cost,
            grounding=grounding_cost,
            redaction=breakdown(),
        ),
    }


async def _extract_markdown(
    request: ActionGroundingRequest,
) -> dict[str, Any]:
    return await asyncio.to_thread(get_backend().extract_markdown, request)


def _optional_image_bytes(image_data_url: str | None) -> bytes | None:
    if not image_data_url:
        return None
    image = data_url_to_image(image_data_url)
    return pil_to_jpeg_bytes(image)


def _data_info(markdown_data: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    data_info = markdown_data.get("data_info")
    if isinstance(data_info, dict):
        return data_info
    return {"num_pages": 1, "pages": [{"width": width, "height": height}]}


def _cost_from_markdown_data(markdown_data: dict[str, Any]) -> CostBreakdown:
    cost = markdown_data.get("cost")
    if isinstance(cost, CostBreakdown):
        return cost
    if isinstance(cost, dict):
        return CostBreakdown.model_validate(cost)
    return breakdown()
