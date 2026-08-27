from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


_OMNIPARSER_SEMAPHORE: asyncio.Semaphore | None = None
_OMNIPARSER_SEMAPHORE_LIMIT: int | None = None


@dataclass(frozen=True)
class OmniParserResult:
    layout_details: list[dict[str, Any]]
    warnings: list[str]


def normalize_bbox(raw_bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def omniparser_base_url() -> str:
    """Base URL of the OmniParser container, from the environment."""
    return os.getenv("OMNIPARSER_URL", "http://action-grounding-omniparser:8080").rstrip("/")


async def parse_layout(image_bytes: bytes, width: int, height: int) -> list[dict[str, Any]]:
    return (await parse_layout_with_status(image_bytes, width, height)).layout_details


async def parse_layout_with_status(image_bytes: bytes, width: int, height: int) -> OmniParserResult:
    url = omniparser_base_url()
    timeout = float(os.getenv("OMNIPARSER_TIMEOUT_SECS", "180"))
    try:
        async with _omniparser_semaphore():
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{url}/parse/file",
                    files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                    data={
                        "box_threshold": os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05"),
                        "iou_threshold": os.getenv("OMNIPARSER_IOU_THRESHOLD", "0.1"),
                        "imgsz": os.getenv("OMNIPARSER_IMGSZ", "640"),
                    },
                )
        response.raise_for_status()
    except Exception as exc:
        warning = f"OmniParser unavailable or unhealthy: {exc}"
        print(warning, flush=True)
        return OmniParserResult(layout_details=[], warnings=[warning])

    payload = response.json()
    parsed_content = payload.get("parsed_content")
    if not isinstance(parsed_content, list):
        return OmniParserResult(
            layout_details=[],
            warnings=["OmniParser returned an invalid response: parsed_content is missing or not a list"],
        )

    layout_items: list[dict[str, Any]] = []
    for index, item in enumerate(parsed_content):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"), width, height)
        if bbox is None:
            continue

        native_label = str(item.get("type", "unknown"))
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        layout_items.append(
            {
                "index": index,
                "label": "text" if native_label == "text" else "image",
                "bbox_2d": bbox,
                "content": content,
                "width": width,
                "height": height,
                "native_label": native_label,
                "provider": "omniparser",
            }
        )
    return OmniParserResult(layout_details=layout_items, warnings=[])


async def check_omniparser_health(timeout: float = 3.0) -> tuple[bool, str]:
    url = omniparser_base_url()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/health")
        response.raise_for_status()
    except Exception as exc:
        return False, f"OmniParser unhealthy at {url}/health: {exc}"
    return True, "ok"


def _omniparser_semaphore() -> asyncio.Semaphore:
    global _OMNIPARSER_SEMAPHORE, _OMNIPARSER_SEMAPHORE_LIMIT
    limit = int(os.getenv("OMNIPARSER_MAX_CONCURRENT_REQUESTS", "2"))
    if _OMNIPARSER_SEMAPHORE is None or _OMNIPARSER_SEMAPHORE_LIMIT != limit:
        _OMNIPARSER_SEMAPHORE = asyncio.Semaphore(limit)
        _OMNIPARSER_SEMAPHORE_LIMIT = limit
    return _OMNIPARSER_SEMAPHORE
