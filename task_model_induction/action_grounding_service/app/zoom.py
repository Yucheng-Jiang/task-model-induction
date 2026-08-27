from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image, ImageDraw

from .image_utils import pil_to_data_url


FLOAT_RE = r"-?\d+(?:\.\d+)?"
NUMBER_PATTERN = re.compile(FLOAT_RE)
CLICK_PATTERN = re.compile(
    rf"(?:click(?:_(?:left|right|middle))?|double_click|right_click|middle_click|triple_click)"
    rf"\(\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*\)"
)
MOVE_PATTERN = re.compile(
    rf"(?:move_to|move|mouse_move|mousemove|move_mouse)\(\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*\)"
)

MAX_BBOX_AREA_RATIO = 0.20
MAX_BBOX_DIM_RATIO = 0.85
OCR_BBOX_PADDING_PX = 24
MIN_OCR_CROP_W = 320
MIN_OCR_CROP_H = 220
FALLBACK_CROP_W = 640
FALLBACK_CROP_H = 420
CURSOR_MARKER_SIZE = 18
ANNOTATION_COLOR = (255, 0, 0)


def build_zoom_views(
    image: Image.Image,
    action: str,
    layout_details: list[Any],
    screen_size: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    points = _parse_action_points(action)
    if not points:
        return []

    img_w, img_h = image.size
    entries = _extract_box_entries(layout_details, img_w, img_h)
    scale_x, scale_y = _coordinate_scale(screen_size, img_w, img_h)

    views: list[dict[str, str]] = []
    for label, (raw_x, raw_y) in points:
        x = _clamp(raw_x * scale_x, 0, max(0, img_w - 1))
        y = _clamp(raw_y * scale_y, 0, max(0, img_h - 1))
        entry = _pick_bbox_for_point(x, y, entries, img_w, img_h)
        bbox = entry["bbox"] if entry else None

        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            crop_w = max(MIN_OCR_CROP_W, int(round((bx2 - bx1) + 2 * OCR_BBOX_PADDING_PX)))
            crop_h = max(MIN_OCR_CROP_H, int(round((by2 - by1) + 2 * OCR_BBOX_PADDING_PX)))
            crop_box = _centered_rect((bx1 + bx2) / 2, (by1 + by2) / 2, crop_w, crop_h, img_w, img_h)
            source_text = "detected UI element"
        else:
            crop_box = _centered_rect(x, y, FALLBACK_CROP_W, FALLBACK_CROP_H, img_w, img_h)
            source_text = "action coordinate"

        crop = image.crop(crop_box)
        if bbox is not None:
            _draw_bbox_on_crop(crop, crop_box, bbox)
        _draw_cursor_marker_on_crop(crop, crop_box, x, y)

        views.append(
            {
                "caption": f"Zoomed view around {label} ({source_text}):",
                "url": pil_to_data_url(crop),
            }
        )
    return views


def _parse_action_points(action: str) -> list[tuple[str, tuple[float, float]]]:
    action_lower = action.lower()
    if "drag" in action_lower:
        named: dict[str, float] = {}
        for key in ("start_x", "start_y", "end_x", "end_y"):
            match = re.search(rf"{key}\s*=\s*({FLOAT_RE})", action)
            if match:
                named[key] = float(match.group(1))
        if {"start_x", "start_y", "end_x", "end_y"}.issubset(named):
            return [
                ("drag start", (named["start_x"], named["start_y"])),
                ("drag end", (named["end_x"], named["end_y"])),
            ]
        values = [float(value) for value in NUMBER_PATTERN.findall(action)]
        if len(values) >= 4:
            return [("drag start", (values[0], values[1])), ("drag end", (values[2], values[3]))]
        if len(values) >= 2:
            return [("drag", (values[0], values[1]))]
        return []

    if "click" in action_lower:
        match = CLICK_PATTERN.search(action)
        return [("click", (float(match.group(1)), float(match.group(2))))] if match else []

    if "move_to(" in action_lower or re.search(r"\bmove\(", action_lower):
        matches = MOVE_PATTERN.findall(action)
        if matches:
            x_str, y_str = matches[-1]
            return [("move", (float(x_str), float(y_str)))]
    return []


def _extract_box_entries(layout_details: list[Any], img_w: int, img_h: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in layout_details:
        if isinstance(entry, dict):
            items.append(entry)
        elif isinstance(entry, list):
            items.extend(item for item in entry if isinstance(item, dict))

    entries: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        bbox = _coerce_bbox_to_xyxy(item.get("bbox_2d"), img_w, img_h)
        if bbox is None:
            bbox = _coerce_bbox_to_xyxy(item.get("bbox"), img_w, img_h)
        if bbox is not None:
            entries.append({"layout_index": index, "bbox": bbox})
    return entries


def _coerce_bbox_to_xyxy(raw_bbox: Any, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    if isinstance(raw_bbox, str):
        try:
            raw_bbox = json.loads(raw_bbox)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_bbox, dict):
        values = [raw_bbox.get("x1"), raw_bbox.get("y1"), raw_bbox.get("x2"), raw_bbox.get("y2")]
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        values = list(raw_bbox)
    else:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= img_w
        x2 *= img_w
        y1 *= img_h
        y2 *= img_h
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1, y1 = max(0.0, min(x1, float(img_w))), max(0.0, min(y1, float(img_h)))
    x2, y2 = max(0.0, min(x2, float(img_w))), max(0.0, min(y2, float(img_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _pick_bbox_for_point(
    x: float,
    y: float,
    entries: list[dict[str, Any]],
    img_w: int,
    img_h: int,
) -> dict[str, Any] | None:
    inside = [
        entry
        for entry in entries
        if _bbox_is_reasonable(entry["bbox"], img_w, img_h)
        and entry["bbox"][0] <= x <= entry["bbox"][2]
        and entry["bbox"][1] <= y <= entry["bbox"][3]
    ]
    if not inside:
        return None
    inside.sort(key=lambda entry: (entry["bbox"][2] - entry["bbox"][0]) * (entry["bbox"][3] - entry["bbox"][1]))
    return inside[0]


def _bbox_is_reasonable(bbox: tuple[float, float, float, float], img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    area_ratio = (bw * bh) / float(max(1, img_w * img_h))
    return (
        area_ratio <= MAX_BBOX_AREA_RATIO
        and bw <= MAX_BBOX_DIM_RATIO * img_w
        and bh <= MAX_BBOX_DIM_RATIO * img_h
    )


def _coordinate_scale(screen_size: dict[str, int] | None, img_w: int, img_h: int) -> tuple[float, float]:
    if not screen_size:
        return 1.0, 1.0
    screen_w = screen_size.get("width")
    screen_h = screen_size.get("height")
    if not screen_w or not screen_h:
        return 1.0, 1.0
    return img_w / float(screen_w), img_h / float(screen_h)


def _centered_rect(cx: float, cy: float, crop_w: int, crop_h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    crop_w = max(1, min(int(round(crop_w)), img_w))
    crop_h = max(1, min(int(round(crop_h)), img_h))
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(x1, img_w - crop_w))
    y1 = max(0, min(y1, img_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def _draw_bbox_on_crop(crop: Image.Image, crop_box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> None:
    draw = ImageDraw.Draw(crop)
    offset_x, offset_y = crop_box[0], crop_box[1]
    x1 = max(0, int(round(bbox[0] - offset_x)))
    y1 = max(0, int(round(bbox[1] - offset_y)))
    x2 = min(crop.width, int(round(bbox[2] - offset_x)))
    y2 = min(crop.height, int(round(bbox[3] - offset_y)))
    if x2 > x1 and y2 > y1:
        draw.rectangle([(x1, y1), (x2, y2)], outline=ANNOTATION_COLOR, width=3)


def _draw_cursor_marker_on_crop(crop: Image.Image, crop_box: tuple[int, int, int, int], x: float, y: float) -> None:
    draw = ImageDraw.Draw(crop)
    local_x = int(round(x - crop_box[0]))
    local_y = int(round(y - crop_box[1]))
    half = max(2, CURSOR_MARKER_SIZE // 2)
    x1, y1 = max(0, local_x - half), max(0, local_y - half)
    x2, y2 = min(crop.width - 1, local_x + half), min(crop.height - 1, local_y + half)
    if x2 > x1 and y2 > y1:
        draw.rectangle([(x1, y1), (x2, y2)], outline=ANNOTATION_COLOR, width=3)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))
