from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from .schemas import CursorLocation
from .zoom import _clamp, _coordinate_scale, _parse_action_points


OVERLAY_COLOR = (255, 0, 0)
MIN_POINT_BOX_HALF_SIZE = 18
POINT_BOX_SCREEN_RATIO = 0.035
DRAG_PADDING_RATIO = 0.025


def annotate_action_overlay(
    image: Image.Image,
    action: str,
    screen_size: dict[str, int] | None,
    cursor_location: CursorLocation | None = None,
) -> Image.Image:
    """Return a copy of image with a temporary action-location overlay."""
    annotated = image.convert("RGB").copy()
    points = _overlay_points(action, annotated.width, annotated.height, screen_size, cursor_location)
    if not points:
        return annotated

    draw = ImageDraw.Draw(annotated)
    line_width = max(4, int(round(min(annotated.width, annotated.height) * 0.007)))
    if len(points) >= 2 and "drag" in action.lower():
        _draw_drag_bbox(draw, points[0], points[1], annotated.width, annotated.height, line_width)
    else:
        for _, point in points:
            _draw_point_bbox(draw, point, annotated.width, annotated.height, line_width)
    return annotated


def _overlay_points(
    action: str,
    img_w: int,
    img_h: int,
    screen_size: dict[str, int] | None,
    cursor_location: CursorLocation | None,
) -> list[tuple[str, tuple[float, float]]]:
    if cursor_location is not None:
        raw_points = [("cursor", (cursor_location.x, cursor_location.y))]
    else:
        raw_points = _parse_action_points(action)
    scale_x, scale_y = _coordinate_scale(screen_size, img_w, img_h)
    return [
        (label, (_clamp(x * scale_x, 0, max(0, img_w - 1)), _clamp(y * scale_y, 0, max(0, img_h - 1))))
        for label, (x, y) in raw_points
    ]


def _draw_point_bbox(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    width: int,
    height: int,
    line_width: int,
) -> None:
    x, y = point
    half_size = max(MIN_POINT_BOX_HALF_SIZE, int(round(min(width, height) * POINT_BOX_SCREEN_RATIO)))
    x1 = int(round(_clamp(x - half_size, 0, width - 1)))
    y1 = int(round(_clamp(y - half_size, 0, height - 1)))
    x2 = int(round(_clamp(x + half_size, 0, width - 1)))
    y2 = int(round(_clamp(y + half_size, 0, height - 1)))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=OVERLAY_COLOR, width=line_width)


def _draw_drag_bbox(
    draw: ImageDraw.ImageDraw,
    start: tuple[Any, tuple[float, float]],
    end: tuple[Any, tuple[float, float]],
    width: int,
    height: int,
    line_width: int,
) -> None:
    sx, sy = start[1]
    ex, ey = end[1]
    pad = max(20, int(round(min(width, height) * DRAG_PADDING_RATIO)))
    x1 = int(round(_clamp(min(sx, ex) - pad, 0, width - 1)))
    y1 = int(round(_clamp(min(sy, ey) - pad, 0, height - 1)))
    x2 = int(round(_clamp(max(sx, ex) + pad, 0, width - 1)))
    y2 = int(round(_clamp(max(sy, ey) + pad, 0, height - 1)))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=OVERLAY_COLOR, width=line_width)
