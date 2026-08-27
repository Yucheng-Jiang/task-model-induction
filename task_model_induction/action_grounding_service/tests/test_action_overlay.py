from __future__ import annotations

from PIL import Image

from action_grounding_service.app.action_overlay import annotate_action_overlay
from action_grounding_service.app.schemas import CursorLocation


def _has_red_pixel_near(image: Image.Image, x: int, y: int, radius: int = 3) -> bool:
    for px in range(max(0, x - radius), min(image.width, x + radius + 1)):
        for py in range(max(0, y - radius), min(image.height, y + radius + 1)):
            red, green, blue = image.getpixel((px, py))
            if red >= 180 and green <= 80 and blue <= 80:
                return True
    return False


def test_action_overlay_draws_click_bbox_without_mutating_source() -> None:
    image = Image.new("RGB", (200, 160), "white")

    annotated = annotate_action_overlay(
        image,
        "click(50, 60)",
        screen_size={"width": 200, "height": 160},
    )

    assert image.getpixel((32, 42)) == (255, 255, 255)
    assert _has_red_pixel_near(annotated, 32, 42)


def test_action_overlay_scales_cursor_location_from_screen_size() -> None:
    image = Image.new("RGB", (100, 80), "white")

    annotated = annotate_action_overlay(
        image,
        "",
        screen_size={"width": 200, "height": 160},
        cursor_location=CursorLocation(x=100, y=80),
    )

    assert _has_red_pixel_near(annotated, 32, 22)
