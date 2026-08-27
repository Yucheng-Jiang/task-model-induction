from __future__ import annotations

import base64
import io

from PIL import Image

from action_grounding_service.app.zoom import build_zoom_views


def _image_from_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def _has_red_pixel_near(image: Image.Image, x: int, y: int, radius: int = 3) -> bool:
    for px in range(max(0, x - radius), min(image.width, x + radius + 1)):
        for py in range(max(0, y - radius), min(image.height, y + radius + 1)):
            red, green, blue = image.getpixel((px, py))
            if red >= 180 and green <= 80 and blue <= 80:
                return True
    return False


def test_zoom_view_marks_cursor_even_when_layout_bbox_matches() -> None:
    image = Image.new("RGB", (400, 300), "white")
    views = build_zoom_views(
        image,
        "click(100, 120)",
        [{"bbox_2d": [80, 90, 180, 160]}],
        screen_size={"width": 400, "height": 300},
    )

    crop = _image_from_data_url(views[0]["url"])

    # The crop is centered on the detected bbox. With this fixture it starts at
    # (0, 15), so the bbox top-left is (80, 75) and the cursor marker top-left
    # is (91, 96) in crop coordinates.
    assert _has_red_pixel_near(crop, 80, 75)
    assert _has_red_pixel_near(crop, 91, 96)
