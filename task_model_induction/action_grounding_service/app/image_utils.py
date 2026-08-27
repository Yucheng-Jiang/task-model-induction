from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from PIL import Image


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    header, encoded = data_url.split(";base64,", 1)
    media_type = header.removeprefix("data:")
    try:
        return media_type, base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 image data") from exc


def data_url_bytes(data_url: str) -> bytes:
    return decode_data_url(data_url)[1]


def data_url_to_image(data_url: str) -> Image.Image:
    media_type, image_bytes = decode_data_url(data_url)
    if not media_type.startswith("image/"):
        raise ValueError(f"Expected image data URL, got {media_type!r}")
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def pil_to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def pil_to_data_url(image: Image.Image, quality: int = 85) -> str:
    encoded = base64.b64encode(pil_to_jpeg_bytes(image, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def data_url_to_suffix(data_url: str) -> str:
    media_type, _ = decode_data_url(data_url)
    suffix = mimetypes.guess_extension(media_type) or ".jpg"
    return ".jpg" if suffix in {".jpe", ".jpeg"} else suffix


@contextmanager
def data_url_as_temp_file(data_url: str) -> Iterator[Path]:
    suffix = data_url_to_suffix(data_url)
    _, image_bytes = decode_data_url(data_url)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(tmp.name)
    try:
        tmp.write(image_bytes)
        tmp.close()
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
