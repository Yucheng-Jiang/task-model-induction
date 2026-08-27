from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Update:
    content: str
    content_type: Literal["input_text", "input_image"]
