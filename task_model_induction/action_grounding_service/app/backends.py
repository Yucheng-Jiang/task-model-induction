from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from .config import apply_service_config_to_env, litellm_call_kwargs, load_service_config
from .costs import breakdown, cost_item_from_litellm_response
from .prompts import OCR_SYSTEM_PROMPT, OCR_USER_PROMPT
from .schemas import ActionGroundingRequest


class OcrBackend(ABC):
    @abstractmethod
    def extract_markdown(self, request: ActionGroundingRequest) -> dict[str, Any]:
        """Return trajectory-compatible OCR data with at least md_results."""


class LiteLlmOcrBackend(OcrBackend):
    def extract_markdown(self, request: ActionGroundingRequest) -> dict[str, Any]:
        import litellm

        config = apply_service_config_to_env(load_service_config())
        ocr = config.ocr
        response = litellm.completion(
            model=ocr.model,
            messages=[
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": request.before_image}},
                    ],
                },
            ],
            max_tokens=ocr.max_tokens,
            timeout=ocr.timeout_secs,
            request_timeout=ocr.timeout_secs,
            **litellm_call_kwargs(ocr),
        )
        markdown = _strip_thinking_text(response.choices[0].message.content or "")
        return {
            "md_results": markdown.strip(),
            "layout_details": [],
            "provider": "litellm",
            "model": ocr.model,
            "cost": breakdown([cost_item_from_litellm_response("ocr_markdown", response, ocr.model)]),
        }


def get_backend() -> OcrBackend:
    return LiteLlmOcrBackend()


def _strip_thinking_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
