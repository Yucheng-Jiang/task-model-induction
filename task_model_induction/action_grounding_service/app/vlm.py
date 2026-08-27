from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .config import apply_service_config_to_env, litellm_call_kwargs, load_service_config
from .costs import breakdown, cost_item_from_litellm_response
from .prompts import CONTEXT_SYSTEM_PROMPT, GOAL_SYSTEM_PROMPT
from .schemas import CostBreakdown


OutputT = TypeVar("OutputT", bound=BaseModel)


class GoalOutput(BaseModel):
    goal: str = Field(default="", description="Immediate user action intent.")


class ContextOutput(BaseModel):
    active_application: str = Field(default="", description="Application/window/tab name visible on screen.")
    visual_content: str = Field(default="", description="Specific focused visual artifact.")


@dataclass(frozen=True)
class VlmRunResult(Generic[OutputT]):
    output: OutputT
    cost: CostBreakdown


async def infer_goal(
    action: str,
    before_image_bytes: bytes,
    after_image_bytes: bytes | None = None,
    model_name: str | None = None,
) -> VlmRunResult[GoalOutput]:
    user_content = _base_content(action, before_image_bytes, after_image_bytes)
    return await _run_litellm("goal_vlm", GOAL_SYSTEM_PROMPT, GoalOutput, user_content, model_name)


async def infer_context(
    action: str,
    before_image_bytes: bytes,
    zoom_views: list[dict[str, str]],
    after_image_bytes: bytes | None = None,
    model_name: str | None = None,
) -> VlmRunResult[ContextOutput]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Action: {action}"},
        {"type": "text", "text": "Screenshot exactly at the moment the action happened:"},
        {"type": "image_url", "image_url": {"url": _image_data_url(before_image_bytes)}},
    ]
    for view in zoom_views:
        user_content.extend(
            [
                {"type": "text", "text": view["caption"]},
                {"type": "image_url", "image_url": {"url": view["url"]}},
            ]
        )
    if after_image_bytes is not None:
        user_content.extend(
            [
                {"type": "text", "text": "Screenshot taken shortly after the action happened:"},
                {"type": "image_url", "image_url": {"url": _image_data_url(after_image_bytes)}},
            ]
        )
    return await _run_litellm("context_vlm", CONTEXT_SYSTEM_PROMPT, ContextOutput, user_content, model_name)


def _base_content(
    action: str,
    before_image_bytes: bytes,
    after_image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"Action: {action}"},
        {"type": "text", "text": "Screenshot exactly at the moment the action happened:"},
        {"type": "image_url", "image_url": {"url": _image_data_url(before_image_bytes)}},
    ]
    if after_image_bytes is not None:
        parts.extend(
            [
                {"type": "text", "text": "Screenshot taken shortly after the action happened:"},
                {"type": "image_url", "image_url": {"url": _image_data_url(after_image_bytes)}},
            ]
        )
    return parts


async def _run_litellm(
    cost_item_name: str,
    instructions: str,
    output_type: type[OutputT],
    user_content: list[dict[str, Any]],
    model_name: str | None,
) -> VlmRunResult[OutputT]:
    import litellm

    config = apply_service_config_to_env(load_service_config())
    vlm = config.vlm
    model = model_name or vlm.model
    completion_kwargs = litellm_call_kwargs(vlm)
    completion_kwargs.setdefault("temperature", 1 if _requires_temperature_one(model) else 0)

    schema_hint = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                instructions
                + "\n\nReturn only a JSON object that conforms to this schema. "
                + "Do not include Markdown, thinking, reasoning, or extra keys.\n"
                + schema_hint
            ),
        },
        {"role": "user", "content": user_content},
    ]

    timeout = float(vlm.timeout_secs)
    response = await asyncio.wait_for(
        asyncio.to_thread(
            litellm.completion,
            model=model,
            messages=messages,
            max_tokens=vlm.max_tokens,
            timeout=timeout,
            request_timeout=timeout,
            **completion_kwargs,
        ),
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    output = _parse_output(content, output_type)
    cost = breakdown([cost_item_from_litellm_response(cost_item_name, response, model)])
    return VlmRunResult(output=output, cost=cost)


def _parse_output(text: str, output_type: type[OutputT]) -> OutputT:
    text = _strip_thinking_text(text)
    try:
        return output_type.model_validate_json(text)
    except ValidationError:
        pass
    except ValueError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return output_type.model_validate_json(match.group(0))
        except Exception:
            pass
    return output_type()


def _image_data_url(image_bytes: bytes) -> str:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


def _requires_temperature_one(model: str) -> bool:
    return "gpt-5" in model.lower()


def _strip_thinking_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
