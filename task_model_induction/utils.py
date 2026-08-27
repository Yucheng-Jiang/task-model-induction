"""Utility helpers for task model induction."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib import request

from PIL import Image

try:
    from task_model_induction.schemas import ActionGroundingOutput, ComputerUseActivityEntry
except ModuleNotFoundError:
    from schemas import ActionGroundingOutput, ComputerUseActivityEntry


_LITELLM_MODEL_OVERRIDES: ContextVar[tuple[tuple[str, dict[str, Any]], ...]] = ContextVar(
    "TMI_LITELLM_MODEL_OVERRIDES",
    default=(),
)
_LITELLM_PROXY_API_KEY_OVERRIDE: ContextVar[str | None] = ContextVar(
    "TMI_LITELLM_PROXY_API_KEY_OVERRIDE",
    default=None,
)
_MANUAL_USAGE_PRICE_OVERRIDES: dict[str, dict[str, Decimal | int]] = {
    # Fallback until genai-prices ships these entries.
    # gpt-5.4: verified from direct-LLM call data (C2 gen: in=8014 out=2587 → $0.0588).
    "gpt-5.4": {
        "input_mtok": Decimal("2.5"),
        "cache_read_mtok": Decimal("0.25"),
        "output_mtok": Decimal("15"),
        "high_context_threshold": 272000,
        "high_context_input_mtok": Decimal("5"),
        "high_context_cache_read_mtok": Decimal("0.5"),
        "high_context_output_mtok": Decimal("22.5"),
    },
    # gpt-5.5: 2× gpt-5.4 across all tiers.
    "gpt-5.5": {
        "input_mtok": Decimal("5"),
        "cache_read_mtok": Decimal("0.5"),
        "output_mtok": Decimal("30"),
        "high_context_threshold": 272000,
        "high_context_input_mtok": Decimal("10"),
        "high_context_cache_read_mtok": Decimal("1"),
        "high_context_output_mtok": Decimal("45"),
    },
}


def set_litellm_proxy_api_key(api_key: str | None) -> Token[str | None]:
    return _LITELLM_PROXY_API_KEY_OVERRIDE.set(api_key.strip() if isinstance(api_key, str) and api_key.strip() else None)


def reset_litellm_proxy_api_key(token: Token[str | None]) -> None:
    _LITELLM_PROXY_API_KEY_OVERRIDE.reset(token)


@contextmanager
def litellm_model_config(
    *,
    model_alias: str | None,
    litellm_params: dict[str, Any] | None,
) -> Iterator[None]:
    if not model_alias or not litellm_params:
        yield
        return
    current = _LITELLM_MODEL_OVERRIDES.get()
    token = _LITELLM_MODEL_OVERRIDES.set((*current, (model_alias, dict(litellm_params))))
    try:
        yield
    finally:
        _LITELLM_MODEL_OVERRIDES.reset(token)


@contextmanager
def litellm_model_configs(
    mappings: list[tuple[str | None, dict[str, Any] | None]],
) -> Iterator[None]:
    additions = tuple(
        (model_alias, dict(params))
        for model_alias, params in mappings
        if model_alias and params
    )
    if not additions:
        yield
        return
    current = _LITELLM_MODEL_OVERRIDES.get()
    token = _LITELLM_MODEL_OVERRIDES.set((*current, *additions))
    try:
        yield
    finally:
        _LITELLM_MODEL_OVERRIDES.reset(token)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_with_retries(
    fn: Callable[[], Any],
    *,
    attempts: int,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Any:
    """Run fn, retrying on any exception up to `attempts` total tries.

    Codex sandbox runs fail transiently (mid-stream crashes, provider-side
    'item not found' errors); a fresh attempt usually succeeds, so root-level
    callers retry instead of failing the whole stage on one bad run.
    """
    attempts = max(1, attempts)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry boundary for opaque subprocess failures
            last = exc
            if attempt < attempts and on_retry is not None:
                on_retry(attempt, exc)
    assert last is not None
    raise last


DIRECT_LLM_MAX_INPUT_BYTES = 1_500_000


def direct_llm_input_too_large(
    *paths: Path | None,
    max_bytes: int = DIRECT_LLM_MAX_INPUT_BYTES,
) -> bool:
    """True when the combined serialized inputs are too large for one direct-LLM request.

    Activity count does not bound payload size — a few dozen activities with heavy
    OCR/visual context can exceed provider message-size and context-window limits —
    so direct-vs-codex routing must also check bytes on disk. Unreadable paths count
    as too large so routing falls back to the codex branch rather than failing later.
    """
    total = 0
    for path in paths:
        if path is None:
            continue
        try:
            total += path.stat().st_size
        except OSError:
            return True
    return total > max_bytes


def sanitize_for_utf8_json(value: Any) -> Any:
    """Return a JSON-like value with lone surrogates made UTF-8 encodable."""
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [sanitize_for_utf8_json(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_utf8_json(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_for_utf8_json(key) if isinstance(key, str) else key: sanitize_for_utf8_json(item)
            for key, item in value.items()
        }
    return value


def litellm_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float | None = None,
    request_timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    model, kwargs = _apply_litellm_model_override(model, kwargs)
    messages = sanitize_for_utf8_json(messages)
    kwargs = sanitize_for_utf8_json(kwargs)
    proxy_base = os.environ.get("TMI_LITELLM_PROXY_API_BASE", "").strip()
    if proxy_base:
        proxy_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"api_key", "api_base", "base_url", "custom_llm_provider"}
        }
        return openai_compatible_completion(
            base_url=proxy_base,
            api_key=_LITELLM_PROXY_API_KEY_OVERRIDE.get()
            or os.environ.get("TMI_LITELLM_PROXY_API_KEY", "").strip(),
            model=model,
            messages=messages,
            timeout=timeout or request_timeout,
            **proxy_kwargs,
        )

    import litellm

    return litellm.completion(
        model=model,
        messages=messages,
        timeout=timeout,
        request_timeout=request_timeout,
        **kwargs,
    )


def openai_compatible_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        key: value
        for key, value in kwargs.items()
        if value is not None
        and key not in {"api_key", "api_base", "base_url", "custom_llm_provider", "request_timeout", "timeout"}
    }
    payload["model"] = model
    payload["messages"] = sanitize_for_utf8_json(messages)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def completion_message_content(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                return sanitize_for_utf8_json(str(message.get("content") or ""))
        return ""
    return sanitize_for_utf8_json(response.choices[0].message.content or "")


def normalize_litellm_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None and hasattr(response, "model_dump"):
        try:
            usage = response.model_dump().get("usage")
        except Exception:
            usage = None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = {
            key: getattr(usage, key)
            for key in (
                "input_tokens",
                "prompt_tokens",
                "cache_read_tokens",
                "cached_input_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
            )
            if hasattr(usage, key)
        }
    if not isinstance(usage, dict):
        return {}
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    cache_read_tokens = _cached_input_tokens(usage)
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens") or input_tokens + output_tokens
    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if cache_read_tokens:
        normalized["cache_read_tokens"] = min(cache_read_tokens, input_tokens) if input_tokens else cache_read_tokens
    return normalized


def estimated_litellm_completion_cost_usd(response: Any, model_name: str) -> float | None:
    usage = normalize_litellm_usage(response)
    reported = litellm_reported_response_cost(response)
    if reported is not None and (reported > 0 or not any(usage.values())):
        return float(reported)
    try:
        import litellm

        cost = litellm.completion_cost(completion_response=response)
        if cost is not None and (float(cost) > 0.0 or not any(usage.values())):
            return float(cost)
    except Exception:
        pass
    return estimate_litellm_usage_cost_usd(usage, _effective_litellm_model_name(model_name))


def estimate_litellm_usage_cost_usd(usage: dict[str, int], model_name: str) -> float | None:
    if not usage:
        return None
    provider, model_ref = split_provider_model(model_name)
    try:
        from genai_prices import Usage, calc_price

        price = calc_price(
            Usage(
                input_tokens=usage.get("input_tokens") or None,
                cache_read_tokens=usage.get("cache_read_tokens") or None,
                output_tokens=usage.get("output_tokens") or None,
            ),
            model_ref,
            provider_id=provider,
        ).total_price
        return float(price)
    except Exception:
        return _manual_usage_price_usd(usage, provider=provider, model_ref=model_ref)


def litellm_reported_response_cost(response: Any) -> Decimal | None:
    hidden = getattr(response, "_hidden_params", None)
    if hidden is None and isinstance(response, dict):
        hidden = response.get("_hidden_params")
    if not isinstance(hidden, dict):
        return None
    value = hidden.get("response_cost")
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def split_provider_model(model: str) -> tuple[str | None, str]:
    if ":" in model:
        provider, model_ref = model.split(":", 1)
        return provider or None, model_ref
    if "/" in model:
        provider, model_ref = model.split("/", 1)
        return provider or None, model_ref
    return None, model


def _apply_litellm_model_override(model: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = _litellm_override_params(model)
    if not params:
        return model, kwargs
    resolved = {
        key: _resolve_env_ref(value)
        for key, value in params.items()
        if key != "model"
    }
    resolved = {key: value for key, value in resolved.items() if value is not None}
    return str(params.get("model") or model), {**resolved, **kwargs}


def _effective_litellm_model_name(model: str) -> str:
    params = _litellm_override_params(model)
    if params and params.get("model"):
        return str(params["model"])
    return model


def _litellm_override_params(model: str) -> dict[str, Any] | None:
    for alias, params in reversed(_LITELLM_MODEL_OVERRIDES.get()):
        if alias == model:
            return params
    raw = os.environ.get("TMI_LITELLM_MODEL_PARAMS_JSON", "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            params = payload.get(model)
            if isinstance(params, dict):
                return params
    return None


def _resolve_env_ref(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("os.environ/"):
        return os.environ.get(value.removeprefix("os.environ/"))
    return value


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def _cached_input_tokens(usage: dict[str, Any]) -> int:
    cached = _usage_int(usage, "cache_read_tokens", "cached_input_tokens")
    if cached:
        return cached
    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict):
            cached = _usage_int(details, "cached_tokens")
            if cached:
                return cached
    return 0


def _manual_usage_price_usd(usage: dict[str, int], *, provider: str | None, model_ref: str) -> float | None:
    normalized_provider = (provider or "").strip().lower() or None
    if normalized_provider not in (None, "openai"):
        return None
    prices = _manual_model_prices(model_ref)
    if prices is None:
        return None
    input_tokens = max(0, int(usage.get("input_tokens") or 0))
    cache_read_tokens = min(max(0, int(usage.get("cache_read_tokens") or 0)), input_tokens)
    output_tokens = max(0, int(usage.get("output_tokens") or 0))
    input_rate = Decimal(prices["input_mtok"])
    cache_read_rate = Decimal(prices["cache_read_mtok"])
    output_rate = Decimal(prices["output_mtok"])
    if input_tokens > int(prices["high_context_threshold"]):
        input_rate = Decimal(prices["high_context_input_mtok"])
        cache_read_rate = Decimal(prices["high_context_cache_read_mtok"])
        output_rate = Decimal(prices["high_context_output_mtok"])
    uncached_input_tokens = input_tokens - cache_read_tokens
    total = (
        Decimal(uncached_input_tokens) * input_rate
        + Decimal(cache_read_tokens) * cache_read_rate
        + Decimal(output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return float(total)


def _manual_model_prices(model_ref: str) -> dict[str, Decimal | int] | None:
    normalized = model_ref.strip().lower().replace("_", "-")
    if normalized.startswith("gpt-5.5") or normalized.startswith("gpt-5-5"):
        return _MANUAL_USAGE_PRICE_OVERRIDES["gpt-5.5"]
    if normalized.startswith("gpt-5.4") or normalized.startswith("gpt-5-4"):
        return _MANUAL_USAGE_PRICE_OVERRIDES["gpt-5.4"]
    return _MANUAL_USAGE_PRICE_OVERRIDES.get(normalized)


def image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def infer_screen_size(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        width, height = image.size
    return {"width": width, "height": height}


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: expected an object.")
            yield raw


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl_objects(path):
        rows.append(row)
    return rows


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def render_json_document(document: Any) -> str:
    return json.dumps(sanitize_for_utf8_json(document), ensure_ascii=False, indent=2) + "\n"


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_iter_atomic(path, rows)


def write_jsonl_iter_atomic(path: Path, rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            for row in rows:
                handle.write(json.dumps(sanitize_for_utf8_json(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            json.dump(sanitize_for_utf8_json(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def find_normalized_usage_dicts(value: Any) -> list[dict[str, int]]:
    found: list[dict[str, int]] = []
    if isinstance(value, dict):
        normalized = normalize_litellm_usage(value)
        if normalized:
            found.append(normalized)
        for child in value.values():
            found.extend(find_normalized_usage_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_normalized_usage_dicts(child))
    return found


def extract_max_usage_from_json_events(*streams: str | None) -> dict[str, int]:
    usages: list[dict[str, int]] = []
    for stream in streams:
        if not isinstance(stream, str):
            continue
        for line in stream.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            usages.extend(find_normalized_usage_dicts(event))
    return max(usages, key=lambda item: item.get("total_tokens", 0)) if usages else {}


def sum_usage_dicts(usages: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in ("llm_requests", "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
            elif isinstance(value, float):
                totals[key] = totals.get(key, 0) + int(value)
    return {key: value for key, value in totals.items() if value}


def row_id(row: dict[str, Any], fallback_idx: int, *, keys: tuple[str, ...] | None = None) -> str:
    for key in keys or ("id", "node_id", "provenounce_id", "provenance_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return f"row_{fallback_idx}"


def string_field(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def safe_action_text(row: dict[str, Any], fallback_idx: int) -> str:
    action = string_field(row, "action")
    if action:
        return action
    parts: list[str] = []
    goal = string_field(row, "goal")
    active_application = string_field(row, "active_application")
    visual_content = string_field(row, "visual_content")
    if goal:
        parts.append(goal)
    if active_application:
        parts.append(f"Active application: {active_application}")
    if visual_content:
        parts.append(f"Visible content: {visual_content}")
    return " | ".join(parts) or f"action_{fallback_idx}"


def is_action_row(row: dict[str, Any]) -> bool:
    if row.get("row_type") == "meta":
        return False
    if row.get("node_type") == "sequence":
        return False
    if row.get("row_type") == "node" and row.get("node_type") not in (None, "action"):
        return False
    return True


def read_activity_jsonl(path: Path) -> list[ComputerUseActivityEntry]:
    return list(iter_activity_jsonl(path))


def iter_activity_jsonl(path: Path) -> Iterator[ComputerUseActivityEntry]:
    for row in iter_jsonl_objects(path):
        yield ComputerUseActivityEntry.model_validate(row)


def write_action_grounding_jsonl(path: Path, outputs: list[ActionGroundingOutput]) -> None:
    write_jsonl_atomic(path, [output.model_dump(mode="json") for output in outputs])


def write_action_grounding_jsonl_iter(path: Path, outputs: Iterator[ActionGroundingOutput]) -> None:
    write_jsonl_iter_atomic(path, (output.model_dump(mode="json") for output in outputs))
