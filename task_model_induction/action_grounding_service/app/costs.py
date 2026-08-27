from __future__ import annotations

from decimal import Decimal
from typing import Any

from .schemas import CostBreakdown, CostItem, CostSummary


def zero_item(name: str, model: str | None = None, provider: str | None = None) -> CostItem:
    return CostItem(name=name, model=model, provider=provider)


def breakdown(items: list[CostItem] | None = None) -> CostBreakdown:
    clean_items = items or []
    return CostBreakdown(total_usd=sum(item.usd for item in clean_items), items=clean_items)


def summarize_costs(
    *,
    ocr: CostBreakdown | None = None,
    grounding: CostBreakdown | None = None,
    redaction: CostBreakdown | None = None,
) -> CostSummary:
    ocr = ocr or breakdown()
    grounding = grounding or breakdown()
    redaction = redaction or breakdown()
    return CostSummary(
        total_usd=ocr.total_usd + grounding.total_usd + redaction.total_usd,
        ocr=ocr,
        grounding=grounding,
        redaction=redaction,
    )


def combine_breakdowns(*parts: CostBreakdown) -> CostBreakdown:
    items: list[CostItem] = []
    for part in parts:
        items.extend(part.items)
    return breakdown(items)


def cost_item_from_litellm_response(name: str, response: Any, model: str) -> CostItem:
    provider, model_ref = split_provider_model(model)
    usage = _extract_response_usage(response)
    counts = _usage_counts(usage)
    reported_cost = _extract_litellm_reported_cost(response)
    if reported_cost is not None:
        return _cost_item(name, reported_cost, model_ref, provider, counts, "litellm")
    return cost_item_from_usage(name, usage, model, default_source="genai-prices")


def cost_item_from_usage(name: str, usage: Any, model: str, default_source: str = "genai-prices") -> CostItem:
    provider, model_ref = split_provider_model(model)
    counts = _usage_counts(usage)
    if not any(counts.values()):
        return zero_item(name, model_ref, provider)
    price = _price_from_counts(counts, model_ref, provider)
    if price is None:
        return _cost_item(name, 0.0, model_ref, provider, counts, "usage_no_price")
    return _cost_item(name, price, model_ref, provider, counts, default_source)


def split_provider_model(model: str) -> tuple[str | None, str]:
    if ":" in model:
        provider, model_ref = model.split(":", 1)
        return provider or None, model_ref
    if "/" in model:
        provider, model_ref = model.split("/", 1)
        return provider or None, model_ref
    return None, model


def _cost_item(
    name: str,
    usd: Decimal | float,
    model: str | None,
    provider: str | None,
    counts: dict[str, int],
    source: str,
) -> CostItem:
    return CostItem(
        name=name,
        usd=float(usd),
        model=model,
        provider=provider,
        input_tokens=counts["input_tokens"],
        output_tokens=counts["output_tokens"],
        total_tokens=counts["total_tokens"],
        requests=counts["requests"],
        source=source,
    )


def _price_from_counts(counts: dict[str, int], model_ref: str, provider: str | None) -> Decimal | None:
    try:
        from genai_prices import Usage, calc_price

        usage = Usage(
            input_tokens=counts["input_tokens"] or None,
            cache_write_tokens=counts["cache_write_tokens"] or None,
            cache_read_tokens=counts["cache_read_tokens"] or None,
            output_tokens=counts["output_tokens"] or None,
            input_audio_tokens=counts["input_audio_tokens"] or None,
            cache_audio_read_tokens=counts["cache_audio_read_tokens"] or None,
            output_audio_tokens=counts["output_audio_tokens"] or None,
        )
        return calc_price(usage, model_ref, provider_id=provider).total_price
    except Exception:
        return None


def _usage_counts(usage: Any) -> dict[str, int]:
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": _usage_int(usage, "requests") or (1 if total_tokens else 0),
        "cache_write_tokens": _usage_int(usage, "cache_write_tokens"),
        "cache_read_tokens": _usage_int(usage, "cache_read_tokens"),
        "input_audio_tokens": _usage_int(usage, "input_audio_tokens"),
        "cache_audio_read_tokens": _usage_int(usage, "cache_audio_read_tokens"),
        "output_audio_tokens": _usage_int(usage, "output_audio_tokens"),
    }


def _usage_int(usage: Any, *names: str) -> int:
    for name in names:
        value = _usage_value(usage, name)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    return 0


def _usage_value(usage: Any, name: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
        if value is not None:
            return value
        if name == "cache_read_tokens":
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            if isinstance(details, dict):
                return details.get("cached_tokens")
        return None
    return getattr(usage, name, None)


def _extract_response_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_litellm_reported_cost(response: Any) -> Decimal | None:
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
