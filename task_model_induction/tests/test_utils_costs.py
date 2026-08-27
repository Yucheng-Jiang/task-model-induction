from __future__ import annotations

import json

from task_model_induction.utils import (
    estimate_litellm_usage_cost_usd,
    normalize_litellm_usage,
    render_json_document,
    sanitize_for_utf8_json,
    write_json_atomic,
)


def test_normalize_litellm_usage_preserves_cached_input_tokens() -> None:
    response = {
        "usage": {
            "input_tokens": 1000,
            "cached_input_tokens": 250,
            "output_tokens": 100,
            "total_tokens": 1100,
        }
    }

    assert normalize_litellm_usage(response) == {
        "input_tokens": 1000,
        "cache_read_tokens": 250,
        "output_tokens": 100,
        "total_tokens": 1100,
    }


def test_estimate_usage_cost_usd_falls_back_for_gpt_55() -> None:
    usage = {
        "input_tokens": 1000,
        "cache_read_tokens": 250,
        "output_tokens": 100,
    }

    estimated = estimate_litellm_usage_cost_usd(usage, "openai/gpt-5.5")

    assert estimated == 0.006875


def test_estimate_usage_cost_usd_uses_high_context_gpt_55_rates() -> None:
    usage = {
        "input_tokens": 300_000,
        "output_tokens": 1_000,
    }

    estimated = estimate_litellm_usage_cost_usd(usage, "openai/gpt-5.5-2026-04-24")

    assert estimated == 3.045


def test_json_helpers_sanitize_lone_surrogates(tmp_path) -> None:
    payload = {"text": "bad \ud83d value"}

    sanitized = sanitize_for_utf8_json(payload)
    assert sanitized == {"text": "bad \\ud83d value"}

    rendered = render_json_document(payload)
    assert json.loads(rendered) == sanitized

    output_path = tmp_path / "payload.json"
    write_json_atomic(output_path, payload)
    assert json.loads(output_path.read_text(encoding="utf-8")) == sanitized
