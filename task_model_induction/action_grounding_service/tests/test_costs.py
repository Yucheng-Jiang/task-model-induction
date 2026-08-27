from types import SimpleNamespace

from action_grounding_service.app.costs import (
    breakdown,
    combine_breakdowns,
    cost_item_from_usage,
    summarize_costs,
)


def test_cost_summary_itemizes_ocr_grounding_and_redaction():
    ocr = breakdown([cost_item_from_usage("ocr_markdown", SimpleNamespace(input_tokens=10, output_tokens=5, requests=1), "openai:gpt-5.4-mini")])
    grounding = combine_breakdowns(
        breakdown([cost_item_from_usage("goal_vlm", SimpleNamespace(input_tokens=20, output_tokens=5, requests=1), "openai:gpt-5.4-mini")]),
        breakdown([cost_item_from_usage("context_vlm", SimpleNamespace(input_tokens=30, output_tokens=6, requests=1), "openai:gpt-5.4-mini")]),
    )

    summary = summarize_costs(ocr=ocr, grounding=grounding)

    assert [item.name for item in summary.ocr.items] == ["ocr_markdown"]
    assert [item.name for item in summary.grounding.items] == ["goal_vlm", "context_vlm"]
    assert summary.redaction.total_usd == 0
    assert summary.total_usd == ocr.total_usd + grounding.total_usd
