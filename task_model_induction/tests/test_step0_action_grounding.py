from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from task_model_induction.schemas import ActionGroundingOutput, ComputerUseActivityEntry
from task_model_induction.step0_action_grounding import (
    DEFAULT_INPUT_FILE_NAME,
    DEFAULT_OUTPUT_FILE_NAME,
    action_grounding,
    cached_output_matches_entry,
    iter_merged_action_grounding_outputs,
    output_from_error,
)
from task_model_induction.step1_semantic_action_induction import load_action_trace


def _legacy_output(output_id: str, *, status: str = "success") -> ActionGroundingOutput:
    return ActionGroundingOutput(
        status=status,
        goal="Cached grounded goal" if status == "success" else None,
        error="grounding failed" if status == "error" else None,
        completed_at="2026-05-06T00:00:00Z",
        provenounce_id=output_id,
    )


def test_grounding_error_preserves_entire_raw_event_and_original_index() -> None:
    entry = ComputerUseActivityEntry(
        id="action_0042",
        action="click(10, 20)",
        state_before="screens/before.png",
        state_after="screens/after.png",
        time_before=10.5,
        time_after="2026-05-06T00:00:01Z",
        time_range=0.75,
    )

    output = output_from_error(entry, "service unavailable", original_index=42)

    assert output.status == "error"
    assert output.error == "service unavailable"
    assert output.provenounce_id == entry.id
    assert output.original_index == 42
    assert output.id == entry.id
    assert output.action == entry.action
    assert output.state_before == entry.state_before
    assert output.state_after == entry.state_after
    assert output.time_before == entry.time_before
    assert output.time_after == entry.time_after
    assert output.time_range == entry.time_range


def test_step1_does_not_drop_grounding_error_rows(tmp_path) -> None:
    output_path = tmp_path / DEFAULT_OUTPUT_FILE_NAME
    rows = [
        output_from_error(
            ComputerUseActivityEntry(id="a1", action="click(1, 2)"),
            "grounding failed",
            original_index=0,
        ).model_dump(mode="json"),
        _legacy_output("a2", status="error").model_dump(mode="json"),
        output_from_error(
            ComputerUseActivityEntry(id="a3", action='type("hello")'),
            "grounding failed",
            original_index=2,
        ).model_dump(mode="json"),
    ]
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    actions = load_action_trace(output_path)

    assert [action.action_id for action in actions] == ["a1", "a2", "a3"]
    assert [action.action for action in actions] == ["click(1, 2)", "action_1", 'type("hello")']


def test_merge_rehydrates_legacy_rows_and_excludes_stale_and_out_of_limit_ids(tmp_path) -> None:
    input_path = tmp_path / DEFAULT_INPUT_FILE_NAME
    raw_rows = [
        {"id": "a1", "action": "first", "state_before": "before-1.png", "time_before": 1.0},
        {"id": "a2", "action": "second", "state_after": "after-2.png", "time_after": 2.0},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    cached = {
        "stale": _legacy_output("stale"),
        "a2": _legacy_output("a2"),
        "a1": _legacy_output("a1"),
    }

    merged = list(iter_merged_action_grounding_outputs(input_path, cached, limits=1))

    assert [output.provenounce_id for output in merged] == ["a1"]
    assert merged[0].id == "a1"
    assert merged[0].action == "first"
    assert merged[0].state_before == "before-1.png"
    assert merged[0].time_before == 1.0
    assert merged[0].original_index == 0


def test_cache_match_rejects_same_id_with_changed_source_event() -> None:
    original = ComputerUseActivityEntry(
        id="a1", action="first", state_before="before-1.png"
    )
    cached = output_from_error(original, "unused", original_index=0).model_copy(
        update={"status": "success", "error": None}
    )

    assert cached_output_matches_entry(cached, original)
    assert not cached_output_matches_entry(
        cached,
        ComputerUseActivityEntry(
            id="a1", action="changed", state_before="before-1.png"
        ),
    )
    assert not cached_output_matches_entry(
        cached,
        ComputerUseActivityEntry(
            id="a1", action="first", state_before="replacement.png"
        ),
    )


def test_merge_rejects_duplicate_raw_event_ids(tmp_path) -> None:
    input_path = tmp_path / DEFAULT_INPUT_FILE_NAME
    input_path.write_text(
        json.dumps({"id": "a1", "action": "first"})
        + "\n"
        + json.dumps({"id": "a1", "action": "second"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate raw activity id"):
        list(
            iter_merged_action_grounding_outputs(
                input_path,
                {"a1": _legacy_output("a1")},
            )
        )


def test_resume_with_no_new_requests_still_prunes_and_rehydrates_output(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / DEFAULT_INPUT_FILE_NAME
    output_path = tmp_path / DEFAULT_OUTPUT_FILE_NAME
    raw_rows = [
        {"id": "a1", "action": "first", "state_before": "before-1.png"},
        {"id": "a2", "action": "second", "state_before": "before-2.png"},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    legacy_rows = [_legacy_output("a1"), _legacy_output("a2"), _legacy_output("deleted")]
    output_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in legacy_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "task_model_induction.step0_action_grounding.load_action_grounding_stage_config",
        lambda: SimpleNamespace(grounding_url="http://unused.invalid", max_concurrent_requests=1),
    )

    outputs = action_grounding(tmp_path, limits=1, no_console=True)

    assert [output.provenounce_id for output in outputs] == ["a1"]
    persisted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["provenounce_id"] for row in persisted] == ["a1"]
    assert persisted[0]["id"] == "a1"
    assert persisted[0]["action"] == "first"
    assert persisted[0]["state_before"] == "before-1.png"
    assert persisted[0]["original_index"] == 0
