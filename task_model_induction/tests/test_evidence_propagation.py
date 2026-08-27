from __future__ import annotations

import json

from task_model_induction.schemas import (
    Activity,
    ActivityInductionMeta,
    ActivityInductionOutput,
    AtomSemanticAction,
    SemanticActionInductionMeta,
    SemanticActionInductionOutput,
)
from task_model_induction.step1_semantic_action_induction import (
    SemanticActionIR,
    action_trace_fingerprint,
    load_action_trace,
    read_semantic_action_output,
    rehydrate_semantic_action_evidence,
    write_semantic_action_output,
)
from task_model_induction.step2_activity_induction import (
    ActivityIR,
    RunStats,
    read_activity_output,
    read_semantic_actions,
    rehydrate_activity_evidence,
    semantic_actions_fingerprint,
    write_activity_output,
)
from task_model_induction.step3_task_threads_induction import TaskThreadInductionBuilder


def _semantic_meta(input_path: str, fingerprint: str | None) -> SemanticActionInductionMeta:
    return SemanticActionInductionMeta(
        created_at="2026-07-09T00:00:00Z",
        model="test-model",
        input_path=input_path,
        input_fingerprint=fingerprint,
        num_actions=2,
        num_semantic_actions=1,
        backward_batch_size=2,
        max_future_semantic_actions=2,
    )


def _activity_meta(input_path: str, output_path: str, fingerprint: str | None) -> ActivityInductionMeta:
    return ActivityInductionMeta(
        created_at="2026-07-09T00:00:00Z",
        model="test-model",
        input_path=input_path,
        input_fingerprint=fingerprint,
        output_path=output_path,
        num_semantic_actions=1,
        num_candidate_segments=1,
        num_activities=1,
        segmentation_batch_size=1,
        merge_batch_size=2,
        merge_batch_overlap=0,
        max_prior_segments=0,
    )


def _grounded_rows() -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "goal": "Edit the release title",
            "active_application": "Issue Tracker",
            "visual_content": "Release title field",
            "ocr_results": {
                "screen_size": {"width": 1440, "height": 900},
                "md_results": "Release title\nDraft",
                "layout_details": [{"label": "Release title"}],
                "data_info": {"source": "ocr"},
                "warnings": [],
            },
            "md_results": "Release title\nDraft",
            "warnings": [],
            "completed_at": "2026-07-09T00:00:00Z",
            "provenounce_id": "a0",
            "original_index": 0,
            "id": "a0",
            "action": "click(100, 200)",
            "state_before": "screens/000-before.png",
            "state_after": "screens/000-after.png",
            "time_before": 1.0,
            "time_after": 2.0,
            "time_range": 1.0,
        },
        {
            "status": "success",
            "goal": "Save the release title",
            "active_application": "Issue Tracker",
            "visual_content": "Save changes button",
            "ocr_results": {
                "screen_size": {"width": 1440, "height": 900},
                "md_results": "Save changes",
                "layout_details": [{"label": "Save changes"}],
                "data_info": {"source": "ocr"},
                "warnings": [],
            },
            "md_results": "Save changes",
            "warnings": [],
            "completed_at": "2026-07-09T00:00:01Z",
            "provenounce_id": "a1",
            "original_index": 1,
            "id": "a1",
            "action": "click(300, 400)",
            "state_before": "screens/001-before.png",
            "state_after": "screens/001-after.png",
            "time_before": 2.0,
            "time_after": 3.0,
            "time_range": 1.0,
        },
    ]


def test_grounding_evidence_survives_real_step1_step2_serialization_and_reaches_step3(tmp_path) -> None:
    grounded_path = tmp_path / "processed_trajectory_with_goals.jsonl"
    grounded_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _grounded_rows()),
        encoding="utf-8",
    )

    raw_actions = load_action_trace(grounded_path)
    fingerprint = action_trace_fingerprint(raw_actions)
    semantic = SemanticActionIR(
        actions=raw_actions,
        semantic_action="Update the release title",
        action_details="Edit the field and save it.",
        start_action_idx=0,
        end_action_idx=1,
    ).to_model(0)
    semantic_path = tmp_path / "atom_semantic_actions.jsonl"
    write_semantic_action_output(
        semantic_path,
        SemanticActionInductionOutput(
            meta=_semantic_meta(str(grounded_path), fingerprint),
            semantic_actions=[semantic],
        ),
    )

    serialized_semantic = read_semantic_action_output(semantic_path).semantic_actions[0]
    assert serialized_semantic.raw_action_ids == ["a0", "a1"]
    assert serialized_semantic.entities == ["Release title field", "Save changes button"]
    assert serialized_semantic.pre_state == "screens/000-before.png"
    assert serialized_semantic.post_state == "screens/001-after.png"
    assert serialized_semantic.actions[0].action == "click(100, 200)"
    assert serialized_semantic.actions[0].grounded_visual_content == "Release title field"
    assert serialized_semantic.actions[0].ocr_results["layout_details"] == [{"label": "Release title"}]

    semantic_rows = read_semantic_actions(semantic_path)
    activity = ActivityIR(
        start_semantic_action_idx=0,
        end_semantic_action_idx=0,
        objective="Publish the revised release title",
        additional_context="The title is edited and saved.",
    ).to_model(0, semantic_rows)
    activity_path = tmp_path / "activity.jsonl"
    write_activity_output(
        activity_path,
        ActivityInductionOutput(
            meta=_activity_meta(
                str(semantic_path),
                str(activity_path),
                semantic_actions_fingerprint(semantic_rows),
            ),
            activities=[activity],
        ),
        stats=RunStats(),
    )

    serialized_activity = read_activity_output(activity_path).activities[0]
    assert serialized_activity.raw_action_ids == ["a0", "a1"]
    assert serialized_activity.ocr_texts == ["Release title\nDraft", "Save changes"]
    assert serialized_activity.source_actions[1].state_after == "screens/001-after.png"

    builder = TaskThreadInductionBuilder("test-model", llm_timeout_secs=1)
    leaf = builder._load_leaf_tasks(str(activity_path))[0]
    prompt_row = json.loads(builder._describe_leaf(leaf))
    assert leaf.raw_action_ids == ["a0", "a1"]
    assert leaf.entities == ["Release title field", "Save changes button"]
    assert leaf.pre_state == "screens/000-before.png"
    assert leaf.post_state == "screens/001-after.png"
    assert prompt_row["ocr_text"] == ["Release title Draft", "Save changes"]
    heuristic = builder._heuristic_discovery_output([leaf], {}, 1, None)
    assert heuristic["new_roots"][0]["deliverable"] == "Publish the revised release title"
    assert "Save changes" in heuristic["new_roots"][0]["success_criteria"]
    assert "screens/001-after.png" not in heuristic["new_roots"][0]["success_criteria"]


def test_cache_reuse_requires_matching_full_evidence_fingerprint(tmp_path) -> None:
    grounded_path = tmp_path / "grounded.jsonl"
    rows = _grounded_rows()
    grounded_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    actions = load_action_trace(grounded_path)
    fingerprint = action_trace_fingerprint(actions)
    semantic = SemanticActionIR(
        actions=actions,
        semantic_action="Update title",
        start_action_idx=0,
        end_action_idx=1,
    ).to_model(0)
    cached = SemanticActionInductionOutput(
        meta=_semantic_meta(str(grounded_path), fingerprint),
        semantic_actions=[semantic],
    )

    assert rehydrate_semantic_action_evidence(cached.model_copy(deep=True), actions) is not None
    changed_rows = _grounded_rows()
    changed_rows[0]["md_results"] = "A different title"
    changed_rows[0]["ocr_results"]["md_results"] = "A different title"
    grounded_path.write_text("".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8")
    changed_actions = load_action_trace(grounded_path)
    assert rehydrate_semantic_action_evidence(cached.model_copy(deep=True), changed_actions) is None

    legacy = cached.model_copy(deep=True)
    legacy.meta.input_fingerprint = None
    assert rehydrate_semantic_action_evidence(legacy, actions) is None


def test_activity_cache_is_rebuilt_from_current_semantic_evidence() -> None:
    from task_model_induction.step1_semantic_action_induction import ActionTraceEntry

    actions = [
        ActionTraceEntry(
            action_id="a0",
            action="click(1, 2)",
            original_index=0,
            visual_content="Target artifact",
            md_results="Saved successfully",
            state_before="before.png",
            state_after="after.png",
        )
    ]
    semantic = SemanticActionIR(
        actions=actions,
        semantic_action="Save artifact",
        start_action_idx=0,
        end_action_idx=0,
    ).to_model(0)
    fingerprint = semantic_actions_fingerprint([semantic])
    cached_activity = Activity(
        activity_id="activity_0000",
        start_semantic_action_idx=0,
        end_semantic_action_idx=0,
        start_semantic_action_id=semantic.semantic_action_id,
        end_semantic_action_id=semantic.semantic_action_id,
        semantic_action_ids=[semantic.semantic_action_id],
        start_action_idx=0,
        end_action_idx=0,
        start_action_id="a0",
        end_action_id="a0",
        objective="Save the artifact",
        semantic_action_count=1,
        event_count=1,
    )
    cached = ActivityInductionOutput(
        meta=_activity_meta("semantic.jsonl", "activity.jsonl", fingerprint),
        activities=[cached_activity],
    )

    refreshed = rehydrate_activity_evidence(cached, [semantic])

    assert refreshed is not None
    assert refreshed.activities[0].raw_action_ids == ["a0"]
    assert refreshed.activities[0].entities == ["Target artifact"]
    assert refreshed.activities[0].ocr_texts == ["Saved successfully"]
    assert refreshed.activities[0].source_actions[0].action == "click(1, 2)"


def test_legacy_artifacts_remain_readable_with_empty_evidence_defaults() -> None:
    semantic = AtomSemanticAction.model_validate(
        {
            "semantic_action_id": "semantic_action_0000",
            "start_action_idx": 0,
            "end_action_idx": 0,
            "semantic_action": "Legacy semantic action",
            "event_count": 1,
        }
    )
    activity = Activity.model_validate(
        {
            "activity_id": "activity_0000",
            "start_semantic_action_idx": 0,
            "end_semantic_action_idx": 0,
            "start_action_idx": 0,
            "end_action_idx": 0,
            "objective": "Legacy activity",
            "semantic_action_count": 1,
            "event_count": 1,
        }
    )

    assert semantic.actions == []
    assert semantic.ocr_texts == []
    assert activity.source_actions == []
    assert activity.raw_action_ids == []
