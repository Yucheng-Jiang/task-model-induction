from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from task_model_induction.schemas import HierarchicalObjectiveNode, ObjectiveSuccessCriterion
from task_model_induction.step3_task_threads_induction import (
    ProvisionalRootIR,
    TaskThreadInductionBuilder,
)
from task_model_induction.step4_objective_model_induction import (
    compact_activity_ids,
    initial_candidate,
    normalize_existing_hierarchy,
    validate_merged_hierarchy_cache,
    validate_hierarchy,
)
from task_model_induction.validate.validate_hierarchy import validate_hierarchy as validate_standalone


def _source(count: int = 3) -> dict:
    return {
        "activities": [
            {
                "activity_id": f"activity_{idx:04d}",
                "semantic_action_ids": [f"semantic_action_{idx:04d}"],
                "start_action_id": f"raw_action_{idx:04d}",
                "end_action_id": f"raw_action_{idx:04d}",
            }
            for idx in range(count)
        ]
    }


def _expand_refs(refs: list[str]) -> list[str]:
    expanded: list[str] = []
    for ref in refs:
        if "-" not in ref:
            expanded.append(ref)
            continue
        start_text, end_text = ref.split("-", 1)
        prefix = start_text.rsplit("_", 1)[0]
        start = int(start_text.rsplit("_", 1)[1])
        end = int(end_text.rsplit("_", 1)[1])
        expanded.extend(f"{prefix}_{idx:04d}" for idx in range(start, end + 1))
    return expanded


def _node(node_id: str, refs: list[str], children: list[dict] | None = None) -> dict:
    evidence_refs = _expand_refs(refs)
    return {
        "id": node_id,
        "objective": f"Accomplish {node_id}",
        "summary": f"Evidence for {node_id}",
        "deliverables": [
            {
                "kind": "state",
                "target": f"outcome {node_id}",
                "expected_state": "ready",
                "evidence_refs": list(evidence_refs),
            }
        ],
        "success_criteria": [
            {
                "predicate": f"outcome {node_id} is ready",
                "verifier": "state_delta",
                "evidence_refs": list(evidence_refs),
                "confidence": 0.8,
            }
        ],
        "observed_outcome": {
            "status": "unknown",
            "description": "Completion is not independently verified.",
            "evidence_refs": [],
        },
        "evidence_refs": list(evidence_refs),
        "subgoal_segments": refs,
        "decomposition": children or [],
    }


def _feedback(candidate: dict, source: dict | None = None):
    return validate_hierarchy(
        candidate,
        source or _source(),
        large_node_review_threshold=20,
        small_decomposition_review_threshold=5,
    )


def test_recursive_grounding_schema_requires_semantic_fields() -> None:
    candidate = _node("C1", ["activity_0000"])
    HierarchicalObjectiveNode.model_validate(candidate)

    del candidate["deliverables"]
    with pytest.raises(ValidationError):
        HierarchicalObjectiveNode.model_validate(candidate)

    with pytest.raises(ValidationError):
        ObjectiveSuccessCriterion(
            predicate="done",
            verifier="state_delta",
            evidence_refs=[],
            confidence=1.1,
        )


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("success_criteria", 0), "verifier"),
        (("success_criteria", 0), "confidence"),
        (("observed_outcome",), "evidence_refs"),
    ],
)
def test_internal_and_standalone_grounding_contracts_reject_missing_keys(
    path: tuple[str | int, ...], field: str
) -> None:
    candidate = _node("C1", ["activity_0000"])
    current: object = candidate
    for part in path:
        current = current[part]  # type: ignore[index]
    assert isinstance(current, dict)
    current.pop(field)

    with pytest.raises(ValidationError):
        HierarchicalObjectiveNode.model_validate(candidate)
    standalone = validate_standalone(candidate, _source(1))
    assert not standalone.valid
    assert any(field in error for error in standalone.errors)


def test_exact_partonomy_and_source_evidence_are_valid() -> None:
    candidate = _node(
        "C1",
        ["activity_0000-activity_0002"],
        [
            _node("C1.1", ["activity_0000-activity_0001"]),
            _node("C1.2", ["activity_0002"]),
        ],
    )
    assert _feedback(candidate).valid
    assert validate_standalone(candidate, _source()).valid


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (
            _node(
                "C1",
                ["activity_0000-activity_0002"],
                [_node("C1.1", ["activity_0000"]), _node("C1.2", ["activity_0001"])],
            ),
            "partition the parent's subgoal_segments exactly",
        ),
        (
            _node(
                "C1",
                ["activity_0000-activity_0002"],
                [
                    _node("C1.1", ["activity_0000-activity_0001"]),
                    _node("C1.2", ["activity_0001-activity_0002"]),
                ],
            ),
            "overlaps sibling",
        ),
        (
            _node(
                "C1",
                ["activity_0000-activity_0002"],
                [
                    _node(
                        "C1.1",
                        ["activity_0000-activity_0001"],
                        [_node("C1.1.1", ["activity_0000"]), _node("C1.1.2", ["activity_0002"])],
                    ),
                    _node("C1.2", ["activity_0002"]),
                ],
            ),
            "must be a subset of its parent",
        ),
    ],
)
def test_invalid_partonomies_are_rejected(candidate: dict, message: str) -> None:
    feedback = _feedback(candidate)
    assert not feedback.valid
    assert any(message in error for error in feedback.errors)
    standalone = validate_standalone(candidate, _source())
    assert not standalone.valid
    assert any(message in error for error in standalone.errors)


def test_root_requires_exact_source_coverage_and_known_evidence() -> None:
    candidate = _node("C1", ["activity_0000-activity_0003"])
    feedback = _feedback(candidate)
    assert not feedback.valid
    assert any("outside the source" in error for error in feedback.errors)

    candidate = _node("C1", ["activity_0000-activity_0002"])
    candidate["evidence_refs"].append("raw_action_9999")
    candidate["deliverables"][0]["evidence_refs"].append("raw_action_9999")
    feedback = _feedback(candidate)
    assert not feedback.valid
    assert any("identifiers not present in the source" in error for error in feedback.errors)
    assert not validate_standalone(candidate, _source()).valid


def test_child_evidence_must_belong_to_its_own_activity_span() -> None:
    candidate = _node(
        "C1",
        ["activity_0000-activity_0002"],
        [
            _node("C1.1", ["activity_0000-activity_0001"]),
            _node("C1.2", ["activity_0002"]),
        ],
    )
    candidate["decomposition"][0]["evidence_refs"].append("raw_action_0002")
    candidate["decomposition"][0]["deliverables"][0]["evidence_refs"].append(
        "raw_action_0002"
    )

    feedback = _feedback(candidate)
    assert not feedback.valid
    assert any("outside this node's subgoal_segments" in error for error in feedback.errors)
    standalone = validate_standalone(candidate, _source())
    assert not standalone.valid
    assert any("outside this node's subgoal_segments" in error for error in standalone.errors)


def test_grounding_and_activity_span_evidence_cannot_be_empty() -> None:
    candidate = _node("C1", ["activity_0000"])
    candidate["evidence_refs"] = []
    candidate["deliverables"][0]["evidence_refs"] = []
    candidate["success_criteria"][0]["evidence_refs"] = []
    with pytest.raises(ValidationError):
        HierarchicalObjectiveNode.model_validate(candidate)
    assert not validate_standalone(candidate, _source(1)).valid

    empty_child = _node(
        "C1",
        ["activity_0000-activity_0001"],
        [_node("C1.1", ["activity_0000"]), _node("C1.2", ["activity_0001"])],
    )
    empty_child["decomposition"][1]["subgoal_segments"] = []
    with pytest.raises(ValidationError):
        HierarchicalObjectiveNode.model_validate(empty_child)
    assert not validate_standalone(empty_child, _source(2)).valid


def test_multi_activity_root_cannot_bypass_decomposition() -> None:
    feedback = _feedback(_node("C1", ["activity_0000-activity_0002"]))
    assert not feedback.valid
    assert any("Root decomposition is empty" in error for error in feedback.errors)


def test_hierarchy_root_id_must_match_declared_task_thread() -> None:
    source = _source(1)
    source["canonical_root_id"] = "C2"
    candidate = _node("C1", ["activity_0000"])

    feedback = _feedback(candidate, source)
    assert not feedback.valid
    assert any("must match source root id 'C2'" in error for error in feedback.errors)
    standalone = validate_standalone(candidate, source)
    assert not standalone.valid
    assert any("must match source root id 'C2'" in error for error in standalone.errors)


def test_compact_activity_ids_preserves_singular_activity_prefix() -> None:
    assert compact_activity_ids(["activity_0001", "activity_0000", "activity_0001"]) == [
        "activity_0000-activity_0001"
    ]


def test_legacy_hierarchy_gets_backward_compatible_grounding_defaults() -> None:
    normalized = normalize_existing_hierarchy(
        {
            "id": "C1",
            "objective": "Prepare the report",
            "summary": "Observed report work.",
            "deliverable": "report.pdf",
            "success_criteria": "report.pdf exists",
            "subgoal_segments": ["activity_0000"],
            "decomposition": [],
        }
    )
    assert normalized is not None
    assert normalized["deliverables"][0]["target"] == "report.pdf"
    assert normalized["success_criteria"][0]["predicate"] == "report.pdf exists"
    assert normalized["observed_outcome"]["status"] == "unknown"


def test_step3_to_step4_bridge_preserves_grounding_strings_and_evidence(tmp_path: Path) -> None:
    activity_path = tmp_path / "activity.jsonl"
    activity = _source(1)["activities"][0]
    activity_path.write_text(json.dumps(activity) + "\n", encoding="utf-8")
    task_threads_path = tmp_path / "task_threads.json"
    task_threads_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "derived"
    builder = TaskThreadInductionBuilder("test-model", llm_timeout_secs=1)
    builder._write_derived_task_thread_objectives(
        output_payload={
            "roots": [
                {
                    "canonical_root_id": "C1",
                    "label": "Prepare report",
                    "objective": "Prepare the report",
                    "deliverable": "report.pdf",
                    "success_criteria": "report.pdf exists and opens",
                    "summary": "A report was prepared.",
                    "activity_id": ["activity_0000"],
                    "semantic_action_id": ["semantic_action_0000"],
                    "raw_action_id": ["raw_action_0000"],
                }
            ]
        },
        activity_path=activity_path,
        task_threads_path=task_threads_path,
        derived_objectives_dir=output_dir,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    derived = json.loads(Path(manifest["roots"][0]["file"]).read_text(encoding="utf-8"))
    assert derived["deliverable"] == "report.pdf"
    assert derived["success_criteria"] == "report.pdf exists and opens"
    assert derived["objective_grounding"]["deliverables"][0]["target"] == "report.pdf"
    assert derived["objective_grounding"]["success_criteria"][0]["predicate"] == (
        "report.pdf exists and opens"
    )
    assert set(derived["objective_grounding"]["evidence_refs"]) == {
        "activity_0000",
        "semantic_action_0000",
        "raw_action_0000",
    }

    preflight = initial_candidate(
        derived,
        derived,
        model="unused",
        llm_timeout_secs=1,
        large_node_review_threshold=20,
        small_decomposition_review_threshold=5,
        preflight_only=True,
        segment_id_by_int={0: "activity_0000"},
    )
    assert preflight["deliverables"][0]["target"] == "report.pdf"
    assert preflight["success_criteria"][0]["predicate"] == "report.pdf exists and opens"
    assert preflight["evidence_refs"] == derived["objective_grounding"]["evidence_refs"]


def test_step3_updates_require_and_apply_grounding() -> None:
    builder = TaskThreadInductionBuilder("test-model", llm_timeout_secs=1)
    payload = {
        "new_roots": [
            {
                "temp_root_id": "new_1",
                "objective": "Prepare report",
                "deliverable": "report.pdf",
                "success_criteria": "report.pdf exists",
            }
        ],
        "assignments": [{"leaf_idx": 0, "assigned_root_id": "new_1"}],
        "root_updates": [
            {
                "root_id": "new_1",
                "objective": "Prepare report",
                "deliverable": "report.pdf",
                "success_criteria": "report.pdf exists",
            }
        ],
    }
    assert builder._validate_discovery_output(payload, [0], []) == []
    del payload["root_updates"][0]["success_criteria"]
    assert any(
        "Missing success_criteria for root_update" in error
        for error in builder._validate_discovery_output(payload, [0], [])
    )

    root = ProvisionalRootIR(root_id="R001")
    root.apply_update(
        {
            "deliverable": "report.pdf",
            "success_criteria": "report.pdf exists",
        }
    )
    assert root.deliverable == "report.pdf"
    assert root.success_criteria == "report.pdf exists"


def test_merged_cache_is_revalidated_against_current_source(tmp_path: Path) -> None:
    source_path = tmp_path / "C1.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    candidate = _node(
        "C1",
        ["activity_0000-activity_0002"],
        [
            _node("C1.1", ["activity_0000-activity_0001"]),
            _node("C1.2", ["activity_0002"]),
        ],
    )
    payload = {
        "meta": {
            "created_at": "2026-07-09T00:00:00Z",
            "model": "test-model",
            "output_dir": str(tmp_path),
            "num_roots": 1,
            "num_succeeded": 1,
            "preflight_only": False,
            "cost": None,
        },
        "roots": [
            {
                "input_file": str(source_path),
                "output_file": str(tmp_path / "hierarchy.json"),
                "ok": True,
                "hierarchy": candidate,
            }
        ],
    }
    validated = validate_merged_hierarchy_cache(
        payload,
        hierarchy_inputs=[source_path],
        large_node_review_threshold=20,
        small_decomposition_review_threshold=5,
    )
    assert validated["roots"][0]["hierarchy"]["id"] == "C1"

    payload["roots"][0]["hierarchy"]["decomposition"][1]["subgoal_segments"] = [
        "activity_0001-activity_0002"
    ]
    with pytest.raises(ValueError, match="failed validation"):
        validate_merged_hierarchy_cache(
            payload,
            hierarchy_inputs=[source_path],
            large_node_review_threshold=20,
            small_decomposition_review_threshold=5,
        )


def test_merged_cache_cannot_reuse_one_root_for_two_current_inputs(tmp_path: Path) -> None:
    first = tmp_path / "C1.json"
    second = tmp_path / "C2.json"
    first.write_text(json.dumps(_source()), encoding="utf-8")
    second.write_text(json.dumps(_source()), encoding="utf-8")
    candidate = _node(
        "C1",
        ["activity_0000-activity_0002"],
        [
            _node("C1.1", ["activity_0000-activity_0001"]),
            _node("C1.2", ["activity_0002"]),
        ],
    )
    root_result = {
        "input_file": str(first),
        "output_file": str(tmp_path / "hierarchy.json"),
        "ok": True,
        "hierarchy": candidate,
    }
    second_result = deepcopy(root_result)
    second_result["hierarchy"] = _node(
        "C2",
        ["activity_0000-activity_0002"],
        [
            _node("C2.1", ["activity_0000-activity_0001"]),
            _node("C2.2", ["activity_0002"]),
        ],
    )
    payload = {
        "meta": {
            "created_at": "2026-07-09T00:00:00Z",
            "model": "test-model",
            "output_dir": str(tmp_path),
            "num_roots": 2,
            "num_succeeded": 2,
            "preflight_only": False,
            "cost": None,
        },
        "roots": [root_result, second_result],
    }

    with pytest.raises(ValueError, match="more than once"):
        validate_merged_hierarchy_cache(
            payload,
            hierarchy_inputs=[first, second],
            large_node_review_threshold=20,
            small_decomposition_review_threshold=5,
        )
