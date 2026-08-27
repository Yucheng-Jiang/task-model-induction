from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

import task_model_induction.step5_procedure_model_induction as step5
from task_model_induction.schemas.procedure_model_induction_output import ProcedureTaskModel
from task_model_induction.schemas.unified_task_model import UnifiedTaskModel
from task_model_induction.validate.validate_procedure_model import validate_procedure_output
from task_model_induction.validate.validate_unified_model import validate_unified_output


def _source(count: int = 2) -> dict:
    return {
        "canonical_root_id": "C1",
        "activities": [
            {
                "activity_id": f"activity_{index:04d}",
                "raw_action_ids": [f"action_{index:04d}"],
            }
            for index in range(count)
        ],
    }


def _procedure_node(
    node_id: str,
    activity_ref: str,
    body: dict,
    *,
    operator: str = "SEQ",
    **extra,
) -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "operator": operator,
        "description": f"Procedure {node_id}",
        "body": body,
        "activity_refs": [activity_ref],
        "evidence_summary": f"Evidence for {node_id}",
        **extra,
    }


def _valid_procedure_model() -> dict:
    return {
        "version": "0.1",
        "root_procedure_id": "P0",
        "procedure_nodes": [
            _procedure_node(
                "P0",
                "activity_0000-activity_0001",
                {
                    "operator": "SEQ",
                    "steps": [
                        {"procedure_node_id": "P1"},
                        {"procedure_node_id": "P2"},
                    ],
                },
            ),
            _procedure_node(
                "P1",
                "activity_0000",
                {"operator": "SEQ", "steps": [{"activity_id": "activity_0000"}]},
            ),
            _procedure_node(
                "P2",
                "activity_0001",
                {"operator": "SEQ", "steps": [{"activity_id": "activity_0001"}]},
            ),
        ],
    }


def _valid_for_procedure_model() -> dict:
    return {
        "version": "0.1",
        "root_procedure_id": "P0",
        "procedure_nodes": [
            _procedure_node(
                "P0",
                "activity_0000-activity_0001",
                {
                    "operator": "SEQ",
                    "steps": [
                        {
                            "name": "Process item",
                            "description": "Apply the operation to each item.",
                            "activity_refs": ["activity_0000-activity_0001"],
                        }
                    ],
                },
                operator="FOR",
                bindings={
                    "iteration_variable": "item",
                    "collection": ["alpha", "beta"],
                },
            )
        ],
    }


def _valid_while_procedure_model() -> dict:
    condition = "The extracted dataset entry is visible in the workspace Explorer"
    return {
        "version": "0.1",
        "root_procedure_id": "P0",
        "procedure_nodes": [
            _procedure_node(
                "P0",
                "activity_0000-activity_0001",
                {
                    "operator": "SEQ",
                    "steps": [
                        {
                            "name": "Repair and verify",
                            "description": "Apply a repair and inspect the resulting state.",
                            "activity_refs": ["activity_0000-activity_0001"],
                        }
                    ],
                },
                operator="WHILE",
                condition=condition,
                condition_grounding={
                    "predicate": condition,
                    "verifier": (
                        "Inspect the workspace Explorer and confirm the named dataset "
                        "entry appears under the extracted folder"
                    ),
                    "evidence_refs": ["activity_0001"],
                    "observed_status": "satisfied",
                },
            )
        ],
    }


def _grounding(evidence_ref: str) -> dict:
    return {
        "deliverables": [
            {
                "kind": "state",
                "target": "target",
                "expected_state": "ready",
                "evidence_refs": [evidence_ref],
            }
        ],
        "success_criteria": [
            {
                "predicate": "The target is ready",
                "verifier": "state_delta",
                "evidence_refs": [evidence_ref],
                "confidence": 0.8,
            }
        ],
        "observed_outcome": {
            "status": "achieved",
            "description": "The target was observed ready.",
            "evidence_refs": [evidence_ref],
        },
        "evidence_refs": [evidence_ref],
    }


def _unified_leaf(node_id: str, index: int) -> dict:
    activity_ref = f"activity_{index:04d}"
    return {
        "id": node_id,
        "objective": f"Complete phase {index}",
        "summary": f"Phase {index}",
        **_grounding(f"action_{index:04d}"),
        "activity_refs": [activity_ref],
        "procedure": {
            "operator": "SEQ",
            "name": f"Phase {index}",
            "body": [{"name": "Execute", "activity_refs": [activity_ref]}],
        },
        "decomposition": [],
    }


def _valid_unified_model() -> dict:
    return {
        "version": "0.2",
        "root": {
            "id": "C1",
            "objective": "Complete the task",
            "summary": "Both phases are complete.",
            **_grounding("action_0000"),
            "activity_refs": ["activity_0000-activity_0001"],
            "procedure": {
                "operator": "SEQ",
                "name": "Main flow",
                "body": [
                    {"name": "Phase zero", "activity_refs": ["activity_0000"]},
                    {"name": "Phase one", "activity_refs": ["activity_0001"]},
                ],
            },
            "decomposition": [
                _unified_leaf("C1.1", 0),
                _unified_leaf("C1.2", 1),
            ],
        },
    }


def test_procedure_validator_accepts_rooted_exact_model() -> None:
    feedback = validate_procedure_output(_valid_procedure_model(), source=_source())
    assert feedback.valid, feedback.as_text()


def test_procedure_validator_rejects_unreachable_node_used_for_coverage() -> None:
    model = _valid_procedure_model()
    model["procedure_nodes"][0]["body"]["steps"] = [{"procedure_node_id": "P1"}]
    model["procedure_nodes"][0]["activity_refs"] = ["activity_0000"]
    # P2 still claims activity_0001, but it is now unreachable from P0.
    feedback = validate_procedure_output(model, source=_source())

    assert not feedback.valid
    assert any("unreachable nodes" in error and "P2" in error for error in feedback.errors)
    assert any("coverage is incomplete" in error for error in feedback.errors)


def test_procedure_validator_rejects_missing_and_extra_for_bindings() -> None:
    missing = _valid_for_procedure_model()
    missing["procedure_nodes"][0]["bindings"].pop("collection")
    missing_feedback = validate_procedure_output(missing, source=_source())
    assert not missing_feedback.valid
    assert any("missing FOR fields" in error for error in missing_feedback.errors)

    extra = _valid_for_procedure_model()
    extra["procedure_nodes"][0]["bindings"]["items"] = ["alpha", "beta"]
    extra_feedback = validate_procedure_output(extra, source=_source())
    assert not extra_feedback.valid
    assert any("unsupported FOR fields" in error for error in extra_feedback.errors)


def test_procedure_validator_requires_grounded_while_exit_condition() -> None:
    valid = validate_procedure_output(_valid_while_procedure_model(), source=_source())
    assert valid.valid, valid.as_text()

    missing = _valid_while_procedure_model()
    missing["procedure_nodes"][0].pop("condition_grounding")
    missing_feedback = validate_procedure_output(missing, source=_source())
    assert not missing_feedback.valid
    assert any("condition_grounding" in error for error in missing_feedback.errors)

    vague = _valid_while_procedure_model()
    vague["procedure_nodes"][0]["condition"] = "until satisfied"
    vague["procedure_nodes"][0]["condition_grounding"]["predicate"] = "until satisfied"
    vague_feedback = validate_procedure_output(vague, source=_source())
    assert not vague_feedback.valid
    assert any("not an observable exit state" in error for error in vague_feedback.errors)


def test_procedure_schema_and_prompts_require_grounded_while_condition() -> None:
    parsed = ProcedureTaskModel.model_validate(_valid_while_procedure_model())
    assert parsed.procedure_nodes[0].condition_grounding is not None
    assert parsed.procedure_nodes[0].condition_grounding.observed_status == "satisfied"

    missing = _valid_while_procedure_model()
    missing["procedure_nodes"][0].pop("condition_grounding")
    with pytest.raises(ValidationError, match="condition_grounding"):
        ProcedureTaskModel.model_validate(missing)

    for prompt in (
        step5.procedure_induction_prompt(),
        step5.procedure_generation_prompt_text(),
        step5.procedure_repair_prompt_text(),
    ):
        assert "condition_grounding" in prompt
        assert "observed_status" in prompt
        assert "until satisfied" in prompt


def test_procedure_validator_rejects_while_evidence_outside_loop() -> None:
    model = _valid_while_procedure_model()
    model["procedure_nodes"][0]["condition_grounding"]["evidence_refs"] = [
        "activity_0002"
    ]
    feedback = validate_procedure_output(model, source=_source(3))

    assert not feedback.valid
    assert any("must be contained in the WHILE activity_refs" in error for error in feedback.errors)


def test_procedure_validator_rejects_dangling_reference() -> None:
    model = _valid_procedure_model()
    model["procedure_nodes"][0]["body"]["steps"][1] = {
        "procedure_node_id": "P404"
    }
    feedback = validate_procedure_output(model, source=_source())

    assert not feedback.valid
    assert any("unknown procedure_node_id 'P404'" in error for error in feedback.errors)


def test_procedure_validator_rejects_node_body_mismatch() -> None:
    model = _valid_procedure_model()
    model["procedure_nodes"][1]["activity_refs"] = ["activity_0000-activity_0001"]
    feedback = validate_procedure_output(model, source=_source())

    assert not feedback.valid
    assert any("not present in the union of its body" in error for error in feedback.errors)


def test_procedure_validator_rejects_cycles_and_multiple_parents() -> None:
    cyclic = _valid_procedure_model()
    cyclic["procedure_nodes"][1]["body"] = {
        "operator": "SEQ",
        "steps": [{"procedure_node_id": "P0"}],
    }
    cyclic_feedback = validate_procedure_output(cyclic, source=_source())
    assert not cyclic_feedback.valid
    assert any("contains a cycle" in error for error in cyclic_feedback.errors)

    multi_parent = _valid_procedure_model()
    multi_parent["procedure_nodes"][2]["body"] = {
        "operator": "SEQ",
        "steps": [
            {"activity_id": "activity_0001"},
            {"procedure_node_id": "P1"},
        ],
    }
    multi_parent_feedback = validate_procedure_output(multi_parent, source=_source())
    assert not multi_parent_feedback.valid
    assert any("multiple parents" in error for error in multi_parent_feedback.errors)


def test_unified_validator_accepts_grounded_exact_partonomy() -> None:
    feedback = validate_unified_output(_valid_unified_model(), source=_source())
    assert feedback.valid, feedback.as_text()


def test_unified_validator_requires_canonical_version_and_grounding() -> None:
    legacy = _valid_unified_model()
    legacy["version"] = "0.1"
    legacy["root"].pop("deliverables")
    feedback = validate_unified_output(legacy, source=_source())

    assert not feedback.valid
    assert any("canonical version '0.2'" in error for error in feedback.errors)
    assert any("missing required fields" in error and "deliverables" in error for error in feedback.errors)


def test_unified_validator_rejects_unknown_activity_refs() -> None:
    model = _valid_unified_model()
    model["root"]["activity_refs"] = ["activity_0000-activity_0002"]
    model["root"]["procedure"]["body"].append(
        {"name": "Unknown phase", "activity_refs": ["activity_0002"]}
    )
    model["root"]["decomposition"].append(_unified_leaf("C1.3", 2))
    feedback = validate_unified_output(model, source=_source())

    assert not feedback.valid
    assert any("absent from the source" in error for error in feedback.errors)


def test_unified_validator_rejects_incomplete_or_extra_for_bindings() -> None:
    base = {
        "version": "0.2",
        "root": _unified_leaf("C1", 0),
    }
    base["root"]["procedure"] = {
        "operator": "FOR",
        "name": "Process items",
        "bindings": {
            "iteration_variable": "item",
            "collection": ["alpha", "beta"],
        },
        "body": [{"name": "Process", "activity_refs": ["activity_0000"]}],
    }
    valid = validate_unified_output(base, source=_source(1))
    assert valid.valid, valid.as_text()

    missing = deepcopy(base)
    missing["root"]["procedure"]["bindings"].pop("collection")
    missing_feedback = validate_unified_output(missing, source=_source(1))
    assert not missing_feedback.valid
    assert any("missing FOR fields" in error for error in missing_feedback.errors)

    extra = deepcopy(base)
    extra["root"]["procedure"]["bindings"]["items"] = ["alpha", "beta"]
    extra_feedback = validate_unified_output(extra, source=_source(1))
    assert not extra_feedback.valid
    assert any("unsupported FOR fields" in error for error in extra_feedback.errors)


def test_unified_validator_requires_grounded_while_exit_condition() -> None:
    condition = "The extracted dataset entry is visible in the workspace Explorer"
    model = {
        "version": "0.2",
        "root": {
            "id": "C1",
            "objective": "Make the extracted dataset available",
            "summary": "The dataset is repaired and inspected.",
            **_grounding("action_0000"),
            "activity_refs": ["activity_0000-activity_0001"],
            "procedure": {
                "operator": "WHILE",
                "name": "Repair dataset staging",
                "condition": condition,
                "condition_grounding": {
                    "predicate": condition,
                    "verifier": (
                        "Inspect the workspace Explorer and confirm the named dataset "
                        "entry appears under the extracted folder"
                    ),
                    "evidence_refs": ["activity_0001"],
                    "observed_status": "satisfied",
                },
                "body": [
                    {
                        "name": "Repair and inspect",
                        "activity_refs": ["activity_0000-activity_0001"],
                    }
                ],
            },
            "decomposition": [],
        },
    }
    valid = validate_unified_output(model, source=_source())
    assert valid.valid, valid.as_text()
    parsed = UnifiedTaskModel.model_validate(model)
    assert parsed.root.procedure.condition_grounding is not None
    assert parsed.root.procedure.condition_grounding.evidence_refs == ["activity_0001"]

    vague = deepcopy(model)
    vague["root"]["procedure"]["condition"] = "until done"
    vague["root"]["procedure"]["condition_grounding"]["predicate"] = "until done"
    vague_feedback = validate_unified_output(vague, source=_source())
    assert not vague_feedback.valid
    assert any("not an observable exit state" in error for error in vague_feedback.errors)


def test_unified_validator_rejects_parent_child_and_body_mismatches() -> None:
    overlapping = _valid_unified_model()
    overlapping["root"]["decomposition"][1]["activity_refs"] = ["activity_0000"]
    overlapping["root"]["decomposition"][1]["procedure"]["body"][0][
        "activity_refs"
    ] = ["activity_0000"]
    overlap_feedback = validate_unified_output(overlapping, source=_source())
    assert not overlap_feedback.valid
    assert any("overlaps a sibling" in error for error in overlap_feedback.errors)

    leaf_mismatch = {
        "version": "0.2",
        "root": _unified_leaf("C1", 0),
    }
    leaf_mismatch["root"]["activity_refs"] = ["activity_0000-activity_0001"]
    mismatch_feedback = validate_unified_output(leaf_mismatch, source=_source())
    assert not mismatch_feedback.valid
    assert any("outside the exact union of procedure.body" in error for error in mismatch_feedback.errors)


def test_unified_validator_rejects_nested_evidence_outside_node() -> None:
    model = _valid_unified_model()
    model["root"]["deliverables"][0]["evidence_refs"] = ["action_not_owned"]
    feedback = validate_unified_output(model, source=_source())

    assert not feedback.valid
    assert any("subset of the node evidence_refs" in error for error in feedback.errors)
