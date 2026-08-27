from __future__ import annotations

import json
from pathlib import Path

import task_model_induction.step6_bidirectional_alignment as step6
from task_model_induction.schemas.bidirectional_alignment_output import (
    AlignedTaskModel,
    BidirectionalAlignmentMergedMeta,
    BidirectionalAlignmentOutput,
    BidirectionalAlignmentRootResult,
)
from task_model_induction.schemas.unified_task_model import (
    UnifiedTaskModel,
    UnifiedTaskModelMergedMeta,
    UnifiedTaskModelOutput,
    UnifiedTaskModelRootResult,
)
from task_model_induction.step6_bidirectional_alignment import ModelPair
from task_model_induction.validate.validate_unified_model import validate_unified_output


def _grounding(evidence_ref: str = "action_0000") -> dict:
    return {
        "deliverables": [
            {
                "kind": "state",
                "target": "target state",
                "expected_state": "ready",
                "evidence_refs": [evidence_ref],
            }
        ],
        "success_criteria": [
            {
                "predicate": "The target state is ready",
                "verifier": "state_delta",
                "evidence_refs": [evidence_ref],
                "confidence": 0.9,
            }
        ],
        "observed_outcome": {
            "status": "achieved",
            "description": "The ready state was observed.",
            "evidence_refs": [evidence_ref],
        },
        "evidence_refs": [evidence_ref],
    }


def _valid_task_model() -> dict:
    return {
        "version": "0.2",
        "root": {
            "id": "C1",
            "objective": "Reach the target state",
            "summary": "The target is made ready.",
            **_grounding(),
            "activity_refs": ["activity_0000"],
            "procedure": {
                "operator": "SEQ",
                "name": "Main flow",
                "body": [{"name": "Prepare target", "activity_refs": ["activity_0000"]}],
            },
            "decomposition": [],
        },
    }


def test_legacy_step6_schema_names_alias_canonical_unified_contract() -> None:
    assert AlignedTaskModel is UnifiedTaskModel
    assert BidirectionalAlignmentRootResult is UnifiedTaskModelRootResult
    assert BidirectionalAlignmentMergedMeta is UnifiedTaskModelMergedMeta
    assert BidirectionalAlignmentOutput is UnifiedTaskModelOutput

    parsed = AlignedTaskModel.model_validate(_valid_task_model())
    assert parsed.version == "0.2"
    assert parsed.root.deliverables[0].target == "target state"


def test_unified_validator_accepts_canonical_task_model() -> None:
    source = {
        "canonical_root_id": "C1",
        "activities": [
            {"activity_id": "activity_0000", "raw_action_ids": ["action_0000"]}
        ],
    }
    feedback = validate_unified_output(_valid_task_model(), source=source)
    assert feedback.valid, feedback.as_text()


def test_reconciliation_prompts_require_grounding_and_canonical_contract() -> None:
    direct_prompt = step6.reconciliation_generation_prompt()
    repair_prompt = step6.reconciliation_repair_prompt()
    codex_prompt = step6.codex_reconciliation_prompt()

    for prompt in (direct_prompt, repair_prompt, codex_prompt):
        assert "deliverables" in prompt
        assert "success_criteria" in prompt
        assert "observed_outcome" in prompt
        assert "evidence_refs" in prompt
        assert "condition_grounding" in prompt
        assert "observed_status" in prompt
        assert "iteration_variable" in prompt
        assert "collection" in prompt
    assert "version` to `0.2" in repair_prompt
    assert "validate_unified_model.py" in codex_prompt
    assert "objective_model.json output/procedure_model.json" not in codex_prompt


def test_unchanged_objective_nodes_preserve_step4_grounding_losslessly() -> None:
    objective = {
        "id": "C1",
        "objective": "Reach the target state",
        "summary": "Grounded objective",
        **_grounding("action_0000"),
        "subgoal_segments": ["activity_0000"],
        "decomposition": [],
    }
    candidate = _valid_task_model()
    candidate["root"].update(_grounding("invented_evidence"))

    preserved = step6.preserve_unchanged_objective_grounding(candidate, objective)

    for field in ("deliverables", "success_criteria", "observed_outcome", "evidence_refs"):
        assert preserved["root"][field] == objective[field]
    assert candidate["root"]["evidence_refs"] == ["invented_evidence"]


def test_run_direct_reconciliation_writes_canonical_unified_output(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.json"
    objective_path = tmp_path / "objective.json"
    procedure_path = tmp_path / "procedure.json"
    output_path = tmp_path / "unified.json"
    source_path.write_text(
        json.dumps(
            {
                "canonical_root_id": "C1",
                "activities": [
                    {"activity_id": "activity_0000", "raw_action_ids": ["action_0000"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    objective_path.write_text(json.dumps({"id": "C1"}), encoding="utf-8")
    procedure_path.write_text(json.dumps({"root_procedure_id": "P0"}), encoding="utf-8")
    pair = ModelPair(
        key="thread.json",
        input_path=source_path,
        objective_path=objective_path,
        procedure_path=procedure_path,
        activity_count=1,
    )
    monkeypatch.setattr(step6, "_call_llm_json", lambda **kwargs: _valid_task_model())

    result = step6.run_direct_reconciliation(
        pair=pair,
        output_path=output_path,
        model="test-model",
        llm_timeout_secs=5.0,
        max_retries=0,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["version"] == "0.2"
    assert written["root"]["success_criteria"][0]["verifier"] == "state_delta"
    assert result["task_model"] == written


def test_load_cached_root_rejects_legacy_or_structurally_invalid_artifact(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    objective_path = tmp_path / "objective.json"
    procedure_path = tmp_path / "procedure.json"
    output_path = tmp_path / "cached.json"
    source_path.write_text(
        json.dumps(
            {
                "canonical_root_id": "C1",
                "activities": [
                    {"activity_id": "activity_0000", "raw_action_ids": ["action_0000"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    objective_path.write_text("{}", encoding="utf-8")
    procedure_path.write_text("{}", encoding="utf-8")
    legacy = _valid_task_model()
    legacy["version"] = "0.1"
    output_path.write_text(json.dumps(legacy), encoding="utf-8")
    pair = ModelPair(
        key="thread.json",
        input_path=source_path,
        objective_path=objective_path,
        procedure_path=procedure_path,
        activity_count=1,
    )

    assert (
        step6.load_cached_root_result(
            pair=pair, output_path=output_path, execution_mode="direct_llm"
        )
        is None
    )


def test_load_cached_root_accepts_valid_canonical_artifact(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    objective_path = tmp_path / "objective.json"
    procedure_path = tmp_path / "procedure.json"
    output_path = tmp_path / "cached.json"
    source_path.write_text(
        json.dumps(
            {
                "canonical_root_id": "C1",
                "activities": [
                    {"activity_id": "activity_0000", "raw_action_ids": ["action_0000"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    objective_path.write_text("{}", encoding="utf-8")
    procedure_path.write_text("{}", encoding="utf-8")
    output_path.write_text(json.dumps(_valid_task_model()), encoding="utf-8")
    pair = ModelPair(
        key="thread.json",
        input_path=source_path,
        objective_path=objective_path,
        procedure_path=procedure_path,
        activity_count=1,
    )

    cached = step6.load_cached_root_result(
        pair=pair, output_path=output_path, execution_mode="direct_llm"
    )
    assert cached is not None
    assert cached["task_model"]["version"] == "0.2"
