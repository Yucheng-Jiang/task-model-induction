from __future__ import annotations

import json
from pathlib import Path

import task_model_induction.step6_bidirectional_alignment as step6


def test_bidirectional_alignment_reuses_cached_per_root_outputs(tmp_path: Path, monkeypatch) -> None:
    objective_dir = tmp_path / "task_thread_objective_model"
    procedure_dir = tmp_path / "task_thread_procedure_model"
    output_dir = tmp_path / "task_thread_task_model"
    objective_dir.mkdir()
    procedure_dir.mkdir()
    output_dir.mkdir()

    (objective_dir / "thread.json").write_text(json.dumps({"id": "C1"}), encoding="utf-8")
    (procedure_dir / "thread.json").write_text(json.dumps({"root_procedure_id": "P0"}), encoding="utf-8")

    cached_task_model = {
        "version": "0.2",
        "root": {
            "id": "C1",
            "objective": "Fix bug",
            "summary": "Resolve the failing behavior.",
            "deliverables": [
                {
                    "kind": "state",
                    "target": "failing behavior",
                    "expected_state": "resolved",
                    "evidence_refs": ["action_0000"],
                }
            ],
            "success_criteria": [
                {
                    "predicate": "The failing behavior no longer occurs",
                    "verifier": "action_result",
                    "evidence_refs": ["action_0000"],
                    "confidence": 0.9,
                }
            ],
            "observed_outcome": {
                "status": "achieved",
                "description": "The corrected behavior was observed.",
                "evidence_refs": ["action_0000"],
            },
            "evidence_refs": ["action_0000"],
            "activity_refs": ["activity_0000"],
            "procedure": {
                "operator": "SEQ",
                "name": "Main flow",
                "body": [{"name": "Investigate", "activity_refs": ["activity_0000"]}],
            },
            "decomposition": [],
        },
    }
    (output_dir / "thread.json").write_text(json.dumps(cached_task_model), encoding="utf-8")
    (output_dir / "thread.json.meta.json").write_text(
        json.dumps(
            {
                "execution_mode": "codex_cli",
                "run_id": "cached-run",
                "session_id": "cached-session",
                "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                "estimated_usd": 0.12,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        step6,
        "run_direct_reconciliation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("direct reconciliation should not run")),
    )
    monkeypatch.setattr(
        step6,
        "run_codex_reconciliation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("codex reconciliation should not run")),
    )

    payload = step6.bidirectional_alignment(
        data_dir=tmp_path,
        objective_output_dir="task_thread_objective_model",
        procedure_output_dir="task_thread_procedure_model",
        output_dir="task_thread_task_model",
        output_file_name="task_model.json",
        direct_model="test-direct-model",
        direct_litellm_params=None,
        direct_llm_max_activities=50,
        codex_config={"model": "test-codex-model"},
        max_retries=1,
        workers=1,
        llm_timeout_secs=30.0,
        reuse_cache=True,
        no_console=True,
    )

    assert payload is not None
    assert payload["meta"]["num_roots"] == 1
    assert payload["meta"]["num_succeeded"] == 1
    assert payload["roots"][0]["output_file"] == str(output_dir / "thread.json")
    assert payload["roots"][0]["run_id"] == "cached-run"
    assert payload["roots"][0]["usage"] == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert payload["roots"][0]["task_model"]["version"] == "0.2"
    assert payload["roots"][0]["task_model"]["root"]["id"] == "C1"
    assert payload["roots"][0]["task_model"]["root"]["objective"] == "Fix bug"
    assert payload["roots"][0]["task_model"]["root"]["activity_refs"] == ["activity_0000"]
