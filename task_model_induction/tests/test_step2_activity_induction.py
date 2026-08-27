from task_model_induction.step2_activity_induction import RunStats
from task_model_induction.schemas.activity_induction_output import ActivityInductionMeta


def test_runstats_record_call_accepts_cache_read_tokens() -> None:
    stats = RunStats()

    stats.record_call(
        operation="segmentation",
        model="openai/gpt-5.4-mini",
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_read_tokens": 40,
        },
        estimated_usd=0.12,
    )

    assert stats.llm_requests == 1
    assert stats.input_tokens == 100
    assert stats.output_tokens == 20
    assert stats.total_tokens == 120
    assert stats.cache_read_tokens == 40
    assert stats.as_meta()["cache_read_tokens"] == 40
    assert stats.cost_breakdown()["total"]["cache_read_tokens"] == 40
    assert stats.cost_breakdown()["by_operation"] == [
        {
            "operation": "segmentation",
            "model": "openai/gpt-5.4-mini",
            "llm_requests": 1,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_read_tokens": 40,
            "estimated_usd": 0.12,
        }
    ]


def test_activity_induction_meta_accepts_cache_read_tokens() -> None:
    meta = ActivityInductionMeta(
        created_at="2026-05-21T22:20:08Z",
        model="openai/gpt-5.4-mini",
        input_path="/tmp/in.jsonl",
        output_path="/tmp/out.jsonl",
        num_semantic_actions=71,
        num_candidate_segments=10,
        num_activities=5,
        segmentation_batch_size=40,
        merge_batch_size=16,
        merge_batch_overlap=2,
        max_prior_segments=8,
        reused_cache=False,
        preflight_only=False,
        elapsed_secs=9.76,
        llm_requests=3,
        input_tokens=18635,
        output_tokens=1195,
        total_tokens=19830,
        cache_read_tokens=0,
        estimated_usd=0.019354,
        cost_breakdown={"total": {"cache_read_tokens": 0}, "by_operation": []},
    )

    assert meta.cache_read_tokens == 0
