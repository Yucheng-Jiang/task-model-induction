import json

from PIL import Image

try:
    from task_model_induction.utils import (
        image_data_url,
        infer_screen_size,
        iter_activity_jsonl,
        read_activity_jsonl,
        write_action_grounding_jsonl_iter,
        write_action_grounding_jsonl,
    )
    from task_model_induction.schemas import (
        ActionGroundingOutput,
        ComputerUseActivityEntry,
    )
except ModuleNotFoundError:
    from utils import (
        image_data_url,
        infer_screen_size,
        iter_activity_jsonl,
        read_activity_jsonl,
        write_action_grounding_jsonl_iter,
        write_action_grounding_jsonl,
    )
    from schemas import ActionGroundingOutput, ComputerUseActivityEntry


def test_client_image_helpers(tmp_path):
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (12, 8), color="white").save(image_path)

    assert infer_screen_size(image_path) == {"width": 12, "height": 8}
    assert image_data_url(image_path).startswith("data:image/png;base64,")


def test_client_reads_typed_activity_jsonl(tmp_path):
    jsonl_path = tmp_path / "activity.jsonl"
    jsonl_path.write_text(
        (
            '{"id": "a1", "action": "click(10, 20)", "state_before": "before.png", '
            '"state_after": "after.png", "time_before": 1.0, "time_after": 2.0, "time_range": 1.0}\n'
        ),
        encoding="utf-8",
    )

    loaded = read_activity_jsonl(jsonl_path)

    assert loaded == [
        ComputerUseActivityEntry(
            id="a1",
            action="click(10, 20)",
            state_before="before.png",
            state_after="after.png",
            time_before=1.0,
            time_after=2.0,
            time_range=1.0,
        )
    ]


def test_client_writes_typed_action_grounding_jsonl(tmp_path):
    jsonl_path = tmp_path / "action_grounding.jsonl"
    output = ActionGroundingOutput(
        status="success",
        goal="Open the menu",
        active_application="Example",
        visual_content="Toolbar",
        ocr_results={"screen_size": {"width": 12, "height": 8}},
        md_results="",
        cost={"total_usd": 0.0},
        completed_at="2026-05-06T00:00:00Z",
        provenounce_id="a1",
    )

    write_action_grounding_jsonl(jsonl_path, [output])

    assert json.loads(jsonl_path.read_text(encoding="utf-8")) == output.model_dump(mode="json")


def test_client_iterates_typed_activity_jsonl(tmp_path):
    jsonl_path = tmp_path / "activity.jsonl"
    jsonl_path.write_text(
        (
            '{"id": "a1", "action": "click(10, 20)"}\n'
            '{"id": "a2", "action": "type(\\"hello\\")"}\n'
        ),
        encoding="utf-8",
    )

    loaded = list(iter_activity_jsonl(jsonl_path))

    assert [entry.id for entry in loaded] == ["a1", "a2"]
    assert [entry.action for entry in loaded] == ["click(10, 20)", 'type("hello")']


def test_client_writes_typed_action_grounding_jsonl_iter(tmp_path):
    jsonl_path = tmp_path / "action_grounding_iter.jsonl"
    outputs = [
        ActionGroundingOutput(
            status="success",
            goal="Open the menu",
            completed_at="2026-05-06T00:00:00Z",
            provenounce_id="a1",
        ),
        ActionGroundingOutput(
            status="error",
            error="service failed",
            completed_at="2026-05-06T00:00:01Z",
            provenounce_id="a2",
        ),
    ]

    write_action_grounding_jsonl_iter(jsonl_path, iter(outputs))

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [output.model_dump(mode="json") for output in outputs]
