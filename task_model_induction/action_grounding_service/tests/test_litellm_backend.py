from types import SimpleNamespace

from action_grounding_service.app.backends import LiteLlmOcrBackend
from action_grounding_service.app.image_utils import pil_to_data_url
from action_grounding_service.app.schemas import ActionGroundingRequest
from PIL import Image


def test_litellm_backend_passes_qwen_endpoint_and_strips_thinking(tmp_path, monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="We should OCR this first.</think>\n\n# Window\n- OK")
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        return SimpleNamespace(choices=[choice], usage=usage)

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[ocr]
model = "openai/qwen3-vl-32b-thinking"
api_key_env = "QWEN_TOKEN"
base_url_env = "QWEN_URL"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))
    monkeypatch.setenv("QWEN_TOKEN", "test-token")
    monkeypatch.setenv("QWEN_URL", "http://example.test:18000/v1")

    request = ActionGroundingRequest.model_validate(
        {
            "before_image": pil_to_data_url(Image.new("RGB", (10, 10), "white")),
            "after_image": None,
            "action": "click(1, 1)",
            "screen_size": {"width": 10, "height": 10},
        }
    )

    result = LiteLlmOcrBackend().extract_markdown(request)

    assert captured["model"] == "openai/qwen3-vl-32b-thinking"
    assert captured["api_key"] == "test-token"
    assert captured["base_url"] == "http://example.test:18000/v1"
    assert result["md_results"] == "# Window\n- OK"
