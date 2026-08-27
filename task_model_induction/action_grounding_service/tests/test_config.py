import os
from pathlib import Path

from action_grounding_service.app.config import (
    apply_service_config_to_env,
    load_service_config,
    merge_preserving_secrets,
    redacted_service_config,
    save_service_config,
)
from task_model_induction.config import resolve_dotenv_path


def test_litellm_ocr_env_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
dotenv_path = ".env"

[ocr]
model = "openai/qwen3-vl-32b-thinking"

[ocr.env]
OPENAI_API_KEY = "test-token"
OPENAI_API_BASE = "http://example.test:18000/v1"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    config = apply_service_config_to_env()

    assert config.ocr.model == "openai/qwen3-vl-32b-thinking"
    assert os.environ["OCR_LITELLM_MODEL"] == "openai/qwen3-vl-32b-thinking"
    assert os.environ["OPENAI_API_KEY"] == "test-token"
    assert os.environ["OPENAI_API_BASE"] == "http://example.test:18000/v1"


def test_litellm_vlm_env_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
dotenv_path = ".env"

[vlm]
model = "openai/gpt-5.4-mini"

[vlm.env]
OPENAI_API_KEY = "abc"
OPENAI_API_BASE = "https://example.test/v1"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    config = apply_service_config_to_env()

    assert config.vlm.model == "openai/gpt-5.4-mini"
    assert os.environ["VLM_LITELLM_MODEL"] == "openai/gpt-5.4-mini"
    assert os.environ["OPENAI_API_KEY"] == "abc"
    assert os.environ["OPENAI_API_BASE"] == "https://example.test/v1"


def test_removed_direct_env_value_does_not_linger(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config_path.write_text(
        """
[ocr]
model = "openai/qwen3-vl-32b-thinking"

[ocr.env]
OPENAI_API_KEY = "first"
""",
        encoding="utf-8",
    )
    apply_service_config_to_env()
    assert os.environ["OPENAI_API_KEY"] == "first"

    config_path.write_text(
        """
[ocr]
model = "openai/qwen3-vl-32b-thinking"
""",
        encoding="utf-8",
    )
    apply_service_config_to_env()
    assert "OPENAI_API_KEY" not in os.environ


def test_legacy_litellm_ocr_config_still_loads(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[ocr]
backend = "litellm"
dotenv_path = ".env"

[ocr.litellm]
model = "openai/gpt-4.1-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))

    config = load_service_config()

    assert config.dotenv_path == ".env"
    assert config.ocr.model == "openai/gpt-4.1-mini"


def test_save_service_config_persists_toml_and_redacts_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))

    config = load_service_config()
    config.ocr.api_key = "secret"
    save_service_config(config)

    loaded = load_service_config()
    redacted = redacted_service_config(loaded)

    assert loaded.ocr.api_key == "secret"
    assert redacted["ocr"]["api_key"] == "********"
    assert "openai/gpt-5.4-mini" in config_path.read_text(encoding="utf-8")


def test_unified_yaml_service_section_round_trip(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
action_grounding_stage:
  grounding_url: "http://localhost:8000"
dotenv_path: ".env"
action_grounding_service:
  ocr:
    model: "openai/qwen3-vl-32b-thinking"
  vlm:
    model: "openai/gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))

    config = load_service_config()
    config.ocr.api_key = "secret"
    config.omniparser.imgsz = 512
    config.omniparser.max_concurrent_requests = 2
    save_service_config(config)

    loaded = load_service_config()
    redacted = redacted_service_config(loaded)
    saved = config_path.read_text(encoding="utf-8")

    assert loaded.ocr.model == "openai/qwen3-vl-32b-thinking"
    assert loaded.ocr.api_key == "secret"
    assert loaded.omniparser.imgsz == 512
    assert loaded.omniparser.max_concurrent_requests == 2
    assert redacted["ocr"]["api_key"] == "********"
    assert "action_grounding_stage:" in saved
    assert 'dotenv_path: ".env"' in saved or "dotenv_path: .env" in saved
    assert "action_grounding_service:" in saved
    assert "  dotenv_path:" not in saved
    assert "max_concurrent_requests: 2" in saved
    assert "imgsz: 512" in saved


def test_merge_preserves_redacted_secret():
    current = load_service_config()
    current.ocr.api_key = "secret"
    current.ocr.env = {"OPENAI_API_KEY": "env-secret"}
    payload = redacted_service_config(current)

    merged = merge_preserving_secrets(payload, current)

    assert merged.ocr.api_key == "secret"
    assert merged.ocr.env["OPENAI_API_KEY"] == "env-secret"


def test_resolve_dotenv_path_searches_parent_directories(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "task_model_induction" / "config.yaml"
    dotenv_path = repo_root / ".env"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dotenv_path: .env\n", encoding="utf-8")
    dotenv_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    monkeypatch.chdir(repo_root / "task_model_induction")

    resolved = resolve_dotenv_path(config_path, ".env")

    assert resolved == dotenv_path
    assert Path(resolved).read_text(encoding="utf-8") == "OPENAI_API_KEY=test\n"
