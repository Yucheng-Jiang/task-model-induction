from __future__ import annotations

from task_model_induction.codex_cli_sandbox.runner import codex_config_toml
from task_model_induction.codex_cli_sandbox.sandbox import CodexCliSandbox


def test_codex_config_toml_includes_provider_block() -> None:
    config = codex_config_toml(
        model="gpt-5.4",
        model_reasoning_effort="medium",
        personality="pragmatic",
        model_provider="sandbox",
        provider_name="Task Model Induction Sandbox",
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
    )

    assert 'model_provider = "sandbox"' in config
    assert "[model_providers.sandbox]" in config
    assert 'base_url = "https://api.openai.com/v1"' in config
    assert 'env_key = "OPENAI_API_KEY"' in config


def test_resolve_codex_config_uses_litellm_env_ref_and_base_url() -> None:
    sandbox = CodexCliSandbox()

    resolved = sandbox._resolve_codex_config(
        {
            "litellm_params": {
                "model": "openai/gpt-5.4",
                "api_key": "os.environ/OPENAI_API_KEY",
                "api_base": "https://example.test/v1",
            }
        }
    )

    assert resolved["model"] == "gpt-5.4"
    assert resolved["env_key"] == "OPENAI_API_KEY"
    assert resolved["base_url"] == "https://example.test/v1"
    assert resolved["model_provider"] == "sandbox"


def test_resolve_codex_config_defaults_base_url_to_openai() -> None:
    sandbox = CodexCliSandbox()

    resolved = sandbox._resolve_codex_config(
        {
            "litellm_params": {
                "model": "openai/gpt-5.4",
                "api_key": "os.environ/CUSTOM_OPENAI_API_KEY",
            }
        }
    )

    assert resolved["model"] == "gpt-5.4"
    assert resolved["env_key"] == "CUSTOM_OPENAI_API_KEY"
    assert resolved["base_url"] == "https://api.openai.com/v1"


def test_resolve_codex_config_strips_openai_prefix() -> None:
    sandbox = CodexCliSandbox()

    resolved = sandbox._resolve_codex_config({"model": "openai/gpt-5.4"})

    assert resolved["model"] == "gpt-5.4"


def test_docker_env_args_use_configured_env_key() -> None:
    sandbox = CodexCliSandbox()

    env_args = sandbox._docker_env_args(
        {
            "env_key": "OVAL_OPENAI_API_KEY",
            "litellm_params": {"api_key": "secret"},
        }
    )

    assert env_args[:2] == ["-e", "OVAL_OPENAI_API_KEY=secret"]
