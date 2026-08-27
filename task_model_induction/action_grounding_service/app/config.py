from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


_APPLIED_ENDPOINT_ENV: dict[str, str] = {}


def resolve_dotenv_path(config_path: str | Path, dotenv_path: str | Path) -> Path:
    """Resolve service dotenv files without importing the parent package.

    The service Docker image intentionally installs only ``action_grounding_service``.
    Keep this small path resolver local so container startup does not depend on the
    rest of ``task_model_induction`` being present.
    """

    config_root = Path(config_path).expanduser().resolve().parent
    candidate = Path(dotenv_path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base_dir in (config_root, *config_root.parents, Path.cwd().resolve()):
        resolved = base_dir / candidate
        if resolved.exists():
            return resolved
    return config_root / candidate


class LiteLlmEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    timeout_secs: float = 120
    max_tokens: int = 4096
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    api_base_env: str | None = None
    api_version: str | None = None
    api_version_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


class OcrConfig(LiteLlmEndpointConfig):
    model: str = "openai/gpt-5.4-mini"
    max_tokens: int = 16384


class VlmConfig(LiteLlmEndpointConfig):
    model: str = "openai/gpt-5.4-mini"
    max_tokens: int = 1024


class OmniParserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrent_requests: int = 2
    timeout_secs: float = 180
    box_threshold: float = 0.05
    iou_threshold: float = 0.1
    imgsz: int = 640


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dotenv_path: str | None = ".env"
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    omniparser: OmniParserConfig = Field(default_factory=OmniParserConfig)
    max_concurrent_requests: int = 32

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        migrated = dict(data)
        ocr = migrated.get("ocr")
        if isinstance(ocr, dict):
            legacy_litellm = ocr.pop("litellm", None)
            if isinstance(legacy_litellm, dict):
                merged_ocr = dict(legacy_litellm)
                merged_ocr.update({key: value for key, value in ocr.items() if key in _endpoint_config_keys()})
                migrated["ocr"] = merged_ocr
            else:
                migrated["ocr"] = {
                    key: value
                    for key, value in ocr.items()
                    if key in _endpoint_config_keys()
                }
            if "dotenv_path" in ocr and "dotenv_path" not in migrated:
                migrated["dotenv_path"] = ocr["dotenv_path"]

        if "backend" in migrated:
            migrated.pop("backend", None)
        for removed_key in ("litellm", "zai", "local_glm"):
            migrated.pop(removed_key, None)

        vlm = migrated.get("vlm")
        if isinstance(vlm, dict):
            vlm = dict(vlm)
            model_settings = vlm.pop("model_settings", None)
            if isinstance(model_settings, dict):
                completion_kwargs = dict(vlm.get("completion_kwargs") or {})
                completion_kwargs.update(model_settings)
                vlm["completion_kwargs"] = completion_kwargs
            migrated["vlm"] = vlm

        return migrated


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Service config file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "action_grounding_service" in data:
            root_data = data
            service_data = data["action_grounding_service"]
            if not isinstance(service_data, dict):
                raise ValueError("action_grounding_service config must be a mapping")
            data = dict(service_data)
            if "dotenv_path" in data:
                raise ValueError("dotenv_path belongs at the top level of config.yaml, not under action_grounding_service")
            if "dotenv_path" in root_data:
                data["dotenv_path"] = root_data["dotenv_path"]
    else:
        raise ValueError("Service config supports TOML and YAML files only")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Service config must be a mapping")
    return data


def _resolve_config_path() -> Path:
    raw_path = os.getenv("ACTION_GROUNDING_CONFIG", "/app/config.yaml")
    return Path(raw_path).expanduser()


def _load_dotenv(config: ServiceConfig, config_path: Path) -> None:
    if not config.dotenv_path:
        return
    dotenv_path = resolve_dotenv_path(config_path, config.dotenv_path)
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)


def load_service_config() -> ServiceConfig:
    path = _resolve_config_path()
    if path.exists():
        return ServiceConfig.model_validate(_read_config_file(path))
    return ServiceConfig()


def save_service_config(config: ServiceConfig) -> ServiceConfig:
    path = _resolve_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_config_file(path, config)
    apply_service_config_to_env(config)
    return config


def apply_service_config_to_env(config: ServiceConfig | None = None) -> ServiceConfig:
    config_path = _resolve_config_path()
    config = config or load_service_config()
    _clear_previous_endpoint_env()
    _load_dotenv(config, config_path)

    _apply_litellm_endpoint_env("OCR", config.ocr)
    _apply_litellm_endpoint_env("VLM", config.vlm)

    os.environ["OMNIPARSER_MAX_CONCURRENT_REQUESTS"] = str(config.omniparser.max_concurrent_requests)
    os.environ["OMNIPARSER_TIMEOUT_SECS"] = str(config.omniparser.timeout_secs)
    os.environ["OMNIPARSER_BOX_THRESHOLD"] = str(config.omniparser.box_threshold)
    os.environ["OMNIPARSER_IOU_THRESHOLD"] = str(config.omniparser.iou_threshold)
    os.environ["OMNIPARSER_IMGSZ"] = str(config.omniparser.imgsz)
    os.environ["MAX_CONCURRENT_REQUESTS"] = str(config.max_concurrent_requests)
    return config


def litellm_call_kwargs(endpoint: LiteLlmEndpointConfig) -> dict[str, Any]:
    kwargs = dict(endpoint.completion_kwargs)
    api_key = _resolve_config_value(endpoint.api_key) or (
        os.getenv(endpoint.api_key_env) if endpoint.api_key_env else None
    )
    api_base = _resolve_config_value(endpoint.api_base) or (
        os.getenv(endpoint.api_base_env) if endpoint.api_base_env else None
    )
    api_version = _resolve_config_value(endpoint.api_version) or (
        os.getenv(endpoint.api_version_env) if endpoint.api_version_env else None
    )
    base_url = _resolve_config_value(endpoint.base_url) or (
        os.getenv(endpoint.base_url_env) if endpoint.base_url_env else None
    )
    if api_key:
        kwargs.setdefault("api_key", api_key)
    if api_base:
        kwargs.setdefault("api_base", api_base.rstrip("/"))
    if api_version:
        kwargs.setdefault("api_version", api_version)
    if base_url:
        kwargs.setdefault("base_url", base_url.rstrip("/"))
    return kwargs


def redacted_service_config(config: ServiceConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    for section in ("ocr", "vlm"):
        endpoint = data.get(section)
        if not isinstance(endpoint, dict):
            continue
        if endpoint.get("api_key"):
            endpoint["api_key"] = "********"
        env = endpoint.get("env")
        if isinstance(env, dict):
            for key, value in list(env.items()):
                if value and _is_sensitive_env_key(key):
                    env[key] = "********"
    return data


def merge_preserving_secrets(payload: dict[str, Any], current: ServiceConfig) -> ServiceConfig:
    data = dict(payload)
    current_data = current.model_dump(mode="json")
    for section in ("ocr", "vlm"):
        endpoint = data.get(section)
        if not isinstance(endpoint, dict):
            continue
        value = endpoint.get("api_key")
        if value in (None, "", "********"):
            current_api_key = current_data.get(section, {}).get("api_key")
            if current_api_key:
                endpoint["api_key"] = current_api_key
            else:
                endpoint.pop("api_key", None)
        env = endpoint.get("env")
        current_env = current_data.get(section, {}).get("env")
        if isinstance(env, dict) and isinstance(current_env, dict):
            for key, env_value in list(env.items()):
                if env_value == "********" and key in current_env:
                    env[key] = current_env[key]
    return ServiceConfig.model_validate(data)


def _apply_litellm_endpoint_env(prefix: str, endpoint: LiteLlmEndpointConfig) -> None:
    os.environ[f"{prefix}_LITELLM_MODEL"] = endpoint.model
    os.environ[f"{prefix}_LITELLM_TIMEOUT_SECS"] = str(endpoint.timeout_secs)
    os.environ[f"{prefix}_LITELLM_MAX_TOKENS"] = str(endpoint.max_tokens)
    os.environ[f"{prefix}_LITELLM_COMPLETION_KWARGS"] = json.dumps(endpoint.completion_kwargs, ensure_ascii=False)
    if endpoint.api_key_env:
        os.environ[f"{prefix}_LITELLM_API_KEY_ENV"] = endpoint.api_key_env
    if endpoint.api_base_env:
        os.environ[f"{prefix}_LITELLM_API_BASE_ENV"] = endpoint.api_base_env
    if endpoint.api_version_env:
        os.environ[f"{prefix}_LITELLM_API_VERSION_ENV"] = endpoint.api_version_env
    if endpoint.base_url_env:
        os.environ[f"{prefix}_LITELLM_BASE_URL_ENV"] = endpoint.base_url_env
    for key, value in endpoint.env.items():
        if value != "":
            os.environ[key] = value
            _APPLIED_ENDPOINT_ENV[key] = value


def _clear_previous_endpoint_env() -> None:
    for key, value in list(_APPLIED_ENDPOINT_ENV.items()):
        if os.environ.get(key) == value:
            os.environ.pop(key, None)
        _APPLIED_ENDPOINT_ENV.pop(key, None)


def _is_sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    return any(part in upper for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))


def _endpoint_config_keys() -> set[str]:
    return {
        "env",
        "model",
        "timeout_secs",
        "max_tokens",
        "api_key",
        "api_key_env",
        "api_base",
        "api_base_env",
        "api_version",
        "api_version_env",
        "base_url",
        "base_url_env",
        "completion_kwargs",
    }


def _resolve_config_value(value: str | None) -> str | None:
    if not value:
        return None
    prefix = "os.environ/"
    if value.startswith(prefix):
        return os.getenv(value.removeprefix(prefix))
    return value


def _to_toml(config: ServiceConfig) -> str:
    lines: list[str] = [
        f"max_concurrent_requests = {config.max_concurrent_requests}",
        f"dotenv_path = {_toml_value(config.dotenv_path)}",
        "",
    ]
    _append_endpoint(lines, "ocr", config.ocr)
    _append_endpoint(lines, "vlm", config.vlm)
    lines.extend(
        [
            "[omniparser]",
            f"max_concurrent_requests = {_toml_value(config.omniparser.max_concurrent_requests)}",
            f"timeout_secs = {_toml_value(config.omniparser.timeout_secs)}",
            f"box_threshold = {_toml_value(config.omniparser.box_threshold)}",
            f"iou_threshold = {_toml_value(config.omniparser.iou_threshold)}",
            f"imgsz = {_toml_value(config.omniparser.imgsz)}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_config_file(path: Path, config: ServiceConfig) -> None:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        path.write_text(_to_toml(config), encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Service config YAML must be a mapping")
        service_data = config.model_dump(mode="json")
        data["dotenv_path"] = service_data.pop("dotenv_path")
        data["action_grounding_service"] = service_data
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return
    raise ValueError("Service config supports TOML and YAML files only")


def _append_endpoint(lines: list[str], name: str, endpoint: LiteLlmEndpointConfig) -> None:
    lines.extend(
        [
            f"[{name}]",
            f"model = {_toml_value(endpoint.model)}",
            f"timeout_secs = {_toml_value(endpoint.timeout_secs)}",
            f"max_tokens = {_toml_value(endpoint.max_tokens)}",
        ]
    )
    for field_name in (
        "api_key",
        "api_key_env",
        "api_base",
        "api_base_env",
        "api_version",
        "api_version_env",
        "base_url",
        "base_url_env",
    ):
        value = getattr(endpoint, field_name)
        if value:
            lines.append(f"{field_name} = {_toml_value(value)}")
    lines.append("")
    if endpoint.env:
        lines.append(f"[{name}.env]")
        for key, value in endpoint.env.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    if endpoint.completion_kwargs:
        lines.append(f"[{name}.completion_kwargs]")
        for key, value in endpoint.completion_kwargs.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")


def _toml_value(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)
