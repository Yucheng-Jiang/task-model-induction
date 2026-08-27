"""Host-side runner for the standalone Codex CLI Docker sandbox."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


DEFAULT_IMAGE = "task-model-induction-codex-cli-sandbox:latest"


def _bind_mountable_tmpdir() -> str:
    """Temp directory whose real path Docker can bind-mount.

    Docker shares resolved paths, not symlinks, so the run directory has to be
    created under a real path: on macOS /tmp is a symlink to /private/tmp, and
    only the latter is bind-mountable. Resolving gets that for free on macOS and
    leaves /tmp untouched on Linux, where /private/tmp does not exist.
    """
    for candidate in (os.sep + "tmp", tempfile.gettempdir()):
        resolved = os.path.realpath(candidate)
        if os.path.isdir(resolved) and os.access(resolved, os.W_OK):
            return resolved
    return tempfile.gettempdir()


class CodexSandboxError(RuntimeError):
    """Raised when the Codex CLI sandbox cannot run."""


class CodexCliSandbox:
    def __init__(self, *, image: str = DEFAULT_IMAGE, package_dir: Path | None = None) -> None:
        self.image = image
        self.package_dir = package_dir or Path(__file__).resolve().parents[1]
        self.repo_root = self.package_dir.parent
        self.dockerfile = self.package_dir / "codex_cli_sandbox" / "Dockerfile"

    def ensure_image(self, *, rebuild: bool = False) -> None:
        self._ensure_docker_available()
        exists = self._run(["docker", "image", "inspect", self.image], check=False).returncode == 0
        if exists and not rebuild:
            return
        self._run(
            [
                "docker",
                "build",
                "-t",
                self.image,
                "-f",
                str(self.dockerfile),
                str(self.repo_root),
            ]
        )

    def run_file_task(
        self,
        *,
        prompt: str,
        files: dict[str, str],
        output_file: str,
        output_files: tuple[str, ...] = (),
        codex_config: dict[str, Any] | None = None,
        rebuild_image: bool = False,
    ) -> dict[str, Any]:
        self.ensure_image(rebuild=rebuild_image)
        effective_config = self._resolve_codex_config(codex_config or {})
        with tempfile.TemporaryDirectory(prefix="tmi-codex-", dir=_bind_mountable_tmpdir()) as temp_dir:
            run_dir = Path(temp_dir)
            request_path = run_dir / "request.json"
            result_path = run_dir / "result.json"
            request = {
                "run_id": str(uuid.uuid4()),
                "prompt": prompt,
                "files": files,
                "output_file": output_file,
                "output_files": list(output_files),
                "codex_config": effective_config,
            }
            request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
            env_args = self._docker_env_args(effective_config)
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/run/tmi-codex",
                *env_args,
                self.image,
            ]
            completed = self._run(command, check=False)
            if completed.returncode != 0 and not result_path.exists():
                raise CodexSandboxError(
                    "Codex sandbox container failed before producing result.json: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            if not result_path.exists():
                raise CodexSandboxError("Codex sandbox did not produce result.json")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise CodexSandboxError("Codex sandbox result.json must contain an object")
            return result

    def _resolve_codex_config(self, config: dict[str, Any]) -> dict[str, Any]:
        params = config.get("litellm_params") if isinstance(config.get("litellm_params"), dict) else {}
        resolved_params = {key: _resolve_env_ref(value) for key, value in params.items()}
        model = _codex_model_name(str(config.get("model") or params.get("model") or "openai/gpt-5.4"))
        env_key = _env_ref_name(params.get("api_key")) or "OPENAI_API_KEY"
        base_url = _normalize_base_url(
            str(
                resolved_params.get("base_url")
                or resolved_params.get("api_base")
                or "https://api.openai.com/v1"
            )
        )
        return {
            **config,
            "model": model,
            "model_provider": str(config.get("model_provider") or "sandbox"),
            "provider_name": str(config.get("provider_name") or "Task Model Induction Sandbox"),
            "base_url": base_url,
            "env_key": env_key,
            "litellm_params": resolved_params,
        }

    def _docker_env_args(self, config: dict[str, Any]) -> list[str]:
        params = config.get("litellm_params") if isinstance(config.get("litellm_params"), dict) else {}
        args: list[str] = []
        env_key = str(config.get("env_key") or "OPENAI_API_KEY")
        api_key = params.get("api_key") or os.environ.get(env_key) or os.environ.get("OPENAI_API_KEY")
        if api_key:
            args.extend(["-e", f"{env_key}={api_key}"])
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            if os.environ.get(name):
                args.extend(["-e", f"{name}={os.environ[name]}"])
        return args

    def _ensure_docker_available(self) -> None:
        try:
            self._run(["docker", "version", "--format", "{{.Server.Version}}"])
        except CodexSandboxError as exc:
            raise CodexSandboxError("Docker is not available. Start Docker and retry.") from exc

    def _run(self, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and completed.returncode != 0:
            raise CodexSandboxError(completed.stderr.strip() or completed.stdout.strip())
        return completed


def _resolve_env_ref(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("os.environ/"):
        return os.environ.get(value.removeprefix("os.environ/"))
    return value


def _env_ref_name(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("os.environ/"):
        return value.removeprefix("os.environ/")
    return None


def _codex_model_name(model: str) -> str:
    if "/" in model:
        provider, _, name = model.partition("/")
        if provider == "openai" and name:
            return name
    return model


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    return normalized or "https://api.openai.com/v1"
