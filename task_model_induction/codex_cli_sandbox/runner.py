#!/usr/bin/env python3
"""Minimal in-container Codex CLI runner for task model induction."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/run/tmi-codex")
REQUEST_PATH = RUN_ROOT / "request.json"
RESULT_PATH = RUN_ROOT / "result.json"
WORKSPACE_DIR = RUN_ROOT / "workspace"
INPUT_DIR = WORKSPACE_DIR / "input"
OUTPUT_DIR = WORKSPACE_DIR / "output"
HOME_DIR = RUN_ROOT / "home"
CODEX_HOME = HOME_DIR / ".codex"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_files(files: dict[str, str]) -> None:
    for relative_name, content in files.items():
        target = INPUT_DIR / relative_name
        resolved = target.resolve()
        if INPUT_DIR.resolve() != resolved and INPUT_DIR.resolve() not in resolved.parents:
            raise ValueError(f"input file escapes input directory: {relative_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")


def read_output(name: str) -> str | None:
    target = OUTPUT_DIR / name
    try:
        resolved = target.resolve()
        root = OUTPUT_DIR.resolve()
    except OSError:
        return None
    if root != resolved and root not in resolved.parents:
        return None
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8")


def codex_config_toml(
    *,
    model: str,
    model_reasoning_effort: str,
    personality: str,
    model_provider: str,
    provider_name: str,
    base_url: str,
    env_key: str,
) -> str:
    provider = model_provider or "sandbox"
    return (
        f"model = {json.dumps(model)}\n"
        f"model_reasoning_effort = {json.dumps(model_reasoning_effort)}\n"
        f"personality = {json.dumps(personality)}\n"
        f"model_provider = {json.dumps(provider)}\n"
        "\n"
        f"[model_providers.{provider}]\n"
        f"name = {json.dumps(provider_name)}\n"
        f"base_url = {json.dumps(base_url)}\n"
        f"env_key = {json.dumps(env_key)}\n"
        f"model = {json.dumps(model)}\n"
    )


def main() -> int:
    started_at = utc_now_iso()
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    codex_config = request.get("codex_config") if isinstance(request.get("codex_config"), dict) else {}
    model = str(codex_config.get("model") or request.get("model") or "gpt-5.4")
    timeout = int(codex_config.get("command_timeout_seconds") or 1200)
    output_file = str(request.get("output_file") or "hierarchy.json")
    extra_output_files = tuple(str(item) for item in request.get("output_files") or ())

    for path in (INPUT_DIR, OUTPUT_DIR, CODEX_HOME):
        path.mkdir(parents=True, exist_ok=True)
    write_files(request.get("files") if isinstance(request.get("files"), dict) else {})
    (CODEX_HOME / "config.toml").write_text(
        codex_config_toml(
            model=model,
            model_reasoning_effort=str(codex_config.get("model_reasoning_effort") or "medium"),
            personality=str(codex_config.get("personality") or "pragmatic"),
            model_provider=str(codex_config.get("model_provider") or "sandbox"),
            provider_name=str(codex_config.get("provider_name") or "Task Model Induction Sandbox"),
            base_url=str(codex_config.get("base_url") or "https://api.openai.com/v1"),
            env_key=str(codex_config.get("env_key") or "OPENAI_API_KEY"),
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["HOME"] = str(HOME_DIR)
    env["CODEX_HOME"] = str(CODEX_HOME)
    command = [
        "codex",
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        str(request.get("prompt") or ""),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=WORKSPACE_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        output_content = read_output(output_file)
        result = {
            "ok": completed.returncode == 0 and output_content is not None,
            "run_id": request.get("run_id"),
            "session_id": request.get("run_id"),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_file": output_file,
            "output_content": output_content,
            "output_files": {name: read_output(name) for name in extra_output_files},
            "error": None if completed.returncode == 0 and output_content is not None else "Codex run did not complete successfully",
            "codex": {
                "model": model,
                "model_reasoning_effort": codex_config.get("model_reasoning_effort") or "medium",
                "personality": codex_config.get("personality") or "pragmatic",
                "command_timeout_seconds": timeout,
            },
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "ok": False,
            "run_id": request.get("run_id"),
            "session_id": request.get("run_id"),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"Codex timed out after {timeout}s",
            "codex": {"model": model},
        }
    write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
