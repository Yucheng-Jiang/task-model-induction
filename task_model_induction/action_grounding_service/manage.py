#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVICE_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_IMAGE = "action-grounding-service:latest"
DEFAULT_OMNIPARSER_IMAGE = "action-grounding-omniparser:latest"
DEFAULT_CONTAINER = "action-grounding-service"
DEFAULT_OMNIPARSER_CONTAINER = "action-grounding-omniparser"
LEGACY_OMNIPARSER_CONTAINER = "omniparser"
DEFAULT_NETWORK = "action-grounding-net"
DEFAULT_OMNIPARSER_WEIGHTS_VOLUME = "action-grounding-omniparser-weights"
DEFAULT_PORT = 8000
DEFAULT_OMNIPARSER_PORT = 8080


class Console:
    def __init__(self) -> None:
        try:
            from rich.console import Console as RichConsole
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn

            self._rich = RichConsole()
            self._panel = Panel
            self._progress_cls = Progress
            self._spinner_col = SpinnerColumn
            self._text_col = TextColumn
        except Exception:
            self._rich = None

    def title(self, text: str) -> None:
        if self._rich:
            self._rich.print(self._panel.fit(text, border_style="cyan"))
        else:
            print(f"\n== {text} ==")

    def step(self, text: str) -> None:
        if self._rich:
            self._rich.print(f"[cyan]→[/cyan] {text}")
        else:
            print(f"-> {text}")

    def ok(self, text: str) -> None:
        if self._rich:
            self._rich.print(f"[green]✓[/green] {text}")
        else:
            print(f"[ok] {text}")

    def warn(self, text: str) -> None:
        if self._rich:
            self._rich.print(f"[yellow]![/yellow] {text}")
        else:
            print(f"[warn] {text}")

    def error(self, text: str) -> None:
        if self._rich:
            self._rich.print(f"[red]✗[/red] {text}")
        else:
            print(f"[error] {text}", file=sys.stderr)

    def command(self, cmd: list[str]) -> None:
        rendered = " ".join(cmd)
        if self._rich:
            self._rich.print(f"[dim]$ {rendered}[/dim]")
        else:
            print(f"$ {rendered}")

    def stream_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        if self._rich:
            self._rich.print(f"[dim]{line}[/dim]")
        else:
            print(line)


console = Console()


def run_stream(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    console.command(cmd)
    if dry_run:
        return
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        console.stream_line(line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def run_capture(cmd: list[str], cwd: Path = REPO_ROOT, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {detail}")
    return proc.stdout.strip()


def docker_available() -> bool:
    try:
        run_capture(["docker", "version", "--format", "{{.Server.Version}}"])
        return True
    except Exception:
        return False


def container_id(name: str) -> str | None:
    output = run_capture(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
        check=False,
    )
    return output.splitlines()[0] if output else None


def running_container_id(name: str) -> str | None:
    output = run_capture(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
        check=False,
    )
    return output.splitlines()[0] if output else None


def remove_container(name: str, dry_run: bool = False) -> None:
    cid = container_id(name)
    if not cid:
        return
    console.step(f"Removing existing container `{name}`")
    run_stream(["docker", "rm", "-f", name], cwd=REPO_ROOT, dry_run=dry_run)


def ensure_network(name: str, dry_run: bool = False) -> None:
    if dry_run:
        console.command(["docker", "network", "create", name])
        return
    exists = run_capture(["docker", "network", "ls", "--filter", f"name=^{name}$", "--format", "{{.Name}}"], check=False)
    if exists:
        return
    console.step(f"Creating Docker network `{name}`")
    run_stream(["docker", "network", "create", name], cwd=REPO_ROOT, dry_run=dry_run)


def wait_for_health(port: int, timeout_secs: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_secs
    url = f"http://localhost:{port}/health"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise TimeoutError(f"Service did not become healthy at {url}: {last_error}")


def build_image(args: argparse.Namespace) -> None:
    cmd = ["docker", "build", "-f", str(SERVICE_DIR / "Dockerfile"), "-t", args.image]
    cmd.append(str(PROJECT_DIR))
    console.step("Building action grounding Docker image")
    run_stream(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def build_omniparser_image(args: argparse.Namespace) -> None:
    cmd = [
        "docker",
        "build",
        "-f",
        str(SERVICE_DIR / "omniparser_server" / "Dockerfile"),
        "-t",
        args.omniparser_image,
        str(SERVICE_DIR / "omniparser_server"),
    ]
    console.step("Building OmniParser Docker image")
    run_stream(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def wait_for_container_health(container: str, timeout_secs: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_secs
    last_error = ""
    probe = (
        "import json, urllib.request; "
        "r=urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2); "
        "print(r.read().decode())"
    )
    while time.monotonic() < deadline:
        output = run_capture(["docker", "exec", container, "python", "-c", probe], check=False)
        if output:
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                last_error = output
        time.sleep(1)
    raise TimeoutError(f"OmniParser did not become healthy in container `{container}`: {last_error}")


def start_omniparser_container(args: argparse.Namespace) -> None:
    remove_container(args.omniparser_container, dry_run=args.dry_run)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        args.omniparser_container,
        "--network",
        args.network,
        "-p",
        f"{args.omniparser_port}:8080",
        "-v",
        f"{args.omniparser_weights_volume}:/weights",
        "-e",
        "OMNIPARSER_WEIGHTS_DIR=/weights",
        args.omniparser_image,
    ]
    console.step("Starting OmniParser container")
    run_stream(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)


def start_container(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    env_file = args.env_file.resolve() if args.env_file else None
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if env_file and not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")

    remove_container(args.container, dry_run=args.dry_run)
    sanitized_env = sanitize_env_file(env_file) if env_file else None
    try:
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            args.container,
            "-p",
            f"{args.port}:8000",
            "--network",
            args.network,
            "-v",
            f"{config_path}:/app/config.yaml",
            "-e",
            "ACTION_GROUNDING_CONFIG=/app/config.yaml",
        ]
        if sanitized_env:
            cmd.extend(["--env-file", str(sanitized_env)])
        cmd.append(args.image)

        console.step("Starting action grounding service container")
        run_stream(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
    finally:
        if sanitized_env and sanitized_env != env_file:
            sanitized_env.unlink(missing_ok=True)


def sanitize_env_file(env_file: Path) -> Path:
    """Write a Docker-compatible env file, tolerating whitespace around keys."""
    lines: list[str] = []
    changed = False
    for raw_line in env_file.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw_line)
            continue
        if "=" not in raw_line:
            lines.append(raw_line)
            continue
        raw_key, raw_value = raw_line.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if key != raw_key or value != raw_value:
            changed = True
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
            changed = True
        if key:
            lines.append(f"{key}={value}")
    if not changed:
        return env_file

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="action-grounding-service-",
        suffix=".env",
        delete=False,
    )
    with tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")
    sanitized = Path(tmp.name)
    console.warn(f"Sanitized env file for Docker: {env_file}")
    return sanitized


def init(args: argparse.Namespace) -> int:
    console.title("Action Grounding Service")
    if not docker_available() and not args.dry_run:
        console.error("Docker is not available. Start Docker Desktop and retry.")
        return 1

    console.ok(f"Repo root: {REPO_ROOT}")
    console.ok(f"Config: {args.config}")
    ensure_network(args.network, dry_run=args.dry_run)
    if args.with_omniparser:
        if args.rebuild or not running_container_id(args.omniparser_container):
            build_omniparser_image(args)
            start_omniparser_container(args)
        else:
            console.ok(f"Container `{args.omniparser_container}` is already running")

    if args.rebuild or not running_container_id(args.container):
        build_image(args)
        start_container(args)
    else:
        console.ok(f"Container `{args.container}` is already running")
    if args.dry_run:
        console.ok("Dry run complete")
        return 0

    if args.with_omniparser:
        console.step("Waiting for OmniParser health check")
        omni_health = wait_for_container_health(args.omniparser_container, args.omniparser_timeout)
        console.ok(f"OmniParser healthy: {omni_health}")

    console.step("Waiting for health check")
    health = wait_for_health(args.port, args.timeout)
    console.ok(f"Service healthy: {health}")
    console.ok(f"Endpoint: http://localhost:{args.port}/ground")
    return 0


def status(args: argparse.Namespace) -> int:
    console.title("Action Grounding Service Status")
    cid = running_container_id(args.container)
    if cid:
        console.ok(f"Running container: {cid}")
        try:
            health = wait_for_health(args.port, timeout_secs=3)
            console.ok(f"Health: {health}")
        except Exception as exc:
            console.warn(f"Container is running but health check failed: {exc}")
        try:
            omni_cid = running_container_id(args.omniparser_container)
            if not omni_cid:
                raise RuntimeError(f"Container `{args.omniparser_container}` is not running")
            omni_health = wait_for_container_health(args.omniparser_container, timeout_secs=3)
            console.ok(f"OmniParser health: {omni_health}")
        except Exception as exc:
            console.warn(f"OmniParser health check failed: {exc}")
        return 0
    console.warn(f"Container `{args.container}` is not running")
    return 1


def stop(args: argparse.Namespace) -> int:
    console.title("Stop Action Grounding Service")
    stopped = False
    for name in (args.container, args.omniparser_container, LEGACY_OMNIPARSER_CONTAINER):
        cid = container_id(name)
        if not cid:
            console.warn(f"Container `{name}` does not exist")
            continue
        run_stream(["docker", "rm", "-f", name], cwd=REPO_ROOT, dry_run=args.dry_run)
        stopped = True
    if not stopped:
        return 0
    console.ok("Stopped")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--omniparser-image", default=DEFAULT_OMNIPARSER_IMAGE)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--omniparser-container", default=DEFAULT_OMNIPARSER_CONTAINER)
    parser.add_argument("--omniparser-weights-volume", default=DEFAULT_OMNIPARSER_WEIGHTS_VOLUME)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--omniparser-port", type=int, default=DEFAULT_OMNIPARSER_PORT)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="action_grounding_service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Build and start the Docker service.")
    add_common_args(init_parser)
    init_parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.yaml")
    init_parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    init_parser.add_argument(
        "--with-omniparser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build and start the separate OmniParser container.",
    )
    init_parser.add_argument("--omniparser-timeout", type=float, default=300)
    init_parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Rebuild images before starting containers. By default, existing running containers are reused.",
    )
    init_parser.add_argument("--timeout", type=float, default=120)
    init_parser.set_defaults(func=init)

    status_parser = subparsers.add_parser("status", help="Check service container and health.")
    add_common_args(status_parser)
    status_parser.set_defaults(func=status)

    stop_parser = subparsers.add_parser("stop", help="Stop and remove the service container.")
    add_common_args(stop_parser)
    stop_parser.set_defaults(func=stop)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, TimeoutError, URLError) as exc:
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
