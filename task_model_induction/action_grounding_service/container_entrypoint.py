from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress


def _start(cmd: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(cmd)


def main() -> int:
    omni_host = os.getenv("OMNIPARSER_HOST", "127.0.0.1")
    omni_port = os.getenv("OMNIPARSER_PORT", "8080")
    api_host = os.getenv("ACTION_GROUNDING_HOST", "0.0.0.0")
    api_port = os.getenv("ACTION_GROUNDING_PORT", "8000")

    children = [
        _start(
            [
                sys.executable,
                "/app/action_grounding_service/omniparser_server/server.py",
                "--host",
                omni_host,
                "--port",
                omni_port,
            ]
        ),
        _start(
            [
                "uvicorn",
                "action_grounding_service.app.main:app",
                "--host",
                api_host,
                "--port",
                api_port,
            ]
        ),
    ]

    stopping = False

    def stop_children(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    def wait_for_shutdown() -> None:
        deadline = time.monotonic() + 10
        for child in children:
            remaining = max(0.1, deadline - time.monotonic())
            if child.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=remaining)
            if child.poll() is None:
                child.kill()
        for child in children:
            if child.poll() is None:
                child.wait()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    while True:
        for child in children:
            code = child.poll()
            if code is not None:
                if not stopping:
                    stop_children(signal.SIGTERM, None)
                wait_for_shutdown()
                return code
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
