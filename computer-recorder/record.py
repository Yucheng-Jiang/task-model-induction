#!/usr/bin/env python3
"""Record a session from the command line, without installing the .dmg.

This drives the exact same backend the ComputerRecorder app drives:
``recorder-ui/launcher.py``, which the app ships as the ``crec-service``
binary. The app spawns it with ``--session-name`` and ``--base-path`` and
nothing else, so running this script with no flags produces a session that is
structurally identical to one recorded through the UI.

Only tested on Apple Silicon macOS.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.util
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

# launcher.py and the consolidation code use PEP 604 unions in type aliases,
# which are evaluated eagerly and so cannot be deferred with __future__.
MIN_PYTHON = (3, 10)

# --- Import paths -----------------------------------------------------------
# launcher.py lives under recorder-ui/ and imports `crec` (from here) plus
# `parse_raw_trace` (from recorder-ui/backend_lib/). It sets up the latter two
# itself, but only once it is importable, so put its directory on the path.
HERE = os.path.dirname(os.path.abspath(__file__))
RECORDER_UI = os.path.join(HERE, "recorder-ui")

for _path in (HERE, RECORDER_UI):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_BASE_PATH = os.path.expanduser("~/Downloads/recorder_sessions")

# --- Permissions ------------------------------------------------------------
# The app gates on these three before it will record. Screen Recording and
# Accessibility are checked by Electron's systemPreferences; Input Monitoring
# is checked by the backend itself via CGPreflightListenEventAccess. All three
# have a C entry point we can call directly from Python.
SETTINGS_ROOT = "x-apple.systempreferences:com.apple.preference.security"


def _screen_recording_pane_name() -> str:
    """macOS 15 renamed this pane; show whichever the user will actually see."""
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return "Screen Recording"
    return "Screen & System Audio Recording" if major >= 15 else "Screen Recording"


PERMISSIONS = (
    {
        "key": "screen_recording",
        "label": "Screen Recording",
        "why": "capture the before/after screenshot for each action",
        "preflight": "CGPreflightScreenCaptureAccess",
        # Argless and does prompt, so we can raise the dialog ourselves.
        "request": "CGRequestScreenCaptureAccess",
        "pane": "Privacy_ScreenCapture",
        "settings_name": _screen_recording_pane_name,
    },
    {
        "key": "accessibility",
        "label": "Accessibility",
        "why": "observe window events; also required for input capture",
        "preflight": "AXIsProcessTrusted",
        # AXIsProcessTrustedWithOptions can raise a prompt but never grants, and
        # it needs a CFDictionary. Sending people to the pane is the honest path.
        "request": None,
        "pane": "Privacy_Accessibility",
        "settings_name": lambda: "Accessibility",
    },
    {
        "key": "input_monitoring",
        "label": "Input Monitoring",
        "why": "receive the mouse and keyboard events that become actions",
        "preflight": "CGPreflightListenEventAccess",
        "request": "CGRequestListenEventAccess",
        "pane": "Privacy_ListenEvent",
        "settings_name": lambda: "Input Monitoring",
    },
)


def _application_services():
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        raise OSError("ApplicationServices framework not found — is this macOS?")
    return ctypes.cdll.LoadLibrary(path)


def _call_bool(symbol: str) -> bool | None:
    """Call an argless C function returning a bool. None if unavailable."""
    try:
        lib = _application_services()
    except OSError:
        return None
    if not hasattr(lib, symbol):
        return None
    fn = getattr(lib, symbol)
    fn.restype = ctypes.c_bool
    fn.argtypes = []
    try:
        return bool(fn())
    except Exception:
        return None


def check_permissions() -> dict[str, bool | None]:
    return {spec["key"]: _call_bool(spec["preflight"]) for spec in PERMISSIONS}


def responsible_app() -> str:
    """Best guess at the app the TCC grant will attach to."""
    term = os.environ.get("TERM_PROGRAM", "")
    return {
        "Apple_Terminal": "Terminal",
        "iTerm.app": "iTerm",
        "vscode": "Visual Studio Code (or your editor)",
        "WarpTerminal": "Warp",
        "ghostty": "Ghostty",
        "WezTerm": "WezTerm",
        "Hyper": "Hyper",
        "kitty": "kitty",
        "Alacritty": "Alacritty",
    }.get(term, "your terminal app")


def multiplexer_note() -> str | None:
    """tmux and screen own the grant, so quitting the terminal achieves nothing."""
    if os.environ.get("TMUX"):
        return (
            "You are inside tmux, so macOS attributes the grant to the tmux "
            "server rather than\nto the terminal. Quitting the terminal will not "
            "help: run `tmux kill-server`\nafter granting, then start a fresh "
            "session."
        )
    if os.environ.get("STY"):
        return (
            "You are inside screen, so macOS attributes the grant to the screen "
            "server rather\nthan to the terminal. Quit screen entirely after "
            "granting, then start a new one."
        )
    return None


def open_settings_pane(pane: str) -> None:
    subprocess.run(["open", f"{SETTINGS_ROOT}?{pane}"], check=False)


# --- Console ----------------------------------------------------------------


class Console:
    """Line output, with an in-place status line when attached to a TTY.

    The timer thread and the asyncio loop both write here, so every method
    takes the lock: an unsynchronized repaint landing between clear_live() and
    print() would leave a stale status line stranded above the output.
    """

    def __init__(self, stream=None, use_color: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.color = self.tty if use_color is None else use_color
        self._live = ""
        self._lock = threading.RLock()

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, text: str) -> str:
        return self._paint(text, "2")

    def bold(self, text: str) -> str:
        return self._paint(text, "1")

    def green(self, text: str) -> str:
        return self._paint(text, "32")

    def red(self, text: str) -> str:
        return self._paint(text, "31")

    def yellow(self, text: str) -> str:
        return self._paint(text, "33")

    def _clear_live_locked(self) -> None:
        if self._live:
            self.stream.write("\r" + " " * _visible_width(self._live) + "\r")
            self.stream.flush()
            self._live = ""

    def clear_live(self) -> None:
        with self._lock:
            self._clear_live_locked()

    def line(self, text: str = "") -> None:
        with self._lock:
            self._clear_live_locked()
            self.stream.write(text + "\n")
            self.stream.flush()

    def status(self, text: str) -> None:
        """Overwrite the current status line. No-op when not a TTY."""
        if not self.tty:
            return
        with self._lock:
            padding = max(0, _visible_width(self._live) - _visible_width(text))
            self.stream.write("\r" + text + " " * padding)
            self.stream.flush()
            self._live = text


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_width(text: str) -> int:
    """Printable width, ignoring the SGR escapes _paint adds."""
    return len(_ANSI_RE.sub("", text))


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --- Session naming ---------------------------------------------------------
# Mirrors createDefaultSessionName / resolveUniqueSessionName / stripZipSuffix
# in recorder-ui/src/main/index.ts so CLI and app sessions are named alike.


def default_session_name(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("session_%Y%m%d_%H%M%S")


def unique_session_name(base_path: str, requested: str) -> str:
    name = requested.strip()
    candidate = name
    suffix = 2
    while os.path.exists(os.path.join(base_path, candidate)) or os.path.exists(
        os.path.join(base_path, f"{candidate}.zip")
    ):
        candidate = f"{name} ({suffix})"
        suffix += 1
    return candidate


def strip_zip_suffix(name: str) -> str:
    return re.sub(r"\.zip$", "", name, flags=re.IGNORECASE)


def validate_session_name(name: str, base_path: str) -> str:
    """Reject any name that would resolve outside base_path.

    A basename check alone is not enough, and the gap is dangerous rather than
    cosmetic: os.path.basename("..") is "..", and a session named ".." ends
    with launcher's zip_session calling shutil.rmtree on the *parent* of the
    sessions directory. This mirrors resolveSessionDirectory in
    src/main/index.ts, which pairs the basename test with a containment check.
    """
    if not name or name in (os.curdir, os.pardir):
        raise SystemExit(f"Invalid session name: {name!r}")
    if os.path.basename(name) != name:
        raise SystemExit(f"Session name must not contain a path separator: {name!r}")
    resolved = os.path.realpath(os.path.join(base_path, name))
    if os.path.dirname(resolved) != os.path.realpath(base_path):
        raise SystemExit(f"Session name would escape {base_path}: {name!r}")
    return name


def resolve_resume_target(base_path: str, requested: str) -> str:
    """Return the session directory to resume into, unpacking its zip if needed."""
    name = validate_session_name(strip_zip_suffix(requested.strip()), base_path)

    session_dir = os.path.join(base_path, name)
    archive = f"{session_dir}.zip"

    if not os.path.isdir(session_dir):
        if not os.path.exists(archive):
            raise SystemExit(
                f'No session named "{name}" under {base_path}.\n'
                f"Available: {', '.join(list_sessions(base_path)) or '(none)'}"
            )
        # ditto is what the app uses to unpack a finished session.
        result = subprocess.run(
            ["ditto", "-x", "-k", archive, base_path],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.isdir(session_dir):
            raise SystemExit(f"Could not unpack {archive}")
        os.remove(archive)

    return name


def list_sessions(base_path: str) -> list[str]:
    if not os.path.isdir(base_path):
        return []
    names = set()
    for entry in os.listdir(base_path):
        if entry.startswith("."):
            continue
        full = os.path.join(base_path, entry)
        if os.path.isdir(full):
            names.add(entry)
        elif entry.endswith(".zip"):
            names.add(entry[: -len(".zip")])
    return sorted(names)


# --- Permission reporting ---------------------------------------------------


def print_permission_report(console: Console, statuses: dict[str, bool | None]) -> bool:
    console.line(console.bold("macOS permissions"))
    console.line()

    host = responsible_app()
    all_granted = True

    for spec in PERMISSIONS:
        granted = statuses[spec["key"]]
        if granted:
            mark = console.green("granted")
        elif granted is None:
            mark = console.yellow("unknown")
            all_granted = False
        else:
            mark = console.red("missing")
            all_granted = False
        console.line(f"  {spec['label']:<18} {mark}")
        console.line(console.dim(f"  {'':<18} {spec['why']}"))

    console.line()

    if all_granted:
        console.line(console.green("All three granted. You're ready to record."))
        return True

    console.line(
        console.bold(f"Grant these to {host}, not to python")
        + " — macOS attaches the grant"
    )
    console.line("to the application that owns the shell.")
    console.line()
    for spec in PERMISSIONS:
        if statuses[spec["key"]]:
            continue
        console.line(
            f"  System Settings > Privacy & Security > {spec['settings_name']()}"
        )
    console.line()
    console.line(
        console.bold("record.py --setup-permissions")
        + f" opens those panes. Add {host} with"
    )
    console.line("the + button if it isn't listed.")
    console.line()

    note = multiplexer_note()
    console.line(
        console.yellow(note)
        if note
        else console.yellow(
            f"Quit {host} completely (Cmd-Q) and reopen it afterwards — "
            "macOS caches\nthe old answer until the application restarts."
        )
    )
    console.line()
    console.line(
        console.dim(
            f"This grants screen and input access to everything you run from "
            f"{host},\nnot just this recorder."
        )
    )
    return False


def setup_permissions(console: Console) -> int:
    statuses = check_permissions()
    missing = [spec for spec in PERMISSIONS if not statuses[spec["key"]]]

    if not missing:
        console.line(console.green("All three permissions are already granted."))
        return 0

    host = responsible_app()
    console.line(f"Requesting the missing permissions for {console.bold(host)}.")
    console.line()

    for spec in missing:
        if spec["request"]:
            console.line(f"  Prompting for {spec['label']}...")
            _call_bool(spec["request"])
        console.line(f"  Opening Settings > {spec['settings_name']()}")
        open_settings_pane(spec["pane"])
        time.sleep(0.6)

    console.line()
    console.line(f"Switch {console.bold(host)} on in each list.")
    note = multiplexer_note()
    if note:
        console.line(console.yellow(note))
    else:
        console.line(console.yellow(f"Then quit {host} completely (Cmd-Q) and reopen."))
    console.line(
        "Re-check with " + console.bold("record.py --check-permissions") + "."
    )
    return 1


# --- Recording --------------------------------------------------------------


class RecordingUI:
    """Renders launcher.py's event stream as human-readable output."""

    def __init__(
        self,
        console: Console,
        json_console: Console | None = None,
        json_mode: bool = False,
    ) -> None:
        self.console = console
        # In --json mode the event stream owns stdout while prose goes to
        # stderr, so the two are not the same Console.
        self.json_console = json_console or console
        self.json_mode = json_mode
        self.session_path: str | None = None
        self.final_path: str | None = None
        self.failed = False
        self.paused = False
        self.consolidating = False
        self.started = threading.Event()
        self._lock = threading.RLock()
        self._base_offset = 0.0
        self._started_at: float | None = None
        self._paused_at: float | None = None
        self._stop_timer = threading.Event()
        self._timer: threading.Thread | None = None

    # -- control, gated on state so we never latch launcher's pause/resume
    # events. launcher sets them unconditionally and only clears the one it is
    # waiting on, so a redundant SIGUSR2 while recording would silently cancel
    # the *next* pause.
    def request_pause(self) -> str | None:
        with self._lock:
            if self.paused:
                return "already paused"
            self.paused = True
        os.kill(os.getpid(), signal.SIGUSR1)
        return None

    def request_resume(self) -> str | None:
        with self._lock:
            if not self.paused:
                return "already recording"
            self.paused = False
        os.kill(os.getpid(), signal.SIGUSR2)
        return None

    # -- timer --
    def _elapsed(self) -> float:
        with self._lock:
            if self._started_at is None:
                return self._base_offset
            end = self._paused_at if self._paused_at is not None else time.time()
            return self._base_offset + (end - self._started_at)

    def _run_timer(self) -> None:
        while not self._stop_timer.wait(0.5):
            state = "PAUSED " if self.paused else "recording"
            self.console.status(
                f"  {state} {format_duration(self._elapsed())}  "
                + self.console.dim("p / r / f then Return")
            )

    def _start_timer(self) -> None:
        if self._timer or not self.console.tty or self.json_mode:
            return
        self._timer = threading.Thread(target=self._run_timer, daemon=True)
        self._timer.start()

    def _end_timer(self) -> None:
        self._stop_timer.set()
        timer, self._timer = self._timer, None
        if timer:
            timer.join(timeout=1.0)
        self.console.clear_live()

    # -- event sink --
    def handle(self, payload: dict) -> None:
        if self.json_mode:
            self.json_console.line(json.dumps(payload))

        kind = payload.get("type")
        message = payload.get("message", "")

        if kind == "started":
            self.session_path = payload.get("path")
            with self._lock:
                self._base_offset = float(payload.get("resumedDurationSeconds") or 0)
                self._started_at = time.time()
            if not self.json_mode:
                self.console.line(
                    self.console.green("* ") + f"Recording to {self.session_path}"
                )
                if self._base_offset:
                    self.console.line(
                        self.console.dim(
                            f"  resuming at {format_duration(self._base_offset)}"
                        )
                    )
                self.console.line()
            self._start_timer()
            self.started.set()

        elif kind == "status":
            if message == "Recording paused.":
                with self._lock:
                    self.paused = True
                    self._paused_at = time.time()
            elif message == "Recording resumed.":
                with self._lock:
                    if self._paused_at is not None:
                        self._started_at = (self._started_at or 0) + (
                            time.time() - self._paused_at
                        )
                        self._paused_at = None
                    self.paused = False
            elif message.startswith("Signal received") and self.consolidating:
                if not self.json_mode:
                    self.console.line(
                        self.console.yellow(
                            "  Already finishing — consolidation cannot be "
                            "interrupted.\n  Your capture is safe on disk; give "
                            "it a moment."
                        )
                    )
                return
            if not self.json_mode and message:
                self.console.line(self.console.dim(f"  {message}"))

        elif kind == "consolidation-progress":
            if not self.consolidating:
                self.consolidating = True
                self._end_timer()
            if not self.json_mode:
                progress = payload.get("progress")
                bar = ""
                if isinstance(progress, (int, float)):
                    filled = int(round(progress * 24))
                    bar = "  [" + "#" * filled + "." * (24 - filled) + "]"
                self.console.status(f"  {message}{bar}")

        elif kind == "error":
            self.failed = True
            self._end_timer()
            if not self.json_mode:
                self.console.line(self.console.red(f"  error: {message}"))

        elif kind == "finished":
            self._end_timer()
            self.final_path = payload.get("path")


def start_key_reader(ui: RecordingUI, console: Console) -> None:
    """Translate typed commands into the same signals the app sends.

    stdin is left in cooked mode, so commands are line-buffered and each needs
    a Return. That keeps this working over a pipe and under nohup, at the cost
    of single-keystroke control.
    """

    def note(text: str) -> None:
        if ui.json_mode:
            return
        console.line(console.dim(f"  {text}"))

    def reader() -> None:
        ui.started.wait()
        for raw in sys.stdin:
            command = raw.strip().lower()
            if not command:
                continue
            if command in ("p", "pause"):
                refusal = ui.request_pause()
                if refusal:
                    note(refusal)
            elif command in ("r", "resume"):
                refusal = ui.request_resume()
                if refusal:
                    note(refusal)
            elif command in ("f", "finish", "q", "quit", "stop"):
                note("finishing...")
                os.kill(os.getpid(), signal.SIGTERM)
                return
            else:
                note(
                    f"unknown command {command!r} — "
                    "p = pause, r = resume, f = finish"
                )

    threading.Thread(target=reader, daemon=True).start()


def preflight_capture_engine(console: Console) -> str | None:
    """Import the capture engine early so a missing dep is a sentence, not a traceback.

    launcher.py imports only stdlib at module level and reaches crec inside the
    coroutine, so without this the failure surfaces as a bare ModuleNotFoundError
    from deep inside asyncio, after we have already claimed to be recording.
    """
    try:
        import crec.observers  # noqa: F401
    except ImportError as exc:
        console.line(console.red(f"The capture engine is not installed: {exc}"))
        console.line()
        console.line("Install it with:")
        console.line(console.bold(f"  pip install -e {HERE}"))
        return str(exc)
    return None


def run_record(args: argparse.Namespace, console: Console, out: Console) -> int:
    if preflight_capture_engine(console):
        return 1

    import launcher

    base_path = os.path.abspath(os.path.expanduser(args.base_path))
    os.makedirs(base_path, exist_ok=True)

    if args.resume:
        session_name = resolve_resume_target(base_path, args.resume)
    else:
        # index.ts strips a .zip suffix before the resume/new fork, so it
        # applies to new sessions too.
        requested = strip_zip_suffix((args.session_name or default_session_name()).strip())
        session_name = unique_session_name(
            base_path, validate_session_name(requested, base_path)
        )

    ui = RecordingUI(console, json_console=out, json_mode=args.json)
    launcher.set_event_sink(ui.handle)

    # Same configuration launcher.setup_logging applies, but on stderr: the app
    # owns stdout, so backend log lines must not land in our prose or --json.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    if not args.json:
        console.line(console.bold(f"Session: {session_name}"))
        if args.record_video:
            console.line(console.yellow("  raw video capture is on"))
        console.line()

    if args.interactive and sys.stdin.isatty():
        start_key_reader(ui, console)
    elif not args.json:
        console.line(
            console.dim(
                "  not a tty — control with: "
                f"kill -USR1 {os.getpid()} (pause), -USR2 (resume), -TERM (finish)"
            )
        )

    try:
        asyncio.run(
            launcher.run_recording(
                session_name,
                base_path,
                args.debug,
                args.scroll_debounce,
                args.scroll_min_distance,
                args.scroll_max_frequency,
                args.scroll_session_timeout,
                args.record_video,
            )
        )
    except KeyboardInterrupt:
        pass

    console.clear_live()

    if ui.failed and not ui.final_path:
        return 1

    if not args.json:
        console.line()
        if ui.final_path:
            console.line(console.green("* ") + f"Saved {ui.final_path}")
        elif ui.session_path:
            console.line(console.green("* ") + f"Saved {ui.session_path}")
    return 0


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record.py",
        description=(
            "Record a macOS session from the command line. Drives the same "
            "backend as the ComputerRecorder app, so sessions are identical."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "While recording, type a command and press Return:\n"
            "  p = pause    r = resume    f = finish   (Ctrl-C also finishes)\n"
            "Scripted:  kill -USR1 <pid> / -USR2 <pid> / -TERM <pid>\n"
        ),
    )

    parser.add_argument(
        "--session-name",
        help="Name for this session (default: session_<timestamp>). "
        "A numbered suffix is added if the name is taken.",
    )
    parser.add_argument(
        "--resume",
        metavar="NAME",
        help="Continue recording into an existing session, unpacking its zip first.",
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=f"Where sessions are written (default: {DEFAULT_BASE_PATH})",
    )
    parser.add_argument(
        "--check-permissions",
        action="store_true",
        help="Report the three macOS permissions and exit.",
    )
    parser.add_argument(
        "--setup-permissions",
        action="store_true",
        help="Trigger the macOS prompts and open the right Settings panes, then exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sessions",
        help="List recorded sessions and exit.",
    )
    parser.add_argument(
        "--skip-permission-check",
        action="store_true",
        help="Record even if a permission looks missing.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_false",
        dest="interactive",
        help="Do not read commands from stdin; use signals only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the backend's raw JSON-lines event stream instead of prose.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose backend logging.")

    advanced = parser.add_argument_group(
        "advanced",
        "The app never sets these. Defaults match what the app uses — "
        "changing them means your session is no longer app-identical.",
    )
    advanced.add_argument(
        "--record-video",
        action="store_true",
        help="Also capture raw screen video to raw_video.mp4 via macOS "
        "screencapture. Adds a large file the pipeline does not read.",
    )
    advanced.add_argument(
        "--scroll-debounce",
        type=float,
        default=0.5,
        help="Minimum seconds between logged scroll events (default: %(default)s).",
    )
    advanced.add_argument(
        "--scroll-min-distance",
        type=float,
        default=5.0,
        help="Minimum scroll distance in pixels to log (default: %(default)s).",
    )
    advanced.add_argument(
        "--scroll-max-frequency",
        type=int,
        default=10,
        help="Maximum scroll events logged per second (default: %(default)s).",
    )
    advanced.add_argument(
        "--scroll-session-timeout",
        type=float,
        default=2.0,
        help="Seconds of stillness that ends a scroll session (default: %(default)s).",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    # --json owns stdout, so prose has to go somewhere else.
    console = Console(sys.stderr if args.json else sys.stdout)
    out = Console(sys.stdout)

    if sys.platform != "darwin":
        console.line(console.red("The recorder only runs on macOS."))
        return 1

    if sys.version_info < MIN_PYTHON:
        want = ".".join(str(part) for part in MIN_PYTHON)
        console.line(
            console.red(
                f"record.py needs Python {want} or newer; this is "
                f"{platform.python_version()}."
            )
        )
        console.line(console.dim(f"  interpreter: {sys.executable}"))
        console.line()
        console.line(
            "macOS ships 3.9 as /usr/bin/python3. Install a newer one "
            "(e.g. `brew install python@3.11`)"
        )
        console.line("and run it explicitly: `python3.11 record.py`.")
        return 1

    if args.list_sessions:
        base_path = os.path.abspath(os.path.expanduser(args.base_path))
        names = list_sessions(base_path)
        if not names:
            out.line(f"No sessions under {base_path}")
        else:
            out.line(out.bold(f"Sessions in {base_path}"))
            for name in names:
                out.line(f"  {name}")
        return 0

    if args.setup_permissions:
        return setup_permissions(console)

    if args.check_permissions:
        return 0 if print_permission_report(console, check_permissions()) else 1

    if not args.skip_permission_check:
        statuses = check_permissions()
        if not all(statuses.values()):
            print_permission_report(console, statuses)
            console.line()
            console.line(
                console.dim(
                    "Recording anyway would produce blank screenshots and an "
                    "empty event\nstream. Pass --skip-permission-check to "
                    "override this gate."
                )
            )
            if args.json:
                out.line(
                    json.dumps(
                        {
                            "type": "error",
                            "reason": "permissions",
                            "missing": [
                                spec["key"]
                                for spec in PERMISSIONS
                                if not statuses[spec["key"]]
                            ],
                        }
                    )
                )
            return 1

    return run_record(args, console, out)


if __name__ == "__main__":
    sys.exit(main())
