# Computer Recorder

Captures a macOS work session — mouse, keyboard, and screenshots — into a
session directory the [induction pipeline](../task_model_induction/) can read.

> [!WARNING]
> **macOS only, and only tested on Apple Silicon.** The capture engine is built
> on macOS screen and input APIs. Linux and Windows are not supported today;
> contributions welcome.

Run it as an app or from the command line. Both drive the same capture backend
and produce the same session.

## Option 1 — the app

A prebuilt, signed `.dmg` ships in this directory:
[`ComputerRecorder-1.0.1-arm64.dmg`](ComputerRecorder-1.0.1-arm64.dmg). For a
screenshot-by-screenshot tour, see the **[Walkthrough](WALKTHROUGH.md)**.

1. Open the `.dmg`, drag **ComputerRecorder** to `/Applications`, and launch it.
   It is signed but not notarized, so a downloaded copy needs a right-click →
   **Open** the first time.
2. Grant **Screen Recording**, **Accessibility**, and **Input Monitoring** when
   prompted. The app checks all three on launch and links to the right pane.
3. Record, pausing and resuming as needed. Finishing consolidates the capture —
   a minute or two on a long session.

### Building the app yourself

```bash
cd recorder-ui
npm install && pip install -e .. && pip install pyinstaller
./build_backend.sh          # -> resources/backend/crec-service/
npm run build:mac           # -> dist/ComputerRecorder-<version>-arm64.dmg
```

See [`recorder-ui/README.md`](recorder-ui/README.md) for the full instructions,
including why `CREC_SIGNING_IDENTITY` matters for permission inheritance.

## Option 2 — the command line

```bash
pip install -e .                          # Python 3.10+
python3 record.py --check-permissions
python3 record.py
```

### Permissions

Grant **Screen Recording**, **Accessibility**, and **Input Monitoring** to your
terminal — Terminal, iTerm, VS Code — not to `python`. macOS attaches the grant
to the application that owns the shell.

`python3 record.py --setup-permissions` opens the three panes. Quit the terminal
completely (Cmd-Q) and reopen it afterwards; macOS caches the old answer until
the application restarts.

> [!CAUTION]
> This grants screen and input access to **everything you run from that
> terminal**, not just this recorder. The `.app` is narrower, which is the main
> reason to prefer Option 1.

### Recording

Type `p`, `r`, or `f` and press Return to pause, resume, or finish. Ctrl-C also
finishes.

```bash
python3 record.py --session-name expense_report_run1
python3 record.py --resume expense_report_run1
python3 record.py --list
```

`--help` covers the rest. The flags grouped under **advanced** already default
to the app's values; changing one takes the session out of parity.

For scripting, `--no-interactive --json` emits the backend's event stream as
JSON lines on stdout, and `kill -USR1` / `-USR2` / `-TERM` pause, resume, and
finish.

## What a session contains

Either path writes to `~/Downloads/recorder_sessions/<session_name>.zip`:

```
<session_name>/
  processed_trajectory.jsonl    one action per line, with before/after state
  processed_trajectory.json     the same actions plus OCR/VLM enrichment
  raw_trace.jsonl               the raw event stream
  screenshots/                  before/after JPEG per action + bounding_boxes.jsonl
  annotated_screenshots/        same frames with the click position marked
```

`processed_trajectory.jsonl` is the file the pipeline reads. Sessions never
leave your machine, and are exactly as sensitive as the work they captured —
review one before you share it.

## Source code

- [`crec/`](crec/) — the capture engine (input hooks, screen capture).
- [`recorder-ui/`](recorder-ui/) — the Electron app, plus `launcher.py` and the
  consolidation code in `backend_lib/`.
- [`record.py`](record.py) — the command-line front end.
