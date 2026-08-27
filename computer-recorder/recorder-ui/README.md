# Recorder UI

The macOS recorder app: an Electron front end over a PyInstaller-packaged
Python backend (`crec-service`) that does the actual capture.

Prebuilt, signed `.dmg` for Apple Silicon:
[`../ComputerRecorder-1.0.1-arm64.dmg`](../ComputerRecorder-1.0.1-arm64.dmg).

## Using the app

1. Drag **ComputerRecorder** to `/Applications` and open it.
2. Grant **Accessibility** and **Screen Recording** when prompted. The app
   checks both on launch and links straight to the right System Settings pane.
3. Record. Pause and resume as needed.
4. On stop, the app consolidates the raw capture into
   `processed_trajectory.jsonl` — this takes a minute or two for a long session
   and is what the induction pipeline consumes.

Sessions are archived to `~/Downloads/recorder_sessions/<session_name>.zip` and
the working directory is removed, so unzip a session before using it. The app's
session list lets you reveal, re-open, or delete them.

## Building from source

Requires Node 18+, Python 3.11, and PyInstaller.

**1. Install dependencies.**

```bash
npm install
pip install -e ..
pip install pyinstaller
```

**2. Set a signing identity** (optional but recommended):

```bash
export CREC_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
```

List what you have with `security find-identity -v -p codesigning`. Signing
matters here beyond the usual reasons: the packaged backend runs as a separate
process, and only a signed helper can *inherit* the Accessibility and Screen
Recording grants from the parent app. Unsigned, macOS treats it as a distinct
program and prompts the user a second time. Both build steps read this one
variable; leave it unset and they skip signing with a warning.

**3. Build the backend, then the app.**

```bash
./build_backend.sh          # -> resources/backend/crec-service/
npm run build:mac           # -> dist/ComputerRecorder-<version>-arm64.dmg
```

`build_backend.sh` bundles `crec` and `backend_lib` with PyInstaller, excluding
the large scientific stack that would otherwise get pulled in. `npm run
build:mac` strips quarantine xattrs from `node_modules/electron`, builds the
renderer, and packages the `.dmg`; `scripts/afterPack.js` re-signs the backend
binaries inside the bundle afterwards.

For UI work without repackaging:

```bash
npm run dev
```

Dev mode expects `resources/backend/crec-service/` to already exist, so run
`./build_backend.sh` at least once first.

## Layout

```
src/main/index.ts            Electron main: process lifecycle, IPC, permissions
src/preload/index.ts         context bridge
src/renderer/src/
  App.tsx                    permissions gate -> recording control
  components/
    PermissionsPage.tsx      Accessibility + Screen Recording checks
    RecordingControl.tsx     record/pause/stop, session browser, progress
launcher.py                  backend entrypoint, speaks JSON lines over stdout
backend_lib/
  parse_raw_trace.py         raw capture -> processed_trajectory.jsonl
  trace_utils.py, language.py
resources/
  entitlements.mac.plist     app entitlements
  entitlements.inherit.plist helper entitlements (TCC inheritance)
  hooks/                     PyInstaller hooks
scripts/afterPack.js         electron-builder hook: re-sign bundled backend
```

The renderer talks to the main process over IPC only; the main process owns the
`crec-service` child and forwards its JSON status lines to the UI.
