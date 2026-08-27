import {
  app,
  shell,
  BrowserWindow,
  ipcMain,
  systemPreferences,
  WebContents,
} from "electron";
import { join } from "path";
import { electronApp, optimizer, is } from "@electron-toolkit/utils";
// import icon from '../../resources/icon.png?asset'

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    // ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      sandbox: false,
    },
  });

  mainWindow.on("ready-to-show", () => {
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url);
    return { action: "deny" };
  });

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env["ELECTRON_RENDERER_URL"]) {
    mainWindow.loadURL(process.env["ELECTRON_RENDERER_URL"]);
  } else {
    mainWindow.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId("com.electron");

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on("browser-window-created", (_, window) => {
    optimizer.watchWindowShortcuts(window);
  });

  createWindow();

  app.on("activate", function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// In this file you can include the rest of your app"s specific main process
// code. You can also put them in separate files and require them here.

import { spawn, spawnSync, ChildProcess } from "child_process";
import fs from "fs";
import path from "path";

type RecorderState = "idle" | "recording" | "paused";
type UiSettings = {
  developerView?: boolean;
};
type RecordedSessionSummary = {
  name: string;
  path: string;
  actionCount: number | null;
  startedAt: number | null;
  updatedAt: number;
  hasProcessedTrajectory: boolean;
  hasRawVideo: boolean;
};
type StartRecordingArgs = {
  sessionName?: string;
  resumeExisting?: boolean;
};

let recorderProcess: ChildProcess | null = null;
let recorderState: RecorderState = "idle";
let recorderSessionName: string | null = null;
let recorderBasePath: string | null = null;
let recorderSender: WebContents | null = null;
let recorderDidSignalFinished = false;

const DEFAULT_UI_SETTINGS: Required<UiSettings> = {
  developerView: false,
};

const getBackendPath = () => {
  // In production, resources are running inside the app bundle or next to it
  if (app.isPackaged) {
    return path.join(
      process.resourcesPath,
      "backend/crec-service/crec-service",
    );
  }
  // In dev, we use the build output
  return path.join(
    __dirname,
    "../../resources/backend/crec-service/crec-service",
  );
};

const getSessionPath = () => {
  if (!recorderBasePath || !recorderSessionName) {
    return null;
  }

  return path.join(recorderBasePath, recorderSessionName);
};

const getSessionsBasePath = () =>
  path.join(app.getPath("downloads"), "recorder_sessions");

const resolveExistingPath = (targetPath: string) => {
  if (!targetPath) {
    return null;
  }

  const resolvedPath = path.resolve(String(targetPath));
  return fs.existsSync(resolvedPath) ? resolvedPath : null;
};

const stripZipSuffix = (name: string) => name.replace(/\.zip$/i, "");

const getSessionArchivePath = (sessionPath: string) => `${sessionPath}.zip`;

const extractSessionArchive = (archivePath: string, basePath: string) => {
  // ditto preserves the archived directory structure on macOS without extra deps.
  const result = spawnSync("ditto", ["-x", "-k", archivePath, basePath]);
  return result.status === 0;
};

const resolveSessionDirectory = (sessionName: string) => {
  const trimmedName = stripZipSuffix(String(sessionName || "").trim());
  if (!trimmedName || path.basename(trimmedName) !== trimmedName) {
    return null;
  }

  const basePath = path.resolve(getSessionsBasePath());
  const targetPath = path.resolve(basePath, trimmedName);
  const relativePath = path.relative(basePath, targetPath);

  if (
    !relativePath ||
    relativePath.startsWith("..") ||
    path.isAbsolute(relativePath)
  ) {
    return null;
  }

  return targetPath;
};

const getSessionTimestamp = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return value;
  }

  if (typeof value === "string") {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      return numeric;
    }
  }

  if (!value || typeof value !== "object") {
    return null;
  }

  const timeRecord = value as { before?: unknown; after?: unknown };
  const before = getSessionTimestamp(timeRecord.before);
  if (before !== null) {
    return before;
  }
  return getSessionTimestamp(timeRecord.after);
};

const summarizeTrajectoryJsonl = (filePath: string) => {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split(/\r?\n/);

  let actionCount = 0;
  let startedAt: number | null = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    let row: Record<string, unknown> | null = null;
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        row = parsed as Record<string, unknown>;
      }
    } catch {
      continue;
    }
    if (!row) {
      continue;
    }

    // Rows the recorder writes today carry no discriminator: they are plain
    // action objects (id/action/state_before/...). Older traces tagged each row
    // with row_type/node_type, so keep accepting those too.
    const isTaggedActionRow =
      row.row_type === "node" && row.node_type === "action";
    const isPlainActionRow =
      row.row_type === undefined &&
      row.node_type === undefined &&
      typeof row.id === "string" &&
      "action" in row;
    if (!isTaggedActionRow && !isPlainActionRow) {
      continue;
    }

    actionCount += 1;
    if (startedAt !== null) {
      continue;
    }

    const inlineNode =
      row.node && typeof row.node === "object"
        ? (row.node as { time?: unknown })
        : null;
    const inlineTimestamp = inlineNode
      ? getSessionTimestamp(inlineNode.time)
      : null;
    if (inlineTimestamp !== null) {
      startedAt = inlineTimestamp;
      continue;
    }

    startedAt = getSessionTimestamp({
      before: row.time_before,
      after: row.time_after,
    });
  }

  return {
    actionCount,
    startedAt,
    hasProcessedTrajectory: actionCount > 0 || lines.some((line) => line.trim()),
  };
};

const readProcessedTrajectorySummary = (sessionPath: string) => {
  const candidateFiles = [
    "processed_trajectory.jsonl",
    "processed_trajectory_with_goals.jsonl",
    "processed_trajectory.json",
    "processed_trajectory_with_goals.json",
  ];

  for (const fileName of candidateFiles) {
    const filePath = path.join(sessionPath, fileName);
    if (!fs.existsSync(filePath)) {
      continue;
    }

    try {
      if (fileName.endsWith(".jsonl")) {
        const summary = summarizeTrajectoryJsonl(filePath);
        return {
          actionCount: summary.actionCount,
          startedAt: summary.startedAt,
          hasProcessedTrajectory: summary.hasProcessedTrajectory,
        };
      }

      const parsed = JSON.parse(fs.readFileSync(filePath, "utf-8")) as
        | { nodes?: unknown[] }
        | unknown[];
      const nodes = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed?.nodes)
          ? parsed.nodes
          : [];
      let startedAt: number | null = null;

      for (const node of nodes) {
        if (!node || typeof node !== "object") {
          continue;
        }
        const timestamp = getSessionTimestamp(
          (node as { time?: unknown }).time,
        );
        if (timestamp !== null) {
          startedAt = timestamp;
          break;
        }
      }

      return {
        actionCount: nodes.length,
        startedAt,
        hasProcessedTrajectory: true,
      };
    } catch (error) {
      console.warn(`Failed to read session metadata from ${filePath}:`, error);
    }
  }

  return {
    actionCount: null,
    startedAt: null,
    hasProcessedTrajectory: false,
  };
};

const buildArchivedSessionSummary = (
  archivePath: string,
): RecordedSessionSummary => {
  const stats = fs.statSync(archivePath);
  const fallbackStartedAt =
    stats.birthtimeMs > 0 ? stats.birthtimeMs / 1000 : stats.ctimeMs / 1000;

  return {
    name: stripZipSuffix(path.basename(archivePath)),
    path: archivePath,
    actionCount: null,
    startedAt: fallbackStartedAt,
    updatedAt: stats.mtimeMs / 1000,
    hasProcessedTrajectory: true,
    hasRawVideo: false,
  };
};

const buildRecordedSessionSummary = (
  sessionPath: string,
): RecordedSessionSummary => {
  const stats = fs.statSync(sessionPath);
  const processedSummary = readProcessedTrajectorySummary(sessionPath);
  const fallbackStartedAt =
    stats.birthtimeMs > 0 ? stats.birthtimeMs / 1000 : stats.ctimeMs / 1000;
  const fileNames = new Set(fs.readdirSync(sessionPath));
  const hasRawVideo =
    fileNames.has("raw_video.mp4") ||
    Array.from(fileNames).some((fileName) =>
      fileName.toLowerCase().endsWith(".mp4"),
    );

  return {
    name: path.basename(sessionPath),
    path: sessionPath,
    actionCount: processedSummary.actionCount,
    startedAt: processedSummary.startedAt ?? fallbackStartedAt,
    updatedAt: stats.mtimeMs / 1000,
    hasProcessedTrajectory: processedSummary.hasProcessedTrajectory,
    hasRawVideo,
  };
};

const createDefaultSessionName = (date = new Date()) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  return `session_${year}${month}${day}_${hours}${minutes}${seconds}`;
};

const resolveUniqueSessionName = (basePath: string, requestedName: string) => {
  const trimmedName = requestedName.trim();
  let candidate = trimmedName;
  let suffix = 2;

  while (
    fs.existsSync(path.join(basePath, candidate)) ||
    fs.existsSync(getSessionArchivePath(path.join(basePath, candidate)))
  ) {
    candidate = `${trimmedName} (${suffix})`;
    suffix += 1;
  }

  return candidate;
};

const sendBackendMessage = (message: Record<string, unknown>) => {
  recorderSender?.send("backend-message", message);
};

const broadcastRecorderState = () => {
  sendBackendMessage({ type: "session-state", state: recorderState });
};

const resetRecorderState = () => {
  recorderProcess = null;
  recorderState = "idle";
  recorderSessionName = null;
  recorderBasePath = null;
  recorderDidSignalFinished = false;
};

const signalRecorderProcess = (signal: NodeJS.Signals) => {
  if (!recorderProcess?.pid) {
    return false;
  }

  try {
    process.kill(recorderProcess.pid, signal);
    return true;
  } catch (error) {
    console.error(`Failed to send ${signal} to recorder process:`, error);
    return false;
  }
};

const finishRecording = () => {
  if (!recorderProcess) {
    return false;
  }

  recorderState = "idle";
  broadcastRecorderState();
  return signalRecorderProcess("SIGINT");
};

ipcMain.on("start-recording", (event, args) => {
  const startArgs =
    args && typeof args === "object" ? (args as StartRecordingArgs) : {};
  const trimmedSessionName = stripZipSuffix(
    String(startArgs.sessionName || "").trim(),
  );
  const resumeExisting = Boolean(startArgs.resumeExisting);
  const backendPath = getBackendPath();
  const uiSettings = getStoredSettings();

  console.log(`Attempting to launch backend: ${backendPath}`);

  if (recorderProcess) {
    console.log("Recorder already running");
    event.reply("backend-message", {
      type: "error",
      message: "A recording session is already running.",
    });
    return;
  }

  try {
    if (!fs.existsSync(backendPath)) {
      event.reply("backend-message", {
        type: "error",
        message: `Backend executable not found at ${backendPath}`,
      });
      return;
    }

    if (!trimmedSessionName) {
      event.reply("backend-message", {
        type: "error",
        message: "Session name cannot be empty.",
      });
      return;
    }

    const basePath = getSessionsBasePath();
    fs.mkdirSync(basePath, { recursive: true });
    let resolvedSessionName = trimmedSessionName;
    if (resumeExisting) {
      const existingSessionPath = resolveSessionDirectory(trimmedSessionName);
      if (existingSessionPath && !fs.existsSync(existingSessionPath)) {
        // Finished sessions are stored as zip archives; unpack before resuming.
        const archivePath = getSessionArchivePath(existingSessionPath);
        if (
          fs.existsSync(archivePath) &&
          extractSessionArchive(archivePath, basePath) &&
          fs.existsSync(existingSessionPath)
        ) {
          fs.rmSync(archivePath, { force: true });
        }
      }
      if (
        !existingSessionPath ||
        !fs.existsSync(existingSessionPath) ||
        !fs.statSync(existingSessionPath).isDirectory()
      ) {
        event.reply("backend-message", {
          type: "error",
          message: `Session "${trimmedSessionName}" could not be resumed.`,
        });
        return;
      }
      resolvedSessionName = path.basename(existingSessionPath);
    } else {
      resolvedSessionName = resolveUniqueSessionName(basePath, trimmedSessionName);
    }

    recorderSender = event.sender;
    recorderSessionName = resolvedSessionName;
    recorderBasePath = basePath;
    recorderDidSignalFinished = false;

    const argsList = [
      "--session-name",
      resolvedSessionName,
      "--base-path",
      basePath,
    ];
    if (uiSettings.developerView) {
      argsList.push("--debug");
    }

    recorderProcess = spawn(backendPath, argsList, {
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    recorderState = "recording";

    event.reply("backend-message", {
      type: "session-name-resolved",
      sessionName: resolvedSessionName,
    });
    event.reply("backend-message", {
      type: "status",
      message: resumeExisting
        ? `Resuming session "${resolvedSessionName}".`
        : "Backend process spawned",
    });

    recorderProcess.stdout?.on("data", (data) => {
      const str = data.toString().trim();
      console.log("STDOUT:", str);
      try {
        // Try parsing JSON lines
        const lines = str.split("\n");
        lines.forEach((line: string) => {
          try {
            const json = JSON.parse(line);
            if (json?.type === "finished") {
              recorderDidSignalFinished = true;
            }
            sendBackendMessage(json);
          } catch {
            // Not JSON, just forward as generic log
            if (line.trim()) {
              sendBackendMessage({ type: "status", message: line });
            }
          }
        });
      } catch (e) {
        sendBackendMessage({ type: "status", message: str });
      }
    });

    recorderProcess.stderr?.on("data", (data) => {
      const str = data.toString().trim();
      console.error("STDERR:", str);
      if (str) {
        sendBackendMessage({ type: "stderr", message: str });
      }
    });

    recorderProcess.on("close", (code, signal) => {
      const sessionPath = getSessionPath();
      console.log(`Backend exited with code ${code} and signal ${signal}`);
      const exitedCleanly =
        code === 0 || signal === "SIGINT" || signal === "SIGTERM";

      if (sessionPath && exitedCleanly && !recorderDidSignalFinished) {
        sendBackendMessage({ type: "finished", path: sessionPath });
      } else if (!exitedCleanly) {
        sendBackendMessage({
          type: "error",
          message: `Recorder exited unexpectedly${code !== null ? ` (code ${code})` : ""}.`,
        });
      }
      resetRecorderState();
      broadcastRecorderState();
    });
  } catch (e) {
    console.error(e);
    resetRecorderState();
    broadcastRecorderState();
    event.reply("backend-message", { type: "error", message: String(e) });
  }
});

ipcMain.on("pause-recording", (event) => {
  if (!recorderProcess || recorderState !== "recording") {
    event.reply("backend-message", {
      type: "error",
      message: "No active recording is available to pause.",
    });
    return;
  }

  recorderSender = event.sender;
  if (!signalRecorderProcess("SIGUSR1")) {
    event.reply("backend-message", {
      type: "error",
      message: "Failed to pause the recording session.",
    });
    return;
  }

  recorderState = "paused";
  broadcastRecorderState();
  sendBackendMessage({ type: "status", message: "Recording paused." });
});

ipcMain.on("resume-recording", (event) => {
  if (!recorderProcess || recorderState !== "paused") {
    event.reply("backend-message", {
      type: "error",
      message: "No paused recording is available to resume.",
    });
    return;
  }

  recorderSender = event.sender;
  if (!signalRecorderProcess("SIGUSR2")) {
    event.reply("backend-message", {
      type: "error",
      message: "Failed to resume the recording session.",
    });
    return;
  }

  recorderState = "recording";
  broadcastRecorderState();
  sendBackendMessage({ type: "status", message: "Recording resumed." });
});

ipcMain.on("stop-recording", () => {
  if (recorderProcess) {
    console.log("Stopping recorder process...");
    finishRecording();
  }
});

app.on("before-quit", () => {
  if (recorderProcess) {
    finishRecording();
  }
});

// Permission persistence logic
const getSettingsPath = () =>
  path.join(app.getPath("userData"), "ui-settings.json");

const getStoredSettings = (): Required<UiSettings> => {
  try {
    const settingsPath = getSettingsPath();
    if (fs.existsSync(settingsPath)) {
      return {
        ...DEFAULT_UI_SETTINGS,
        ...JSON.parse(fs.readFileSync(settingsPath, "utf-8")),
      };
    }
  } catch (e) {
    console.error("Failed to read settings:", e);
  }
  return { ...DEFAULT_UI_SETTINGS };
};

const saveStoredSettings = (settings: UiSettings) => {
  try {
    const settingsPath = getSettingsPath();
    fs.writeFileSync(
      settingsPath,
      JSON.stringify({ ...DEFAULT_UI_SETTINGS, ...settings }, null, 2),
    );
  } catch (e) {
    console.error("Failed to write settings:", e);
  }
};

const checkInputMonitoring = async (): Promise<boolean> => {
  return new Promise((resolve) => {
    const backendPath = getBackendPath();

    // If the backend executable doesn't exist (e.g. not built yet), we can't check
    if (!fs.existsSync(backendPath)) {
      console.warn(
        `Backend not found at ${backendPath}, assuming false for input monitoring`,
      );
      resolve(false);
      return;
    }

    const proc = spawn(backendPath, ["--check-permissions"]);
    let output = "";

    proc.stdout.on("data", (data) => {
      output += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output.trim());
          resolve(!!result.input_monitoring);
        } catch (e) {
          console.error(
            "Failed to parse backend permission check output:",
            output,
          );
          resolve(false);
        }
      } else {
        console.warn(`Backend permission check exited with code ${code}`);
        resolve(false);
      }
    });

    proc.on("error", (err) => {
      console.error("Failed to spawn backend for permission check:", err);
      resolve(false);
    });
  });
};

ipcMain.handle("check-permissions", async () => {
  const screen = systemPreferences.getMediaAccessStatus("screen");
  const ax = systemPreferences.isTrustedAccessibilityClient(false);

  // Real check via backend
  const inputMonitoring = await checkInputMonitoring();

  return {
    screenRecording: screen === "granted",
    accessibility: ax,
    inputMonitoring: inputMonitoring,
  };
});

ipcMain.handle("confirm-input-monitoring", async () => {
  // Just re-check, basically
  const granted = await checkInputMonitoring();
  return granted;
});

ipcMain.handle("reset-settings", async () => {
  saveStoredSettings({ ...DEFAULT_UI_SETTINGS });
  return true;
});

ipcMain.handle("get-ui-settings", async () => {
  return getStoredSettings();
});

ipcMain.handle("save-ui-settings", async (_, nextSettings: UiSettings) => {
  const mergedSettings = {
    ...getStoredSettings(),
    ...nextSettings,
  };
  saveStoredSettings(mergedSettings);
  return mergedSettings;
});

ipcMain.handle("get-default-session-name", async () => {
  const basePath = getSessionsBasePath();
  return resolveUniqueSessionName(basePath, createDefaultSessionName());
});

ipcMain.handle("list-recorded-sessions", async () => {
  const basePath = getSessionsBasePath();
  if (!fs.existsSync(basePath)) {
    return [];
  }

  const entries = fs
    .readdirSync(basePath, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith("."));
  const directorySummaries = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) =>
      buildRecordedSessionSummary(path.join(basePath, entry.name)),
    );
  const directoryNames = new Set(
    directorySummaries.map((summary) => summary.name),
  );
  const archiveSummaries = entries
    .filter(
      (entry) =>
        entry.isFile() &&
        entry.name.toLowerCase().endsWith(".zip") &&
        !directoryNames.has(stripZipSuffix(entry.name)),
    )
    .map((entry) => buildArchivedSessionSummary(path.join(basePath, entry.name)));

  return [...directorySummaries, ...archiveSummaries]
    .sort((left, right) => {
      const leftTimestamp = left.startedAt ?? left.updatedAt;
      const rightTimestamp = right.startedAt ?? right.updatedAt;
      return rightTimestamp - leftTimestamp;
    });
});

ipcMain.handle("delete-recorded-session", async (_, sessionName: string) => {
  const sessionPath = resolveSessionDirectory(sessionName);
  if (!sessionPath) {
    throw new Error("Invalid session name.");
  }

  const archivePath = getSessionArchivePath(sessionPath);
  const hasDirectory =
    fs.existsSync(sessionPath) && fs.statSync(sessionPath).isDirectory();
  const hasArchive = fs.existsSync(archivePath);
  if (!hasDirectory && !hasArchive) {
    throw new Error("Session not found.");
  }

  const activeSessionPath = getSessionPath();
  if (
    activeSessionPath &&
    path.resolve(activeSessionPath) === path.resolve(sessionPath)
  ) {
    throw new Error("Cannot delete the session that is currently recording.");
  }

  if (hasDirectory) {
    await fs.promises.rm(sessionPath, { recursive: true, force: false });
  }
  if (hasArchive) {
    await fs.promises.rm(archivePath, { force: false });
  }
  return { success: true };
});

ipcMain.handle("get-backend-path", async () => {
  return getBackendPath();
});

ipcMain.handle("reveal-backend", async () => {
  const backendPath = getBackendPath();
  await shell.showItemInFolder(backendPath);
  return backendPath;
});

ipcMain.handle("path-exists", async (_, targetPath: string) => {
  return Boolean(resolveExistingPath(targetPath));
});

ipcMain.handle("reveal-path", async (_, targetPath: string) => {
  const resolvedPath = resolveExistingPath(targetPath);
  if (!resolvedPath) {
    return false;
  }

  shell.showItemInFolder(resolvedPath);
  return true;
});

ipcMain.handle("open-settings", async (_, type: string) => {
  let url = "";
  switch (type) {
    case "screen":
      url =
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";
      break;
    case "accessibility":
      url =
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
      break;
    case "input":
      url =
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";
      break;
  }
  if (url) {
    await shell.openExternal(url);
  }
});
