import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  Clock3,
  Disc,
  FolderOpen,
  History,
  Info,
  MousePointerClick,
  Pause,
  Play,
  RefreshCw,
  Settings2,
  Square,
  Terminal,
  Trash2,
  X,
} from "lucide-react";

type SessionPhase =
  | "setup"
  | "starting"
  | "recording"
  | "paused"
  | "finishing"
  | "consolidating"
  | "finished";
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
type PostProcessStageState = "pending" | "active" | "complete";
type PostProcessStage = {
  key:
    | "stop_capture"
    | "build_trajectory"
    | "annotate_screenshots"
    | "finalize_session";
  label: string;
  detail: string;
  progress: number;
  state: PostProcessStageState;
  indeterminate?: boolean;
};

type BackendMessage =
  | { type: "status"; message: string }
  | { type: "started"; path: string; resumedDurationSeconds?: number }
  | { type: "finished"; path: string; processedPath?: string }
  | { type: "error"; message: string }
  | { type: "consolidation-progress"; message: string; progress?: number }
  | { type: "session-state"; state: "idle" | "recording" | "paused" }
  | { type: "stderr"; message: string }
  | { type: "session-name-resolved"; sessionName: string };

const MAX_LOG_LINES = 60;
const BUILD_TRAJECTORY_PROGRESS_END = 0.9;
const ANNOTATE_SCREENSHOTS_PROGRESS_START = 0.9;
const ANNOTATE_SCREENSHOTS_PROGRESS_END = 0.95;
const FINALIZE_SESSION_PROGRESS_START = 0.95;

const createFallbackSessionName = () => {
  const date = new Date();
  return `session_${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, "0")}${String(date.getDate()).padStart(2, "0")}_${String(date.getHours()).padStart(2, "0")}${String(date.getMinutes()).padStart(2, "0")}${String(date.getSeconds()).padStart(2, "0")}`;
};

const clampProgress = (value: number) => Math.min(1, Math.max(0, value));

const getPathTail = (fullPath: string) => {
  const parts = fullPath.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? fullPath;
};

const getProgressWithinRange = (value: number, start: number, end: number) => {
  const span = end - start;
  if (span <= 0) {
    return value >= end ? 1 : 0;
  }
  return clampProgress((value - start) / span);
};

const formatPostProcessMessage = (message: string, fallback: string) => {
  const trimmed = message.trim();
  if (!trimmed) {
    return fallback;
  }

  const trajectoryMatch = trimmed.match(
    /^Consolidating raw data and deriving processed_trajectory\.jsonl? \((\d+)\/(\d+)\)$/,
  );
  if (trajectoryMatch) {
    return `Building processed trajectory ${trajectoryMatch[1]} of ${trajectoryMatch[2]}`;
  }

  const annotationMatch = trimmed.match(
    /^Creating annotated screenshots(?:\.\.\.)?(?: \((\d+)\/(\d+)\))?$/,
  );
  if (annotationMatch) {
    const [, current, total] = annotationMatch;
    if (current && total) {
      return `Creating annotated screenshots ${current} of ${total}`;
    }
    return "Creating annotated screenshots";
  }

  if (
    trimmed === "processed_trajectory.jsonl is ready." ||
    trimmed === "processed_trajectory.json is ready."
  ) {
    return "Writing final session files";
  }
  if (trimmed.startsWith("Loaded ")) {
    return "Loaded recorder data";
  }
  if (trimmed.startsWith("Matched screenshots")) {
    return "Matched screenshots to captured actions";
  }
  if (trimmed.startsWith("Pruned trace")) {
    return "Filtered unusable actions from the trace";
  }
  if (trimmed.startsWith("Derived ")) {
    return "Derived the structured interaction trajectory";
  }
  if (trimmed.startsWith("Skipping optional OCR/VLM enrichment.")) {
    return "Skipping optional screen enrichment";
  }

  return trimmed;
};

const buildPostProcessStages = ({
  phase,
  status,
  consolidationProgress,
}: {
  phase: SessionPhase;
  status: string;
  consolidationProgress: number;
}): PostProcessStage[] => {
  const normalizedProgress = clampProgress(consolidationProgress);
  const hasEnteredConsolidation =
    phase === "consolidating" || phase === "finished";
  const hasFinished = phase === "finished";
  const noAnnotations =
    /No bounding_boxes\.jsonl found\. Skipping annotations\./.test(status);

  const stopCaptureState: PostProcessStageState = hasEnteredConsolidation
    ? "complete"
    : phase === "finishing"
      ? "active"
      : "pending";

  const stages: PostProcessStage[] = [
    {
      key: "stop_capture",
      label: "Stop capture",
      detail:
        stopCaptureState === "active"
          ? "Stopping the recorder and flushing pending input events."
          : "Capture closed cleanly.",
      progress:
        stopCaptureState === "complete"
          ? 1
          : stopCaptureState === "active"
            ? 0.64
            : 0,
      state: stopCaptureState,
      indeterminate: stopCaptureState === "active",
    },
  ];

  const buildTrajectoryComplete =
    hasFinished || normalizedProgress >= BUILD_TRAJECTORY_PROGRESS_END;
  const buildTrajectoryActive =
    phase === "consolidating" && !buildTrajectoryComplete;
  stages.push({
    key: "build_trajectory",
    label: "Build trajectory",
    detail: buildTrajectoryActive
      ? formatPostProcessMessage(status, "Building processed trajectory")
      : "Match screenshots to actions and derive the structured session trace.",
    progress: buildTrajectoryComplete
      ? 1
      : buildTrajectoryActive
        ? getProgressWithinRange(
            normalizedProgress,
            0,
            BUILD_TRAJECTORY_PROGRESS_END,
          )
        : 0,
    state: buildTrajectoryComplete
      ? "complete"
      : buildTrajectoryActive
        ? "active"
        : "pending",
  });

  const annotateComplete =
    hasFinished || normalizedProgress >= ANNOTATE_SCREENSHOTS_PROGRESS_END;
  const annotateActive =
    phase === "consolidating" &&
    normalizedProgress >= ANNOTATE_SCREENSHOTS_PROGRESS_START &&
    !annotateComplete;
  stages.push({
    key: "annotate_screenshots",
    label: "Annotate screenshots",
    detail: annotateActive
      ? formatPostProcessMessage(status, "Creating annotated screenshots")
      : noAnnotations
        ? "No click overlays were generated for this session."
        : "Overlay captured click targets on saved screenshots.",
    progress: annotateComplete
      ? 1
      : annotateActive
        ? getProgressWithinRange(
            normalizedProgress,
            ANNOTATE_SCREENSHOTS_PROGRESS_START,
            ANNOTATE_SCREENSHOTS_PROGRESS_END,
          )
        : 0,
    state: annotateComplete
      ? "complete"
      : annotateActive
        ? "active"
        : "pending",
  });

  const finalizeSessionActive =
    phase === "consolidating" &&
    normalizedProgress >= FINALIZE_SESSION_PROGRESS_START &&
    !hasFinished;
  stages.push({
    key: "finalize_session",
    label: "Finalize session",
    detail: finalizeSessionActive
      ? formatPostProcessMessage(status, "Writing final session files")
      : "Write processed session files and mark the recording complete.",
    progress: hasFinished
      ? 1
      : finalizeSessionActive
        ? getProgressWithinRange(
            normalizedProgress,
            FINALIZE_SESSION_PROGRESS_START,
            1.0,
          )
        : 0,
    state: hasFinished
      ? "complete"
      : finalizeSessionActive
        ? "active"
        : "pending",
  });

  return stages;
};

const getPostProcessProgress = (
  phase: SessionPhase,
  stages: PostProcessStage[],
  consolidationProgress: number,
) => {
  if (phase === "finished") {
    return 1;
  }

  if (phase === "consolidating") {
    return Math.max(0.18, clampProgress(consolidationProgress));
  }

  if (stages.length === 0) {
    return 0;
  }

  const averageProgress =
    stages.reduce((sum, stage) => sum + stage.progress, 0) / stages.length;
  return Math.max(phase === "finishing" ? 0.08 : 0.18, averageProgress);
};

const sessionDateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const formatSessionStartedAt = (timestamp: number | null) => {
  if (timestamp === null || !Number.isFinite(timestamp)) {
    return "Start time unavailable";
  }

  return sessionDateFormatter.format(new Date(timestamp * 1000));
};

const formatActionCount = (count: number | null) => {
  if (count === null) {
    return "Action count unavailable";
  }

  return `${count.toLocaleString()} action${count === 1 ? "" : "s"}`;
};

const ProgressRing: React.FC<{
  progress: number;
  tone: "blue" | "green";
  spinning?: boolean;
  label: string;
  sublabel?: string;
  icon?: React.ReactNode;
}> = ({ progress, tone, spinning = false, label, sublabel, icon }) => {
  const radius = 56;
  const normalizedProgress = clampProgress(progress);
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - normalizedProgress);
  const toneClass = tone === "green" ? "text-green-500" : "text-blue-600";

  return (
    <div className="relative h-40 w-40">
      <svg
        className="h-full w-full -rotate-90"
        viewBox="0 0 140 140"
        fill="none"
      >
        <circle
          cx="70"
          cy="70"
          r={radius}
          stroke="currentColor"
          strokeWidth="10"
          className="text-gray-200"
        />
        {spinning ? (
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="currentColor"
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.32} ${circumference}`}
            className={`${toneClass} animate-spin`}
            style={{ transformOrigin: "50% 50%" }}
          />
        ) : (
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="currentColor"
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className={`${toneClass} transition-all duration-500 ease-out`}
          />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <div className={`flex items-center justify-center ${toneClass}`}>
          {icon ?? (
            <span className="text-3xl font-semibold tracking-tight">
              {label}
            </span>
          )}
        </div>
        {icon ? (
          <div className="mt-2 text-sm font-semibold tracking-tight text-gray-900">
            {label}
          </div>
        ) : null}
        {sublabel ? (
          <div className="mt-1 text-[11px] font-medium tracking-[0.08em] text-gray-500">
            {sublabel}
          </div>
        ) : null}
      </div>
    </div>
  );
};

const RecordingControl: React.FC = () => {
  const [sessionName, setSessionName] = useState(createFallbackSessionName());
  const [sessionNameEdited, setSessionNameEdited] = useState(false);
  const [phase, setPhase] = useState<SessionPhase>("setup");
  const [duration, setDuration] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Ready");
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [savedSessionPath, setSavedSessionPath] = useState<string | null>(null);
  const [savedPathAvailable, setSavedPathAvailable] = useState(true);
  const [consolidationProgress, setConsolidationProgress] = useState<number>(0);
  const [developerView, setDeveloperView] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [recordedSessions, setRecordedSessions] = useState<
    RecordedSessionSummary[]
  >([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [sessionPendingDelete, setSessionPendingDelete] =
    useState<RecordedSessionSummary | null>(null);
  const [deletingSessionName, setDeletingSessionName] = useState<string | null>(
    null,
  );

  const appendLog = (message: string) => {
    setLogs((previous) => [...previous.slice(-(MAX_LOG_LINES - 1)), message]);
  };

  const fetchRecordedSessions = async () => {
    const result = (await window.electron.ipcRenderer.invoke(
      "list-recorded-sessions",
    )) as RecordedSessionSummary[] | undefined;
    return Array.isArray(result) ? result : [];
  };

  const refreshRecordedSessions = async () => {
    setSessionsLoading(true);
    setSessionsError(null);

    try {
      const nextSessions = await fetchRecordedSessions();
      setRecordedSessions(nextSessions);
      return nextSessions;
    } catch {
      setSessionsError("Could not load recorded sessions.");
      return [];
    } finally {
      setSessionsLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadSettings = async () => {
      try {
        const storedSettings = (await window.electron.ipcRenderer.invoke(
          "get-ui-settings",
        )) as UiSettings | undefined;
        if (!cancelled) {
          setDeveloperView(Boolean(storedSettings?.developerView));
        }
      } catch {
        if (!cancelled) {
          setDeveloperView(false);
        }
      }
    };

    void loadSettings();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (phase === "setup" || libraryOpen) {
      void refreshRecordedSessions();
    }
  }, [phase, libraryOpen]);

  useEffect(() => {
    if (phase !== "setup" || sessionNameEdited) {
      return;
    }

    let cancelled = false;

    const syncDefaultSessionName = async () => {
      try {
        const nextName = await window.electron.ipcRenderer.invoke(
          "get-default-session-name",
        );
        if (!cancelled && typeof nextName === "string" && nextName.trim()) {
          setSessionName(nextName);
          return;
        }
      } catch {
        // Fall back to a renderer-generated timestamp if IPC is unavailable.
      }

      if (!cancelled) {
        setSessionName(createFallbackSessionName());
      }
    };

    void syncDefaultSessionName();
    const interval = setInterval(() => {
      void syncDefaultSessionName();
    }, 1000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [phase, sessionNameEdited]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined;
    if (phase === "recording") {
      interval = setInterval(() => {
        setDuration((current) => current + 1);
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [phase]);

  useEffect(() => {
    const handleBackendMessage = (_event: unknown, data: BackendMessage) => {
      if (data.type === "status") {
        setStatus(data.message);
        appendLog(data.message);
        return;
      }

      if (data.type === "session-name-resolved") {
        setSessionName(data.sessionName);
        return;
      }

      if (data.type === "started") {
        setStatus(`Recording to ${data.path}`);
        setSavedSessionPath(data.path);
        setSavedPath(data.path);
        setSavedPathAvailable(true);
        setSessionName(getPathTail(data.path));
        if (
          typeof data.resumedDurationSeconds === "number" &&
          data.resumedDurationSeconds > 0
        ) {
          setDuration(data.resumedDurationSeconds);
        }
        setPhase("recording");
        appendLog(`Recording to ${data.path}`);
        return;
      }

      if (data.type === "finished") {
        setStatus("Session saved successfully.");
        setSavedSessionPath(data.path);
        setSavedPath(data.processedPath ?? data.path);
        setSavedPathAvailable(true);
        setConsolidationProgress(1);
        setPhase("finished");
        appendLog(`Saved to ${data.processedPath ?? data.path}`);
        return;
      }

      if (data.type === "error") {
        setStatus(`Error: ${data.message}`);
        setPhase("setup");
        setSavedPath(null);
        setSavedSessionPath(null);
        setSavedPathAvailable(true);
        setConsolidationProgress(0);
        appendLog(`ERROR: ${data.message}`);
        return;
      }

      if (data.type === "stderr") {
        appendLog(`STDERR: ${data.message}`);
        return;
      }

      if (data.type === "consolidation-progress") {
        setStatus(data.message);
        setPhase("consolidating");
        setConsolidationProgress(data.progress ?? 0);
        appendLog(data.message);
        return;
      }

      if (data.type === "session-state") {
        if (data.state === "paused") {
          setPhase("paused");
          return;
        }

        if (data.state === "recording") {
          setPhase("recording");
        }
      }
    };

    window.electron.ipcRenderer.on("backend-message", handleBackendMessage);
    return () => {
      window.electron.ipcRenderer.removeListener(
        "backend-message",
        handleBackendMessage,
      );
    };
  }, []);

  useEffect(() => {
    const targetPath = savedSessionPath ?? savedPath;
    if (phase !== "finished" || !targetPath) {
      setSavedPathAvailable(true);
      return;
    }

    let cancelled = false;

    const syncSavedPathAvailability = async () => {
      try {
        const exists = await window.electron.ipcRenderer.invoke(
          "path-exists",
          targetPath,
        );
        if (!cancelled) {
          setSavedPathAvailable(Boolean(exists));
        }
      } catch {
        if (!cancelled) {
          setSavedPathAvailable(true);
        }
      }
    };

    const handleFocus = () => {
      void syncSavedPathAvailability();
    };
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void syncSavedPathAvailability();
      }
    };

    void syncSavedPathAvailability();
    const interval = window.setInterval(() => {
      void syncSavedPathAvailability();
    }, 2000);

    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [phase, savedPath, savedSessionPath]);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}:${String(mins).padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    }

    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const beginRecording = ({
    nextSessionName,
    resumeExisting = false,
  }: {
    nextSessionName: string;
    resumeExisting?: boolean;
  }) => {
    const trimmedSessionName = nextSessionName.trim();
    if (!trimmedSessionName) {
      setStatus("Session name cannot be empty.");
      appendLog("ERROR: Session name cannot be empty.");
      return;
    }

    setSessionName(trimmedSessionName);
    setSessionNameEdited(true);
    setDuration(0);
    setSavedPath(null);
    setSavedSessionPath(null);
    setSavedPathAvailable(true);
    setConsolidationProgress(0);
    setStatus(resumeExisting ? "Resuming session..." : "Starting backend...");
    appendLog(
      resumeExisting
        ? `Resuming session "${trimmedSessionName}"...`
        : "Starting recording...",
    );
    setPhase("starting");
    setLibraryOpen(false);
    setSessionsError(null);
    setSessionPendingDelete(null);
    setDeletingSessionName(null);
    window.electron.ipcRenderer.send("start-recording", {
      sessionName: trimmedSessionName,
      resumeExisting,
    });
  };

  const startRecording = () => {
    beginRecording({
      nextSessionName: sessionName,
    });
  };

  const resumeSavedSession = () => {
    if (!savedSessionPath) {
      return;
    }

    beginRecording({
      nextSessionName: getPathTail(savedSessionPath),
      resumeExisting: true,
    });
  };

  const pauseRecording = () => {
    setStatus("Pausing recording...");
    window.electron.ipcRenderer.send("pause-recording");
  };

  const resumeRecording = () => {
    setStatus("Resuming recording...");
    window.electron.ipcRenderer.send("resume-recording");
  };

  const finishRecording = () => {
    setStatus("Saving session...");
    setConsolidationProgress(0);
    setPhase("finishing");
    window.electron.ipcRenderer.send("stop-recording");
  };

  const resetSession = () => {
    setPhase("setup");
    setDuration(0);
    setStatus("Ready");
    setLogs([]);
    setSavedPath(null);
    setSavedSessionPath(null);
    setSavedPathAvailable(true);
    setConsolidationProgress(0);
    setSessionName(createFallbackSessionName());
    setSessionNameEdited(false);
    setLibraryOpen(false);
    setSessionsError(null);
    setSessionPendingDelete(null);
    setDeletingSessionName(null);
  };

  const openLibrary = () => {
    setSettingsOpen(false);
    setLibraryOpen(true);
  };

  const toggleDeveloperView = async () => {
    const nextValue = !developerView;
    setDeveloperView(nextValue);

    try {
      const storedSettings = (await window.electron.ipcRenderer.invoke(
        "save-ui-settings",
        {
          developerView: nextValue,
        },
      )) as UiSettings | undefined;
      setDeveloperView(Boolean(storedSettings?.developerView));
    } catch {
      setDeveloperView(!nextValue);
      setStatus("Could not save settings.");
      appendLog("ERROR: Could not save settings.");
    }
  };

  const revealSavedPath = async () => {
    const targetPath = savedSessionPath ?? savedPath;
    if (!targetPath) {
      return;
    }

    try {
      const revealed = await window.electron.ipcRenderer.invoke(
        "reveal-path",
        targetPath,
      );
      if (!revealed) {
        setSavedPathAvailable(false);
        setStatus("Saved session data is no longer available.");
        appendLog("ERROR: Saved session data is no longer available.");
        return;
      }
    } catch {
      setStatus("Could not reveal the saved session.");
      appendLog("ERROR: Could not reveal the saved session.");
    }
  };

  const revealRecordedSession = async (session: RecordedSessionSummary) => {
    try {
      const revealed = await window.electron.ipcRenderer.invoke(
        "reveal-path",
        session.path,
      );
      if (!revealed) {
        setStatus(`Session "${session.name}" is no longer available.`);
        appendLog(`ERROR: Session "${session.name}" is no longer available.`);
        void refreshRecordedSessions();
      }
    } catch {
      setStatus(`Could not reveal "${session.name}".`);
      appendLog(`ERROR: Could not reveal "${session.name}".`);
    }
  };

  const resumeRecordedSession = (session: RecordedSessionSummary) => {
    beginRecording({
      nextSessionName: session.name,
      resumeExisting: true,
    });
  };

  const deleteRecordedSession = async () => {
    if (!sessionPendingDelete) {
      return;
    }

    const targetSession = sessionPendingDelete;
    setDeletingSessionName(targetSession.name);
    setSessionsError(null);

    try {
      await window.electron.ipcRenderer.invoke(
        "delete-recorded-session",
        targetSession.name,
      );
      setStatus(`Deleted "${targetSession.name}".`);
      appendLog(`Deleted session "${targetSession.name}".`);
      setSessionPendingDelete(null);
      await refreshRecordedSessions();
    } catch (error) {
      const message =
        error instanceof Error && error.message
          ? error.message
          : `Could not delete "${targetSession.name}".`;
      setSessionsError(message);
      setStatus(message);
      appendLog(`ERROR: ${message}`);
    } finally {
      setDeletingSessionName(null);
    }
  };

  const isBusy =
    phase === "starting" || phase === "finishing" || phase === "consolidating";
  const postProcessStages = buildPostProcessStages({
    phase,
    status,
    consolidationProgress,
  });
  const activePostProcessStage = postProcessStages.find(
    (stage) => stage.state === "active",
  );
  const postProcessProgress = getPostProcessProgress(
    phase,
    postProcessStages,
    consolidationProgress,
  );
  const savedRevealPath = savedSessionPath ?? savedPath;
  const savedPathMissing = Boolean(savedRevealPath) && !savedPathAvailable;
  const savedPathMissingMessage = "Saved session data was removed from disk.";
  const canResumeFromLibrary = phase === "setup" || phase === "finished";
  const headerDotClass =
    phase === "recording"
      ? "bg-red-500 animate-pulse"
      : phase === "paused"
        ? "bg-amber-400"
        : phase === "consolidating"
          ? "bg-blue-600 animate-pulse"
          : phase === "starting" || phase === "finishing"
            ? "bg-blue-500 animate-pulse"
            : phase === "finished"
              ? "bg-green-500"
              : "bg-gray-300";
  const phaseBadgeLabel =
    phase === "starting"
      ? "Starting"
      : phase === "paused"
        ? "Paused"
        : phase === "finishing"
          ? "Saving"
          : phase === "consolidating"
            ? "Processing"
            : phase === "finished"
              ? "Saved"
              : "Recording";
  const phaseSubtitle =
    phase === "paused"
      ? "Session paused"
      : phase === "consolidating"
        ? "Post-processing captured session"
        : phase === "finishing"
          ? "Stopping capture and preparing session files"
          : phase === "finished"
            ? "Session complete"
            : phase === "setup"
              ? "Ready for a new session"
              : "Session in progress";

  const renderStatusLog = (emptyMessage: string) => {
    if (!developerView) {
      return null;
    }

    return (
      <section className="overflow-hidden rounded-[28px] border border-gray-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50 px-4 py-3">
          <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-gray-500">
            <Terminal className="h-3.5 w-3.5" />
            Status Log
          </span>
          <span className="text-xs text-gray-400">{logs.length} entries</span>
        </div>
        <div className="max-h-64 space-y-2 overflow-y-auto px-4 py-4 font-mono text-xs text-gray-600">
          {logs.length === 0 ? (
            <span className="italic text-gray-300">{emptyMessage}</span>
          ) : (
            logs.map((log, index) => (
              <div key={`${log}-${index}`} className="break-all">
                {log}
              </div>
            ))
          )}
        </div>
      </section>
    );
  };

  return (
    <div className="flex min-h-screen flex-col bg-gray-50 text-gray-900">
      <header className="flex items-center justify-between border-b border-gray-100 bg-white px-6 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className={`h-3 w-3 rounded-full ${headerDotClass}`} />
          <div>
            <h1 className="text-lg font-semibold">Computer Recorder</h1>
            <p className="text-xs text-gray-500">{phaseSubtitle}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={openLibrary}
            className="inline-flex items-center gap-2 rounded-2xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50"
          >
            <History className="h-4 w-4" />
            Library
          </button>
          <button
            onClick={() => {
              setLibraryOpen(false);
              setSettingsOpen(true);
            }}
            className="inline-flex items-center gap-2 rounded-2xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50"
          >
            <Settings2 className="h-4 w-4" />
            Settings
          </button>
        </div>
      </header>

      <main
        className={`flex flex-1 justify-center p-6 ${
          phase === "setup" ? "items-start overflow-y-auto" : "items-center"
        }`}
      >
        {phase === "setup" ? (
          <div className="w-full max-w-3xl space-y-6">
            <section className="rounded-[28px] border border-gray-200 bg-white p-8 shadow-sm">
              <div className="max-w-xl">
                <h2 className="text-3xl font-semibold tracking-tight text-gray-900">
                  New recording
                </h2>
              </div>

              <div className="mt-10 space-y-6">
                <div className="space-y-2">
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-gray-500">
                    Session Name
                  </label>
                  <input
                    type="text"
                    value={sessionName}
                    onChange={(event) => {
                      setSessionName(event.target.value);
                      setSessionNameEdited(true);
                    }}
                    className="w-full rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3 font-mono text-sm text-gray-900 outline-none transition focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-100"
                    placeholder="session_YYYYMMDD_HHMMSS"
                  />
                </div>

                <button
                  onClick={startRecording}
                  className="flex w-full items-center justify-center gap-3 rounded-2xl bg-gray-900 px-5 py-4 text-sm font-medium text-white shadow-lg shadow-gray-900/10 transition hover:bg-gray-800"
                >
                  <Disc className="h-4 w-4" />
                  Start Recording
                </button>
              </div>
            </section>

            {renderStatusLog("Ready to record...")}
          </div>
        ) : (
          <div className="w-full max-w-3xl space-y-6">
            <section className="rounded-[28px] border border-gray-200 bg-white p-8 shadow-sm">
              <div className="flex flex-col items-center text-center">
                {phase === "finished" ? null : (
                  <div className="mb-5 rounded-full bg-gray-100 px-4 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-gray-500">
                    {phaseBadgeLabel}
                  </div>
                )}
                <p className="text-sm font-medium text-gray-500">Session</p>
                <h2 className="mt-1 max-w-lg break-all text-2xl font-semibold tracking-tight text-gray-900">
                  {sessionName}
                </h2>

                {phase === "finishing" || phase === "consolidating" ? (
                  <>
                    <div className="mt-8">
                      <ProgressRing
                        progress={postProcessProgress}
                        tone="blue"
                        label={`${Math.round(postProcessProgress * 100)}%`}
                        sublabel={
                          activePostProcessStage?.label ?? "Saving session"
                        }
                      />
                    </div>
                    <p className="mt-4 max-w-md text-sm text-gray-500">
                      {activePostProcessStage?.detail ??
                        "Saving your session and preparing the final files."}
                    </p>
                    <div className="mt-8 w-full max-w-2xl space-y-4 text-left">
                      {postProcessStages.map((stage) => {
                        const stageStateLabel =
                          stage.state === "complete"
                            ? "Done"
                            : stage.state === "active"
                              ? stage.indeterminate
                                ? "Working"
                                : `${Math.round(clampProgress(stage.progress) * 100)}%`
                              : "Pending";
                        const fillWidth =
                          stage.indeterminate && stage.state === "active"
                            ? "62%"
                            : `${Math.round(clampProgress(stage.progress) * 100)}%`;

                        return (
                          <div
                            key={stage.key}
                            className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-4"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="min-w-0">
                                <div className="text-sm font-semibold text-gray-900">
                                  {stage.label}
                                </div>
                                <div className="mt-1 text-sm text-gray-500">
                                  {stage.detail}
                                </div>
                              </div>
                              <div
                                className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium tracking-[0.08em] ${
                                  stage.state === "complete"
                                    ? "bg-green-100 text-green-700"
                                    : stage.state === "active"
                                      ? "bg-blue-100 text-blue-700"
                                      : "bg-gray-200 text-gray-500"
                                }`}
                              >
                                {stageStateLabel}
                              </div>
                            </div>
                            <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200">
                              <div
                                className={`h-full rounded-full transition-[width,background-color] duration-700 ease-out ${
                                  stage.state === "complete"
                                    ? "bg-green-500"
                                    : stage.state === "active"
                                      ? "bg-blue-600"
                                      : "bg-gray-300"
                                } ${stage.indeterminate ? "animate-pulse" : ""}`}
                                style={{ width: fillWidth }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : phase === "finished" ? (
                  <>
                    <div className="mt-8">
                      <ProgressRing
                        progress={1}
                        tone="green"
                        label="Saved"
                        sublabel="Complete"
                        icon={<Check className="h-9 w-9" />}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="mt-8 font-mono text-6xl font-semibold tracking-tight text-gray-900">
                      {formatTime(duration)}
                    </div>
                    {developerView ? (
                      <p className="mt-3 max-w-md text-sm text-gray-500">
                        {status}
                      </p>
                    ) : null}
                  </>
                )}
              </div>

              {phase === "finished" ? (
                <div className="mt-10 flex flex-col items-center justify-center gap-3">
                  <div className="flex w-full flex-col items-center justify-center gap-3 sm:flex-row">
                    {savedSessionPath ? (
                      <button
                        onClick={resumeSavedSession}
                        className="flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-blue-500 sm:w-auto"
                      >
                        <Play className="h-4 w-4" />
                        Resume This Session
                      </button>
                    ) : null}
                    {savedRevealPath ? (
                      <div
                        className="group relative w-full sm:w-auto"
                        title={
                          savedPathMissing
                            ? savedPathMissingMessage
                            : "Show the saved session in Finder."
                        }
                      >
                        <button
                          onClick={() => void revealSavedPath()}
                          disabled={savedPathMissing}
                          className="flex w-full items-center justify-center gap-2 rounded-2xl border border-gray-200 bg-white px-5 py-3 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:pointer-events-none disabled:border-gray-200 disabled:bg-gray-100 disabled:text-gray-400 disabled:shadow-none sm:w-auto"
                        >
                          <FolderOpen className="h-4 w-4" />
                          Show in Finder
                        </button>
                        {savedPathMissing ? (
                          <div className="pointer-events-none absolute left-1/2 top-full z-10 mt-2 hidden w-max max-w-56 -translate-x-1/2 rounded-xl bg-gray-900 px-3 py-2 text-center text-xs text-white shadow-lg group-hover:block">
                            {savedPathMissingMessage}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    <button
                      onClick={resetSession}
                      className="flex w-full items-center justify-center gap-2 rounded-2xl bg-gray-900 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-gray-800 sm:w-auto"
                    >
                      <Disc className="h-4 w-4" />
                      Start Another Session
                    </button>
                  </div>
                </div>
              ) : phase === "finishing" || phase === "consolidating" ? null : (
                <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
                  {phase === "starting" ? (
                    <button
                      disabled
                      className="flex w-full items-center justify-center gap-2 rounded-2xl border border-gray-200 bg-white px-5 py-3 text-sm font-medium text-gray-500 shadow-sm sm:w-auto"
                    >
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Starting...
                    </button>
                  ) : phase === "paused" ? (
                    <button
                      onClick={resumeRecording}
                      disabled={isBusy}
                      className="flex w-full items-center justify-center gap-2 rounded-2xl bg-blue-600 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
                    >
                      <Play className="h-4 w-4" />
                      Resume
                    </button>
                  ) : phase === "recording" ? (
                    <button
                      onClick={pauseRecording}
                      className="flex w-full items-center justify-center gap-2 rounded-2xl border border-gray-200 bg-white px-5 py-3 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 sm:w-auto"
                    >
                      <Pause className="h-4 w-4" />
                      Pause
                    </button>
                  ) : null}

                  <button
                    onClick={finishRecording}
                    disabled={phase === "starting"}
                    className="flex w-full items-center justify-center gap-2 rounded-2xl bg-gray-900 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
                  >
                    <Square className="h-4 w-4" />
                    Finish
                  </button>
                </div>
              )}
            </section>

            {renderStatusLog(
              phase === "finished"
                ? "Session complete."
                : "Waiting for recorder output...",
            )}
          </div>
        )}
      </main>

      {sessionPendingDelete ? (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-gray-900/30 p-6 backdrop-blur-sm"
          onClick={() => {
            if (!deletingSessionName) {
              setSessionPendingDelete(null);
            }
          }}
        >
          <div
            className="w-full max-w-md rounded-[28px] border border-gray-200 bg-white p-6 shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start gap-4">
              <div className="rounded-2xl bg-red-100 p-3 text-red-600">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 className="text-xl font-semibold text-gray-900">
                  Delete recorded session?
                </h2>
                <p className="mt-2 text-sm text-gray-500">
                  <span className="font-medium text-gray-700">
                    {sessionPendingDelete.name}
                  </span>{" "}
                  will be permanently deleted from your recorder sessions
                  folder. This cannot be recovered. Are you sure you want to
                  proceed?
                </p>
              </div>
              <button
                onClick={() => setSessionPendingDelete(null)}
                disabled={Boolean(deletingSessionName)}
                className="rounded-full p-2 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600 disabled:cursor-not-allowed disabled:opacity-60"
                aria-label="Close delete confirmation"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
              <button
                onClick={() => setSessionPendingDelete(null)}
                disabled={Boolean(deletingSessionName)}
                className="inline-flex items-center justify-center rounded-2xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={() => void deleteRecordedSession()}
                disabled={Boolean(deletingSessionName)}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-red-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {deletingSessionName ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                Delete Permanently
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {libraryOpen ? (
        <div
          className="fixed inset-0 z-50 overflow-y-auto bg-gray-900/20 p-6 backdrop-blur-sm"
          onClick={() => setLibraryOpen(false)}
        >
          <div className="flex min-h-full items-center justify-center">
            <div
              className="flex min-h-0 max-h-[min(720px,calc(100vh-48px))] w-full max-w-4xl flex-col overflow-hidden rounded-[28px] border border-gray-200 bg-white p-6 shadow-xl"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-gray-500">
                    <History className="h-4 w-4 text-gray-400" />
                    Library
                  </div>
                  <h2 className="mt-2 text-2xl font-semibold text-gray-900">
                    Recorded sessions
                  </h2>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => void refreshRecordedSessions()}
                    disabled={sessionsLoading}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <RefreshCw
                      className={`h-4 w-4 ${sessionsLoading ? "animate-spin" : ""}`}
                    />
                    Refresh
                  </button>
                  <button
                    onClick={() => setLibraryOpen(false)}
                    className="rounded-full p-2 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
                    aria-label="Close library"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="mt-6 flex items-center justify-between gap-3 rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3">
                <span className="text-sm font-medium text-gray-700">
                  {recordedSessions.length} saved session
                  {recordedSessions.length === 1 ? "" : "s"}
                </span>
                <span className="text-xs text-gray-500">
                  {sessionsLoading
                    ? "Refreshing library..."
                    : "Stored in Downloads/recorder_sessions"}
                </span>
              </div>

              {sessionsError ? (
                <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {sessionsError}
                </div>
              ) : null}

              <div className="mt-6 flex min-h-0 flex-1 flex-col overflow-hidden">
                {sessionsLoading && recordedSessions.length === 0 ? (
                  <div className="flex h-full min-h-40 items-center justify-center rounded-2xl border border-dashed border-gray-200 bg-gray-50 text-sm text-gray-500">
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Loading recorded sessions...
                  </div>
                ) : recordedSessions.length === 0 ? (
                  <div className="flex h-full min-h-40 items-center justify-center rounded-2xl border border-dashed border-gray-200 bg-gray-50 px-6 text-center text-sm text-gray-500">
                    No recorded sessions found yet.
                  </div>
                ) : (
                  <div className="min-h-0 flex-1 space-y-3 overflow-y-auto overscroll-contain pr-1">
                    {recordedSessions.map((recordedSession) => {
                      const isDeletingSession =
                        deletingSessionName === recordedSession.name;

                      return (
                        <div
                          key={recordedSession.path}
                          className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-4"
                        >
                          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                            <div className="min-w-0">
                              <div className="break-all text-sm font-semibold text-gray-900">
                                {recordedSession.name}
                              </div>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <span className="inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600">
                                  <MousePointerClick className="h-3.5 w-3.5" />
                                  {formatActionCount(
                                    recordedSession.actionCount,
                                  )}
                                </span>
                                <span className="inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600">
                                  <Clock3 className="h-3.5 w-3.5" />
                                  {formatSessionStartedAt(
                                    recordedSession.startedAt,
                                  )}
                                </span>
                                {recordedSession.hasRawVideo ? (
                                  <span className="inline-flex items-center rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600">
                                    Raw video
                                  </span>
                                ) : null}
                                {!recordedSession.hasProcessedTrajectory ? (
                                  <span className="inline-flex items-center rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700">
                                    Metadata incomplete
                                  </span>
                                ) : null}
                              </div>
                            </div>

                            <div className="flex flex-col gap-2 sm:flex-row">
                              <button
                                onClick={() =>
                                  resumeRecordedSession(recordedSession)
                                }
                                disabled={
                                  isDeletingSession || !canResumeFromLibrary
                                }
                                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                <Play className="h-4 w-4" />
                                Resume
                              </button>
                              <button
                                onClick={() =>
                                  void revealRecordedSession(recordedSession)
                                }
                                disabled={isDeletingSession}
                                className="inline-flex items-center justify-center gap-2 rounded-2xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                <FolderOpen className="h-4 w-4" />
                                Show in Finder
                              </button>
                              <button
                                onClick={() =>
                                  setSessionPendingDelete(recordedSession)
                                }
                                disabled={Boolean(deletingSessionName)}
                                className="inline-flex items-center justify-center gap-2 rounded-2xl border border-red-200 bg-white px-4 py-2 text-sm font-medium text-red-700 shadow-sm transition hover:border-red-300 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                {isDeletingSession ? (
                                  <RefreshCw className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Trash2 className="h-4 w-4" />
                                )}
                                Delete
                              </button>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {settingsOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/20 p-6 backdrop-blur-sm"
          onClick={() => setSettingsOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-[28px] border border-gray-200 bg-white p-6 shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Settings
                </h2>
              </div>
              <button
                onClick={() => setSettingsOpen(false)}
                className="rounded-full p-2 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
                aria-label="Close settings"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-6">
              <div className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-sm font-medium text-gray-900">
                      Developer view
                      <button
                        type="button"
                        title="Shows STATUS LOG and backend diagnostics, including internal stderr output."
                        aria-label="Developer view help"
                        className="rounded-full p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
                      >
                        <Info className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={developerView}
                    onClick={() => void toggleDeveloperView()}
                    className={`relative inline-flex h-7 w-12 shrink-0 items-center rounded-full transition ${
                      developerView ? "bg-blue-600" : "bg-gray-300"
                    }`}
                  >
                    <span
                      className={`inline-block h-5 w-5 rounded-full bg-white transition ${
                        developerView ? "translate-x-6" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default RecordingControl;
