import "server-only";
import { promises as fs } from "fs";
import path from "path";
import os from "os";
import type {
  Activity,
  Manifest,
  ObjectiveModelNode,
  ProcedureModel,
  UnifiedTaskModel,
  SessionData,
  SessionResult,
  ThreadBundle,
} from "./types";

// Override with SESSION_DIR to skip typing a path on every load.
export const DEFAULT_SESSION_DIR =
  process.env.SESSION_DIR ?? "~/Downloads/recorder_sessions";

export function resolveDir(dir: string): string {
  let p = dir.trim();
  if (p.startsWith("~")) p = path.join(os.homedir(), p.slice(1));
  return path.resolve(p);
}

async function readJson<T>(file: string): Promise<T> {
  const raw = await fs.readFile(file, "utf8");
  return JSON.parse(raw) as T;
}

async function readJsonl<T>(file: string): Promise<T[]> {
  const raw = await fs.readFile(file, "utf8");
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as T);
}

async function tryRead<T>(file: string): Promise<T | null> {
  try {
    return await readJson<T>(file);
  } catch {
    return null;
  }
}


export async function loadSession(dir: string): Promise<SessionResult> {
  const resolvedDir = resolveDir(dir);

  let manifest: Manifest;
  try {
    manifest = await readJson<Manifest>(
      path.join(resolvedDir, "derived_task_thread_objectives", "manifest.json"),
    );
  } catch (err) {
    return {
      ok: false,
      resolvedDir,
      error: `Could not read manifest at ${resolvedDir}/derived_task_thread_objectives/manifest.json. ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }

  let activities: Activity[];
  try {
    activities = await readJsonl<Activity>(
      path.join(resolvedDir, "activity.jsonl"),
    );
  } catch (err) {
    return {
      ok: false,
      resolvedDir,
      error: `Could not read activity.jsonl. ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }

  const activityIndex: Record<string, number> = {};
  activities.forEach((a, i) => {
    activityIndex[a.activity_id] = i;
  });

  const threads: ThreadBundle[] = [];
  const threadByObjective: Record<string, string> = {};

  for (const root of manifest.roots) {
    const baseName = path.basename(root.file, ".json");
    const derived = await readJson<{
      canonical_root_id: string;
      label: string;
      task_thread_objective: string;
      activities: Activity[];
    }>(path.join(resolvedDir, "derived_task_thread_objectives", `${baseName}.json`));

    const objectiveModel = await tryRead<ObjectiveModelNode>(
      path.join(resolvedDir, "task_thread_objective_model", `${baseName}.json`),
    );
    const procedureModel = await tryRead<ProcedureModel>(
      path.join(resolvedDir, "task_thread_procedure_model", `${baseName}.json`),
    );
    const unifiedModel = await tryRead<UnifiedTaskModel>(
      path.join(resolvedDir, "task_thread_task_model", `${baseName}.json`),
    );

    const ids = derived.activities.map((a) => a.activity_id);
    for (const id of ids) threadByObjective[id] = root.canonical_root_id;

    threads.push({
      id: root.canonical_root_id,
      label: root.label,
      task_thread_objective: derived.task_thread_objective,
      localObjectiveIds: ids,
      objectiveModel,
      procedureModel,
      unifiedModel,
    });
  }

  const data: SessionData = {
    dir,
    resolvedDir,
    manifest,
    activities,
    threads,
    threadByObjective,
    activityIndex,
  };
  return { ok: true, data };
}
