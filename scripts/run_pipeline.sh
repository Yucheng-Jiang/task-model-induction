#!/usr/bin/env bash
#
# Run the task model induction pipeline over one recorded session.
#
# A session directory is any directory containing processed_trajectory.jsonl.
# The macOS recorder writes one per recording, by default under
# ~/Downloads/recorder_sessions/<session_name>, but any path works.
#
# Each step reads the previous step's output from the same directory, so the
# whole pipeline is resumable: re-run with --from N to pick up where you
# stopped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$REPO_ROOT/task_model_induction"

FIRST_STEP=0
LAST_STEP=6
CONFIG_PATH=""
PREFLIGHT_ONLY=0
NO_CONSOLE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh <session_dir> [options]

Options:
  --from <n>        First step to run (default: 0)
  --to <n>          Last step to run (default: 6)
  --config <path>   Config file to use (default: task_model_induction/config.yaml)
  --preflight       Validate inputs and model access without calling any model
  --no-console      Plain structured logs instead of the live status panel
  -h, --help        Show this message

Steps:
  0  action grounding            processed_trajectory.jsonl
                                   -> processed_trajectory_with_goals.jsonl
  1  semantic action induction   -> atom_semantic_actions.jsonl
  2  activity induction          -> activity.jsonl
  3  task thread induction       -> task_threads.json
  4  objective model induction   -> hierarchy.json, task_thread_objective_model/
  5  procedure model induction   -> task_thread_procedure_model/
  6  bidirectional alignment     -> task_model.json, task_thread_task_model/

Examples:
  scripts/run_pipeline.sh <path/to/session>
  scripts/run_pipeline.sh <path/to/session> --from 3
  scripts/run_pipeline.sh <path/to/session> --preflight
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 1
fi

SESSION_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --from) FIRST_STEP="${2:?--from needs a step number}"; shift 2 ;;
    --to) LAST_STEP="${2:?--to needs a step number}"; shift 2 ;;
    --config) CONFIG_PATH="${2:?--config needs a path}"; shift 2 ;;
    --preflight) PREFLIGHT_ONLY=1; shift ;;
    --no-console) NO_CONSOLE=1; shift ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)
      if [[ -n "$SESSION_DIR" ]]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      SESSION_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$SESSION_DIR" ]]; then
  echo "Missing <session_dir>." >&2
  usage >&2
  exit 1
fi

SESSION_DIR="${SESSION_DIR/#\~/$HOME}"
if [[ ! -d "$SESSION_DIR" ]]; then
  echo "Session directory does not exist: $SESSION_DIR" >&2
  exit 1
fi
SESSION_DIR="$(cd "$SESSION_DIR" && pwd)"

for step in "$FIRST_STEP" "$LAST_STEP"; do
  if [[ ! "$step" =~ ^[0-6]$ ]]; then
    echo "Step must be between 0 and 6, got: $step" >&2
    exit 1
  fi
done
if (( FIRST_STEP > LAST_STEP )); then
  echo "--from ($FIRST_STEP) is after --to ($LAST_STEP)." >&2
  exit 1
fi

if (( FIRST_STEP == 0 )) && [[ ! -f "$SESSION_DIR/processed_trajectory.jsonl" ]]; then
  echo "No processed_trajectory.jsonl in $SESSION_DIR." >&2
  echo "Record a session first, or start from a later step with --from." >&2
  exit 1
fi

if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_PATH="${CONFIG_PATH/#\~/$HOME}"
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file does not exist: $CONFIG_PATH" >&2
    exit 1
  fi
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
  # step0 reads the config from the environment rather than a flag.
  export TASK_MODEL_INDUCTION_CONFIG="$CONFIG_PATH"
fi

cd "$PIPELINE_DIR"

echo "Session: $SESSION_DIR"
echo "Steps:   $FIRST_STEP..$LAST_STEP"
echo "Config:  ${CONFIG_PATH:-$PIPELINE_DIR/config.yaml}"

for (( step = FIRST_STEP; step <= LAST_STEP; step++ )); do
  shopt -s nullglob
  matches=( step"$step"_*.py )
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one script for step $step, found ${#matches[@]}." >&2
    exit 1
  fi
  script="${matches[0]}"

  args=( --data_dir "$SESSION_DIR" )
  # step0 has no --config flag; it resolves the config from the environment.
  if [[ -n "$CONFIG_PATH" && "$step" -ne 0 ]]; then
    args+=( --config "$CONFIG_PATH" )
  fi
  (( PREFLIGHT_ONLY )) && args+=( --preflight_only )
  (( NO_CONSOLE )) && args+=( --no_console )

  echo
  echo "==> step $step: ${script%.py}"
  uv run python "$script" "${args[@]}"
done

echo
if (( PREFLIGHT_ONLY )); then
  echo "Preflight passed for steps $FIRST_STEP..$LAST_STEP."
else
  echo "Finished steps $FIRST_STEP..$LAST_STEP for $SESSION_DIR"
  if (( LAST_STEP == 6 )); then
    echo "Task model:  $SESSION_DIR/task_model.json"
    echo "Per thread:  $SESSION_DIR/task_thread_task_model/"
  fi
fi
