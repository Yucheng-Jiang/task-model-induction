"""Build processed_trajectory.jsonl from recorder raw trace artifacts.

Two files are written side by side in the session directory:

* ``processed_trajectory.jsonl`` -- the induction pipeline's input. One JSON
  object per line, each an action with ``id``, ``action``, ``state_before``,
  ``state_after``, ``time_before``, ``time_after`` and ``time_range``.
  Screenshot paths are relative to the session directory so a session stays
  readable after it is moved or unzipped elsewhere.
* ``processed_trajectory.json`` -- the recorder's own richer artifact: a single
  ``SequenceNode`` document that also carries OCR/VLM enrichment. ``--patch``
  reads and rewrites this file.
"""

import argparse
import importlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from PIL import Image, ImageDraw
from tqdm import tqdm
from trace_utils import (
    is_click_action, is_keyboard_action, is_scroll_action, is_drag_action,
    get_key_input, compose_key_input,
    encode_image,
)
from language import ActionNode, SequenceNode
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv.main import load_dotenv
load_dotenv()

VLM_SYSTEM_PROMPT = """You are a screen content extractor. Given a screenshot of a computer screen, extract ALL visible text and UI content into well-structured Markdown.

Rules:
- Use headings (#, ##, ###) for window titles, section headers, and tab labels.
- Use lists (- or 1.) for menu items, sidebar entries, and list elements.
- Use tables for tabular data.
- Use code blocks for code or terminal content.
- Preserve the spatial hierarchy: top-level UI elements first, then nested content.
- Include button labels, form field labels and values, status bar text, etc.
- In case there is image/video visible, describe it."""


ProgressCallback = Callable[[str, float | None], None] | None


def emit_progress(message: str, progress: float | None = None, progress_callback: ProgressCallback = None) -> None:
    if progress_callback is not None:
        progress_callback(message, progress)
    else:
        print(message)


def process_node_vlm(node, model_name: str = "openai/gpt-5-mini") -> str | None:
    """Use a VLM to extract page content as markdown from a screenshot.

    Args:
        node: ActionNode whose before-state screenshot to process.
        model_name: The litellm model identifier (e.g. openai/gpt-5-mini).

    Returns:
        Markdown string of the page content, or None if no screenshot.
    """
    if not (node.state.before and os.path.exists(node.state.before)):
        return None

    image_url = encode_image(node.state.before, return_url=True)
    content = [
        {"type": "text", "text": "Extract all visible content from this screenshot as Markdown."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    llm_module = importlib.import_module("ut" "ils")
    result = llm_module.call_llm(
        prompt=VLM_SYSTEM_PROMPT,
        content=content,
        model_name=model_name,
    )
    return result if result else None


# %% Bounding Box Annotation

def load_bounding_boxes(bbox_path: str) -> dict[str, dict]:
    """Load bounding boxes from JSONL file.

    Returns a dict mapping screenshot filename to bounding box data.
    """
    bbox_map = {}
    if not os.path.exists(bbox_path):
        return bbox_map

    with open(bbox_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                bbox_map[entry['screenshot_path']] = entry
    return bbox_map


def draw_bounding_box_on_image(image_path: str, box: dict, output_path: str,
                               box_color: str = "red", box_width: int = 10) -> str:
    """Draw a bounding box on an image and save to output path.

    Args:
        image_path: Path to source image
        box: Dict with x1, y1, x2, y2 keys
        output_path: Path to save annotated image
        box_color: Color of the bounding box
        box_width: Width of the bounding box outline

    Returns:
        Path to the saved annotated image
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']],
                   outline=box_color, width=box_width)
    image.save(output_path, "JPEG", quality=70, optimize=True)
    del draw
    del image
    return output_path


def annotate_screenshots(
    screenshot_dir: str,
    annotated_dir: str,
    bbox_map: dict,
    show_progress: bool = True,
) -> dict[str, str]:
    """Create annotated versions of screenshots with bounding boxes.

    Only click actions get bounding boxes drawn on them.

    Args:
        screenshot_dir: Directory containing raw screenshots
        annotated_dir: Directory to save annotated screenshots
        bbox_map: Dict mapping screenshot filename to bounding box data

    Returns:
        Dict mapping original screenshot path to annotated screenshot path
    """
    os.makedirs(annotated_dir, exist_ok=True)
    path_mapping = {}

    for filename, bbox_data in tqdm(
        bbox_map.items(),
        desc="Creating annotated screenshots",
        disable=not show_progress,
    ):
        src_path = os.path.join(screenshot_dir, filename)
        dst_path = os.path.join(annotated_dir, filename)

        if os.path.exists(src_path):
            # Only draw bounding box for click actions
            if 'click' in filename:
                draw_bounding_box_on_image(src_path, bbox_data['box'], dst_path)
            else:
                # Copy the image without bounding box
                image = Image.open(src_path)
                image.save(dst_path, "JPEG", quality=70, optimize=True)
                del image
            path_mapping[src_path] = dst_path

    return path_mapping


def _coerce_timestamp(raw_timestamp) -> float:
    if isinstance(raw_timestamp, (int, float)):
        return float(raw_timestamp)

    text = str(raw_timestamp).strip()
    if not text:
        raise ValueError("Empty timestamp")

    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_actions_from_jsonl(log_dir: str, trace_name: str = "raw_trace.jsonl") -> list[dict]:
    trace_path = os.path.expanduser(os.path.join(log_dir, trace_name))
    actions = []

    with open(trace_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            action = entry.get("content")
            timestamp = entry.get("timestamp")
            if action is None or timestamp is None:
                continue

            actions.append({
                "action": str(action),
                "timestamp": float(timestamp),
            })

    actions.sort(key=lambda action: action["timestamp"])
    return actions


def load_actions_from_legacy_db(log_dir: str, db_name: str = "actions.db") -> list[dict]:
    db_path = os.path.expanduser(os.path.join(log_dir, db_name))
    actions = []

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT content, created_at FROM observations ORDER BY created_at ASC, id ASC"
        ).fetchall()

    for content, created_at in rows:
        actions.append({
            "action": str(content),
            "timestamp": _coerce_timestamp(created_at),
        })

    return actions


def load_actions(log_dir: str) -> list[dict]:
    trace_path = os.path.expanduser(os.path.join(log_dir, "raw_trace.jsonl"))
    if os.path.exists(trace_path):
        return load_actions_from_jsonl(log_dir)

    legacy_db_path = os.path.expanduser(os.path.join(log_dir, "actions.db"))
    if os.path.exists(legacy_db_path):
        return load_actions_from_legacy_db(log_dir)

    raise FileNotFoundError(
        f"No raw trace found in {log_dir}. Expected raw_trace.jsonl or actions.db."
    )


def hotkey_in_action(action: str) -> bool:
    """Check if the action contains a hotkey."""
    return any(hotkey in action for hotkey in [".cmd", ".enter", ".tab", ".up", ".down"])


def trigger_close_buffer(action: str, buffer_actions: list[dict], enable_hotkey: bool = False) -> bool:
    """Time to close the buffer: 
    - Current buffer is non-empty
    - Next new key/scroll action is different from the last action in the buffer."""
    if len(buffer_actions) == 0:
        return False

    last_action = buffer_actions[-1]["action"]
    if is_keyboard_action(last_action) and (not is_keyboard_action(action)):
        return True
    if is_scroll_action(last_action) and (not is_scroll_action(action)):
        return True
    if enable_hotkey and is_keyboard_action(action) and hotkey_in_action(action):
        return True
    return False


def trigger_add_buffer(action: str, buffer_actions: list[dict]) -> bool:
    """Should add the new action to the buffer.
    - Is keyboard or scroll action
    - (i) buffer is empty; (ii) last action in buffer is the same type as the new action.
    """
    if not (is_keyboard_action(action) or is_scroll_action(action)):
        return False
    if len(buffer_actions) == 0:
        return True

    last_action = buffer_actions[-1]["action"]
    if is_keyboard_action(action) and is_keyboard_action(last_action):
        # print(f"Event 1: {action} | {last_action}")
        return True
    if is_scroll_action(action) and is_scroll_action(last_action):
        return True
    return False


def merge_actions(actions: list[dict], enable_hotkey: bool = False) -> tuple[list[dict], list[dict]]:
    """Merge adjacent keyboard and scrolling actions into a single action."""
    original_actions, merged_actions = [], []
    buffer_actions = []
    for action in actions:
        close_buffer_flag = trigger_close_buffer(action["action"], buffer_actions, enable_hotkey=enable_hotkey)
        if close_buffer_flag:
            if buffer_actions and is_keyboard_action(buffer_actions[0]["action"]):  # keypress buffer
                assert all([is_keyboard_action(action["action"]) for action in buffer_actions])
                original_actions.append({
                    "before": buffer_actions[0]["action"],
                    "after": buffer_actions[-1]["action"],
                    "timestamp_before": buffer_actions[0]["timestamp"],
                    "timestamp_after": buffer_actions[-1]["timestamp"],
                    "constituents": buffer_actions
                })

                buffer_values = [get_key_input(action["action"]) for action in buffer_actions]
                keyboard_input = compose_key_input(buffer_values)
                merged_actions.append({
                    "action": f"key_press('{keyboard_input}')",
                    "timestamp_before": buffer_actions[0]["timestamp"],
                    "timestamp_after": buffer_actions[-1]["timestamp"],
                    "constituents": buffer_actions
                })
                # print("[KeyPress] :", merged_actions[-1]["action"])
            elif buffer_actions and is_scroll_action(buffer_actions[0]["action"]):  # scroll buffer
                assert all([is_scroll_action(action["action"]) for action in buffer_actions])
                for ba in buffer_actions:
                    if len(merged_actions) == 0 or ba["action"] != merged_actions[-1]["action"]:
                        original_actions.append({
                            "before": ba["action"],
                            "after": ba["action"],
                            "timestamp_before": ba["timestamp"],
                            "timestamp_after": ba["timestamp"],
                            "constituents": [ba]
                        })
                        merged_actions.append({
                            "action": ba["action"],
                            "timestamp_before": ba["timestamp"],
                            "timestamp_after": ba["timestamp"],
                            "constituents": [ba]
                        })
                        # print("[Scroll] :", merged_actions[-1]["action"])
            buffer_actions = []

        add_buffer_flag = trigger_add_buffer(action["action"], buffer_actions)
        if add_buffer_flag:
            buffer_actions.append(action)
        else:
            merged_actions.append({
                "action": action["action"],
                "timestamp_before": action["timestamp"],
                "timestamp_after": action["timestamp"],
                "constituents": [action]
            })
            original_actions.append({
                "before": action["action"],
                "after": action["action"],
                "timestamp_before": action["timestamp"],
                "timestamp_after": action["timestamp"],
                "constituents": [action]
            })

    # Flush any remaining buffered actions
    if buffer_actions:
        if is_keyboard_action(buffer_actions[0]["action"]):
            assert all([is_keyboard_action(a["action"]) for a in buffer_actions])
            original_actions.append({
                "before": buffer_actions[0]["action"],
                "after": buffer_actions[-1]["action"],
                "timestamp_before": buffer_actions[0]["timestamp"],
                "timestamp_after": buffer_actions[-1]["timestamp"],
                "constituents": buffer_actions
            })
            buffer_values = [get_key_input(a["action"]) for a in buffer_actions]
            keyboard_input = compose_key_input(buffer_values)
            merged_actions.append({
                "action": f"key_press('{keyboard_input}')",
                "timestamp_before": buffer_actions[0]["timestamp"],
                "timestamp_after": buffer_actions[-1]["timestamp"],
                "constituents": buffer_actions
            })
        elif is_scroll_action(buffer_actions[0]["action"]):
            assert all([is_scroll_action(a["action"]) for a in buffer_actions])
            for ba in buffer_actions:
                if len(merged_actions) == 0 or ba["action"] != merged_actions[-1]["action"]:
                    original_actions.append({
                        "before": ba["action"],
                        "after": ba["action"],
                        "timestamp_before": ba["timestamp"],
                        "timestamp_after": ba["timestamp"],
                        "constituents": [ba]
                    })
                    merged_actions.append({
                        "action": ba["action"],
                        "timestamp_before": ba["timestamp"],
                        "timestamp_after": ba["timestamp"],
                        "constituents": [ba]
                    })

    return original_actions, merged_actions


# %% State

def find_screenshot(screenshot_paths: list[str], action: str, preferred_suffix: str, timestamp: float, time_threshold: float = 2.0) -> tuple[str, list[str]]:
    """Find the screenshot path for the given action.
    Prioritize:
    1. Timestamp proximity (within threshold)
    2. Preferred suffix match
    """
    candidates = []

    # First, collect all candidates that contain the action string
    for i, sp in enumerate(screenshot_paths):
        # We need to handle potential partial matches or escaping differences?
        # For now, stick to "action in sp"
        if action in sp:
            try:
                ss_time = parse_time_from_path(sp)
                diff = abs(ss_time - timestamp)
                if diff < time_threshold:
                    candidates.append({
                        "path": sp,
                        "idx": i,
                        "diff": diff,
                        "suffix_match": sp.endswith(preferred_suffix)
                    })
            except:
                continue

    if not candidates:
        return None, screenshot_paths

    candidates.sort(key=lambda x: (not x["suffix_match"], x["diff"]))
    best = candidates[0]

    return best["path"], screenshot_paths[: best["idx"]] + screenshot_paths[best["idx"]+1:]


def get_states(actions: list[dict], screenshot_dir: str, is_windows: bool = False) -> list[dict[str, str]]:
    """Get before/after states (screenshots) associate with each action.
    """
    screenshot_paths = sorted(os.listdir(screenshot_dir), key=lambda x: x.split('_')[0])  # sort by timestamp
    screenshot_paths = [os.path.join(screenshot_dir, p) for p in screenshot_paths]

    states = []
    # We iterate through actions. We maintain the full list of remaining screenshots to search from.
    # But since we might have out-of-order slightly, we rely on timestamp search.
    # To prevent O(N^2), the find_screenshot reduces the list. We should ensure we don't skip too aggressively.

    current_screenshot_paths = screenshot_paths

    for action_dict in actions:
        # print(action_dict)

        # Get constituents (fallback to self if missing, though merge_actions ensures it)
        constituents = action_dict.get("constituents", [action_dict])

        # --- BEFORE STATE ---
        # Try to find a valid 'before' screenshot from the constituents, starting from the first.
        before_path = None
        for constituent in constituents:
            # Determine suffix based on the constituent action type
            c_action = constituent["action"] if isinstance(constituent, dict) else constituent
            c_time = constituent["timestamp"] if isinstance(constituent, dict) else action_dict["timestamp_before"]

            suffix_before = "_first.jpg" if is_keyboard_action(c_action) else "_before.jpg"

            # Try to find it
            path, new_paths = find_screenshot(
                current_screenshot_paths,
                c_action,
                suffix_before,
                c_time
            )

            if path:
                before_path = path
                current_screenshot_paths = new_paths
                break

        # --- AFTER STATE ---
        # Try to find a valid 'after' screenshot from the constituents, starting from the LAST.
        after_path = None
        for constituent in reversed(constituents):
            c_action = constituent["action"] if isinstance(constituent, dict) else constituent
            c_time = constituent["timestamp"] if isinstance(constituent, dict) else action_dict["timestamp_after"]

            suffix_after = "_final.jpg" if is_keyboard_action(c_action) else "_after.jpg"

            path, new_paths = find_screenshot(
                current_screenshot_paths,
                c_action,
                suffix_after,
                c_time
            )

            if path:
                after_path = path
                # current_screenshot_paths = new_paths # Do not update for 'after', to be safe?
                # Actually if we found it, we can update. But if we skipped some 'end' actions, we might skip their screenshots too.
                # Since we are moving forward, updating is correct.
                current_screenshot_paths = new_paths
                break

        state = {"before": before_path, "after": after_path}
        states.append(state)

    return states


def adjust_states(actions: list[dict], states: list[dict]) -> list[dict]:
    """Adjust the states to reflect more accurate changes.

    For non-keyboard actions:
    - Use the action's own before screenshot if it exists
    - Only fall back to previous action's after screenshot if own before is None

    For keyboard actions:
    - Always use the action's own before screenshot (since key typing happens 
      immediately and shouldn't inherit screen state from previous action)
    """
    adjusted_states = []
    for i, (action_item, state) in enumerate(zip(actions, states)):
        # Handle both dict (with timestamp) and legacy str actions
        action_str = action_item["action"] if isinstance(action_item, dict) else action_item

        if (i == 0) or is_keyboard_action(action_str):
            before_state = state["before"]
        else:
            # For non-keyboard actions: prefer own before state if available,
            # only fall back to previous action's after state if own before is None
            if state["before"] is not None:
                before_state = state["before"]
            else:
                before_state = states[i-1]["after"]

        # For drag actions: keep the action's own after screenshot (the drag
        # result is captured immediately, so the next action's before may show
        # a completely different state).
        # For other actions: use next action's before as this action's after,
        # which captures the actual result after the UI has had time to update.
        if is_drag_action(action_str):
            after_state = state.get("after", state["before"])
        elif i < len(actions) - 1:
            after_state = states[i+1]["before"] or state.get("after", state["before"])
        else:
            after_state = state.get("after", state["before"])

        adjusted_states.append({"before": before_state, "after": after_state})

    return adjusted_states


# %% Time

def parse_screenshot_path(path: str) -> tuple[str, str]:
    """Parse the screenshot path into action and timestamp."""
    parts = path.split('/')[-1].split('_')
    timestamp = parts[0]
    if "key" in parts:
        action = '_'.join(parts[1:]).rstrip(".jpg")
        tag = "before"
    else:
        action = '_'.join(parts[1:-1])
        tag = parts[-1].split('.')[0]
    return {"timestamp": timestamp, "action": action, "tag": tag}


# %% Merge Click Actions

def parse_click_coords(action: str) -> tuple[float, float]:
    """Parse the coordinates from the action."""
    x, y = action.split('(')[1].split(')')[0].split(',')
    return float(x), float(y)


def is_double_click(step_1: ActionNode, step_2: ActionNode, time_threshold: float = 0.5, distance_threshold: float = 10) -> bool:
    """Check if the two click actions constitute a double click."""
    if not (is_click_action(step_1.action) and is_click_action(step_2.action)):
        return False
    if step_2.time.diff > time_threshold:
        return False

    x1, y1 = parse_click_coords(step_1.action)
    x2, y2 = parse_click_coords(step_2.action)
    dx, dy = x2 - x1, y2 - y1
    distance = (dx * dx + dy * dy) ** 0.5
    return distance < distance_threshold


def merge_double_clicks(node_list: list[ActionNode], verbose: bool = True) -> list[ActionNode]:
    """Merge double clicks into a single click."""
    merged_node_list = []
    i, N = 0, len(node_list) - 1
    while i < N:
        step, next_step = node_list[i], node_list[i+1]
        if is_double_click(step, next_step):
            coords_str = '(' + step.action.split('(')[1]
            merged_action = "double_click" + coords_str

            data = {
                "action": merged_action,
                "state": {
                    "before": step.state.before,
                    "after": next_step.state.after,
                },
                "time": {
                    "before": step.time.before,
                    "after": next_step.time.after,
                    "range": step.time.range + next_step.time.range,
                    "diff": step.time.diff + next_step.time.diff
                }
            }
            merged_node_list.append(ActionNode.from_json(data=data))
            i += 2
        else:
            merged_node_list.append(step)
            i += 1
    if verbose:
        print(f"Double clicks merged: #{len(node_list)} -> #{len(merged_node_list)} steps.")
    return merged_node_list


# %% Time

def parse_time_from_path(path: str) -> float:
    """Parse the time from the path."""
    return float(path.split('/')[-1].split('_')[0])


def measure_time_from_states(states: list[dict]) -> list[dict]:
    """Measure the time from the states."""
    time_list = []
    for i, state in enumerate(states):
        # calculate time range
        before_time = 0
        after_time = 0

        # Try to parse any available timestamp
        if state["before"]:
            try:
                before_time = parse_time_from_path(state["before"])
            except Exception:
                pass

        if state.get("after"):
            try:
                after_time = parse_time_from_path(state["after"])
            except Exception:
                pass

        # If one is missing, use the other
        if before_time == 0 and after_time != 0:
            before_time = after_time
        if after_time == 0 and before_time != 0:
            after_time = before_time

        time_range = after_time - before_time

        # calculate time diff
        if i == 0:
            time_diff = 0
        else:
            try:
                # Get last time from previous state
                prev_state = states[i-1]
                last_time = 0

                # Try after, then before
                if prev_state.get("after"):
                    last_time = parse_time_from_path(prev_state["after"])
                elif prev_state.get("before"):
                    last_time = parse_time_from_path(prev_state["before"])

                if last_time != 0 and before_time != 0:
                    time_diff = before_time - last_time
                else:
                    time_diff = 0
            except:
                print(f"Error parsing time from path: {states[i-1]}")
                time_diff = 0

        time_list.append({
            "before": before_time, "after": after_time,
            "range": time_range, "diff": time_diff,
        })
    return time_list


# %% Patch
def patch_missing_ocr(args):
    """Load existing processed_trajectory.json, find nodes with missing OCR
    results that have valid before-screenshots, re-run OCR/VLM on only those
    nodes, and save the patched trajectory back."""
    traj_path = os.path.join(args.data_dir, "processed_trajectory.json")
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found. Run without --patch first.")
        return

    print(f"Loading existing trajectory from {traj_path}...")
    root = SequenceNode.from_json(path=traj_path)

    # Collect all ActionNodes that are missing OCR but have a valid screenshot
    def _is_missing_ocr(node):
        ocr = node.ocr_results
        return ocr is None or ocr == {} or ocr == []

    nodes_to_process = []
    for node in root.nodes:
        if isinstance(node, ActionNode):
            if _is_missing_ocr(node) and node.state.before and os.path.exists(node.state.before):
                nodes_to_process.append(node)

    if not nodes_to_process:
        print("All nodes already have OCR results (or have no screenshots). Nothing to patch.")
        return

    print(f"Found {len(nodes_to_process)} nodes missing OCR with valid screenshots. Re-processing...")

    if args.vlm:
        print(f"Running VLM ({args.vlm_model}) on {len(nodes_to_process)} missing nodes...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_node = {
                executor.submit(process_node_vlm, node, args.vlm_model): node
                for node in nodes_to_process
            }
            for future in tqdm(as_completed(future_to_node), total=len(nodes_to_process), desc="VLM Patching"):
                node = future_to_node[future]
                try:
                    md_content = future.result()
                    if md_content:
                        if node.ocr_results is None:
                            node.ocr_results = {}
                        node.ocr_results["vlm_md_results"] = md_content
                except Exception as e:
                    print(f"Error processing VLM for a node: {e}")
    else:
        print(f"Running OCR on {len(nodes_to_process)} missing nodes...")
        try:
            process_image = importlib.import_module("glm" "_ocr").process_image
        except Exception as error:
            print(f"GLM OCR unavailable ({error}). Skipping OCR patching.")
            return

        def process_node_ocr(node):
            if node.state.before and os.path.exists(node.state.before):
                return process_image(input_image_path=node.state.before)
            return None

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_node = {
                executor.submit(process_node_ocr, node): node
                for node in nodes_to_process
            }
            for future in tqdm(as_completed(future_to_node), total=len(nodes_to_process), desc="OCR Patching"):
                node = future_to_node[future]
                try:
                    ocr_data = future.result()
                    if ocr_data:
                        node.ocr_results = ocr_data
                except Exception as e:
                    print(f"Error processing OCR for a node: {e}")

    # Check how many were successfully patched
    still_missing = sum(1 for n in nodes_to_process if _is_missing_ocr(n))
    print(f"Patched {len(nodes_to_process) - still_missing}/{len(nodes_to_process)} nodes. "
          f"Still missing: {still_missing}")

    print(f"Saving patched trajectory to {traj_path}...")
    root.to_json(traj_path)
    print("Done.")


# %% Pipeline handoff
def relative_session_path(path: str | None, data_dir: str) -> str | None:
    """Rewrite an absolute screenshot path as one relative to the session dir.

    Paths that fall outside the session directory are kept verbatim.
    """
    if not path:
        return None
    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(data_dir))
    except ValueError:
        return path
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return path
    return relative


def write_trajectory_jsonl(traj_path: str, node_list: list, data_dir: str) -> str:
    """Write the action nodes as the pipeline's processed_trajectory.jsonl.

    One JSON object per line, matching the schema step 0 of the induction
    pipeline validates against.
    """
    with open(traj_path, "w", encoding="utf-8") as handle:
        index = 0
        for node in node_list:
            if not isinstance(node, ActionNode):
                continue
            index += 1
            node_time = node.time
            row = {
                "id": f"n{index}",
                "action": node.action,
                "state_before": relative_session_path(node.state.before, data_dir),
                "state_after": relative_session_path(node.state.after, data_dir),
                "time_before": node_time.before if node_time is not None else None,
                "time_after": node_time.after if node_time is not None else None,
                "time_range": node_time.range if node_time is not None else None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return traj_path


def build_processed_trajectory(
    data_dir: str,
    screenshot_dir: str | None = None,
    threads: int = 1,
    vlm: bool = False,
    ocr: bool = True,
    vlm_model: str = "openai/gpt-5-mini",
    progress_callback: ProgressCallback = None,
) -> str:
    screenshot_dir = screenshot_dir or os.path.join(data_dir, "screenshots")

    emit_progress("Loading raw trace...", 0.05, progress_callback)
    actions = load_actions(data_dir)
    emit_progress(f"Loaded {len(actions)} raw actions.", 0.1, progress_callback)

    original_actions, actions = merge_actions(actions, enable_hotkey=True)
    states = get_states(original_actions, screenshot_dir)
    time_list = measure_time_from_states(states)
    assert len(actions) == len(states) == len(time_list)
    emit_progress(f"Matched screenshots for {len(actions)} actions.", 0.25, progress_callback)

    first_idx = 0
    for i, (_, state, _) in enumerate(zip(actions, states, time_list)):
        if state["before"] is not None and state["after"] is not None:
            first_idx = i
            break
    actions = actions[first_idx:]
    states = states[first_idx:]
    time_list = time_list[first_idx:]

    last_idx = len(actions) - 1
    for i in range(len(actions) - 1, -1, -1):
        if states[i]["before"] is not None and states[i]["after"] is not None:
            last_idx = i
            break
    actions = actions[: last_idx + 1]
    states = states[: last_idx + 1]
    time_list = time_list[: last_idx + 1]
    emit_progress(
        f"Pruned trace to {len(actions)} usable actions.",
        0.35,
        progress_callback,
    )

    states = adjust_states(actions, states)
    adjusted_time_list = measure_time_from_states(states)
    for time_dict, adjusted in zip(time_list, adjusted_time_list):
        time_dict["range"] = adjusted["range"]

    node_list = []
    for action, state, time_dict in zip(actions, states, time_list):
        action_str = action["action"] if isinstance(action, dict) else action
        node_list.append(ActionNode(action=action_str, state=state, time=time_dict))
    node_list = merge_double_clicks(node_list, verbose=progress_callback is None)
    emit_progress(f"Derived {len(node_list)} trajectory nodes.", 0.45, progress_callback)

    if vlm:
        emit_progress(f"Extracting screen content with {vlm_model}...", 0.5, progress_callback)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_node = {
                executor.submit(process_node_vlm, node, vlm_model): node
                for node in node_list
            }

            total = max(len(node_list), 1)
            for index, future in enumerate(
                tqdm(
                    as_completed(future_to_node),
                    total=len(node_list),
                    desc="VLM Processing",
                    disable=progress_callback is not None,
                ),
                start=1,
            ):
                node = future_to_node[future]
                try:
                    md_content = future.result()
                    if md_content:
                        if node.ocr_results is None:
                            node.ocr_results = {}
                        node.ocr_results["vlm_md_results"] = md_content
                except Exception as error:
                    print(f"Error processing VLM for a node: {error}")

                emit_progress(
                    f"Consolidating raw data and deriving processed_trajectory.jsonl ({index}/{len(node_list)})",
                    0.5 + (0.35 * index / total),
                    progress_callback,
                )
    elif ocr:
        emit_progress("Running OCR on screenshots...", 0.5, progress_callback)
        try:
            process_image = importlib.import_module("glm" "_ocr").process_image
        except Exception as error:
            emit_progress(
                f"GLM OCR unavailable ({error}). Continuing without OCR results.",
                0.85,
                progress_callback,
            )
            process_image = None

        def process_node_ocr(node):
            if process_image is None:
                return None
            if node.state.before and os.path.exists(node.state.before):
                return process_image(input_image_path=node.state.before)
            return None

        if process_image is not None:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                future_to_node = {
                    executor.submit(process_node_ocr, node): node
                    for node in node_list
                }

                total = max(len(node_list), 1)
                for index, future in enumerate(
                    tqdm(
                        as_completed(future_to_node),
                        total=len(node_list),
                        desc="OCR Processing",
                        disable=progress_callback is not None,
                    ),
                    start=1,
                ):
                    node = future_to_node[future]
                    try:
                        ocr_data = future.result()
                        if ocr_data:
                            node.ocr_results = ocr_data
                    except Exception as error:
                        print(f"Error processing OCR for a node: {error}")

                    emit_progress(
                        f"Consolidating raw data and deriving processed_trajectory.jsonl ({index}/{len(node_list)})",
                        0.5 + (0.35 * index / total),
                        progress_callback,
                    )
    else:
        emit_progress(
            "Skipping optional OCR/VLM enrichment.",
            0.85,
            progress_callback,
        )

    bbox_path = os.path.join(screenshot_dir, "bounding_boxes.jsonl")
    if os.path.exists(bbox_path):
        emit_progress("Creating annotated screenshots...", 0.9, progress_callback)
        bbox_map = load_bounding_boxes(bbox_path)
        annotated_dir = os.path.join(data_dir, "annotated_screenshots")
        path_mapping = annotate_screenshots(
            screenshot_dir,
            annotated_dir,
            bbox_map,
            show_progress=progress_callback is None,
        )
        emit_progress(
            f"Created {len(path_mapping)} annotated screenshots.",
            0.95,
            progress_callback,
        )
    else:
        emit_progress("No bounding_boxes.jsonl found. Skipping annotations.", 0.95, progress_callback)

    root = SequenceNode(nodes=node_list)
    root.to_json(os.path.join(data_dir, "processed_trajectory.json"))

    traj_path = os.path.join(data_dir, "processed_trajectory.jsonl")
    write_trajectory_jsonl(traj_path, node_list, data_dir)
    emit_progress("processed_trajectory.jsonl is ready.", 1.0, progress_callback)
    return traj_path


def main(parsed_args):
    if parsed_args.patch:
        patch_missing_ocr(parsed_args)
        return

    build_processed_trajectory(
        data_dir=parsed_args.data_dir,
        screenshot_dir=parsed_args.screenshot_dir,
        threads=parsed_args.threads,
        vlm=parsed_args.vlm,
        ocr=not parsed_args.skip_ocr,
        vlm_model=parsed_args.vlm_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The directory containing the raw trajectory data.")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for parallel OCR/VLM processing.")
    parser.add_argument("--vlm", action="store_true", default=False,
                        help="Use VLM (openai/gpt-5-mini) to extract page content as markdown "
                             "instead of glm_ocr. Results saved to vlm_md_results.")
    parser.add_argument("--vlm_model", type=str, default="openai/gpt-5-mini",
                        help="VLM model name (default: openai/gpt-5-mini). Only used with --vlm.")
    parser.add_argument("--patch", action="store_true", default=False,
                        help="Patch mode: load existing processed_trajectory.json, "
                             "re-run OCR/VLM only on nodes with missing results, and save back.")
    parser.add_argument("--skip-ocr", action="store_true", default=False,
                        help="Skip optional OCR enrichment and only derive processed_trajectory.jsonl.")

    parsed_args = parser.parse_args()
    parsed_args.screenshot_dir = os.path.join(parsed_args.data_dir, "screenshots")
    main(parsed_args)
