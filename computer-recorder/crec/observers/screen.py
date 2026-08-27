from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import gc
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

import asyncio

# — Third-party —
import mss
import Quartz
from PIL import Image
from pynput import mouse, keyboard           # still synchronous

# — Local —
from .observer import Observer
from ..schemas import Update

###############################################################################
# Window‑geometry helpers                                                     #
###############################################################################


def _get_global_bounds() -> tuple[float, float, float, float]:
    """Return a bounding box enclosing **all** physical displays.

    Returns
    -------
    (min_x, min_y, max_x, max_y) tuple in Quartz global coordinates.
    """
    err, ids, cnt = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != Quartz.kCGErrorSuccess:  # pragma: no cover (defensive)
        raise OSError(f"CGGetActiveDisplayList failed: {err}")

    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    for did in ids[:cnt]:
        r = Quartz.CGDisplayBounds(did)
        x0, y0 = r.origin.x, r.origin.y
        x1, y1 = x0 + r.size.width, y0 + r.size.height
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    return min_x, min_y, max_x, max_y


def _rect_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> Optional[tuple[float, float, float, float]]:
    """Intersection of two (x0, y0, x1, y1) rects, or None if disjoint."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _rects_union_area(rects: List[tuple[float, float, float, float]]) -> float:
    """Total area covered by a set of (x0, y0, x1, y1) rects, counting overlaps once.

    Uses coordinate compression over the rect edges; window counts are small,
    so the O(n^3) grid scan is negligible.
    """
    if not rects:
        return 0.0

    xs = sorted({edge for rect in rects for edge in (rect[0], rect[2])})
    ys = sorted({edge for rect in rects for edge in (rect[1], rect[3])})

    area = 0.0
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            cx = (xs[xi] + xs[xi + 1]) / 2
            cy = (ys[yi] + ys[yi + 1]) / 2
            if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                area += (xs[xi + 1] - xs[xi]) * (ys[yi + 1] - ys[yi])
    return area


def _get_visible_windows() -> List[tuple[dict, float]]:
    """List *onscreen* windows with their visible‑area ratio.

    Each tuple is ``(window_info_dict, visible_ratio)`` where *visible_ratio*
    is in ``[0.0, 1.0]``.  Internal system windows (Dock, WindowServer, …) are
    ignored.
    """
    _, _, _, gmax_y = _get_global_bounds()

    opts = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListOptionIncludingWindow
    )
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)

    occupied: list[tuple[float, float, float, float]] = []  # opaque regions above the current window
    result: list[tuple[dict, float]] = []

    for info in wins:
        owner = info.get("kCGWindowOwnerName", "")
        if owner in ("Dock", "WindowServer", "Window Server"):
            continue

        bounds = info.get("kCGWindowBounds", {})
        x, y, w, h = (
            bounds.get("X", 0),
            bounds.get("Y", 0),
            bounds.get("Width", 0),
            bounds.get("Height", 0),
        )
        if w <= 0 or h <= 0:
            continue  # hidden or minimised

        inv_y = gmax_y - y - h  # Quartz→cartesian Y‑flip
        rect = (x, inv_y, x + w, inv_y + h)

        rect_area = w * h
        covered = _rects_union_area(
            [
                overlap
                for overlap in (_rect_intersection(rect, other) for other in occupied)
                if overlap is not None
            ]
        )
        visible_area = rect_area - covered
        if visible_area > 0:
            result.append((info, visible_area / rect_area))
            occupied.append(rect)

    return result


def _is_app_visible(names: Iterable[str]) -> bool:
    """Return *True* if **any** window from *names* is at least partially visible."""
    targets = set(names)
    return any(
        info.get("kCGWindowOwnerName", "") in targets and ratio > 0
        for info, ratio in _get_visible_windows()
    )

###############################################################################
# Screen observer                                                             #
###############################################################################


class Screen(Observer):
    """
    Capture before/after screenshots around user interactions.
    Blocking work (Quartz, mss, Pillow, OpenAI Vision) is executed in
    background threads via `asyncio.to_thread`.

    Keyboard events are optimized to save disk space:
    - Only the first and last screenshots are kept for consecutive key presses
    - Intermediate screenshots are automatically deleted
    - A keyboard session ends after `keyboard_timeout` seconds of inactivity
    """

    _CAPTURE_FPS: int = 5  # Reduced from 10 to 5 to reduce memory pressure
    _PERIODIC_SEC: int = 30
    _DEBOUNCE_SEC: int = 1
    _MON_START: int = 1     # first real display in mss
    _MEMORY_CLEANUP_INTERVAL: int = 30  # Force GC every 30 frames instead of 50
    _MAX_WORKERS: int = 4  # Limit thread pool size to prevent exhaustion

    # Scroll filtering constants
    _SCROLL_DEBOUNCE_SEC: float = 0.8  # Minimum time between scroll events
    _SCROLL_MIN_DISTANCE: float = 8.0  # Minimum scroll distance to log
    _SCROLL_MAX_FREQUENCY: int = 8  # Max scroll events per second
    _SCROLL_SESSION_TIMEOUT: float = 3.0  # Timeout for scroll sessions

    # ─────────────────────────────── construction
    def __init__(
        self,
        screenshots_dir: str = "~/Downloads/records/screenshots",
        skip_when_visible: Optional[str | list[str]] = None,
        history_k: int = 10,
        debug: bool = False,
        keyboard_timeout: float = 2.0,
        scroll_debounce_sec: float = 0.5,
        scroll_min_distance: float = 5.0,
        scroll_max_frequency: int = 10,
        scroll_session_timeout: float = 2.0,
    ) -> None:

        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])

        self.debug = debug

        # Custom thread pool to prevent exhaustion
        self._thread_pool = ThreadPoolExecutor(max_workers=self._MAX_WORKERS)

        # Scroll filtering configuration
        self._scroll_debounce_sec = scroll_debounce_sec
        self._scroll_min_distance = scroll_min_distance
        self._scroll_max_frequency = scroll_max_frequency
        self._scroll_session_timeout = scroll_session_timeout

        # state shared with worker
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._pending_event: Optional[dict] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._paused: bool = False

        # keyboard activity tracking
        self._key_activity_start: Optional[float] = None
        self._key_activity_timeout: float = keyboard_timeout  # seconds of inactivity to consider session ended
        self._key_screenshots: List[str] = []  # track intermediate screenshots for cleanup
        self._key_activity_lock = asyncio.Lock()

        # scroll activity tracking
        self._scroll_last_time: Optional[float] = None
        self._scroll_last_position: Optional[tuple[float, float]] = None
        self._scroll_session_start: Optional[float] = None
        self._scroll_event_count: int = 0
        self._scroll_lock = asyncio.Lock()

        # call parent
        super().__init__()

        # Adjust settings for high-DPI displays
        if self._detect_high_dpi():
            self._CAPTURE_FPS = 3  # Even lower FPS for high-DPI displays
            self._MEMORY_CLEANUP_INTERVAL = 20  # More frequent cleanup
            if self.debug:
                logging.getLogger("Screen").info("High-DPI display detected, using conservative settings")

    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a function in the custom thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, lambda: func(*args, **kwargs))

    def _detect_high_dpi(self) -> bool:
        """Detect if running on a high-DPI display and adjust settings."""
        try:
            # Check if any monitor has high resolution (likely Retina)
            with mss.mss() as sct:
                for monitor in sct.monitors[1:]:  # Skip monitor 0 (all monitors)
                    if monitor['width'] > 2560 or monitor['height'] > 1600:
                        return True
        except Exception:
            pass
        return False

    def _should_log_scroll(self, x: float, y: float, dx: float, dy: float) -> bool:
        """
        Determine if a scroll event should be logged based on filtering criteria.

        Returns True if the scroll event should be logged, False otherwise.
        """
        current_time = time.time()

        # Check if this is a new scroll session
        if (self._scroll_session_start is None or
                current_time - self._scroll_session_start > self._scroll_session_timeout):
            # Start new session
            self._scroll_session_start = current_time
            self._scroll_event_count = 0
            self._scroll_last_position = (x, y)
            self._scroll_last_time = current_time
            return True

        # Check debounce time
        if (self._scroll_last_time is not None and
                current_time - self._scroll_last_time < self._scroll_debounce_sec):
            return False

        # Check minimum distance
        if self._scroll_last_position is not None:
            distance = ((x - self._scroll_last_position[0]) ** 2 +
                        (y - self._scroll_last_position[1]) ** 2) ** 0.5
            if distance < self._scroll_min_distance:
                return False

        # Check frequency limit
        self._scroll_event_count += 1
        session_duration = current_time - self._scroll_session_start
        if session_duration > 0:
            frequency = self._scroll_event_count / session_duration
            if frequency > self._scroll_max_frequency:
                return False

        # Update tracking state
        self._scroll_last_position = (x, y)
        self._scroll_last_time = current_time

        return True

    async def _cleanup_key_screenshots(self) -> None:
        """Clean up intermediate keyboard screenshots, keeping only first and last."""
        if len(self._key_screenshots) <= 2:
            return

        # Keep first and last, delete the rest
        to_delete = self._key_screenshots[1:-1]
        self._key_screenshots = [self._key_screenshots[0], self._key_screenshots[-1]]

        for path in to_delete:
            try:
                await self._run_in_thread(os.remove, path)
                if self.debug:
                    logging.getLogger("Screen").info(f"Deleted intermediate screenshot: {path}")
            except OSError:
                pass  # File might already be deleted

    async def pause(self) -> None:
        self._paused = True
        self._pending_event = None

        async with self._scroll_lock:
            self._scroll_last_time = None
            self._scroll_last_position = None
            self._scroll_session_start = None
            self._scroll_event_count = 0

        async with self._key_activity_lock:
            if len(self._key_screenshots) > 1:
                last_path = self._key_screenshots[-1]
                final_path = last_path.replace("_intermediate", "_final")
                try:
                    await self._run_in_thread(os.rename, last_path, final_path)
                    self._key_screenshots[-1] = final_path
                except OSError:
                    pass
                await self._cleanup_key_screenshots()
            self._key_activity_start = None
            self._key_screenshots = []

    async def resume(self) -> None:
        self._paused = False

    # ─────────────────────────────── I/O helpers
    def _get_bounding_box_path(self) -> str:
        """Return the path to the bounding boxes JSONL file."""
        return os.path.join(self.screens_dir, "bounding_boxes.jsonl")

    async def _append_bounding_box(self, screenshot_path: str, x: float, y: float, frame_width: int, frame_height: int, timestamp: float) -> None:
        """Append bounding box data to the JSONL file."""
        # Calculate bounding box coordinates
        x1, x2 = max(0, x - 30), min(frame_width, x + 30)
        y1, y2 = max(0, y - 20), min(frame_height, y + 20)

        entry = {
            "screenshot_path": os.path.basename(screenshot_path),
            "x": float(x),
            "y": float(y),
            "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "frame_size": {"width": frame_width, "height": frame_height},
            "timestamp": timestamp
        }

        bbox_path = self._get_bounding_box_path()
        await self._run_in_thread(self._write_bbox_entry, bbox_path, entry)

    @staticmethod
    def _write_bbox_entry(path: str, entry: dict) -> None:
        """Write a single bounding box entry to JSONL file (thread-safe)."""
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def _save_frame(self, frame, x, y, tag: str, box_color: str = "red", box_width: int = 10) -> str:
        ts = f"{time.time():.5f}"
        path = os.path.join(self.screens_dir, f"{ts}_{tag}.jpg")
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)

        # Save raw screenshot without bounding box
        await self._run_in_thread(
            image.save,
            path,
            "JPEG",
            quality=70,
            optimize=True,
        )

        # Save bounding box data to JSONL file
        await self._append_bounding_box(path, x, y, frame.width, frame.height, float(ts))

        # Explicitly delete image object to free memory
        del image

        return path

    async def _process_and_emit(
        self, before_path: str, after_path: str | None,
        action: str | None, ev: dict | None,
    ) -> None:
        if "scroll" in action:
            # Include scroll delta information
            scroll_info = ev.get("scroll", (0, 0))
            step = f"scroll({ev['position'][0]:.1f}, {ev['position'][1]:.1f}, dx={scroll_info[0]:.2f}, dy={scroll_info[1]:.2f})"
            await self.update_queue.put(Update(content=step, content_type="input_text"))
        elif "drag" in action:
            end_pos = ev.get("end_position", (0, 0))
            step = f"{action}(start_x={ev['position'][0]:.1f}, start_y={ev['position'][1]:.1f}, end_x={end_pos[0]:.1f}, end_y={end_pos[1]:.1f})"
            await self.update_queue.put(Update(content=step, content_type="input_text"))
        elif "click" in action:
            step = f"{action}({ev['position'][0]:.1f}, {ev['position'][1]:.1f})"
            await self.update_queue.put(Update(content=step, content_type="input_text"))
        else:
            step = f"{action}({ev['text']})"
            await self.update_queue.put(Update(content=step, content_type="input_text"))

    async def stop(self) -> None:
        """Stop the observer and clean up resources."""
        await super().stop()

        # Final cleanup of any remaining keyboard session
        if self._key_activity_start is not None and len(self._key_screenshots) > 1:
            async with self._key_activity_lock:
                last_path = self._key_screenshots[-1]
                final_path = last_path.replace("_intermediate", "_final")
                try:
                    await self._run_in_thread(os.rename, last_path, final_path)
                    logging.getLogger("Screen").info(f"Final keyboard session cleanup, renamed: {final_path}")
                except OSError:
                    pass
                await self._cleanup_key_screenshots()

        # Clean up frame objects
        async with self._frame_lock:
            for frame in self._frames.values():
                if frame is not None:
                    del frame
            self._frames.clear()

        # Force garbage collection
        await self._run_in_thread(gc.collect)

        # Shutdown thread pool
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)

    # ─────────────────────────────── skip guard
    def _skip(self) -> bool:
        return _is_app_visible(self._guard) if self._guard else False

    # ─────────────────────────────── main async worker
    async def _worker(self) -> None:          # overrides base class
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [Screen] %(message)s", datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        CAP_FPS = self._CAPTURE_FPS
        PERIOD = self._PERIODIC_SEC
        DEBOUNCE = self._DEBOUNCE_SEC

        loop = asyncio.get_running_loop()

        key_event_count = 0

        # ------------------------------------------------------------------
        # All calls to mss / Quartz are wrapped in `to_thread`
        # ------------------------------------------------------------------
        with mss.mss() as sct:
            mons = sct.monitors[self._MON_START:]

            # ---- mouse callbacks (pynput is sync → schedule into loop) ----
            drag_state = {"pos": None, "btn": None, "mon": None, "before": None}
            
            def schedule_mouse_click(x: float, y: float, btn: str, prs: bool):
                if self._paused:
                    return
                if prs:
                    asyncio.run_coroutine_threadsafe(mouse_press_event(x, y, btn), loop)
                else:
                    asyncio.run_coroutine_threadsafe(mouse_release_event(x, y, btn), loop)

            def schedule_scroll_event(x: float, y: float, dx: float, dy: float):
                if self._paused:
                    return
                asyncio.run_coroutine_threadsafe(scroll_event(x, y, dx, dy), loop)

            def schedule_key_event(key, typ: str):
                if self._paused:
                    return
                asyncio.run_coroutine_threadsafe(key_event(key, typ), loop)

            mouse_listener = mouse.Listener(
                on_click=lambda x, y, btn, prs: schedule_mouse_click(x, y, btn.name, prs),
                on_scroll=lambda x, y, dx, dy: schedule_scroll_event(x, y, dx, dy),
            )
            key_listener = keyboard.Listener(
                on_press=lambda key: schedule_key_event(key, "press"),
            )
            mouse_listener.start()
            key_listener.start()

            # ---- nested helper inside the async context ----
            async def flush():
                if self._paused or self._pending_event is None:
                    return
                if self._skip():
                    self._pending_event = None
                    return

                ev = self._pending_event
                try:
                    aft = await self._run_in_thread(sct.grab, mons[ev["mon"] - 1])
                except Exception as e:
                    if self.debug:
                        logging.getLogger("Screen").error(f"Failed to capture after frame: {e}")
                    self._pending_event = None
                    return

                # Calculate scaling factor for the specific monitor
                mon_info = mons[ev["mon"] - 1]
                frame = ev["before"]
                scale_x = frame.width / mon_info["width"]
                scale_y = frame.height / mon_info["height"]
                scaled_x = ev["position"][0] * scale_x
                scaled_y = ev["position"][1] * scale_y

                if "scroll" in ev["type"]:
                    scroll_info = ev.get("scroll", (0, 0))
                    step = f"scroll({ev['position'][0]:.1f}, {ev['position'][1]:.1f}, dx={scroll_info[0]:.2f}, dy={scroll_info[1]:.2f})"
                elif "drag" in ev["type"]:
                    end_pos = ev.get("end_position", (0, 0))
                    step = f"{ev['type']}(start_x={ev['position'][0]:.1f}, start_y={ev['position'][1]:.1f}, end_x={end_pos[0]:.1f}, end_y={end_pos[1]:.1f})"
                else:
                    step = f"{ev['type']}({ev['position'][0]:.1f}, {ev['position'][1]:.1f})"

                bef_path = await self._save_frame(ev["before"], scaled_x, scaled_y, f"{step}_before")
                aft_path = await self._save_frame(aft, scaled_x, scaled_y, f"{step}_after")
                await self._process_and_emit(bef_path, aft_path, ev["type"], ev)

                log.info(f"{ev['type']} captured on monitor {ev['mon']}")
                self._pending_event = None

            # def debounce_flush():
            #     # callback from loop.call_later → must create task
            #     asyncio.create_task(flush())

            # ---- keyboard event reception ----
            async def key_event(key, typ: str):
                if self._paused:
                    return
                # Get current mouse position to determine active monitor
                x, y = mouse.Controller().position
                idx = self._mon_for(x, y, mons)
                if idx is None:
                    return

                mon = mons[idx - self._MON_START]
                x = x - mon["left"]
                y = y - mon["top"]
                log.info(f"Key {typ}: {str(key)} on monitor {idx}")

                step = f"key_{typ}({str(key)})"
                await self.update_queue.put(
                    Update(content=step, content_type="input_text")
                )

                async with self._key_activity_lock:
                    current_time = time.time()

                    # Calculate scaling factor for the specific monitor
                    frame = self._frames.get(idx)
                    # If frame is missing, we can't save it anyway, so skip or error would happen.
                    # Original code assumed existence, we'll access safely but rely on existing logic flow
                    if frame:
                        scale_x = frame.width / mon["width"]
                        scale_y = frame.height / mon["height"]
                        scaled_x = x * scale_x
                        scaled_y = y * scale_y
                    else:
                        scaled_x = x * 2
                        scaled_y = y * 2

                    # Check if this is the start of a new keyboard session
                    if (self._key_activity_start is None or
                            current_time - self._key_activity_start > self._key_activity_timeout):
                        # Start new session - save first screenshot
                        self._key_activity_start = current_time
                        self._key_screenshots = []
                        if frame:
                            screenshot_path = await self._save_frame(frame, scaled_x, scaled_y, f"{step}_first")
                            self._key_screenshots.append(screenshot_path)
                            log.info(f"Started new keyboard session, saved first screenshot: {screenshot_path}")
                    else:
                        # Continue existing session - save intermediate screenshot
                        if frame:
                            screenshot_path = await self._save_frame(frame, scaled_x, scaled_y, f"{step}_intermediate")
                            self._key_screenshots.append(screenshot_path)
                            log.info(f"Continued keyboard session, saved intermediate screenshot: {screenshot_path}")

                    # Schedule cleanup of previous intermediate screenshots
                    if len(self._key_screenshots) > 2:
                        asyncio.create_task(self._cleanup_key_screenshots())

            # ---- scroll event reception ----
            async def scroll_event(x: float, y: float, dx: float, dy: float):
                if self._paused:
                    return
                # Apply scroll filtering
                async with self._scroll_lock:
                    if not self._should_log_scroll(x, y, dx, dy):
                        if self.debug:
                            log.info(f"Scroll filtered out: dx={dx:.2f}, dy={dy:.2f}")
                        return

                idx = self._mon_for(x, y, mons)
                if idx is None:
                    return

                mon = mons[idx - self._MON_START]
                x = x - mon["left"]
                y = y - mon["top"]

                # Only log significant scroll movements
                scroll_magnitude = (dx**2 + dy**2)**0.5
                if scroll_magnitude < 1.0:  # Very small scrolls
                    if self.debug:
                        log.info(f"Scroll too small: magnitude={scroll_magnitude:.2f}")
                    return

                log.info(f"Scroll @({x:7.1f},{y:7.1f}) dx={dx:.2f} dy={dy:.2f} → mon={idx}")

                if self._skip():
                    return

                async with self._frame_lock:
                    bf = self._frames[idx]
                    if bf is None:
                        return
                    self._pending_event = {"type": "scroll", "position": (
                        x, y), "mon": idx, "before": bf, "scroll": (dx, dy)}

                # Process event immediately
                await flush()

            # ---- mouse event reception ----
            async def mouse_press_event(x: float, y: float, btn_name: str):
                if self._paused:
                    return
                idx = self._mon_for(x, y, mons)
                if idx is None:
                    return
                mon = mons[idx - self._MON_START]
                local_x = x - mon["left"]
                local_y = y - mon["top"]
                log.info(
                    f"press_{btn_name} @({local_x:7.1f},{local_y:7.1f}) → mon={idx} {'(guarded)' if self._skip() else ''}"
                )
                if self._skip():
                    return

                async with self._frame_lock:
                    bf = self._frames[idx]
                    if bf is None:
                        return
                    drag_state["pos"] = (local_x, local_y)
                    drag_state["btn"] = btn_name
                    drag_state["mon"] = idx
                    drag_state["before"] = bf

            async def mouse_release_event(x: float, y: float, btn_name: str):
                if self._paused:
                    return
                if drag_state["pos"] is None or drag_state["before"] is None:
                    return
                if btn_name != drag_state["btn"]:
                    return

                idx = self._mon_for(x, y, mons)
                if idx is None:
                    return
                mon = mons[idx - self._MON_START]
                local_x = x - mon["left"]
                local_y = y - mon["top"]

                log.info(
                    f"release_{btn_name} @({local_x:7.1f},{local_y:7.1f}) → mon={idx} {'(guarded)' if self._skip() else ''}"
                )
                if self._skip():
                    return

                start_x, start_y = drag_state["pos"]
                if idx != drag_state["mon"]:
                    dist = 0.0
                else:
                    dist = ((local_x - start_x)**2 + (local_y - start_y)**2)**0.5

                if dist > 5.0 and idx == drag_state["mon"]:
                    typ = f"drag_{btn_name}"
                    self._pending_event = {
                        "type": typ,
                        "position": (start_x, start_y),
                        "end_position": (local_x, local_y),
                        "mon": drag_state["mon"],
                        "before": drag_state["before"]
                    }
                else:
                    typ = f"click_{btn_name}"
                    self._pending_event = {
                        "type": typ,
                        "position": (start_x, start_y),
                        "mon": drag_state["mon"],
                        "before": drag_state["before"]
                    }

                drag_state["pos"] = None
                drag_state["before"] = None

                # Process event immediately instead of using debounce
                await flush()

            # ---- main capture loop ----
            log.info(f"Screen observer started — guarding {self._guard or '∅'}")
            last_periodic = time.time()
            frame_count = 0

            while self._running:                         # flag from base class
                t0 = time.time()

                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # refresh 'before' buffers
                for idx, m in enumerate(mons, 1):
                    old_frame = None
                    async with self._frame_lock:
                        old_frame = self._frames.get(idx)

                    # Capture new frame using custom thread pool
                    try:
                        frame = await self._run_in_thread(sct.grab, m)
                    except Exception as e:
                        # Always log capture errors as they indicate permission issues or broken state
                        logging.getLogger("Screen").error(f"Failed to capture frame: {e}")
                        continue

                    async with self._frame_lock:
                        self._frames[idx] = frame

                    # Explicitly delete old frame to free memory
                    if old_frame is not None:
                        del old_frame

                    frame_count += 1

                    # Force garbage collection every 30 frames to prevent memory buildup
                    if frame_count % self._MEMORY_CLEANUP_INTERVAL == 0:
                        await self._run_in_thread(gc.collect)

                # Check for keyboard session timeout
                current_time = time.time()
                if (self._key_activity_start is not None and
                    current_time - self._key_activity_start > self._key_activity_timeout and
                        len(self._key_screenshots) > 1):
                    # Session ended - rename last screenshot to indicate it's the final one
                    async with self._key_activity_lock:
                        if len(self._key_screenshots) > 1:
                            last_path = self._key_screenshots[-1]
                            final_path = last_path.replace("_intermediate", "_final")
                            try:
                                await self._run_in_thread(os.rename, last_path, final_path)
                                self._key_screenshots[-1] = final_path
                                log.info(f"Keyboard session ended, renamed final screenshot: {final_path}")
                            except OSError:
                                pass
                        self._key_activity_start = None
                        self._key_screenshots = []

                # fps throttle
                dt = time.time() - t0
                await asyncio.sleep(max(0, (1 / CAP_FPS) - dt))

            # shutdown
            mouse_listener.stop()
            key_listener.stop()

            # Final cleanup of any remaining keyboard session
            if self._key_activity_start is not None and len(self._key_screenshots) > 1:
                async with self._key_activity_lock:
                    last_path = self._key_screenshots[-1]
                    final_path = last_path.replace("_intermediate", "_final")
                    try:
                        await self._run_in_thread(os.rename, last_path, final_path)
                        log.info(f"Final keyboard session cleanup, renamed: {final_path}")
                    except OSError:
                        pass
                    await self._cleanup_key_screenshots()

            # if self._debounce_handle:
            #     self._debounce_handle.cancel()
