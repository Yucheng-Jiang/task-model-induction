import argparse
import asyncio
import ctypes
import json
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import threading
from ctypes import util
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
backend_lib_dir = os.path.join(current_dir, "backend_lib")

for import_path in (parent_dir, backend_lib_dir):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


_event_sink = None


def set_event_sink(sink) -> None:
    """Route backend events to `sink` instead of printing JSON lines.

    The Electron app reads JSON lines off stdout, so that stays the default.
    The command-line front end (../record.py) installs a sink to render the
    same events as prose without forking this file's logic.
    """
    global _event_sink
    _event_sink = sink


def emit_json(payload: dict) -> None:
    if _event_sink is not None:
        _event_sink(payload)
        return
    print(json.dumps(payload))
    sys.stdout.flush()


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def consolidate_session(full_path: str) -> str:
    from parse_raw_trace import build_processed_trajectory

    def progress_callback(message: str, progress: float | None = None) -> None:
        payload = {
            "type": "consolidation-progress",
            "message": message,
        }
        if progress is not None:
            payload["progress"] = progress
        emit_json(payload)

    return build_processed_trajectory(
        data_dir=full_path,
        screenshot_dir=os.path.join(full_path, "screenshots"),
        threads=1,
        ocr=False,
        progress_callback=progress_callback,
    )


def get_recorded_duration_seconds(full_path: str) -> int:
    """Elapsed recording time of an existing session, from raw trace timestamps."""
    trace_path = os.path.join(full_path, "raw_trace.jsonl")
    first_ts = None
    last_ts = None
    try:
        with open(trace_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    timestamp = json.loads(line).get("timestamp")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not isinstance(timestamp, (int, float)):
                    continue
                if first_ts is None:
                    first_ts = timestamp
                last_ts = timestamp
    except OSError:
        return 0

    if first_ts is None or last_ts is None or last_ts <= first_ts:
        return 0
    return int(last_ts - first_ts)


def zip_session(full_path: str) -> str:
    session_dir = full_path.rstrip(os.sep)
    archive_path = shutil.make_archive(
        session_dir,
        "zip",
        root_dir=os.path.dirname(session_dir),
        base_dir=os.path.basename(session_dir),
    )
    shutil.rmtree(session_dir)
    return archive_path


def get_video_segments_dir(full_path: str) -> str:
    return os.path.join(full_path, "raw_video_segments")


def get_video_segment_path(full_path: str, segment_index: int) -> str:
    return os.path.join(get_video_segments_dir(full_path), f"segment_{segment_index:03d}.mp4")


async def start_video_recording(full_path: str, segment_index: int) -> tuple[asyncio.subprocess.Process, str]:
    os.makedirs(get_video_segments_dir(full_path), exist_ok=True)
    video_path = get_video_segment_path(full_path, segment_index)
    if segment_index == 1:
        emit_json({"type": "status", "message": f"Starting raw video recording to {video_path}"})
    else:
        emit_json({"type": "status", "message": f"Resuming raw video recording with {video_path}"})
    # screencapture is chatty while recording; piping its output without reading
    # can stall the recorder and make the captured video appear frozen.
    proc = await asyncio.create_subprocess_exec(
        "screencapture",
        "-x",
        "-v",
        video_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return proc, video_path


async def stop_video_recording(video_proc: asyncio.subprocess.Process, reason: str) -> None:
    if video_proc.returncode is not None:
        return

    if reason == "pause":
        emit_json({"type": "status", "message": "Pausing raw video recording..."})
    else:
        emit_json({"type": "status", "message": "Saving raw video recording..."})

    try:
        video_proc.send_signal(signal.SIGINT)
        await asyncio.wait_for(video_proc.wait(), timeout=3.0)
        return
    except ProcessLookupError:
        return
    except asyncio.TimeoutError:
        emit_json({"type": "status", "message": "Raw video is slow to stop, forcing finalization..."})
    except Exception as error:
        emit_json({"type": "error", "message": f"Error stopping video cleanly: {error}"})

    for action, timeout in ((video_proc.terminate, 1.5), (video_proc.kill, 1.0)):
        if video_proc.returncode is not None:
            return

        try:
            action()
            await asyncio.wait_for(video_proc.wait(), timeout=timeout)
            return
        except ProcessLookupError:
            return
        except asyncio.TimeoutError:
            continue
        except Exception as error:
            emit_json({"type": "error", "message": f"Error forcing video recorder shutdown: {error}"})
            return

    if video_proc.returncode is None:
        emit_json({"type": "error", "message": "Raw video recorder did not stop before timeout."})


def write_video_manifest(full_path: str, segment_paths: list[str]) -> str:
    manifest_path = os.path.join(full_path, "raw_video_segments.json")
    payload = {
        "segment_count": len(segment_paths),
        "segments": [os.path.basename(path) for path in segment_paths],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path


def merge_video_segments(segment_paths: list[str], output_path: str) -> None:
    from AVFoundation import (
        AVAssetExportPresetPassthrough,
        AVAssetExportSession,
        AVAssetExportSessionStatusCompleted,
        AVFileTypeMPEG4,
        AVMediaTypeAudio,
        AVMediaTypeVideo,
        AVMutableComposition,
        AVURLAsset,
    )
    from CoreMedia import CMTimeAdd, CMTimeRangeMake, kCMTimeZero
    from Foundation import NSURL

    composition = AVMutableComposition.alloc().init()
    video_track = composition.addMutableTrackWithMediaType_preferredTrackID_(AVMediaTypeVideo, 0)
    audio_track = composition.addMutableTrackWithMediaType_preferredTrackID_(AVMediaTypeAudio, 0)
    cursor = kCMTimeZero

    for segment_path in segment_paths:
        asset = AVURLAsset.URLAssetWithURL_options_(NSURL.fileURLWithPath_(segment_path), None)
        duration = asset.duration()
        time_range = CMTimeRangeMake(kCMTimeZero, duration)

        source_video_tracks = asset.tracksWithMediaType_(AVMediaTypeVideo)
        if source_video_tracks:
            ok, error = video_track.insertTimeRange_ofTrack_atTime_error_(
                time_range,
                source_video_tracks[0],
                cursor,
                None,
            )
            if not ok:
                raise RuntimeError(f"Could not append video track for {segment_path}: {error}")

        source_audio_tracks = asset.tracksWithMediaType_(AVMediaTypeAudio)
        if source_audio_tracks:
            ok, error = audio_track.insertTimeRange_ofTrack_atTime_error_(
                time_range,
                source_audio_tracks[0],
                cursor,
                None,
            )
            if not ok:
                raise RuntimeError(f"Could not append audio track for {segment_path}: {error}")

        cursor = CMTimeAdd(cursor, duration)

    if os.path.exists(output_path):
        os.remove(output_path)

    exporter = AVAssetExportSession.alloc().initWithAsset_presetName_(composition, AVAssetExportPresetPassthrough)
    exporter.setOutputURL_(NSURL.fileURLWithPath_(output_path))
    exporter.setOutputFileType_(AVFileTypeMPEG4)

    done = threading.Event()
    exporter.exportAsynchronouslyWithCompletionHandler_(done.set)
    done.wait()

    if exporter.status() != AVAssetExportSessionStatusCompleted:
        raise RuntimeError(str(exporter.error() or "Unknown AVFoundation export failure"))


def finalize_video_segments(full_path: str, segment_paths: list[str]) -> tuple[str | None, str | None]:
    valid_segments = [path for path in segment_paths if os.path.exists(path) and os.path.getsize(path) > 0]
    if not valid_segments:
        return None, None

    output_path = os.path.join(full_path, "raw_video.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)
    if len(valid_segments) == 1:
        shutil.move(valid_segments[0], output_path)
        shutil.rmtree(get_video_segments_dir(full_path), ignore_errors=True)
        return output_path, None

    manifest_path = write_video_manifest(full_path, valid_segments)
    try:
        merge_video_segments(valid_segments, output_path)
    except Exception:
        return None, manifest_path

    shutil.rmtree(get_video_segments_dir(full_path), ignore_errors=True)
    try:
        os.remove(manifest_path)
    except OSError:
        pass
    return output_path, None


async def run_recording(
    session_name,
    base_path,
    debug=False,
    scroll_debounce=0.5,
    scroll_min_dist=5.0,
    scroll_max_freq=10,
    scroll_timeout=2.0,
    record_video=False,
):
    from crec.crec import crec
    from crec.observers import Screen

    if not base_path:
        base_path = os.path.expanduser("~/Downloads/recorder_sessions")

    full_path = os.path.join(base_path, session_name)
    os.makedirs(full_path, exist_ok=True)
    previous_duration = get_recorded_duration_seconds(full_path)
    emit_json({"type": "status", "message": f"Initializing recording in {full_path}"})

    screenshots_dir = os.path.join(full_path, "screenshots")
    screen_observer = Screen(
        screenshots_dir=screenshots_dir,
        debug=debug,
        scroll_debounce_sec=scroll_debounce,
        scroll_min_distance=scroll_min_dist,
        scroll_max_frequency=scroll_max_freq,
        scroll_session_timeout=scroll_timeout,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    resume_event = asyncio.Event()

    def stop_handler():
        emit_json({"type": "status", "message": "Signal received, stopping..."})
        stop_event.set()

    def pause_handler():
        pause_event.set()

    def resume_handler():
        resume_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_handler)

    if hasattr(signal, "SIGUSR1"):
        loop.add_signal_handler(signal.SIGUSR1, pause_handler)
    if hasattr(signal, "SIGUSR2"):
        loop.add_signal_handler(signal.SIGUSR2, resume_handler)

    video_proc = None
    video_segment_paths: list[str] = []
    next_video_segment_index = 1
    if record_video:
        video_proc, segment_path = await start_video_recording(full_path, next_video_segment_index)
        video_segment_paths.append(segment_path)
        next_video_segment_index += 1

    try:
        async with crec(
            "user",
            screen_observer,
            data_directory=full_path,
            verbosity=logging.DEBUG if debug else logging.INFO,
        ) as recorder:
            emit_json({
                "type": "started",
                "path": full_path,
                "resumedDurationSeconds": previous_duration,
            })
            paused = False

            while True:
                if stop_event.is_set():
                    emit_json({"type": "stopping"})
                    break

                wait_tasks = [asyncio.create_task(stop_event.wait())]
                if paused:
                    wait_tasks.append(asyncio.create_task(resume_event.wait()))
                else:
                    wait_tasks.append(asyncio.create_task(pause_event.wait()))

                done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()

                if stop_event.is_set():
                    emit_json({"type": "stopping"})
                    break

                if not paused and pause_event.is_set():
                    pause_event.clear()
                    await recorder.pause()
                    if video_proc:
                        await stop_video_recording(video_proc, "pause")
                        video_proc = None
                    paused = True
                    emit_json({"type": "status", "message": "Recording paused."})
                    continue

                if paused and resume_event.is_set():
                    resume_event.clear()
                    await recorder.resume()
                    if record_video:
                        video_proc, segment_path = await start_video_recording(full_path, next_video_segment_index)
                        video_segment_paths.append(segment_path)
                        next_video_segment_index += 1
                    paused = False
                    emit_json({"type": "status", "message": "Recording resumed."})
    except Exception as error:
        emit_json({"type": "error", "message": str(error)})
        if video_proc:
            await stop_video_recording(video_proc, "stop")
        sys.exit(1)

    if video_proc:
        await stop_video_recording(video_proc, "stop")

    if video_segment_paths:
        try:
            raw_video_path, manifest_path = await asyncio.to_thread(finalize_video_segments, full_path, video_segment_paths)
            if raw_video_path:
                emit_json({"type": "status", "message": f"Raw video saved to {raw_video_path}"})
            elif manifest_path:
                emit_json({
                    "type": "status",
                    "message": f"Raw video was saved as paused segments listed in {manifest_path}",
                })
        except Exception as error:
            emit_json({"type": "error", "message": f"Error finalizing raw video: {error}"})

    try:
        emit_json({
            "type": "consolidation-progress",
            "message": "Consolidating raw data and deriving processed_trajectory.jsonl",
            "progress": 0.0,
        })
        processed_path = await asyncio.to_thread(consolidate_session, full_path)
    except Exception as error:
        emit_json({
            "type": "error",
            "message": f"Recording saved in {full_path}, but consolidation failed: {error}",
        })
        sys.exit(1)

    archive_path = None
    try:
        emit_json({
            "type": "consolidation-progress",
            "message": "Compressing session into a zip archive",
            "progress": 0.97,
        })
        archive_path = await asyncio.to_thread(zip_session, full_path)
        emit_json({"type": "status", "message": f"Session archived to {archive_path}"})
    except Exception as error:
        emit_json({
            "type": "status",
            "message": f"Could not zip the session, keeping the folder instead: {error}",
        })

    emit_json({
        "type": "finished",
        "path": archive_path or full_path,
        "processedPath": archive_path or processed_path,
    })


def main():
    # PyInstaller's multiprocessing hook expects frozen children to call
    # freeze_support() before argparse sees the synthetic interpreter flags.
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Computer Recorder Launcher")
    parser.add_argument("--session-name", type=str, default=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--base-path", type=str, help="Base directory for sessions")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-permissions", action="store_true", help="Check input monitoring permissions and exit")
    parser.add_argument("--record-video", action="store_true", help="Record raw screen video")
    parser.add_argument("--scroll-debounce", type=float, default=0.5)
    parser.add_argument("--scroll-min-distance", type=float, default=5.0)
    parser.add_argument("--scroll-max-frequency", type=int, default=10)
    parser.add_argument("--scroll-session-timeout", type=float, default=2.0)

    args = parser.parse_args()

    if args.check_permissions:
        try:
            app_services_path = util.find_library("ApplicationServices")
            result = False
            if app_services_path:
                core_graphics = ctypes.cdll.LoadLibrary(app_services_path)
                if hasattr(core_graphics, "CGPreflightListenEventAccess"):
                    result = bool(core_graphics.CGPreflightListenEventAccess())

            emit_json({"input_monitoring": result})
            sys.exit(0)
        except Exception as error:
            emit_json({"error": str(error), "input_monitoring": False})
            sys.exit(1)

    setup_logging(args.debug)

    try:
        asyncio.run(
            run_recording(
                args.session_name,
                args.base_path,
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


if __name__ == "__main__":
    main()
