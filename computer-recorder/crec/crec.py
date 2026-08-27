from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable

from .observers import Observer
from .schemas import Update


class crec:
    def __init__(
        self,
        user_name: str,
        *observers: Observer,
        data_directory: str = "~/Downloads/records",
        trace_name: str = "raw_trace.jsonl",
        max_concurrent_updates: int = 4,
        verbosity: int = logging.WARNING,
    ):
        data_directory = os.path.expanduser(data_directory)
        os.makedirs(data_directory, exist_ok=True)

        self.user_name = user_name
        self.observers: list[Observer] = list(observers)

        self.logger = logging.getLogger("crec")
        self.logger.setLevel(verbosity)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self._data_directory = data_directory
        self._trace_path = os.path.join(self._data_directory, trace_name)
        self._write_lock = asyncio.Lock()
        self._update_sem = asyncio.Semaphore(max_concurrent_updates)
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: asyncio.Task | None = None
        self.update_handlers: list[Callable[[Observer, Update], None]] = []

    def start_update_loop(self):
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._update_loop())

    async def stop_update_loop(self):
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def __aenter__(self):
        self.start_update_loop()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop_update_loop()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        for obs in self.observers:
            await obs.stop()

    async def pause(self) -> None:
        for obs in self.observers:
            await obs.pause()

    async def resume(self) -> None:
        for obs in self.observers:
            await obs.resume()

    async def _update_loop(self):
        while True:
            gets = {
                asyncio.create_task(obs.update_queue.get()): obs
                for obs in self.observers
            }

            done, _ = await asyncio.wait(gets.keys(), return_when=asyncio.FIRST_COMPLETED)

            for future in done:
                update: Update = future.result()
                observer = gets[future]
                task = asyncio.create_task(self._run_with_gate(observer, update))
                self._tasks.add(task)

    async def _run_with_gate(self, observer: Observer, update: Update):
        async with self._update_sem:
            try:
                await self._default_handler(observer, update)
            finally:
                self._tasks.discard(asyncio.current_task())

    @staticmethod
    def _write_trace_entry(path: str, entry: dict) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    async def _append_trace_entry(self, observer: Observer, update: Update) -> None:
        entry = {
            "observer_name": observer.name,
            "content": update.content,
            "content_type": update.content_type,
            "timestamp": time.time(),
        }
        async with self._write_lock:
            await asyncio.to_thread(self._write_trace_entry, self._trace_path, entry)

    async def _default_handler(self, observer: Observer, update: Update) -> None:
        self.logger.info(f"Processing update from {observer.name}")
        self.logger.info(f"Content ({update.content_type}): {update.content[:10]}")
        await self._append_trace_entry(observer, update)

    def add_observer(self, observer: Observer):
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def register_update_handler(self, fn: Callable[[Observer, Update], None]):
        self.update_handlers.append(fn)
