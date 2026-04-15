"""VaultWatcher: auto-compile + index on file changes via watchdog."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from vault_rag.config import VaultConfig

logger = logging.getLogger(__name__)


class _NoteEventHandler(FileSystemEventHandler):
    """Filter .md file events, debounce, and fire callback."""

    def __init__(
        self,
        on_change: Callable[[Path], None],
        excluded_dirs: tuple[str, ...],
        debounce_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self._on_change = on_change
        self._excluded_dirs = excluded_dirs
        self._debounce = debounce_seconds
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _is_excluded(self, path: str) -> bool:
        parts = Path(path).parts
        return any(ex in parts for ex in self._excluded_dirs)

    def _schedule(self, path: str) -> None:
        if not path.endswith(".md") or self._is_excluded(path):
            return

        with self._lock:
            existing = self._pending.pop(path, None)
            if existing is not None:
                existing.cancel()

            timer = threading.Timer(self._debounce, self._fire, args=(path,))
            self._pending[path] = timer
            timer.start()

    def _fire(self, path: str) -> None:
        with self._lock:
            self._pending.pop(path, None)

        logger.info("Change detected: %s", path)
        self._on_change(Path(path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def cancel_all(self) -> None:
        """Cancel all pending timers."""
        with self._lock:
            for timer in self._pending.values():
                timer.cancel()
            self._pending.clear()


class VaultWatcher:
    """Watch vault directory for .md changes and trigger callbacks."""

    def __init__(
        self,
        config: VaultConfig,
        on_change: Callable[[Path], None],
        debounce_seconds: float = 2.0,
    ) -> None:
        self._config = config
        self._handler = _NoteEventHandler(
            on_change=on_change,
            excluded_dirs=config.excluded_dirs,
            debounce_seconds=debounce_seconds,
        )
        self._observer = Observer()

    def start(self) -> None:
        """Start watching (blocking). Ctrl+C to stop."""
        self._observer.schedule(
            self._handler,
            str(self._config.vault_path),
            recursive=True,
        )
        self._observer.start()
        try:
            self._observer.join()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop watching and clean up."""
        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join(timeout=5)
