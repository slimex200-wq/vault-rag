"""Tests for VaultWatcher (file change detection)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.config import VaultConfig
from vault_rag.engine.watcher import VaultWatcher, _NoteEventHandler

# ---------------------------------------------------------------------------
# _NoteEventHandler unit tests
# ---------------------------------------------------------------------------


def test_handler_fires_on_md_file() -> None:
    """on_change callback is invoked for .md file creation."""
    called = threading.Event()
    received_path: list[Path] = []

    def on_change(p: Path) -> None:
        received_path.append(p)
        called.set()

    handler = _NoteEventHandler(
        on_change=on_change,
        excluded_dirs=(".obsidian",),
        debounce_seconds=0.1,
    )

    event = MagicMock()
    event.is_directory = False
    event.src_path = "/vault/notes/test.md"
    handler.on_created(event)

    assert called.wait(timeout=2.0), "Callback was not fired"
    assert received_path[0] == Path("/vault/notes/test.md")
    handler.cancel_all()


def test_handler_ignores_non_md_files() -> None:
    """Non-.md files should not trigger callback."""
    called = threading.Event()

    handler = _NoteEventHandler(
        on_change=lambda p: called.set(),
        excluded_dirs=(),
        debounce_seconds=0.1,
    )

    event = MagicMock()
    event.is_directory = False
    event.src_path = "/vault/notes/image.png"
    handler.on_created(event)

    assert not called.wait(timeout=0.5), "Callback should not fire for non-.md"
    handler.cancel_all()


def test_handler_ignores_excluded_dirs() -> None:
    """Files in excluded directories should not trigger callback."""
    called = threading.Event()

    handler = _NoteEventHandler(
        on_change=lambda p: called.set(),
        excluded_dirs=(".obsidian", "_trash"),
        debounce_seconds=0.1,
    )

    event = MagicMock()
    event.is_directory = False
    event.src_path = "/vault/.obsidian/config.md"
    handler.on_modified(event)

    assert not called.wait(timeout=0.5), "Callback should not fire for excluded dir"
    handler.cancel_all()


def test_handler_debounce() -> None:
    """Rapid successive changes to the same file fire callback only once."""
    call_count: list[int] = [0]
    done = threading.Event()

    def on_change(p: Path) -> None:
        call_count[0] += 1
        done.set()

    handler = _NoteEventHandler(
        on_change=on_change,
        excluded_dirs=(),
        debounce_seconds=0.3,
    )

    event = MagicMock()
    event.is_directory = False
    event.src_path = "/vault/rapid.md"

    # Fire 5 rapid events
    for _ in range(5):
        handler.on_modified(event)
        time.sleep(0.05)

    assert done.wait(timeout=2.0), "Callback was not fired"
    # Allow extra time for any spurious fires
    time.sleep(0.5)
    assert call_count[0] == 1, f"Expected 1 call, got {call_count[0]}"
    handler.cancel_all()


def test_handler_ignores_directory_events() -> None:
    """Directory events should be ignored."""
    called = threading.Event()

    handler = _NoteEventHandler(
        on_change=lambda p: called.set(),
        excluded_dirs=(),
        debounce_seconds=0.1,
    )

    event = MagicMock()
    event.is_directory = True
    event.src_path = "/vault/new_folder"
    handler.on_created(event)

    assert not called.wait(timeout=0.3)
    handler.cancel_all()


# ---------------------------------------------------------------------------
# VaultWatcher lifecycle
# ---------------------------------------------------------------------------


def test_watcher_stop(tmp_path: Path) -> None:
    """Watcher can be created and stopped without errors."""
    config = VaultConfig(vault_path=tmp_path)
    watcher = VaultWatcher(config=config, on_change=lambda p: None, debounce_seconds=0.1)

    # Start in a thread, then stop
    t = threading.Thread(target=watcher.start, daemon=True)
    t.start()
    time.sleep(0.3)
    watcher.stop()
    t.join(timeout=3.0)
    assert not t.is_alive()
