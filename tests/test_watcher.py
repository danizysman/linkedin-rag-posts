"""Tests for the debounced data/ JSON watcher and lockfile."""

from __future__ import annotations

import fcntl
from pathlib import Path

from scripts.watch_data import is_valid_json_file, run_ingest_with_lock, watch_loop


def test_is_valid_json_file(tmp_path: Path) -> None:
    valid = tmp_path / "export.json"
    valid.write_text("[]", encoding="utf-8")
    partial = tmp_path / "export.json.crdownload"
    partial.write_text("[]", encoding="utf-8")
    hidden = tmp_path / ".hidden.json"
    hidden.write_text("[]", encoding="utf-8")
    other = tmp_path / "notes.txt"
    other.write_text("hi", encoding="utf-8")

    assert is_valid_json_file(valid) is True
    assert is_valid_json_file(partial) is False
    assert is_valid_json_file(hidden) is False
    assert is_valid_json_file(other) is False


def test_run_ingest_with_lock_skips_when_locked(tmp_path: Path) -> None:
    lock_file = tmp_path / "ingest.lock"
    calls: list[Path] = []

    with lock_file.open("w", encoding="utf-8") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            ran = run_ingest_with_lock(
                tmp_path,
                lock_file=lock_file,
                runner=lambda p: calls.append(p) or 0,
            )
            assert ran is False
            assert calls == []
        finally:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    ran_after = run_ingest_with_lock(
        tmp_path,
        lock_file=lock_file,
        runner=lambda p: calls.append(p) or 0,
    )
    assert ran_after is True
    assert calls == [tmp_path]


def test_watch_loop_debounces_and_triggers_ingest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lock_file = tmp_path / "ingest.lock"
    calls: list[Path] = []

    # Pre-existing file shouldn't trigger initial ingest; write new file after start
    watch_loop(
        watch_dir=data_dir,
        poll_interval=0.01,
        debounce_seconds=0.02,
        lock_file=lock_file,
        runner=lambda p: calls.append(p) or 0,
        max_iterations=2,
    )
    assert calls == []

    (data_dir / "new.json").write_text('[{"post_urn": "urn:1"}]', encoding="utf-8")
    watch_loop(
        watch_dir=data_dir,
        poll_interval=0.02,
        debounce_seconds=0.02,
        lock_file=lock_file,
        runner=lambda p: calls.append(p) or 0,
        max_iterations=4,
        initial_snapshot={},
    )
    assert calls == [data_dir]
