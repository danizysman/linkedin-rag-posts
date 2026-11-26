"""Tests for the debounced data/ JSON watcher, lockfile, and notifications."""

from __future__ import annotations

import fcntl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.watch_data import (
    is_valid_json_file,
    parse_ingest_summary,
    run_ingest_with_lock,
    watch_loop,
)


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


def test_parse_ingest_summary() -> None:
    stdout = '2026-11-25 INFO: Upserted\n{"ingested": 7, "skipped": 5820, "failed": 0}\n'
    assert parse_ingest_summary(stdout) == {"ingested": 7, "skipped": 5820, "failed": 0}
    assert parse_ingest_summary("no json here") is None


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
                notify=False,
            )
            assert ran is False
            assert calls == []
        finally:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    ran_after = run_ingest_with_lock(
        tmp_path,
        lock_file=lock_file,
        runner=lambda p: calls.append(p) or 0,
        notify=False,
    )
    assert ran_after is True
    assert calls == [tmp_path]


def test_run_ingest_with_lock_sends_notification(tmp_path: Path) -> None:
    lock_file = tmp_path / "ingest.lock"
    fake_proc = SimpleNamespace(
        returncode=0,
        stdout='{"ingested": 3, "skipped": 10, "failed": 0}\n',
        stderr="",
    )

    with (
        patch("scripts.watch_data.subprocess.run", return_value=fake_proc),
        patch("scripts.watch_data.send_notification") as mock_notify,
    ):
        ran = run_ingest_with_lock(tmp_path, lock_file=lock_file, notify=True)

    assert ran is True
    mock_notify.assert_called_once_with(
        "LinkedIn Saved RAG",
        "Ingested 3, skipped 10, failed 0",
    )


def test_watch_loop_debounces_and_triggers_ingest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lock_file = tmp_path / "ingest.lock"
    calls: list[Path] = []

    watch_loop(
        watch_dir=data_dir,
        poll_interval=0.01,
        debounce_seconds=0.02,
        lock_file=lock_file,
        runner=lambda p: calls.append(p) or 0,
        max_iterations=2,
        notify=False,
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
        notify=False,
    )
    assert calls == [data_dir]
