"""Debounced directory watcher that triggers ingestion when new JSON exports land."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

IGNORED_SUFFIXES = (".crdownload", ".part", ".tmp", ".download", ".swp")
DEFAULT_LOCK_FILE = Path("/tmp/linkedin-saved-rag-ingest.lock")


def is_valid_json_file(path: Path) -> bool:
    """Return True if path is a completed, non-temporary .json file."""
    name = path.name
    if not path.is_file():
        return False
    if name.startswith(".") or name.endswith(IGNORED_SUFFIXES):
        return False
    return path.suffix.lower() == ".json"


def snapshot_directory(watch_dir: Path) -> dict[Path, tuple[int, int]]:
    """Return {path: (size, mtime_ns)} for valid JSON files in watch_dir."""
    if not watch_dir.exists():
        return {}
    snap: dict[Path, tuple[int, int]] = {}
    for entry in sorted(watch_dir.iterdir()):
        if is_valid_json_file(entry):
            stat = entry.stat()
            snap[entry] = (stat.st_size, stat.st_mtime_ns)
    return snap


def parse_ingest_summary(stdout: str) -> dict[str, int] | None:
    """Extract the JSON summary dict printed by src.ingest from stdout."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and {"ingested", "skipped", "failed"} <= data.keys():
            return data
    return None


def send_notification(title: str, message: str) -> None:
    """Send a native macOS Notification Center alert via osascript."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                "on run argv",
                "-e",
                "display notification (item 2 of argv) with title (item 1 of argv)",
                "-e",
                "end run",
                title,
                message,
            ],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        logger.debug("Failed to send macOS notification: %s", exc)


def run_ingest_with_lock(
    watch_dir: Path,
    lock_file: Path = DEFAULT_LOCK_FILE,
    runner: Callable[[Path], int] | None = None,
    notify: bool = True,
) -> bool:
    """Acquire non-blocking lockfile and run ingestion. Returns False if already locked."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w", encoding="utf-8") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Ingestion already running (lock held on %s); skipping.", lock_file)
            return False

        summary: dict[str, int] | None = None
        try:
            if runner is not None:
                code = runner(watch_dir)
            else:
                proc = subprocess.run(
                    ["uv", "run", "--no-sync", "python", "-m", "src.ingest", str(watch_dir)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                code = proc.returncode
                if proc.stdout:
                    logger.info("%s", proc.stdout.strip())
                if proc.stderr:
                    logger.info("%s", proc.stderr.strip())
                summary = parse_ingest_summary(proc.stdout or "")
            if code != 0:
                logger.warning("Ingestion exited with status %d", code)
            if notify:
                if code == 0:
                    msg = (
                        f"Ingested {summary['ingested']}, "
                        f"skipped {summary['skipped']}, "
                        f"failed {summary['failed']}"
                        if summary
                        else "Ingestion complete"
                    )
                    send_notification("LinkedIn Saved RAG", msg)
                else:
                    send_notification(
                        "LinkedIn Saved RAG (Error)", f"Ingestion failed (exit {code})"
                    )
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    return True


def watch_loop(
    watch_dir: Path,
    poll_interval: float = 1.0,
    debounce_seconds: float = 2.0,
    lock_file: Path = DEFAULT_LOCK_FILE,
    runner: Callable[[Path], int] | None = None,
    max_iterations: int | None = None,
    initial_snapshot: dict[Path, tuple[int, int]] | None = None,
    notify: bool = True,
) -> None:
    """Poll watch_dir for new/updated JSON files, debounce until stable, and trigger ingest."""
    last_ingested = (
        snapshot_directory(watch_dir) if initial_snapshot is None else dict(initial_snapshot)
    )
    pending_snapshot: dict[Path, tuple[int, int]] | None = None
    last_change_ts = 0.0
    iterations = 0

    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        current = snapshot_directory(watch_dir)
        now = time.monotonic()

        if current != last_ingested:
            if pending_snapshot != current:
                pending_snapshot = current
                last_change_ts = now
            elif now - last_change_ts >= debounce_seconds:
                logger.info("Detected stable JSON changes in %s; starting ingestion.", watch_dir)
                if run_ingest_with_lock(
                    watch_dir,
                    lock_file=lock_file,
                    runner=runner,
                    notify=notify,
                ):
                    last_ingested = current
                    pending_snapshot = None
        else:
            pending_snapshot = None

        if max_iterations is None or iterations < max_iterations:
            time.sleep(poll_interval)


def main() -> None:
    """CLI entrypoint for the data/ JSON watcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Watch data/ for new LinkedIn JSON exports.")
    parser.add_argument("--dir", type=Path, default=Path("data"), help="Directory to watch")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds")
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Wait time after last file change before ingesting",
    )
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE, help="Lockfile path")
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable macOS Notification Center alerts",
    )
    args = parser.parse_args()

    watch_loop(
        watch_dir=args.dir,
        poll_interval=args.interval,
        debounce_seconds=args.debounce,
        lock_file=args.lock_file,
        notify=not args.no_notify,
    )


if __name__ == "__main__":
    main()
