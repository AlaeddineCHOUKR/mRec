"""Run manifests. Written before the work starts, finalized after it succeeds."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mrec.utils.paths import REPO_ROOT


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def git_dirty() -> bool | None:
    """Whether the tree differs from HEAD, ignoring results a run is itself writing.

    A manifest is written after its run directory exists, so counting that directory
    would make every run declare itself dirty and the flag would mean nothing. Only
    *untracked* files under a `results/` directory are excused: a tracked result that
    has been edited by hand is exactly the thing this flag should catch.
    """
    try:
        out = subprocess.run(
            # -uall so untracked files are listed individually; git otherwise
            # collapses them to the containing directory and the path test below
            # would not see the `results/` segment
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uall"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line[:2].strip(), line[3:].strip('"')
        if status == "??" and "results/" in path:
            continue
        return True
    return False


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class Manifest:
    """A json file describing how an artifact was produced.

    Written to disk on `start()` so that an interrupted run leaves evidence of what
    it was attempting, and rewritten by `finish()` with counts and wall time.
    """

    def __init__(self, path: Path, kind: str, config: Any = None):
        self.path = Path(path)
        lock = REPO_ROOT / "uv.lock"
        self.data: dict[str, Any] = {
            "kind": kind,
            "status": "running",
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "uv_lock_sha256": file_sha256(lock) if lock.exists() else None,
            "python": platform.python_version(),
            "host": platform.node(),
            "config": asdict(config) if is_dataclass(config) else config,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._t0 = time.perf_counter()

    def update(self, **fields: Any) -> None:
        self.data.update(fields)

    def start(self) -> Manifest:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        return self

    def finish(self, **fields: Any) -> None:
        self.data.update(fields)
        self.data["status"] = "complete"
        self.data["wall_seconds"] = round(time.perf_counter() - self._t0, 1)
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write()

    def _write(self) -> None:
        # `default=str` so a Path or a numpy scalar in a config never fails a run
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=False, default=str) + "\n"
        )
