"""Filesystem layout of the repository. One source of truth for every path."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MREC_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROC_DIR = DATA_ROOT / "proc"

TARBALL_NAME = "deezer-recsys25.tar.gz"
CHECKSUM_NAME = "deezer-recsys25.checksums.json"
EXTRACT_DIR = RAW_DIR / "deezer-recsys25"
