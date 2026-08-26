"""Run directories: one per evaluation, holding everything needed to trust its numbers."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from mrec.eval.protocol import EVAL_SEEDS, N_EVAL_USERS, SEQLEN, K
from mrec.utils.paths import PROC_DIR, REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class RunConfig:
    model: str
    split: str = "test"
    k: int = K
    seqlen: int = SEQLEN
    n_users: int = N_EVAL_USERS
    seeds: tuple[int, ...] = EVAL_SEEDS
    batch_size: int = 100
    proc_dir: str = str(PROC_DIR)
    results_dir: str = str(RESULTS_DIR)
    model_params: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        overlay = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(overlay) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        if "seeds" in overlay:
            overlay["seeds"] = tuple(overlay["seeds"])
        return replace(cls(model=overlay.pop("model")), **overlay)


def new_run_dir(cfg: RunConfig, stamp: str | None = None) -> Path:
    """`results/<model>_<split>_<timestamp>/`, created empty."""
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / f"{cfg.model}_{cfg.split}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))
    return run_dir


def log_to_run_dir(run_dir: Path) -> logging.Handler:
    """Tee the root logger into `stdout.log` for the life of the run."""
    handler = logging.FileHandler(run_dir / "stdout.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler
