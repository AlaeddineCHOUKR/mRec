"""Evaluate one baseline under the protocol. Thin: every decision lives in `mrec`.

python scripts/run_baseline.py --config .../configs/gtop.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import polars as pl
import torch

from mrec.data.sessions import load_sessions
from mrec.eval.popularity import train_popularity
from mrec.eval.runner import evaluate, summarize
from mrec.models import build_model
from mrec.utils.manifest import Manifest
from mrec.utils.run import RunConfig, log_to_run_dir, new_run_dir

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-users", type=int, default=None, help="override, for smoke runs")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RunConfig.from_yaml(args.config)
    if args.n_users is not None:
        cfg = replace(cfg, n_users=args.n_users)

    run_dir = new_run_dir(cfg)
    log_to_run_dir(run_dir)
    manifest = Manifest(run_dir / "manifest.json", kind="eval", config=cfg).start()
    logger.info("run dir %s", run_dir)

    sessions = load_sessions(cfg.proc_dir)
    popularity = train_popularity(sessions)
    params = dict(cfg.model_params)
    if args.no_progress:
        params.setdefault("progress", False)
    model = build_model(cfg.model, **params).fit(sessions)
    if getattr(model, "log", None):
        pl.DataFrame(model.log).write_csv(run_dir / "train_log.csv")
        torch.save(model.model.state_dict(), run_dir / "model.pt")

    metrics = evaluate(
        model,
        sessions,
        which=cfg.split,
        k=cfg.k,
        seqlen=cfg.seqlen,
        n_users=cfg.n_users,
        seeds=cfg.seeds,
        batch_size=cfg.batch_size,
        popularity=popularity,
        progress=not args.no_progress,
    )
    metrics.write_csv(run_dir / "metrics.csv")
    summary = summarize(metrics)
    logger.info("\n%s", summary)

    manifest.finish(
        counts={"n_users": sessions.n_users, "n_items": sessions.n_items},
        training=getattr(model, "summary", None),
        summary={row["metric"]: row["mean"] for row in summary.iter_rows(named=True)},
    )
    print(summary)


if __name__ == "__main__":
    main()
