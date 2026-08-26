"""Does the `exp` metric measure exploration, or measure slot allocation?

`ndcg_exp@k` and `recall_exp@k` are computed over a list the model fills however it
likes. On this data 84% of targets are repeats, so a model that ranks well spends its
ten slots on tracks the user already knows and the `exp` column collapses -- whether or
not the model could have ranked unheard tracks perfectly.

This script separates the two by scoring one fitted model twice: once as it is, and
once forbidden from returning anything in the user's history. Same weights, same
protocol, same cohort; only the composition of the list changes. The gap is how much of
the published `exp` number is allocation rather than ability.

python scripts/explore_ceiling.py \
    --config configs/pisa.yaml
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
from mrec.models.mixture import ExploreOnly
from mrec.utils.manifest import Manifest
from mrec.utils.run import RunConfig, log_to_run_dir, new_run_dir

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("results")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="restore into model.model after fit; pair with training.epochs=0 in the config",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="override the model's beam width; for a generative model the masked list "
        "can only be filled if the beam reaches k unheard identifiers",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RunConfig.from_yaml(args.config)
    if args.n_users is not None:
        cfg = replace(cfg, n_users=args.n_users)
    suffix = "_ceiling" if args.beam_width is None else f"_ceiling_b{args.beam_width}"
    cfg = replace(cfg, results_dir=str(args.out), model=f"{cfg.model}{suffix}")

    run_dir = new_run_dir(cfg)
    log_to_run_dir(run_dir)
    manifest = Manifest(run_dir / "manifest.json", kind="explore_ceiling", config=cfg).start()
    logger.info("run dir %s", run_dir)

    sessions = load_sessions(cfg.proc_dir)
    popularity = train_popularity(sessions)
    params = dict(cfg.model_params)
    if args.no_progress:
        params.setdefault("progress", False)
    model_name = cfg.model.split("_ceiling")[0]
    inner = build_model(model_name, **params).fit(sessions)
    if args.checkpoint is not None:
        inner.model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
        logger.info("restored %s", args.checkpoint)
    if args.beam_width is not None:
        inner.beam_width = args.beam_width
        logger.info("beam width overridden to %d", args.beam_width)

    frames = []
    for variant, model in (("natural", inner), ("explore_only", ExploreOnly(inner, cfg.k))):
        if isinstance(model, ExploreOnly):
            model.attach(sessions)  # already fitted; the mask must not pay for training
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
        summary = summarize(metrics)
        logger.info("%s\n%s", variant, summary)
        frames.append(metrics.with_columns(pl.lit(variant).alias("variant")))
        if isinstance(model, ExploreOnly) and model.thin:
            logger.warning("%d requests had fewer than k unheard tracks", model.thin)

    pl.concat(frames).write_csv(run_dir / "metrics.csv")

    wide = (
        pl.concat(frames)
        .group_by(["variant", "metric"])
        .agg(pl.col("value").mean())
        .pivot(on="variant", values="value", index="metric")
        .sort("metric")
    )
    wide.write_csv(run_dir / "summary.csv")
    manifest.finish(counts={"n_users": sessions.n_users, "n_items": sessions.n_items})
    with pl.Config(tbl_rows=-1):
        print(wide)


if __name__ == "__main__":
    main()
