"""Is a model's gap to the post-hoc frontier significant, or is it noise?

The protocol asks for a paired bootstrap over users against the named baseline, with the
p-value reported. "Better on average across 5 seeds" is not a result.
`mrec.eval.bootstrap.paired_bootstrap` has existed and been tested since milestone 2 and
nothing has used it. This does.

The model and the frontier are scored on the *same requests in the same pass*, so the
pairing is exact: for each user, that user's macro NDCG@10 under the model and under a
splice holding the same number of repeat slots. Users are the resampling unit, which is
also the unit the protocol's macro aggregation ends on.

A user's value does not depend on which cohort they were drawn into, so the five
protocol cohorts are unioned rather than scored five times over.

python scripts/frontier_significance.py \
    --model-config configs/pisa_ceiling.yaml \
    --pisa-checkpoint results/pisa_test_20260831-094523/model.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from mrec.data.sessions import load_sessions
from mrec.eval.aggregate import score_sessions
from mrec.eval.bootstrap import paired_bootstrap
from mrec.eval.protocol import EVAL_SEEDS, N_EVAL_USERS, SEQLEN, K, sample_eval_users
from mrec.eval.runner import build_requests, top_k
from mrec.models import build_model
from mrec.models.actr import ACTR
from mrec.models.mixture import explore_only, mix
from mrec.models.pisa_recommender import PisaRecommender
from mrec.utils.manifest import Manifest
from mrec.utils.paths import PROC_DIR, REPO_ROOT
from mrec.utils.run import RunConfig, log_to_run_dir

logger = logging.getLogger(__name__)

PISA_CONFIG = {
    "embedding_dim": 128, "seqlen": 30, "num_blocks": 2, "num_heads": 2,
    "dropout": 0.0, "num_favs": 20, "flatten_actr": 0.5,
    "lbda_task": 0.9, "lbda_pos": 0.9, "lbda_ls": 0.4,
}


def per_user(ranked, requests, targets, histories, k, metric):
    """Macro value per user: mean over that user's held-out sessions. `# dict[user, v]`"""
    scores = score_sessions(
        (
            (r.user, r.target, row, target, histories[r.user])
            for r, row, target in zip(requests, ranked, targets, strict=True)
        ),
        k=k,
    )
    table = (
        scores.filter(pl.col("metric") == metric)
        .group_by("user")
        .agg(pl.col("value").mean())
        .sort("user")
    )
    return dict(zip(table["user"].to_list(), table["value"].to_list(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--pisa-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-users", type=int, default=N_EVAL_USERS)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RunConfig.from_yaml(args.model_config)
    metric = f"ndcg_all@{K}"

    run_dir = args.out / f"significance_{cfg.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_to_run_dir(run_dir)
    config = {
        "model": cfg.model, "model_config": str(args.model_config), "metric": metric,
        "ratios": args.ratios, "n_users": args.n_users, "seeds": list(EVAL_SEEDS),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    manifest = Manifest(run_dir / "manifest.json", kind="significance", config=config).start()

    sessions = load_sessions(str(PROC_DIR))
    logger.info("fitting frontier components")
    repeat_model = ACTR(progress=False).fit(sessions)
    explore_model = PisaRecommender(
        model=PISA_CONFIG, checkpoint=str(args.pisa_checkpoint), progress=False
    ).fit(sessions)

    logger.info("fitting the model under test: %s", cfg.model)
    params = dict(cfg.model_params)
    params.setdefault("progress", False)
    model = build_model(cfg.model, **params).fit(sessions)
    if args.model_checkpoint is not None:
        import torch

        model.model.load_state_dict(torch.load(args.model_checkpoint, map_location="cuda"))
        logger.info("restored %s", args.model_checkpoint)

    # a user's value is independent of the cohort they were drawn into
    cohort = np.unique(
        np.concatenate(
            [sample_eval_users(np.arange(sessions.n_users), args.n_users, s) for s in EVAL_SEEDS]
        )
    )
    requests = build_requests(sessions, cohort, cfg.split, SEQLEN)
    logger.info(
        "%d users in the union of the five cohorts, %d requests", len(cohort), len(requests)
    )

    histories = {int(u): sessions.history(int(u)) for u in cohort}
    targets = [sessions.session_items(r.target) for r in requests]
    model_top = np.empty((len(requests), K), dtype=np.int64)
    repeat_top = np.empty((len(requests), K), dtype=np.int64)
    explore_top = np.empty((len(requests), K), dtype=np.int64)

    for start in range(0, len(requests), args.batch_size):
        batch = requests[start : start + args.batch_size]
        end = start + len(batch)
        batch_histories = [histories[r.user] for r in batch]
        model_top[start:end] = top_k(model.score(batch), K)
        repeat_top[start:end] = top_k(repeat_model.score(batch), K)
        explore_top[start:end] = top_k(
            explore_only(explore_model.score(batch), batch_histories), K
        )
        if start % (args.batch_size * 100) == 0:
            logger.info("scored %d/%d", start, len(requests))

    model_values = per_user(model_top, requests, targets, histories, K, metric)
    model_repr = per_user(model_top, requests, targets, histories, K, f"repr@{K}")
    logger.info(
        "%s: %s %.4f, repr@%d %.4f",
        cfg.model, metric, np.mean(list(model_values.values())), K,
        np.mean(list(model_repr.values())),
    )

    rows = []
    for ratio in args.ratios:
        n_rep = int(round(ratio * K))
        frontier_top = mix(repeat_top, explore_top, n_rep=n_rep, k=K, ordering="repeat_first")
        frontier_values = per_user(frontier_top, requests, targets, histories, K, metric)

        users = sorted(set(model_values) & set(frontier_values))
        a = np.array([model_values[u] for u in users])
        b = np.array([frontier_values[u] for u in users])
        difference, p_value = paired_bootstrap(a, b, seed=0)
        rows.append(
            {
                "ratio": ratio, "n_rep": n_rep, "n_users": len(users),
                "model": float(a.mean()), "frontier": float(b.mean()),
                "difference": difference, "p_value": p_value,
            }
        )
        logger.info(
            "ratio %.1f: model %.4f  frontier %.4f  diff %+.4f  p = %.4g",
            ratio, a.mean(), b.mean(), difference, p_value,
        )

    table = pl.DataFrame(rows)
    table.write_csv(run_dir / "significance.csv")
    manifest.finish(counts={"n_users": len(cohort), "n_requests": len(requests)})
    with pl.Config(tbl_rows=-1, float_precision=5):
        print(table)


if __name__ == "__main__":
    main()
