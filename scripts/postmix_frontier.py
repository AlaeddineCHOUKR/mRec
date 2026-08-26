"""The post-hoc controllability frontier: what a decode-time control token must beat.

ACT-R supplies the repeat list (its candidates are the user's history, so `repr@10`
is 1 by construction) and PISA with that history masked supplies the explore list.
Splicing `n_rep` slots from the first and the rest from the second hits any repeat
ratio exactly, with no retraining and no conditioning.

So the frontier below is the honest baseline for `the results below`:
a learned control token is only worth anything if, at a matched repeat ratio, it
scores higher than this. Run it before building the model, not after.

Both component models are scored once per request and every ratio is derived from
the same two lists, so the curve costs one pass rather than one pass per point.

python scripts/postmix_frontier.py \
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
from tqdm import tqdm

from mrec.data.sessions import load_sessions
from mrec.eval.aggregate import macro, score_sessions
from mrec.eval.bootstrap import bootstrap_ci
from mrec.eval.popularity import train_popularity
from mrec.eval.protocol import EVAL_SEEDS, N_EVAL_USERS, SEQLEN, K, sample_eval_users
from mrec.eval.runner import build_requests, top_k
from mrec.models.actr import ACTR
from mrec.models.mixture import ORDERINGS, explore_only, mix
from mrec.models.pisa_recommender import PisaRecommender
from mrec.utils.manifest import Manifest
from mrec.utils.paths import PROC_DIR, REPO_ROOT
from mrec.utils.run import log_to_run_dir

logger = logging.getLogger(__name__)

DEFAULT_RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
PISA_CONFIG = {
    "embedding_dim": 128, "seqlen": 30, "num_blocks": 2, "num_heads": 2,
    "dropout": 0.0, "num_favs": 20, "flatten_actr": 0.5,
    "lbda_task": 0.9, "lbda_pos": 0.9, "lbda_ls": 0.4,
}


def component_lists(repeat_model, explore_model, sessions, requests, k, batch_size, progress):
    """Top-k from each component for every request, plus the per-request targets.

    One pass. `repeat_top` comes from a model whose candidate set is the user's
    history; `explore_top` from one with that history masked out, so the two are
    disjoint and splicing them needs no de-duplication.
    """
    repeat_top = np.empty((len(requests), k), dtype=np.int64)
    explore_top = np.empty((len(requests), k), dtype=np.int64)
    histories: dict[int, np.ndarray] = {}
    thin = 0

    batches = range(0, len(requests), batch_size)
    for start in tqdm(batches, desc="scoring components", disable=not progress):
        batch = requests[start : start + batch_size]
        for request in batch:
            if request.user not in histories:
                histories[request.user] = sessions.history(request.user)
        batch_histories = [histories[r.user] for r in batch]
        thin += sum(1 for h in batch_histories if len(h) < k)

        repeat_top[start : start + len(batch)] = top_k(repeat_model.score(batch), k)
        explore_scores = explore_only(explore_model.score(batch), batch_histories)
        explore_top[start : start + len(batch)] = top_k(explore_scores, k)

    if thin:
        logger.warning("%d requests from users with fewer than %d known tracks", thin, k)
    return repeat_top, explore_top, histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pisa-checkpoint", type=Path, required=True)
    parser.add_argument("--n-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--orderings", nargs="+", default=list(ORDERINGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results"
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    progress = not args.no_progress
    k = K

    run_dir = args.out / f"postmix_frontier_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_to_run_dir(run_dir)
    config = {
        "split": args.split, "k": k, "seqlen": SEQLEN, "n_users": args.n_users,
        "seeds": args.seeds, "ratios": args.ratios, "orderings": args.orderings,
        "batch_size": args.batch_size, "pisa_checkpoint": str(args.pisa_checkpoint),
        "repeat_model": "actr", "explore_model": "pisa (history masked)",
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    manifest = Manifest(run_dir / "manifest.json", kind="postmix_frontier", config=config).start()
    logger.info("run dir %s", run_dir)

    sessions = load_sessions(str(PROC_DIR))
    popularity = train_popularity(sessions)

    logger.info("fitting the repeat component (ACT-R)")
    repeat_model = ACTR(progress=progress).fit(sessions)
    logger.info("loading the explore component (PISA) from %s", args.pisa_checkpoint)
    explore_model = PisaRecommender(
        model=PISA_CONFIG, checkpoint=str(args.pisa_checkpoint), progress=progress
    ).fit(sessions)

    rows = []
    all_users = np.arange(sessions.n_users)
    for seed in args.seeds:
        cohort = sample_eval_users(all_users, args.n_users, seed)
        requests = build_requests(sessions, cohort, args.split, SEQLEN)
        logger.info("seed %d: %d requests over %d users", seed, len(requests), len(cohort))

        repeat_top, explore_top, histories = component_lists(
            repeat_model, explore_model, sessions, requests, k, args.batch_size, progress
        )
        targets = [sessions.session_items(r.target) for r in requests]

        for ordering in args.orderings:
            for ratio in args.ratios:
                n_rep = int(round(ratio * k))
                ranked = mix(repeat_top, explore_top, n_rep=n_rep, k=k, ordering=ordering)
                scores = score_sessions(
                    (
                        (r.user, r.target, row, target, histories[r.user])
                        for r, row, target in zip(requests, ranked, targets, strict=True)
                    ),
                    k=k,
                    popularity=popularity,
                )
                summary = macro(scores)
                summary["rep_bias"] = summary[f"repr@{k}"] - summary["repratio_gt"]
                for metric, value in summary.items():
                    rows.append(
                        {
                            "seed": seed, "ordering": ordering, "ratio": ratio,
                            "n_rep": n_rep, "metric": metric, "k": k, "value": value,
                        }
                    )
                logger.info(
                    "seed %d %s ratio %.1f: ndcg %.4f rep %.4f exp %.4f repr %.4f pop %.4f",
                    seed, ordering, ratio, summary[f"ndcg_all@{k}"],
                    summary.get(f"ndcg_rep@{k}", float("nan")),
                    summary.get(f"ndcg_exp@{k}", float("nan")),
                    summary[f"repr@{k}"], summary.get(f"pop@{k}", float("nan")),
                )

    frontier = pl.DataFrame(rows)
    frontier.write_csv(run_dir / "frontier.csv")

    summary_rows = []
    for key, group in frontier.group_by(["ordering", "ratio", "metric"], maintain_order=True):
        mean, lo, hi = bootstrap_ci(group["value"].to_numpy(), seed=0)
        summary_rows.append(
            {"ordering": key[0], "ratio": key[1], "metric": key[2],
             "mean": mean, "ci_lo": lo, "ci_hi": hi}
        )
    summary = pl.DataFrame(summary_rows).sort(["ordering", "ratio", "metric"])
    summary.write_csv(run_dir / "summary.csv")

    manifest.finish(
        counts={"n_users": sessions.n_users, "n_items": sessions.n_items},
        explore_model=explore_model.summary,
    )
    headline = summary.filter(
        pl.col("metric").is_in([f"ndcg_all@{k}", f"ndcg_rep@{k}", f"ndcg_exp@{k}", f"repr@{k}"])
    )
    with pl.Config(tbl_rows=-1):
        print(headline)


if __name__ == "__main__":
    main()
