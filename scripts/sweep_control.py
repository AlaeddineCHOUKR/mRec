"""Decode a trained control-token retriever under each mode, from one checkpoint.

GLIDE reports (§5.2.3) that swapping the control token at inference shifts the output
distribution towards a discovery horizon without retraining: +11.8% Recall@30 on
unfamiliar content, +4.9% on familiar. The analogue here is the repeat/explore axis,
and the quantity to watch is `repr@10` -- the realized repeat ratio -- against the
accuracy it costs.

The comparison that matters is not free vs repeat vs explore. It is this curve against
`postmix_frontier.py`, which hits any repeat ratio exactly by splicing two lists and
needs no model at all. Read them on the same axes.

Reuses a run's checkpoint; the tokenizer is refit from the same seed, which reproduces
its codes exactly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl
import torch
import yaml

from mrec.data.sessions import load_sessions
from mrec.eval.popularity import train_popularity
from mrec.eval.runner import evaluate, summarize
from mrec.models import build_model
from mrec.models.genrec import MODES
from mrec.utils.manifest import Manifest

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a finished genrec run directory")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load((args.run / "config.yaml").read_text())
    params = dict(cfg["model_params"])
    params["progress"] = False
    if not params.get("control"):
        raise SystemExit(f"{args.run} was not trained with a control token")

    n_users = args.n_users or cfg["n_users"]
    seeds = args.seeds or cfg["seeds"]

    sessions = load_sessions(cfg["proc_dir"])
    popularity = train_popularity(sessions)

    model = build_model(cfg["model"], **{**params, "training": {**params["training"], "epochs": 0}})
    model.fit(sessions)
    model.model.load_state_dict(torch.load(args.run / "model.pt", map_location="cuda"))
    logger.info("restored %s", args.run / "model.pt")

    out_dir = args.run / "control_sweep"
    out_dir.mkdir(exist_ok=True)
    manifest = Manifest(
        out_dir / "manifest.json",
        kind="control_sweep",
        config={"run": str(args.run), "modes": args.modes, "n_users": n_users, "seeds": seeds},
    ).start()

    rows = []
    for mode in args.modes:
        model.decode_mode = mode
        metrics = evaluate(
            model, sessions, which="test", k=cfg["k"], seqlen=cfg["seqlen"],
            n_users=n_users, seeds=seeds, batch_size=cfg["batch_size"],
            popularity=popularity, progress=False,
        )
        summary = summarize(metrics)
        for row in summary.iter_rows(named=True):
            rows.append({"mode": mode, **row})
        logger.info(
            "mode %-8s ndcg %.4f  rep %.4f  exp %.4f  repr %.4f  rep_bias %+.4f",
            mode,
            *[
                summary.filter(pl.col("metric") == m)["mean"][0]
                for m in ("ndcg_all@10", "ndcg_rep@10", "ndcg_exp@10", "repr@10", "rep_bias")
            ],
        )

    table = pl.DataFrame(rows)
    table.write_csv(out_dir / "control_sweep.csv")
    manifest.finish(modes=args.modes, n_users=n_users)
    with pl.Config(tbl_rows=-1):
        print(table.filter(pl.col("metric").is_in(["ndcg_all@10", "repr@10", "rep_bias"])))


if __name__ == "__main__":
    main()
