"""Put the post-hoc frontier and the committed models on one axis.

Every model in `results/` sits at some realized repeat ratio, and the
frontier says what accuracy a two-list splice reaches at that same ratio. A model is
worth its training only if it lands *above* the curve at its own `repr@10`.

Reads committed `metrics.csv` files only, so every number here has a run_id behind it.

python scripts/compare_frontier.py \
    --frontier results/postmix_frontier_test_*/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

METRICS = ("ndcg_all@10", "ndcg_rep@10", "ndcg_exp@10", "repr@10", "repratio_gt", "pop@10")


def model_rows(results_dirs: list[Path]) -> pl.DataFrame:
    """One row per committed evaluation run: its means and where it sits on the axis."""
    rows = []
    for results in results_dirs:
        for run in sorted(results.glob("*/")):
            metrics_path, manifest_path = run / "metrics.csv", run / "manifest.json"
            if not (metrics_path.exists() and manifest_path.exists()):
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("kind") != "eval" or manifest.get("status") != "complete":
                continue
            frame = pl.read_csv(metrics_path)
            if "variant" in frame.columns:
                continue
            means = {
                m: float(frame.filter(pl.col("metric") == m)["value"].mean())
                for m in METRICS
                if (frame["metric"] == m).any()
            }
            rows.append(
                {
                    "run_id": run.name,
                    "model": manifest["config"]["model"],
                    "n_users": manifest["config"]["n_users"],
                    **means,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--ordering", default="repeat_first")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[
            Path("results"),
            Path("results"),
        ],
    )
    args = parser.parse_args()

    frontier = (
        pl.read_csv(args.frontier / "frontier.csv")
        .filter(pl.col("ordering") == args.ordering)
        .group_by(["ratio", "metric"])
        .agg(pl.col("value").mean())
        .pivot(on="metric", values="value", index="ratio")
        .sort("ratio")
    )
    print(f"\n=== post-hoc frontier ({args.ordering}) ===")
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(frontier.select(["ratio", *[c for c in METRICS if c in frontier.columns]]))

    models = model_rows([p for p in args.results if p.exists()])
    if models.is_empty():
        return

    # interpolate the curve at each model's own realized repeat ratio
    ratios = frontier["ratio"].to_numpy()
    curve = frontier["ndcg_all@10"].to_numpy()
    models = models.with_columns(
        pl.col("repr@10")
        .map_elements(lambda r: float(np.interp(r, ratios, curve)), return_dtype=pl.Float64)
        .alias("frontier@repr")
    ).with_columns(
        (pl.col("ndcg_all@10") - pl.col("frontier@repr")).alias("above_frontier"),
        (pl.col("ndcg_all@10") / pl.col("frontier@repr") - 1).alias("relative"),
    )
    print("\n=== each model against the frontier at its own repeat ratio ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(
            models.select(
                [
                    "model",
                    "n_users",
                    "repr@10",
                    "ndcg_all@10",
                    "frontier@repr",
                    "above_frontier",
                    "relative",
                    "run_id",
                ]
            ).sort("ndcg_all@10", descending=True)
        )


if __name__ == "__main__":
    main()
