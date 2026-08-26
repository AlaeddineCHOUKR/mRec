"""Rep/exp decomposition and macro aggregation.

The aggregation order is per held-out session -> per user -> over users, so heavy
users do not dominate. A session whose filtered target set is empty is skipped for
that variant rather than scored zero, and a user with no scorable session drops out
of that variant entirely — both are the reference's behaviour and both are tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from mrec.eval.metrics import as_item_set, ndcg_at_k, pop_at_k, recall_at_k, repr_at_k

VARIANTS = ("all", "rep", "exp")


def split_targets(targets: np.ndarray, history: np.ndarray) -> dict[str, np.ndarray]:
    """Targets split by whether the user already knows the track."""
    targets = as_item_set(targets)
    known = np.isin(targets, history)
    return {"all": targets, "rep": targets[known], "exp": targets[~known]}


def score_session(
    ranked: np.ndarray,
    targets: np.ndarray,
    history: np.ndarray,
    k: int,
    popularity: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Every metric for one held-out session; `None` where the variant is unscorable."""
    scores: dict[str, float | None] = {}
    for variant, subset in split_targets(targets, history).items():
        scorable = len(subset) > 0
        scores[f"ndcg_{variant}@{k}"] = ndcg_at_k(ranked, subset, k) if scorable else None
        scores[f"recall_{variant}@{k}"] = recall_at_k(ranked, subset, k) if scorable else None
    scores[f"repr@{k}"] = repr_at_k(ranked, history, k)
    # the published tables' RepRatio-GT: what share of the truth was a repeat. RepBias
    # is `repr@k - repratio_gt`, so both live in metrics.csv rather than in a script.
    targets = as_item_set(targets)
    scores["repratio_gt"] = float(np.isin(targets, history).mean()) if len(targets) else None
    if popularity is not None:
        scores[f"pop@{k}"] = pop_at_k(ranked, popularity, k)
    return scores


def score_sessions(
    records: Iterable[tuple[int, int, Sequence[int], Sequence[int], np.ndarray]],
    k: int,
    popularity: np.ndarray | None = None,
) -> pl.DataFrame:
    """Score `(user, session, ranked, targets, history)` records into a long table."""
    rows = []
    for user, session, ranked, targets, history in records:
        for metric, value in score_session(ranked, targets, history, k, popularity).items():
            rows.append({"user": user, "session": session, "metric": metric, "value": value})
    schema = {"user": pl.Int64, "session": pl.Int64, "metric": pl.Utf8, "value": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def macro(scores: pl.DataFrame) -> dict[str, float]:
    """Mean over sessions within a user, then over users. Nulls are skipped, not zeroed."""
    per_user = scores.group_by("metric", "user").agg(pl.col("value").mean())
    aggregated = (
        per_user.group_by("metric").agg(pl.col("value").mean().alias("macro")).sort("metric")
    )
    return {row["metric"]: row["macro"] for row in aggregated.iter_rows(named=True)}


def metrics_rows(scores: pl.DataFrame, seed: int, k: int) -> pl.DataFrame:
    """One row per (seed, metric, k), the shape `results/<run_id>/metrics.csv` expects."""
    values = macro(scores)
    return pl.DataFrame(
        {
            "seed": [seed] * len(values),
            "metric": list(values.keys()),
            "k": [k] * len(values),
            "value": list(values.values()),
        }
    )
