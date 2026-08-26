"""The evaluation loop: contexts in, full-catalog rankings out, `metrics.csv` on disk.

One loop for every model, so a number from one baseline is comparable to a number
from another by construction. Ranking is over the whole 50k catalog — no sampled
negatives, ever (`EVALUATION_PROTOCOL.md` §5) — and previously heard tracks are not
filtered out, since repetition is the phenomenon under study.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl
from tqdm import tqdm

from mrec.data.sessions import Sessions
from mrec.eval.aggregate import macro, score_sessions
from mrec.eval.bootstrap import bootstrap_ci
from mrec.eval.protocol import sample_eval_users

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    """One held-out session to predict, with the history the model may look at."""

    user: int
    target: int  # session id of the held-out session
    context: np.ndarray  # [<= seqlen] session ids preceding it, chronological


class Recommender(Protocol):
    name: str

    def fit(self, sessions: Sessions) -> Recommender: ...

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        ...


def build_requests(sessions: Sessions, users: np.ndarray, which: str, seqlen: int):
    """Held-out sessions of `users`, each with the `seqlen` sessions preceding it.

    The context is `sessions[:target]` truncated to `seqlen`, exactly as the
    reference sampler does — which means a late target's context legitimately
    contains *other* held-out sessions. The target itself never appears in it, and
    the rep/exp history is a different, earlier cut.
    """
    requests = []
    for u in users:
        u = int(u)
        first = int(sessions.user_ptr[u])
        for target in sessions.held_out_session_ids(u, which):
            target = int(target)
            requests.append(Request(u, target, np.arange(max(first, target - seqlen), target)))
    return requests


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row, ties broken by ascending item index. `# [B, k]`

    The tie-break matters: the reference sorts on `(-popularity, item_id)`, and with
    a 50k catalog a score of exactly zero is the common case, not an edge case.
    """
    if scores.shape[1] < k:
        raise ValueError(f"cannot take top-{k} of {scores.shape[1]} items")
    threshold = np.partition(scores, -k, axis=1)[:, -k]
    out = np.empty((len(scores), k), dtype=np.int64)
    for i, row in enumerate(scores):
        candidates = np.flatnonzero(row >= threshold[i])
        out[i] = candidates[np.lexsort((candidates, -row[candidates]))][:k]
    return out


def evaluate_cohort(
    model: Recommender,
    sessions: Sessions,
    users: np.ndarray,
    which: str,
    k: int,
    seqlen: int,
    batch_size: int,
    popularity: np.ndarray | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """Score one cohort of users, returning the long per-session table."""
    requests = build_requests(sessions, users, which, seqlen)
    records = []
    histories: dict[int, np.ndarray] = {}
    batches = range(0, len(requests), batch_size)
    for start in tqdm(batches, desc=f"{model.name}:{which}", disable=not progress):
        batch = requests[start : start + batch_size]
        ranked = top_k(model.score(batch), k)
        for request, row in zip(batch, ranked, strict=True):
            if request.user not in histories:
                histories[request.user] = sessions.history(request.user)
            records.append(
                (
                    request.user,
                    request.target,
                    row,
                    sessions.session_items(request.target),
                    histories[request.user],
                )
            )
    return score_sessions(records, k=k, popularity=popularity)


def evaluate(
    model: Recommender,
    sessions: Sessions,
    which: str,
    k: int,
    seqlen: int,
    n_users: int,
    seeds: Sequence[int],
    batch_size: int,
    popularity: np.ndarray | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """One row per (seed, metric, k) — the committed `metrics.csv`."""
    all_users = np.arange(sessions.n_users)
    rows = []
    for seed in seeds:
        cohort = sample_eval_users(all_users, n_users, seed)
        scores = evaluate_cohort(
            model, sessions, cohort, which, k, seqlen, batch_size, popularity, progress
        )
        for metric, value in macro(scores).items():
            rows.append({"seed": seed, "metric": metric, "k": k, "value": value})
        logger.info("seed %d: %s", seed, {m: round(v, 4) for m, v in macro(scores).items()})
    return pl.DataFrame(rows)


def summarize(metrics: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Mean ± 95% bootstrap CI over the evaluation cohorts, one row per metric.

    `rep_bias` is derived here rather than measured: the published tables report
    `repr@k - RepRatio-GT`, and deriving it per seed keeps its interval honest.
    """
    wide = metrics.pivot(on="metric", index="seed", values="value")
    repr_col = next((c for c in wide.columns if c.startswith("repr@")), None)
    if repr_col is not None and "repratio_gt" in wide.columns:
        derived = wide.select(
            pl.col("seed"),
            pl.lit("rep_bias").alias("metric"),
            pl.lit(metrics["k"][0]).alias("k"),
            (pl.col(repr_col) - pl.col("repratio_gt")).alias("value"),
        )
        metrics = pl.concat([metrics, derived], how="vertical_relaxed")

    rows = []
    for metric, group in metrics.group_by("metric", maintain_order=True):
        mean, lo, hi = bootstrap_ci(group["value"].to_numpy(), seed=seed)
        rows.append({"metric": metric[0], "mean": mean, "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(rows).sort("metric")
