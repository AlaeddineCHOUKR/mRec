"""Per-session metrics, in the semantics of the reference implementation.

Verified line by line against `au2actr/eval/metrics/{ndcg,recall,repr,pop}.py` in
the reference implementation. Two details are load-bearing and easy to "fix" by
accident: recall divides by `|T|` rather than `min(k, |T|)`, and the ideal DCG runs
over `min(|T|, k)` hits.

`T` is a *set*: the reference builds it with `set(ss['track_ids'])`, so a track
repeated inside a held-out session counts once.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def as_item_set(items: Iterable[int] | np.ndarray) -> np.ndarray:
    """Distinct items, sorted. The canonical form of a target set or a history."""
    arr = np.fromiter(items, dtype=np.int64) if not isinstance(items, np.ndarray) else items
    return np.unique(np.asarray(arr, dtype=np.int64))


def _top_k(ranked: np.ndarray, k: int) -> np.ndarray:
    ranked = np.asarray(ranked, dtype=np.int64)
    if ranked.ndim != 1:
        raise ValueError(f"ranked list must be 1-d, got shape {ranked.shape}")
    if len(ranked) < k:
        raise ValueError(f"ranked list has {len(ranked)} items, need at least k={k}")
    top = ranked[:k]
    if len(np.unique(top)) != k:
        raise ValueError("ranked list contains duplicate items in its top-k")
    return top


def hits(ranked: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """Binary relevance of the top-k, in rank order. `# [k]`"""
    return np.isin(_top_k(ranked, k), targets).astype(np.float64)


def dcg(rel: np.ndarray) -> float:
    """Discounted cumulative gain with binary gains and the standard log2 discount."""
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def recall_at_k(ranked: np.ndarray, targets: np.ndarray, k: int) -> float:
    """`|R ∩ T| / |T|` — the denominator is `|T|`, so recall is capped when `|T| > k`."""
    if len(targets) == 0:
        raise ValueError("recall is undefined for an empty target set; skip the session instead")
    return float(hits(ranked, targets, k).sum() / len(targets))


def ndcg_at_k(ranked: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Binary-gain NDCG whose ideal ranking places `min(|T|, k)` hits at the top."""
    if len(targets) == 0:
        raise ValueError("ndcg is undefined for an empty target set; skip the session instead")
    ideal = np.zeros(k)
    ideal[: min(len(targets), k)] = 1.0
    return dcg(hits(ranked, targets, k)) / dcg(ideal)


def repr_at_k(ranked: np.ndarray, history: np.ndarray, k: int) -> float:
    """Realized repeat ratio of the *list*: the share of the top-k the user already knows.

    Not an accuracy metric — it says what the model did, not whether it was right.
    """
    return float(np.isin(_top_k(ranked, k), history).mean())


def pop_at_k(ranked: np.ndarray, popularity: np.ndarray, k: int) -> float:
    """Mean training popularity of the top-k, global variant. Lower is more long-tail.

    The reference averages only over items that appear in its popularity dict; ours
    is a dense vector in which a training-unseen item scores 0. The two agree
    whenever the recommended items appear in training — true for every baseline in
    the reproduction gate — and ours stays meaningful for the cold items P3 emits,
    where the reference expression degenerates.
    """
    return float(np.mean(popularity[_top_k(ranked, k)]))
