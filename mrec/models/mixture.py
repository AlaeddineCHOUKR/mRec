"""Post-hoc list mixing: the baseline a decode-time repeat/explore control must beat.

A repeat model and an explore model each produce a top-k list. Taking `n_rep` slots
from the first and the rest from the second hits any repeat ratio *exactly*, without
retraining, without conditioning, and without touching either model. `repr@k` becomes
`n_rep / k` by construction, so controllability alone is not evidence of anything.

That makes this the falsifier rather than a strawman. The question a learned control
token has to answer is not "can the repeat ratio be moved" but "at a given repeat
ratio, is the accuracy higher than this". `repeat_first` is the strongest ordering
available to the baseline on this data — repeat targets are hit several times more
often than explore targets — so it is the one to beat.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ORDERINGS = ("repeat_first", "interleaved")


def explore_only(scores: np.ndarray, histories: Sequence[np.ndarray]) -> np.ndarray:
    """Mask every track the user has already heard, leaving an explore-only ranking.

    The history is the same cut the metrics use (`sessions[:-20]`), so a list drawn
    from this matrix scores `repr@k = 0` and `ndcg_rep@k` is undefined, not merely low.
    """
    masked = scores.copy()
    for row, history in enumerate(histories):
        masked[row, history] = -np.inf
    return masked


def mix(
    repeat_top: np.ndarray,
    explore_top: np.ndarray,
    n_rep: int,
    k: int,
    ordering: str = "repeat_first",
) -> np.ndarray:
    """Splice two top-k lists into one of length k with exactly `n_rep` repeat slots.

    The two sources are disjoint by construction: `repeat_top` is drawn from the
    user's history and `explore_top` from its complement, so no de-duplication is
    needed and `repr@k` lands on `n_rep / k` exactly.
    """
    if not 0 <= n_rep <= k:
        raise ValueError(f"n_rep {n_rep} outside [0, {k}]")
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering `{ordering}`; known: {ORDERINGS}")
    if repeat_top.shape[1] < n_rep or explore_top.shape[1] < k - n_rep:
        raise ValueError("component lists are shorter than the slots asked of them")

    take_repeat = _slot_mask(n_rep, k, ordering)
    out = np.empty((len(repeat_top), k), dtype=np.int64)
    out[:, take_repeat] = repeat_top[:, :n_rep]
    out[:, ~take_repeat] = explore_top[:, : k - n_rep]
    return out


def _slot_mask(n_rep: int, k: int, ordering: str) -> np.ndarray:
    """Which of the k slots the repeat list fills. `# [k] bool`"""
    if ordering == "repeat_first":
        mask = np.zeros(k, dtype=bool)
        mask[:n_rep] = True
        return mask
    # Spread the repeat slots evenly: slot i is a repeat slot whenever the running
    # quota crosses an integer. `ceil` front-loads, so the larger stream takes slot 0
    # rather than being pushed behind the smaller one.
    counts = np.ceil(np.arange(1, k + 1) * n_rep / k).astype(int)
    return np.diff(np.concatenate([[0], counts])) > 0


class ExploreOnly:
    """Any recommender, forbidden from returning a track the user has already heard.

    Wrapping a model this way separates two things the published `exp` metrics
    conflate: whether a model *can* rank unheard tracks, and whether it chooses to
    spend its ten slots on them. `ndcg_exp@k` is computed over a list the model fills
    however it likes, so on repeat-dominated data it measures slot allocation at least
    as much as it measures exploration.
    """

    def __init__(self, inner, k: int = 10) -> None:
        self.inner = inner
        self.name = f"{inner.name}_explore"
        self.k = k
        self.sessions = None
        self.n_items = 0
        # A masked list can only be filled if at least k items still carry a finite
        # score. For a scoring model on a 50k catalog that is automatic. For a
        # *generative* one it is not: the beam reaches a few dozen identifiers, and
        # masking the user's history can leave fewer than k of them -- at which point
        # `top_k` falls back to -inf entries and heard tracks re-enter the list. That
        # is measured, not assumed, because a masked list containing repeats is not the
        # thing the measurement claims to be.
        self.thin = 0

    def fit(self, sessions) -> ExploreOnly:
        self.inner = self.inner.fit(sessions)
        return self.attach(sessions)

    def attach(self, sessions) -> ExploreOnly:
        """Wrap a model that is already fitted, so the mask costs no training."""
        self.sessions = sessions
        self.n_items = sessions.n_items
        self._history: dict[int, np.ndarray] = {}
        return self

    def score(self, requests: Sequence) -> np.ndarray:
        assert self.sessions is not None
        for request in requests:
            if request.user not in self._history:
                self._history[request.user] = self.sessions.history(request.user)
        histories = [self._history[r.user] for r in requests]
        masked = explore_only(self.inner.score(requests), histories)
        self.thin += int((np.isfinite(masked).sum(axis=1) < self.k).sum())
        return masked

    @property
    def log(self):
        return getattr(self.inner, "log", [])

    @property
    def summary(self):
        return getattr(self.inner, "summary", {})
