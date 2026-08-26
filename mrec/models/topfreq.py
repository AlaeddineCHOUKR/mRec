"""Frequency baselines: recommend what is played most, globally or by this user.

Ports `au2actr/models/baselines/freq/topfreq.py`. Both variants are untrained and
context-free — their whole score vector depends on the user alone — which makes them
the cheapest end-to-end check that the protocol plumbing is right.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from mrec.data.sessions import Sessions
from mrec.eval.popularity import train_popularity
from mrec.eval.runner import Request

logger = logging.getLogger(__name__)


class GlobalTopFreq:
    """`gtop`: one ranking for everyone, by training-set popularity."""

    name = "gtop"

    def __init__(self, progress: bool = True) -> None:
        self.progress = progress
        self.popularity: np.ndarray | None = None

    def fit(self, sessions: Sessions) -> GlobalTopFreq:
        self.popularity = train_popularity(sessions, progress=self.progress)
        return self

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        if self.popularity is None:
            raise RuntimeError("fit() first")
        return np.broadcast_to(self.popularity, (len(requests), len(self.popularity)))


class PersonalTopFreq:
    """`ptop`: each user's own training play counts, catalog-wide.

    The reference ranks only the tracks a user has played; scoring the rest zero and
    breaking ties on ascending item index reproduces that order exactly while keeping
    the ranking full-catalog.
    """

    name = "ptop"

    def __init__(self, progress: bool = True) -> None:
        self.progress = progress
        self.counts: np.ndarray | None = None  # [n_users, n_items] float32, sparse in practice
        self.n_items = 0

    def fit(self, sessions: Sessions) -> PersonalTopFreq:
        self.n_items = sessions.n_items
        self.counts = np.zeros((sessions.n_users, sessions.n_items), dtype=np.float32)
        for u in range(sessions.n_users):
            train = sessions.train_session_ids(u)
            lo, hi = sessions.session_ptr[train[0]], sessions.session_ptr[train[-1] + 1]
            items, counts = np.unique(sessions.event_item[lo:hi], return_counts=True)
            self.counts[u, items] = counts
        return self

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        if self.counts is None:
            raise RuntimeError("fit() first")
        return self.counts[[r.user for r in requests]]


MODELS = {"gtop": GlobalTopFreq, "ptop": PersonalTopFreq}
