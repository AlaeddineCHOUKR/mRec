"""ACT-R: declarative memory as a recommender.

A track is activated by two ACT-R mechanisms, ported from
`au2actr/data/datasets/actr_weights.py` and `au2actr/models/baselines/actr.py`:

- **base-level learning (BLL)** — recency and frequency. Every past session in which
  the track appeared contributes `(t_ref - t)^-d`, summed and logged, then softmaxed
  over the user's own items. `d = 0.5`, `t` is the session's first-event timestamp.
- **spreading activation** — the track's normalized co-occurrence with whatever the
  user just played, over a session-level adjacency matrix built from training
  sessions only.

The final score is `bll + spread`, ranked over the user's **training** history alone.
Like `ptop`, ACT-R is therefore structurally incapable of exploration; that is the
model, not a limitation of this port.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from mrec.data.sessions import Sessions
from mrec.eval.runner import Request

logger = logging.getLogger(__name__)

BLL_DECAY = 0.5
# pairs buffered before folding into the running matrix; sparse addition costs O(nnz),
# so few large folds beat many small ones
CHUNK_PAIRS = 40_000_000


def session_cooccurrence(sessions: Sessions, progress: bool = True) -> sp.csr_matrix:
    """Session-level co-occurrence over training sessions, symmetrically normalized.

    Counts `(item in previous session, item in this session)` pairs, the previous
    session deduplicated and the current one not — the reference's asymmetry, kept.
    Self-pairs are dropped. Only training sessions contribute, so the matrix carries
    no held-out information.
    """
    n_items = sessions.n_items
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    total = sp.csr_matrix((n_items, n_items), dtype=np.float64)
    pending = 0

    def flush() -> sp.csr_matrix:
        if not rows:
            return sp.csr_matrix((n_items, n_items), dtype=np.float64)
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        rows.clear()
        cols.clear()
        return sp.coo_matrix(
            (np.ones(len(row), dtype=np.float64), (row, col)), shape=(n_items, n_items)
        ).tocsr()

    for u in tqdm(range(sessions.n_users), desc="co-occurrence", disable=not progress):
        train = sessions.train_session_ids(u)
        for s in train[1:]:
            previous = np.unique(sessions.session_items(s - 1))
            current = sessions.session_items(s)
            row = np.repeat(previous, len(current))
            col = np.tile(current, len(previous))
            keep = row != col
            rows.append(row[keep])
            cols.append(col[keep])
            pending += int(keep.sum())
        if pending >= CHUNK_PAIRS:
            total = total + flush()
            pending = 0
    total = total + flush()
    return normalize_adjacency(total)


def normalize_adjacency(adj: sp.spmatrix) -> sp.csr_matrix:
    """`D^-1/2 A^T D^-1/2` with `D` the row sums of `A`.

    The transpose is the reference's, arising from how it chains the two `dot` calls;
    the spreading step then reads rows of this matrix for the tracks just played.
    """
    row_sum = np.asarray(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    return adj.dot(d).transpose().dot(d).tocsr()


def bll_activation(
    sessions: Sessions, user: int, target: int, decay: float = BLL_DECAY
) -> np.ndarray:
    """Softmaxed base-level activation of everything `user` played before `target`.

    History runs to the target session itself, which for a late target includes
    earlier held-out sessions — faithful to the reference sampler, and never the
    target itself. Timestamps are the *session's* first event, repeated once per
    occurrence of the track in that session.
    """
    first = int(sessions.user_ptr[user])
    history = np.arange(first, target)
    t_ref = int(sessions.event_ts[sessions.session_ptr[target]])

    lo, hi = sessions.session_ptr[first], sessions.session_ptr[target]
    items = np.asarray(sessions.event_item[lo:hi])
    counts = np.diff(sessions.session_ptr[first : target + 1])
    stamps = np.repeat(sessions.event_ts[sessions.session_ptr[history]], counts)

    elapsed = np.maximum(t_ref - stamps, 1).astype(np.float64)
    activation = np.bincount(items, weights=elapsed**-decay, minlength=sessions.n_items)

    seen = np.flatnonzero(activation)
    weights = np.log(activation[seen])
    weights = np.exp(weights - weights.max())
    out = np.zeros(sessions.n_items, dtype=np.float64)
    out[seen] = weights / weights.sum()
    return out


class ACTR:
    """`actr`: BLL plus spreading activation, untrained."""

    name = "actr"

    def __init__(self, decay: float = BLL_DECAY, progress: bool = True) -> None:
        self.decay = decay
        self.progress = progress
        self.sessions: Sessions | None = None
        self.adjacency: sp.csr_matrix | None = None
        self.candidates: list[np.ndarray] = []

    def fit(self, sessions: Sessions) -> ACTR:
        self.sessions = sessions
        self.adjacency = session_cooccurrence(sessions, progress=self.progress)
        self.candidates = [sessions.history(u) for u in range(sessions.n_users)]
        return self

    def bll(self, user: int, target: int) -> np.ndarray:
        assert self.sessions is not None
        return bll_activation(self.sessions, user, target, self.decay)

    def spread(self, target: int) -> np.ndarray:
        """Co-occurrence mass reaching each track from the single preceding session."""
        assert self.sessions is not None and self.adjacency is not None
        previous = self.sessions.session_items(target - 1)
        return np.asarray(self.adjacency[previous].sum(axis=0)).flatten()

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        assert self.sessions is not None
        scores = np.full((len(requests), self.sessions.n_items), -np.inf)
        for i, request in enumerate(requests):
            activation = self.bll(request.user, request.target) + self.spread(request.target)
            candidates = self.candidates[request.user]
            scores[i, candidates] = activation[candidates]
        return scores


MODELS = {"actr": ACTR}
