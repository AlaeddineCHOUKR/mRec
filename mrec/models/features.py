"""Training windows and the ACT-R feature tensors PISA consumes.

One training example is a window of `seqlen` consecutive sessions, and PISA predicts
*the session after each of them* — so a single example carries `seqlen` next-session
targets, not one. Every tensor is `[seqlen, SESSION_LEN]`.

Item ids here are **1-based**, with 0 reserved for padding, matching the reference's
zero-padded embedding table. Everything outside this module speaks 0-based `item_idx`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mrec.data.sessions import Sessions
from mrec.models.actr import BLL_DECAY, bll_activation

SESSION_LEN = 10
SEQLEN = 30
NUM_FAVS = 20
SAMPLES_STEP = 20


def session_item_matrix(sessions: Sessions) -> np.ndarray:
    """Every session as a padded row of 0-based item ids, `-1` for padding. `# [n_sessions, S]`"""
    out = np.full((sessions.n_sessions, SESSION_LEN), -1, dtype=np.int64)
    counts = np.diff(sessions.session_ptr)
    session_of_event = np.repeat(np.arange(sessions.n_sessions), counts)
    position = np.arange(sessions.n_events) - sessions.session_ptr[session_of_event]
    keep = position < SESSION_LEN
    out[session_of_event[keep], position[keep]] = np.asarray(sessions.event_item)[keep]
    return out


def precompute_spread(
    sessions: Sessions, adjacency: sp.csr_matrix, chunk: int = 300_000
) -> np.ndarray:
    """Spreading activation of every session against the one before it. `# [n_sessions, S]`

    This is a property of a *session*, not of the window that contains it, so the whole
    dataset is computed once instead of per training example — the difference between
    half an hour and half a minute. Entries are looked up by encoding `(row, column)`
    into a single sorted key and binary-searching the matrix's own index arrays.

    Indexed `adjacency[current, previous]`, the orientation `_build_spread_weights` uses;
    stand-alone ACT-R reads the transpose. Both are the reference's.
    """
    n_items = sessions.n_items
    adjacency = adjacency.tocsr()
    adjacency.sort_indices()
    adj_keys = (
        np.repeat(np.arange(n_items, dtype=np.int64), np.diff(adjacency.indptr)) * n_items
        + adjacency.indices
    )
    adj_data = adjacency.data

    items = session_item_matrix(sessions)
    out = np.zeros((sessions.n_sessions, SESSION_LEN), dtype=np.float32)
    # a user's first session has no predecessor of its own
    has_previous = np.ones(sessions.n_sessions, dtype=bool)
    has_previous[sessions.user_ptr[:-1]] = False
    targets = np.flatnonzero(has_previous)

    for start in range(0, len(targets), chunk):
        block = targets[start : start + chunk]
        current = items[block]  # [B, S]
        previous = items[block - 1]  # [B, S]
        rows = np.repeat(current, SESSION_LEN, axis=1)  # [B, S*S]
        cols = np.tile(previous, SESSION_LEN)  # [B, S*S]
        valid = (rows >= 0) & (cols >= 0)
        keys = np.where(valid, rows * n_items + cols, -1)

        found = np.searchsorted(adj_keys, keys)
        np.clip(found, 0, len(adj_keys) - 1, out=found)
        hit = valid & (adj_keys[found] == keys)
        values = np.where(hit, adj_data[found], 0.0)
        out[block] = values.reshape(len(block), SESSION_LEN, SESSION_LEN).sum(axis=2)
    return out


def training_windows(sessions: Sessions, seqlen: int = SEQLEN, step: int = SAMPLES_STEP):
    """`(user, target)` pairs for training, on the reference's stride.

    `_load_train_session_indexes` walks `range(len(sessions) - 22, seqlen, -step)`.
    Note the `- 22`, not `- 21`: the session immediately before the held-out tail is
    never used as a target, and matching that keeps epoch sizes comparable.
    """
    windows = []
    for u in range(sessions.n_users):
        first = int(sessions.user_ptr[u])
        n_sessions = sessions.n_user_sessions(u)
        last = n_sessions - sessions.split.n_held_out - 2
        for offset in range(last, seqlen, -step):
            windows.append((u, first + offset))
    return windows


def session_windows(sessions: Sessions, windows, seqlen: int = SEQLEN):
    """Input and target sessions as 0-based item ids, `-1` for padding. `# 2 x [N, L, S]`

    The identifier-only view of a window: no ACT-R tensors, for models that read the
    sessions themselves rather than their activations.
    """
    matrix = session_item_matrix(sessions)
    starts = np.array([target - seqlen for _, target in windows])
    offsets = np.arange(seqlen)
    index = starts[:, None] + offsets[None, :]
    return matrix[index], matrix[index + 1]


@dataclass(frozen=True)
class Example:
    """One PISA example. `pos`/`neg` are absent for evaluation requests."""

    user: int
    seq_in: np.ndarray  # [L, S] int64, 1-based ids
    seq_bll: np.ndarray  # [L, S] float32
    seq_spread: np.ndarray  # [L, S] float32
    fav_ids: np.ndarray  # [F] int64
    fav_bll: np.ndarray  # [F] float32
    pos: np.ndarray | None = None  # [L, S] int64
    pos_bll: np.ndarray | None = None  # [L, S] float32
    pos_spread: np.ndarray | None = None  # [L, S] float32
    neg: np.ndarray | None = None  # [L, S] int64


class ActrFeatures:
    """Builds PISA's inputs from the session store and the co-occurrence matrix."""

    def __init__(
        self,
        sessions: Sessions,
        adjacency: sp.csr_matrix,
        seqlen: int = SEQLEN,
        num_favs: int = NUM_FAVS,
        decay: float = BLL_DECAY,
        spread_table: np.ndarray | None = None,
    ) -> None:
        self.sessions = sessions
        self.adjacency = adjacency
        self.seqlen = seqlen
        self.num_favs = num_favs
        self.decay = decay
        self.spread = spread_table if spread_table is not None else precompute_spread(
            sessions, adjacency
        )

    def _session_matrix(self, session_ids: np.ndarray) -> np.ndarray:
        """Sessions as `[n, SESSION_LEN]` 1-based ids, zero-padded. `# [n, S]`"""
        out = np.zeros((len(session_ids), SESSION_LEN), dtype=np.int64)
        for i, s in enumerate(session_ids):
            items = self.sessions.session_items(int(s))[:SESSION_LEN]
            out[i, : len(items)] = np.asarray(items) + 1
        return out

    def _window_sessions(self, target: int) -> np.ndarray:
        """The `seqlen` sessions preceding `target`, right-aligned like the reference."""
        return np.arange(target - self.seqlen, target)

    def _favourites(self, activation: np.ndarray):
        """The `num_favs` items of highest base-level activation, zero-padded."""
        ids = np.zeros(self.num_favs, dtype=np.int64)
        scores = np.zeros(self.num_favs, dtype=np.float32)
        seen = np.flatnonzero(activation)
        top = seen[np.argsort(-activation[seen], kind="stable")][: self.num_favs]
        ids[: len(top)] = top + 1
        scores[: len(top)] = activation[top]
        return ids, scores

    def evaluation_example(self, user: int, target: int) -> Example:
        activation = bll_activation(self.sessions, user, target, self.decay)
        window = self._window_sessions(target)
        seq_in = self._session_matrix(window)
        fav_ids, fav_bll = self._favourites(activation)
        return Example(
            user=user,
            seq_in=seq_in,
            seq_bll=self._lookup(activation, seq_in),
            seq_spread=self.spread[window],
            fav_ids=fav_ids,
            fav_bll=fav_bll,
        )

    def training_example(
        self, user: int, target: int, rng: np.random.Generator, with_negatives: bool = True
    ) -> Example:
        """A window whose `i`-th position predicts the session after input `i`."""
        activation = bll_activation(self.sessions, user, target, self.decay)
        window = self._window_sessions(target)
        following = window + 1

        seq_in = self._session_matrix(window)
        pos = self._session_matrix(following)
        fav_ids, fav_bll = self._favourites(activation)
        history = self.sessions.history(user)
        return Example(
            user=user,
            seq_in=seq_in,
            seq_bll=self._lookup(activation, seq_in),
            seq_spread=self.spread[window],
            fav_ids=fav_ids,
            fav_bll=fav_bll,
            pos=pos,
            pos_bll=self._lookup(activation, pos),
            pos_spread=self.spread[following],
            neg=self._negatives(history, rng) if with_negatives else None,
        )

    @staticmethod
    def _lookup(activation: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Activation of each id, zero where the slot is padding. `# [L, S]`"""
        out = np.zeros(ids.shape, dtype=np.float32)
        filled = ids > 0
        out[filled] = activation[ids[filled] - 1]
        return out

    def precompute(self, windows, progress: bool = True) -> dict[str, np.ndarray]:
        """Materialize every window's ACT-R tensors once.

        Each window needs only ~620 floats, so the whole training set fits in under a
        gigabyte — far cheaper than recomputing base-level activation every epoch, which
        dominates the step time otherwise.
        """
        from tqdm import tqdm

        n = len(windows)
        shape = (n, self.seqlen, SESSION_LEN)
        out = {
            "user": np.zeros(n, dtype=np.int32),
            "seq_in": np.zeros(shape, dtype=np.int32),
            "seq_bll": np.zeros(shape, dtype=np.float32),
            "seq_spread": np.zeros(shape, dtype=np.float32),
            "pos": np.zeros(shape, dtype=np.int32),
            "pos_bll": np.zeros(shape, dtype=np.float32),
            "pos_spread": np.zeros(shape, dtype=np.float32),
            "fav_ids": np.zeros((n, self.num_favs), dtype=np.int32),
            "fav_bll": np.zeros((n, self.num_favs), dtype=np.float32),
        }
        rng = np.random.default_rng(0)  # unused: negatives are drawn per epoch instead
        for i, (user, target) in enumerate(tqdm(windows, desc="features", disable=not progress)):
            example = self.training_example(user, target, rng, with_negatives=False)
            out["user"][i] = user
            for field in ("seq_in", "seq_bll", "seq_spread", "pos", "pos_bll", "pos_spread"):
                out[field][i] = getattr(example, field)
            out["fav_ids"][i] = example.fav_ids
            out["fav_bll"][i] = example.fav_bll
        return out

    def _negatives(self, history: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Uniform items the user has never played, resampled until none is known.

        The reference rejects against the user's *whole* training history rather than
        against the target session, so a negative is genuinely unheard.
        """
        n_items = self.sessions.n_items
        out = rng.integers(0, n_items, size=(self.seqlen, SESSION_LEN))
        known = np.isin(out, history)
        while known.any():
            out[known] = rng.integers(0, n_items, size=int(known.sum()))
            known = np.isin(out, history)
        return out + 1


def repeat_labels(sessions: Sessions) -> np.ndarray:
    """Whether each played track was already heard by that user. `# [n_sessions, S] bool`

    The label a repeat/explore control token is trained against. It uses the *prefix*
    history — every earlier session of the same user — not the `sessions[:-20]` cut the
    metrics use, because a training target at session `t` must not be told about
    sessions after `t`.

    The two cuts agree in the direction that matters: a track in the prefix history is
    also in the evaluation history, so a target the model was taught to call a repeat is
    scored as one. The two cuts differ, and this is where that is resolved rather than
    assumed away.
    """
    labels = np.zeros((sessions.n_sessions, SESSION_LEN), dtype=bool)
    for user in range(sessions.n_users):
        first, last = int(sessions.user_ptr[user]), int(sessions.user_ptr[user + 1])
        if last <= first:
            continue
        bounds = sessions.session_ptr[first : last + 1]
        start = int(bounds[0])
        items = sessions.event_item[start : int(bounds[-1])]
        if not len(items):
            continue

        counts = np.diff(bounds)
        session_of_event = np.repeat(np.arange(last - first), counts)
        slot_of_event = np.arange(len(items)) - np.repeat(bounds[:-1] - start, counts)

        _, first_index, inverse = np.unique(items, return_index=True, return_inverse=True)
        seen_earlier = session_of_event > session_of_event[first_index][inverse]
        labels[first + session_of_event, slot_of_event] = seen_earlier
    return labels


def history_mean_embedding(sessions: Sessions, table: np.ndarray) -> np.ndarray:
    """Each user's training history as one vector. `# [n_users, d] float32`

    The plain alternative the ACT-R soft prompt has to beat: GLIDE injects a long-term
    collaborative user embedding, and averaging the user's own item vectors is the
    version of that available here. Computed over `sessions[:-20]` only.
    """
    out = np.zeros((sessions.n_users, table.shape[1]), dtype=np.float32)
    for user in range(sessions.n_users):
        history = sessions.history(user)
        if len(history):
            out[user] = table[history].mean(axis=0)
    return out
