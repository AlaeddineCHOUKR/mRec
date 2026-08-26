"""Training-set popularity, computed from training sessions only — `EVALUATION_PROTOCOL.md` §6."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from mrec.data.sessions import Sessions


def train_popularity(sessions: Sessions, progress: bool = False) -> np.ndarray:
    """Global popularity: the fraction of users whose training history holds the item.

    This is a user-coverage statistic, not an event count — the reference computes
    it over the distinct training tracks of each user and divides by the user count.
    """
    counts = np.zeros(sessions.n_items, dtype=np.int64)
    users = range(sessions.n_users)
    for u in tqdm(users, desc="train popularity", disable=not progress):
        counts[sessions.history(u)] += 1
    return counts / sessions.n_users
