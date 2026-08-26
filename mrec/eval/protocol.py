"""Protocol constants and the seeded evaluation cohort.

The five seeds vary *which users are scored*, not model initialization: the reference
shuffles the user list with `RandomState(seed)` and takes the first `n_users`, so a
reported number is a mean over five cohorts drawn from one trained model.
"""

from __future__ import annotations

import numpy as np

EVAL_SEEDS = (1013, 2791, 4357, 6199, 7907)
N_EVAL_USERS = 3000
SEQLEN = 30
K = 10


def sample_eval_users(user_idx: np.ndarray, n_users: int, seed: int) -> np.ndarray:
    """The evaluation cohort for one seed."""
    shuffled = np.array(user_idx, copy=True)
    np.random.RandomState(seed).shuffle(shuffled)
    return shuffled[:n_users]
