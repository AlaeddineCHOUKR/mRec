"""The one interface the three projects share, and its collision policy.

`encode` must accept embeddings of items never seen during `fit` — that single
requirement is what makes P3's day-0 cold start possible, and it is why learned and
classical quantizers sit behind one protocol instead of being written per experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

COLLISION_POLICIES = ("error", "extra_token", "allow")


@runtime_checkable
class Tokenizer(Protocol):
    n_levels: int
    codebook_size: int

    def fit(self, embeddings: np.ndarray) -> Tokenizer: ...

    def encode(self, embeddings: np.ndarray) -> np.ndarray:  # [n, n_levels] int32
        ...

    def decode(self, codes: np.ndarray) -> np.ndarray:  # [n, d]
        ...


@dataclass(frozen=True)
class CollisionReport:
    n_items: int
    n_distinct: int
    n_colliding: int
    largest_bucket: int

    @property
    def collision_rate(self) -> float:
        """Share of items that do not own their code outright."""
        return self.n_colliding / self.n_items if self.n_items else 0.0


def collisions(codes: np.ndarray) -> CollisionReport:
    """How many items share a semantic ID with another item."""
    _, inverse, counts = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    sizes = counts[inverse]
    return CollisionReport(
        n_items=len(codes),
        n_distinct=len(counts),
        n_colliding=int((sizes > 1).sum()),
        largest_bucket=int(counts.max()) if len(counts) else 0,
    )


def apply_collision_policy(codes: np.ndarray, policy: str = "extra_token") -> np.ndarray:
    """Resolve identical codes, returning `[n, n_levels (+1)]`.

    - `error` — refuse; a collision means the tokenizer cannot address the catalog.
    - `extra_token` — append TIGER's disambiguation index, so every item is addressable.
    - `allow` — leave them, and let the decoder be unable to separate the items.

    The choice changes what the decoder can express, so it is a measured variable in P1
    rather than an implementation detail.
    """
    if policy not in COLLISION_POLICIES:
        raise ValueError(f"unknown collision policy `{policy}`; known: {COLLISION_POLICIES}")
    report = collisions(codes)
    if policy == "error":
        if report.n_colliding:
            raise ValueError(
                f"{report.n_colliding} of {report.n_items} items collide "
                f"(largest bucket {report.largest_bucket})"
            )
        return codes
    if policy == "allow":
        return codes

    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    extra = np.empty(len(codes), dtype=codes.dtype)
    ranks = np.zeros(inverse.max() + 1, dtype=np.int64)
    for index in order:
        bucket = inverse[index]
        extra[index] = ranks[bucket]
        ranks[bucket] += 1
    return np.concatenate([codes, extra[:, None]], axis=1)
