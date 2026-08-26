"""Seeding. Randomness flows from an explicit integer, never from global state."""

from __future__ import annotations

import random

import numpy as np


def rng(seed: int) -> np.random.Generator:
    """A fresh generator; the only randomness source new code should use."""
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    """Seed the legacy global RNGs, for code that cannot take a generator."""
    random.seed(seed)
    np.random.seed(seed)
