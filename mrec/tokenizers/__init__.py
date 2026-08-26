"""Tokenizers: one protocol, several families."""

from __future__ import annotations

from mrec.tokenizers.base import (
    COLLISION_POLICIES,
    CollisionReport,
    Tokenizer,
    apply_collision_policy,
    collisions,
)
from mrec.tokenizers.quantizers import (
    ArtistHierarchy,
    ProductQuantizer,
    RandomCodes,
    ResidualKMeans,
)

TOKENIZERS = {
    "residual_kmeans": ResidualKMeans,
    "product_quantizer": ProductQuantizer,
    "random": RandomCodes,
    "artist": ArtistHierarchy,
}

__all__ = [
    "COLLISION_POLICIES",
    "TOKENIZERS",
    "ArtistHierarchy",
    "CollisionReport",
    "ProductQuantizer",
    "RandomCodes",
    "ResidualKMeans",
    "Tokenizer",
    "apply_collision_policy",
    "collisions",
]
