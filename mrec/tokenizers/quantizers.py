"""Classical quantizer families behind the `Tokenizer` protocol.

All of them are k-means at heart, differing in *what* they cluster:

- `ResidualKMeans` — level `l` clusters what levels `0..l-1` failed to explain. This is
  the classical stand-in for RQ-VAE and the family TIGER-style semantic IDs come from.
- `ProductQuantizer` — each level owns a disjoint slice of the dimensions, so codes are
  independent rather than hierarchical.
- `RandomCodes` — the null tokenizer. It is in the table to bound how much of a
  downstream metric is the *decoder* rather than the clustering.
- `ArtistHierarchy` — metadata rather than geometry: artist, then position within it.

Fitting uses faiss so the GPU does the work; every one of them encodes unseen items,
since encoding is nearest-centroid and never a lookup.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _kmeans(x: np.ndarray, k: int, seed: int, n_iter: int = 25, gpu: bool = True) -> np.ndarray:
    """Centroids of `x`, `[min(k, len(x)), d]`. faiss on GPU, with a numpy fallback."""
    k = min(k, len(x))
    x = np.ascontiguousarray(x, dtype=np.float32)
    try:
        import faiss

        km = faiss.Kmeans(x.shape[1], k, niter=n_iter, seed=seed, gpu=gpu, verbose=False)
        km.train(x)
        return np.asarray(km.centroids, dtype=np.float32)
    except ImportError:  # pragma: no cover - exercised only where faiss is absent
        logger.warning("faiss unavailable; falling back to numpy k-means")
        return _numpy_kmeans(x, k, seed, n_iter)


def _numpy_kmeans(x: np.ndarray, k: int, seed: int, n_iter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(n_iter):
        assignment = _assign(x, centroids)
        for j in range(k):
            members = x[assignment == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return centroids


def _assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid per row, by squared euclidean distance. `# [n]`"""
    distances = (x**2).sum(1)[:, None] - 2 * x @ centroids.T + (centroids**2).sum(1)
    return np.argmin(distances, axis=1).astype(np.int32)


class ResidualKMeans:
    """Hierarchical codes: each level quantizes the previous level's residual."""

    def __init__(
        self, n_levels: int = 3, codebook_size: int = 256, seed: int = 0, gpu: bool = True
    ) -> None:
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.seed = seed
        self.gpu = gpu
        self.codebooks: list[np.ndarray] = []

    def fit(self, embeddings: np.ndarray) -> ResidualKMeans:
        residual = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.codebooks = []
        for level in range(self.n_levels):
            centroids = _kmeans(residual, self.codebook_size, self.seed + level, gpu=self.gpu)
            self.codebooks.append(centroids)
            residual = residual - centroids[_assign(residual, centroids)]
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:  # [n, n_levels]
        residual = np.ascontiguousarray(embeddings, dtype=np.float32)
        codes = np.empty((len(embeddings), self.n_levels), dtype=np.int32)
        for level, centroids in enumerate(self.codebooks):
            codes[:, level] = _assign(residual, centroids)
            residual = residual - centroids[codes[:, level]]
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:  # [n, d]
        out = np.zeros((len(codes), self.codebooks[0].shape[1]), dtype=np.float32)
        for level, centroids in enumerate(self.codebooks):
            out += centroids[codes[:, level]]
        return out


class ProductQuantizer:
    """Each level quantizes its own slice of the dimensions — flat, not hierarchical."""

    def __init__(
        self, n_levels: int = 3, codebook_size: int = 256, seed: int = 0, gpu: bool = True
    ) -> None:
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.seed = seed
        self.gpu = gpu
        self.codebooks: list[np.ndarray] = []
        self.slices: list[slice] = []

    def fit(self, embeddings: np.ndarray) -> ProductQuantizer:
        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        bounds = np.linspace(0, x.shape[1], self.n_levels + 1).astype(int)
        self.slices = [slice(bounds[i], bounds[i + 1]) for i in range(self.n_levels)]
        self.codebooks = [
            _kmeans(x[:, part], self.codebook_size, self.seed + i, gpu=self.gpu)
            for i, part in enumerate(self.slices)
        ]
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        codes = np.empty((len(x), self.n_levels), dtype=np.int32)
        for level, (part, centroids) in enumerate(zip(self.slices, self.codebooks, strict=True)):
            codes[:, level] = _assign(x[:, part], centroids)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        dim = sum(part.stop - part.start for part in self.slices)
        out = np.zeros((len(codes), dim), dtype=np.float32)
        for level, (part, centroids) in enumerate(zip(self.slices, self.codebooks, strict=True)):
            out[:, part] = centroids[codes[:, level]]
        return out


class RandomCodes:
    """The null tokenizer: codes carry no information about the embedding.

    Unseen items are assigned deterministically from their own content, so the protocol
    holds — but the assignment is a hash, not a clustering.
    """

    def __init__(self, n_levels: int = 3, codebook_size: int = 256, seed: int = 0) -> None:
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.seed = seed
        self.projection: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> RandomCodes:
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(size=(embeddings.shape[1], self.n_levels)).astype(np.float32)
        self.mean = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.projection is not None
        scores = np.asarray(embeddings, dtype=np.float32) @ self.projection
        digits = np.abs(np.round(scores * 1e6).astype(np.int64))
        return (digits % self.codebook_size).astype(np.int32)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        return np.repeat(self.mean[None, :], len(codes), axis=0)


class ArtistHierarchy:
    """Metadata semantic ID: artist at level 0, position within the artist at level 1.

    The catalog has only 538 artists over 50k tracks, so level 0 is a coarse 538-way
    split over 538 artists. An unseen item needs its artist supplied; without
    one it falls into a reserved bucket rather than failing.
    """

    def __init__(self, artists: np.ndarray, codebook_size: int = 1024) -> None:
        self.artists = np.asarray(artists)
        self.n_levels = 2
        self.codebook_size = codebook_size
        self.artist_index: dict[int, int] = {}
        self.position: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> ArtistHierarchy:
        unique = np.unique(self.artists)
        self.artist_index = {int(a): i for i, a in enumerate(unique)}
        self.position = np.zeros(len(self.artists), dtype=np.int32)
        for artist in unique:
            members = np.flatnonzero(self.artists == artist)
            self.position[members] = np.arange(len(members)) % self.codebook_size
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.position is not None
        if len(embeddings) != len(self.artists):
            raise ValueError(
                "ArtistHierarchy encodes the catalog it was fit on; "
                "an unseen item must arrive with its artist"
            )
        level0 = np.array([self.artist_index[int(a)] for a in self.artists], dtype=np.int32)
        return np.stack([level0, self.position], axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError("a metadata hierarchy has no reconstruction")
