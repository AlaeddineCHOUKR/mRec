"""Tokenizers: the protocol's promises, and the properties each family must have."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.tokenizers import (
    ProductQuantizer,
    RandomCodes,
    ResidualKMeans,
    Tokenizer,
    apply_collision_policy,
    collisions,
)

N, DIM = 400, 16
FAMILIES = [
    lambda: ResidualKMeans(n_levels=3, codebook_size=8, seed=0, gpu=False),
    lambda: ProductQuantizer(n_levels=2, codebook_size=8, seed=0, gpu=False),
    lambda: RandomCodes(n_levels=3, codebook_size=8, seed=0),
]


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(6, DIM)) * 3
    return (centres[rng.integers(0, 6, N)] + rng.normal(size=(N, DIM))).astype(np.float32)


@pytest.mark.parametrize("build", FAMILIES)
def test_every_family_satisfies_the_protocol(build, embeddings):
    tokenizer = build().fit(embeddings)
    assert isinstance(tokenizer, Tokenizer)
    codes = tokenizer.encode(embeddings)
    assert codes.shape == (N, tokenizer.n_levels)
    assert codes.dtype == np.int32
    assert codes.min() >= 0 and codes.max() < tokenizer.codebook_size


@pytest.mark.parametrize("build", FAMILIES)
def test_encoding_is_deterministic(build, embeddings):
    tokenizer = build().fit(embeddings)
    assert np.array_equal(tokenizer.encode(embeddings), tokenizer.encode(embeddings))


@pytest.mark.parametrize("build", FAMILIES)
def test_unseen_items_encode_without_error(build, embeddings):
    """The requirement that makes P3's day-0 cold start possible."""
    tokenizer = build().fit(embeddings[: N // 2])
    codes = tokenizer.encode(embeddings[N // 2 :])
    assert codes.shape == (N // 2, tokenizer.n_levels)
    assert np.isfinite(codes).all()


def test_residual_levels_reduce_reconstruction_error_monotonically(embeddings):
    """The defining property of a residual family: each level explains what is left."""
    errors = []
    for levels in (1, 2, 3, 4):
        tokenizer = ResidualKMeans(n_levels=levels, codebook_size=8, seed=0, gpu=False)
        tokenizer.fit(embeddings)
        reconstruction = tokenizer.decode(tokenizer.encode(embeddings))
        errors.append(float(np.mean((embeddings - reconstruction) ** 2)))
    assert all(a >= b for a, b in zip(errors[:-1], errors[1:], strict=True))


def test_product_quantizer_reconstructs_each_slice_independently(embeddings):
    tokenizer = ProductQuantizer(n_levels=2, codebook_size=8, seed=0, gpu=False).fit(embeddings)
    reconstruction = tokenizer.decode(tokenizer.encode(embeddings))
    assert reconstruction.shape == embeddings.shape
    assert np.mean((embeddings - reconstruction) ** 2) < np.mean(embeddings**2)


def test_random_codes_carry_no_geometry(embeddings):
    """The null tokenizer must not reconstruct: it bounds what the decoder alone buys."""
    tokenizer = RandomCodes(n_levels=3, codebook_size=8, seed=0).fit(embeddings)
    reconstruction = tokenizer.decode(tokenizer.encode(embeddings))
    assert np.allclose(reconstruction, embeddings.mean(axis=0), atol=1e-5)


def test_collisions_are_counted_per_item_not_per_bucket():
    codes = np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1]])
    report = collisions(codes)
    assert report.n_distinct == 3
    assert report.n_colliding == 5  # the lone [0, 1] owns its code; the other five share
    assert report.largest_bucket == 3
    assert report.collision_rate == pytest.approx(5 / 6)


def test_extra_token_policy_makes_every_code_unique():
    codes = np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1]])
    resolved = apply_collision_policy(codes, "extra_token")
    assert resolved.shape == (6, 3)
    assert collisions(resolved).n_colliding == 0
    assert np.array_equal(resolved[:, :2], codes)


def test_error_policy_refuses_a_colliding_catalog():
    codes = np.array([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="collide"):
        apply_collision_policy(codes, "error")
    assert apply_collision_policy(np.array([[0, 0], [0, 1]]), "error").shape == (2, 2)


def test_allow_policy_leaves_the_codes_alone():
    codes = np.array([[0, 0], [0, 0]])
    assert np.array_equal(apply_collision_policy(codes, "allow"), codes)


def test_unknown_collision_policy_fails_loudly():
    with pytest.raises(ValueError, match="unknown collision policy"):
        apply_collision_policy(np.zeros((2, 2), dtype=np.int32), "ignore")
