"""PISA: shapes, masking, and the pieces where a port silently goes wrong."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrec.models.pisa import PISA, PisaConfig, pisa_loss

DIM, ITEMS = 8, 40
CFG = PisaConfig(embedding_dim=DIM, seqlen=4, session_len=3, num_blocks=2, num_heads=2, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    table = rng.normal(size=(ITEMS, DIM)).astype(np.float32)
    return PISA(ITEMS, table, CFG)


def batch(rng, batch_size=3, pad_last=True):
    L, S, F = CFG.seqlen, CFG.session_len, CFG.num_favs
    ids = rng.integers(1, ITEMS + 1, size=(batch_size, L, S))
    if pad_last:
        ids[:, 0, :] = 0  # an entirely padded (absent) session at the front
    out = {
        "seq_in": torch.as_tensor(ids),
        "seq_bll": torch.rand(batch_size, L, S),
        "seq_spread": torch.rand(batch_size, L, S),
        "fav_ids": torch.as_tensor(rng.integers(1, ITEMS + 1, size=(batch_size, F))),
        "fav_bll": torch.rand(batch_size, F),
    }
    return out


def test_padding_row_of_the_embedding_table_stays_zero(model):
    assert torch.equal(model.items[0], torch.zeros(DIM))
    model.item_embedding.data.add_(1.0)
    assert torch.equal(model.items[0], torch.zeros(DIM))


def test_forward_shapes(model):
    short, long = model(**batch(np.random.default_rng(0)))
    assert short.shape == (3, CFG.seqlen, DIM)
    assert long.shape == (3, CFG.seqlen, DIM)
    assert torch.isfinite(short).all() and torch.isfinite(long).all()


def test_representations_are_unit_norm_where_the_session_exists(model):
    inputs = batch(np.random.default_rng(1))
    short, _ = model(**inputs)
    present = (inputs["seq_in"] != 0).sum(-1) != 0
    norms = short.norm(dim=-1)
    assert torch.allclose(norms[present], torch.ones_like(norms[present]), atol=1e-5)


def test_session_weights_are_a_normalized_distribution(model):
    inputs = batch(np.random.default_rng(2), pad_last=False)
    rep, emb = model.session_representation(
        inputs["seq_in"], inputs["seq_bll"], inputs["seq_spread"], inputs["seq_in"]
    )
    # the session vector is a convex combination, so it cannot exceed the item norms
    assert rep.shape == (3, CFG.seqlen, DIM)
    assert (rep.norm(dim=-1) <= emb.norm(dim=-1).max(dim=-1).values + 1e-4).all()


def test_causality_hides_the_future(model):
    """Changing a later session must not move an earlier position's representation."""
    rng = np.random.default_rng(3)
    inputs = batch(rng, batch_size=1, pad_last=False)
    model.eval()
    with torch.no_grad():
        before, _ = model(**inputs)
        changed = {k: v.clone() for k, v in inputs.items()}
        changed["seq_in"][0, -1] = torch.as_tensor(
            rng.integers(1, ITEMS + 1, size=CFG.session_len)
        )
        after, _ = model(**changed)
    assert torch.allclose(before[0, :-1], after[0, :-1], atol=1e-5)
    assert not torch.allclose(before[0, -1], after[0, -1], atol=1e-5)


def test_catalog_scores_cover_every_item(model):
    short, long = model(**batch(np.random.default_rng(4)))
    scores = model.score_catalog(short, long)
    assert scores.shape == (3, ITEMS + 1)
    assert torch.isfinite(scores).all()


def test_loss_is_finite_and_has_gradients(model):
    rng = np.random.default_rng(5)
    inputs = batch(rng)
    short, long = model(**inputs)
    example = dict(inputs)
    example["pos"] = torch.as_tensor(
        rng.integers(1, ITEMS + 1, size=(3, CFG.seqlen, CFG.session_len))
    )
    example["neg"] = torch.as_tensor(
        rng.integers(1, ITEMS + 1, size=(3, CFG.seqlen, CFG.session_len))
    )
    example["pos_bll"] = torch.rand(3, CFG.seqlen, CFG.session_len)
    example["pos_spread"] = torch.rand(3, CFG.seqlen, CFG.session_len)
    example["long"] = long

    loss = pisa_loss(model, short, example)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.actr_weights.grad is not None


def test_a_training_step_reduces_the_loss_on_one_batch(model):
    """Smoke only: the model must be able to fit a single batch."""
    rng = np.random.default_rng(6)
    inputs = batch(rng)
    example = dict(inputs)
    example["pos"] = torch.as_tensor(
        rng.integers(1, ITEMS + 1, size=(3, CFG.seqlen, CFG.session_len))
    )
    example["neg"] = torch.as_tensor(
        rng.integers(1, ITEMS + 1, size=(3, CFG.seqlen, CFG.session_len))
    )
    example["pos_bll"] = torch.rand(3, CFG.seqlen, CFG.session_len)
    example["pos_spread"] = torch.rand(3, CFG.seqlen, CFG.session_len)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.98))
    losses = []
    for _ in range(20):
        short, long = model(**inputs)
        example["long"] = long
        loss = pisa_loss(model, short, example)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
