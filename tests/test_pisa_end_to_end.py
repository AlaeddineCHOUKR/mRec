"""PISA trained and evaluated end to end on the synthetic fixture, on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.data.build import build
from mrec.data.sessions import load_sessions
from mrec.eval.runner import build_requests, evaluate, top_k
from mrec.models import build_model

SEQLEN, FAVS = 3, 4


@pytest.fixture
def sessions(raw_config):
    return load_sessions(build(raw_config, skip_fetch=True))


@pytest.fixture
def trained(sessions):
    model = build_model(
        "pisa",
        model={
            "embedding_dim": 4,
            "seqlen": SEQLEN,
            "session_len": 10,
            "num_blocks": 1,
            "num_heads": 2,
            "num_favs": FAVS,
        },
        training={"epochs": 3, "batch_size": 4, "device": "cpu", "patience": 3},
        n_valid_users=2,
        progress=False,
    )
    return model.fit(sessions)


def test_training_produces_a_loss_curve_and_a_best_epoch(trained):
    assert len(trained.log) >= 1
    assert all(np.isfinite(row["train_loss"]) for row in trained.log)
    assert all(np.isfinite(row["valid_loss"]) for row in trained.log)
    assert trained.summary["best_epoch"] >= 0


def test_scores_cover_the_catalog_and_can_be_ranked(trained, sessions):
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=SEQLEN)
    scores = trained.score(requests[:4])
    assert scores.shape == (4, sessions.n_items)
    assert np.isfinite(scores).all()
    ranked = top_k(scores, 3)
    assert ranked.shape == (4, 3)
    assert ranked.min() >= 0 and ranked.max() < sessions.n_items


def test_unseen_tracks_are_rankable_not_masked_out(trained, sessions):
    """Unlike `ptop` and `actr`, PISA scores the whole catalog, so exploration is reachable.

    The fixture catalog is too small for an unseen track to actually win a slot, so the
    property under test is that unseen items carry real scores rather than `-inf`.
    """
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=SEQLEN)
    scores = trained.score(requests)
    for request, row in zip(requests, scores, strict=True):
        unseen = np.setdiff1d(np.arange(sessions.n_items), sessions.history(request.user))
        assert len(unseen) > 0
        assert np.isfinite(row[unseen]).all()


def test_evaluate_runs_the_full_protocol_loop(trained, sessions):
    metrics = evaluate(
        trained,
        sessions,
        which="test",
        k=3,
        seqlen=SEQLEN,
        n_users=2,
        seeds=(1013,),
        batch_size=2,
        progress=False,
    )
    assert metrics.columns == ["seed", "metric", "k", "value"]
    assert metrics["value"].is_finite().all()
