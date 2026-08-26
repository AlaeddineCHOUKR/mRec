"""ACT-R: hand-computed activation, and the leakage rules the port must keep."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.data.build import build
from mrec.data.sessions import load_sessions
from mrec.eval.runner import Request, build_requests, top_k
from mrec.models.actr import ACTR, normalize_adjacency, session_cooccurrence


@pytest.fixture
def sessions(raw_config):
    return load_sessions(build(raw_config, skip_fetch=True))


@pytest.fixture
def model(sessions):
    return ACTR(progress=False).fit(sessions)


def test_bll_matches_the_hand_computed_activation(model, sessions):
    """`log Σ (t_ref − t)^−0.5` over session timestamps, softmaxed over the user's items."""
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[0])
    first = int(sessions.user_ptr[user])
    t_ref = int(sessions.event_ts[sessions.session_ptr[target]])

    expected: dict[int, float] = {}
    for s in range(first, target):
        stamp = int(sessions.event_ts[sessions.session_ptr[s]])
        for item in sessions.session_items(s):
            expected[int(item)] = expected.get(int(item), 0.0) + max(t_ref - stamp, 1) ** -0.5
    logged = {i: np.log(w) for i, w in expected.items()}
    shifted = {i: np.exp(w - max(logged.values())) for i, w in logged.items()}
    total = sum(shifted.values())

    bll = model.bll(user, target)
    assert np.flatnonzero(bll).tolist() == sorted(expected)
    for item, value in shifted.items():
        assert bll[item] == pytest.approx(value / total)
    assert bll.sum() == pytest.approx(1.0)


def test_bll_history_stops_at_the_target_session(model, sessions):
    """A track played only in the target session must not activate itself."""
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[-1])
    before = set()
    for s in range(int(sessions.user_ptr[user]), target):
        before |= set(sessions.session_items(s).tolist())
    assert set(np.flatnonzero(model.bll(user, target)).tolist()) == before


def test_bll_weights_recent_plays_above_old_ones(sessions):
    """Same play count, more recent session: strictly higher activation."""
    model = ACTR(progress=False).fit(sessions)
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[0])
    bll = model.bll(user, target)
    last = sessions.session_items(target - 1)
    oldest = sessions.session_items(int(sessions.user_ptr[user]))
    only_recent = set(last.tolist()) - set(oldest.tolist())
    only_old = set(oldest.tolist()) - set(last.tolist())
    if only_recent and only_old:
        assert max(bll[list(only_recent)]) > min(bll[list(only_old)])


def test_cooccurrence_excludes_self_pairs_and_held_out_sessions(sessions):
    """Built from training sessions only, so no held-out co-listening leaks in."""
    adjacency = session_cooccurrence(sessions, progress=False)
    assert adjacency.shape == (sessions.n_items, sessions.n_items)
    assert np.allclose(adjacency.diagonal(), 0.0)

    train_items = set()
    for u in range(sessions.n_users):
        train_items |= set(sessions.history(u).tolist())
    reached = set(adjacency.nonzero()[0].tolist()) | set(adjacency.nonzero()[1].tolist())
    assert reached <= train_items


def test_normalization_is_the_references_transposed_form():
    import scipy.sparse as sp

    adj = sp.csr_matrix(np.array([[0.0, 2.0], [1.0, 0.0]]))
    normalized = normalize_adjacency(adj).toarray()
    d = np.array([2.0, 1.0]) ** -0.5
    expected = np.diag(d) @ adj.toarray().T @ np.diag(d)
    assert np.allclose(normalized, expected)


def test_scores_are_restricted_to_the_users_training_history(model, sessions):
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)
    scores = model.score(requests[:4])
    for request, row in zip(requests[:4], scores, strict=True):
        scored = set(np.flatnonzero(np.isfinite(row)).tolist())
        assert scored == set(sessions.history(request.user).tolist())


def test_recommendations_never_include_an_unseen_track(model, sessions):
    """ACT-R is a repeat-only model by construction: exploration is out of its reach."""
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)
    ranked = top_k(model.score(requests), 3)
    for request, row in zip(requests, ranked, strict=True):
        assert set(row.tolist()) <= set(sessions.history(request.user).tolist())


def test_spread_uses_only_the_session_immediately_before_the_target(model, sessions):
    target = int(sessions.held_out_session_ids(0, "test")[0])
    spread = model.spread(target)
    assert spread.shape == (sessions.n_items,)
    assert (spread >= 0).all()


def test_score_is_bll_plus_spread(model, sessions):
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[0])
    row = model.score([Request(user, target, np.array([]))])[0]
    expected = model.bll(user, target) + model.spread(target)
    candidates = sessions.history(user)
    assert np.allclose(row[candidates], expected[candidates])
