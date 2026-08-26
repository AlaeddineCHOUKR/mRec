"""Runner semantics: context windows, deterministic top-k, and the per-seed table."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mrec.data.build import build
from mrec.data.sessions import load_sessions
from mrec.eval.runner import Request, build_requests, evaluate, summarize, top_k
from mrec.models.topfreq import GlobalTopFreq, PersonalTopFreq

K = 3


@pytest.fixture
def sessions(raw_config):
    return load_sessions(build(raw_config, skip_fetch=True))


def test_top_k_breaks_ties_on_ascending_item_index():
    """Zero-score ties are the common case over a 50k catalog, not an edge case."""
    scores = np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.9, 0.9, 0.1]])
    assert top_k(scores, 2).tolist() == [[0, 1], [1, 2]]


def test_top_k_orders_by_descending_score():
    scores = np.array([[0.1, 0.4, 0.2, 0.3]])
    assert top_k(scores, 4).tolist() == [[1, 3, 2, 0]]


def test_top_k_rejects_a_catalog_smaller_than_k():
    with pytest.raises(ValueError):
        top_k(np.zeros((1, 2)), 3)


def test_context_is_the_sessions_immediately_preceding_the_target(sessions):
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)
    for request in requests:
        assert len(request.context) <= 3
        assert request.target not in request.context
        assert list(request.context) == sorted(request.context)
        if len(request.context):
            assert request.context[-1] == request.target - 1


def test_context_is_truncated_by_the_start_of_the_user_not_padded(sessions):
    """A user with fewer than `seqlen` prior sessions yields a shorter context, not zeros."""
    requests = build_requests(sessions, np.array([0]), "test", seqlen=10_000)
    first = min(requests, key=lambda r: r.target)
    assert first.context[0] == sessions.user_ptr[0]


def test_context_may_contain_other_held_out_sessions(sessions):
    """Faithful to the reference: only the target itself is withheld from the input."""
    held_out = set(sessions.held_out_session_ids(0, "test").tolist()) | set(
        sessions.held_out_session_ids(0, "valid").tolist()
    )
    requests = build_requests(sessions, np.array([0]), "test", seqlen=30)
    latest = max(requests, key=lambda r: r.target)
    assert held_out & set(latest.context.tolist())


def test_one_request_per_held_out_session_per_user(sessions):
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=30)
    assert len(requests) == sessions.n_users * sessions.split.n_test
    assert len({(r.user, r.target) for r in requests}) == len(requests)


def test_evaluate_emits_one_row_per_seed_and_metric(sessions):
    model = GlobalTopFreq().fit(sessions)
    seeds = (1013, 2791)
    metrics = evaluate(
        model,
        sessions,
        which="test",
        k=K,
        seqlen=3,
        n_users=2,
        seeds=seeds,
        batch_size=2,
        progress=False,
    )
    assert set(metrics["seed"].to_list()) == set(seeds)
    assert metrics.columns == ["seed", "metric", "k", "value"]
    per_seed = metrics.group_by("seed").len()["len"].to_list()
    assert len(set(per_seed)) == 1

    summary = summarize(metrics)
    assert summary.columns == ["metric", "mean", "ci_lo", "ci_hi"]
    assert (summary["ci_lo"] <= summary["mean"]).all()
    assert (summary["mean"] <= summary["ci_hi"]).all()


def test_global_topfreq_recommends_the_same_list_to_everyone(sessions):
    model = GlobalTopFreq().fit(sessions)
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)
    ranked = top_k(model.score(requests), K)
    assert len({tuple(row) for row in ranked.tolist()}) == 1


def test_personal_topfreq_ranks_a_users_own_training_plays(sessions):
    model = PersonalTopFreq().fit(sessions)
    for u in range(sessions.n_users):
        request = Request(u, int(sessions.held_out_session_ids(u, "test")[0]), np.array([]))
        scores = model.score([request])[0]
        train = sessions.train_session_ids(u)
        lo, hi = sessions.session_ptr[train[0]], sessions.session_ptr[train[-1] + 1]
        items, counts = np.unique(sessions.event_item[lo:hi], return_counts=True)
        assert np.array_equal(np.flatnonzero(scores), items)
        assert np.array_equal(scores[items], counts.astype(np.float32))


def test_personal_topfreq_never_scores_a_held_out_only_track(sessions):
    """The counts come through `train_session_ids`, so the leakage rule holds here too."""
    model = PersonalTopFreq().fit(sessions)
    assert model.counts is not None
    for u in range(sessions.n_users):
        scored = set(np.flatnonzero(model.counts[u]).tolist())
        assert scored <= set(sessions.history(u).tolist())


def test_metrics_table_is_writable_as_the_committed_csv(sessions, tmp_path):
    model = PersonalTopFreq().fit(sessions)
    metrics = evaluate(
        model,
        sessions,
        which="valid",
        k=K,
        seqlen=3,
        n_users=2,
        seeds=(1013,),
        batch_size=4,
        progress=False,
    )
    path = tmp_path / "metrics.csv"
    metrics.write_csv(path)
    assert pl.read_csv(path).shape == metrics.shape
