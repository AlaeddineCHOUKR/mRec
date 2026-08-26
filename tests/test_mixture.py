"""Post-hoc mixing must hit the requested repeat ratio exactly, or it is not a baseline."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.eval.metrics import repr_at_k
from mrec.models.mixture import ORDERINGS, explore_only, mix


@pytest.fixture
def lists():
    repeat = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    explore = np.array([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
    return repeat, explore


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("n_rep", range(11))
def test_mix_places_exactly_n_rep_repeat_slots(lists, ordering, n_rep):
    repeat, explore = lists
    out = mix(repeat, explore, n_rep=n_rep, k=10, ordering=ordering)
    assert out.shape == (1, 10)
    assert len(set(out[0].tolist())) == 10
    assert int(np.isin(out[0], repeat[0]).sum()) == n_rep


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_realized_repeat_ratio_is_the_requested_one(lists, ordering):
    """`repr@k` is the metric the frontier is plotted against, so pin it directly."""
    repeat, explore = lists
    history = repeat[0]
    for n_rep in range(11):
        out = mix(repeat, explore, n_rep=n_rep, k=10, ordering=ordering)
        assert repr_at_k(out[0], history, k=10) == pytest.approx(n_rep / 10)


def test_repeat_first_preserves_each_component_order(lists):
    repeat, explore = lists
    out = mix(repeat, explore, n_rep=4, k=10, ordering="repeat_first")
    np.testing.assert_array_equal(out[0], [0, 1, 2, 3, 100, 101, 102, 103, 104, 105])


def test_interleaved_gives_the_first_slot_to_the_larger_stream(lists):
    repeat, explore = lists
    assert mix(repeat, explore, n_rep=7, k=10, ordering="interleaved")[0, 0] == 0
    assert mix(repeat, explore, n_rep=3, k=10, ordering="interleaved")[0, 0] == 0
    assert mix(repeat, explore, n_rep=0, k=10, ordering="interleaved")[0, 0] == 100


def test_explore_only_masks_the_whole_history():
    scores = np.arange(12, dtype=float).reshape(2, 6)
    masked = explore_only(scores, [np.array([0, 1]), np.array([5])])
    assert np.isneginf(masked[0, [0, 1]]).all()
    assert np.isneginf(masked[1, 5])
    assert masked[0, 2] == scores[0, 2]
    np.testing.assert_array_equal(scores, np.arange(12, dtype=float).reshape(2, 6))


def test_mix_rejects_impossible_requests(lists):
    repeat, explore = lists
    with pytest.raises(ValueError):
        mix(repeat, explore, n_rep=11, k=10)
    with pytest.raises(ValueError):
        mix(repeat, explore, n_rep=-1, k=10)
    with pytest.raises(ValueError):
        mix(repeat, explore, n_rep=5, k=10, ordering="alphabetical")
    with pytest.raises(ValueError):
        mix(repeat[:, :2], explore, n_rep=5, k=10)


def test_explore_only_wrapper_never_returns_a_known_track(raw_config):
    """The wrapper must change the list, not the protocol around it."""
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions
    from mrec.eval.runner import build_requests, top_k
    from mrec.models import build_model
    from mrec.models.mixture import ExploreOnly

    sessions = load_sessions(build(raw_config, skip_fetch=True))
    inner = build_model("gtop")
    wrapped = ExploreOnly(inner).fit(sessions)
    assert wrapped.name == "gtop_explore"

    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)[:4]
    ranked = top_k(wrapped.score(requests), 3)
    for request, row in zip(requests, ranked, strict=True):
        history = sessions.history(request.user)
        if sessions.n_items - len(history) >= 3:
            assert not np.isin(row, history).any()


def test_explore_only_counts_users_it_cannot_fill(raw_config):
    """A user with fewer than k unheard tracks is counted, never silently tolerated."""
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions
    from mrec.eval.runner import build_requests
    from mrec.models import build_model
    from mrec.models.mixture import ExploreOnly

    sessions = load_sessions(build(raw_config, skip_fetch=True))
    wrapped = ExploreOnly(build_model("gtop"), k=sessions.n_items).fit(sessions)
    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)[:4]
    wrapped.score(requests)
    assert wrapped.thin == len(requests), "every user is thin when k is the whole catalog"


def test_thin_counts_rows_the_mask_leaves_unfillable(raw_config):
    """A generative model reaches few identifiers; masking can leave fewer than k."""
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions
    from mrec.eval.runner import Request
    from mrec.models.mixture import ExploreOnly

    sessions = load_sessions(build(raw_config, skip_fetch=True))

    class Sparse:
        """Scores only three items finitely, like a narrow beam."""

        name = "sparse"

        def fit(self, _sessions):
            return self

        def score(self, requests):
            out = np.full((len(requests), sessions.n_items), -np.inf)
            out[:, :3] = [3.0, 2.0, 1.0]
            return out

    wrapped = ExploreOnly(Sparse(), k=10).fit(sessions)
    wrapped.score([Request(0, 0, np.array([]))])
    assert wrapped.thin == 1, "three finite scores cannot fill a list of ten"
