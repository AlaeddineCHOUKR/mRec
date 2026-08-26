"""PISA's inputs: window placement, ACT-R tensors, and the leakage rules."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.data.build import build
from mrec.data.sessions import load_sessions
from mrec.models.actr import bll_activation, session_cooccurrence
from mrec.models.features import (
    SESSION_LEN,
    ActrFeatures,
    history_mean_embedding,
    repeat_labels,
    training_windows,
)

SEQLEN = 3
FAVS = 4


@pytest.fixture
def sessions(raw_config):
    return load_sessions(build(raw_config, skip_fetch=True))


@pytest.fixture
def features(sessions):
    adjacency = session_cooccurrence(sessions, progress=False)
    return ActrFeatures(sessions, adjacency, seqlen=SEQLEN, num_favs=FAVS)


def test_training_windows_stop_two_sessions_before_the_held_out_tail(sessions):
    """The reference's `len(sessions) - 22`: the session at -21 is never a target."""
    windows = training_windows(sessions, seqlen=SEQLEN, step=1)
    for user, target in windows:
        offset = target - int(sessions.user_ptr[user])
        assert offset <= sessions.n_user_sessions(user) - sessions.split.n_held_out - 2
        assert offset > SEQLEN


def test_training_windows_never_target_a_held_out_session(sessions):
    windows = training_windows(sessions, seqlen=SEQLEN, step=1)
    for user, target in windows:
        held_out = set(sessions.held_out_session_ids(user, "valid").tolist()) | set(
            sessions.held_out_session_ids(user, "test").tolist()
        )
        assert target not in held_out
        assert target + 1 not in held_out  # its own target session is a training session


def test_training_windows_follow_the_stride(sessions):
    windows = training_windows(sessions, seqlen=SEQLEN, step=2)
    per_user = {}
    for user, target in windows:
        per_user.setdefault(user, []).append(target)
    for targets in per_user.values():
        # windows walk backwards from the latest usable target
        assert all(a - b == 2 for a, b in zip(targets[:-1], targets[1:], strict=True))


def test_ids_are_one_based_with_zero_as_padding(features, sessions):
    example = features.evaluation_example(0, int(sessions.held_out_session_ids(0, "test")[0]))
    assert example.seq_in.shape == (SEQLEN, SESSION_LEN)
    filled = example.seq_in[example.seq_in > 0]
    assert filled.min() >= 1
    assert filled.max() <= sessions.n_items
    # the fixture's sessions are shorter than SESSION_LEN, so every row is padded
    assert (example.seq_in == 0).any()


def test_window_is_the_sessions_immediately_before_the_target(features, sessions):
    target = int(sessions.held_out_session_ids(0, "test")[-1])
    example = features.evaluation_example(0, target)
    for i, s in enumerate(range(target - SEQLEN, target)):
        items = sessions.session_items(s)
        assert example.seq_in[i][: len(items)].tolist() == (np.asarray(items) + 1).tolist()


def test_bll_tensor_matches_the_activation_of_each_slot(features, sessions):
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[0])
    activation = bll_activation(sessions, user, target)
    example = features.evaluation_example(user, target)
    filled = example.seq_in > 0
    assert np.allclose(example.seq_bll[filled], activation[example.seq_in[filled] - 1])
    assert (example.seq_bll[~filled] == 0).all()


def test_favourites_are_the_top_activations_in_order(features, sessions):
    user, target = 0, int(sessions.held_out_session_ids(0, "test")[0])
    activation = bll_activation(sessions, user, target)
    example = features.evaluation_example(user, target)
    scores = example.fav_bll[example.fav_ids > 0]
    assert (np.diff(scores) <= 1e-12).all()
    assert scores[0] == pytest.approx(activation.max())


def test_training_example_predicts_the_session_after_each_input(features, sessions):
    user, target = 0, training_windows(sessions, seqlen=SEQLEN, step=1)[0][1]
    example = features.training_example(user, target, np.random.default_rng(0))
    assert example.pos is not None
    assert np.array_equal(example.pos[:-1], example.seq_in[1:])
    items = sessions.session_items(target)
    assert example.pos[-1][: len(items)].tolist() == (np.asarray(items) + 1).tolist()


def test_negatives_are_never_tracks_the_user_knows(features, sessions):
    user, target = 0, training_windows(sessions, seqlen=SEQLEN, step=1)[0][1]
    example = features.training_example(user, target, np.random.default_rng(0))
    assert example.neg is not None
    history = set(sessions.history(user).tolist())
    assert not (set((example.neg - 1).flatten().tolist()) & history)


def test_spread_is_zero_where_the_slot_is_padding(features, sessions):
    example = features.evaluation_example(0, int(sessions.held_out_session_ids(0, "test")[0]))
    assert (example.seq_spread[example.seq_in == 0] == 0).all()
    assert (example.seq_spread >= 0).all()


def test_repeat_labels_mark_only_tracks_heard_in_an_earlier_session(raw_config):
    """First play is explore, every later play is repeat, and users do not bleed."""
    build(raw_config)
    sessions = load_sessions(raw_config.proc_dir)
    labels = repeat_labels(sessions)

    for user in range(sessions.n_users):
        seen: set[int] = set()
        for session in sessions.user_session_ids(user):
            items = sessions.session_items(int(session))
            for slot, item in enumerate(items):
                assert bool(labels[int(session), slot]) == (int(item) in seen), (
                    user, int(session), slot, int(item)
                )
            seen.update(int(i) for i in items)


def test_repeat_labels_ignore_other_users(raw_config):
    """A track one user has played is still new to the next user."""
    build(raw_config)
    sessions = load_sessions(raw_config.proc_dir)
    labels = repeat_labels(sessions)

    for user in range(sessions.n_users):
        first_session = int(sessions.user_ptr[user])
        assert not labels[first_session].any(), "nothing can repeat in a user's first session"


def test_history_mean_embedding_uses_training_sessions_only(raw_config):
    build(raw_config)
    sessions = load_sessions(raw_config.proc_dir)
    table = np.zeros((sessions.n_items, 2), dtype=np.float32)
    table[:, 0] = np.arange(sessions.n_items)

    vectors = history_mean_embedding(sessions, table)
    assert vectors.shape == (sessions.n_users, 2)
    for user in range(sessions.n_users):
        expected = table[sessions.history(user)].mean(axis=0)
        np.testing.assert_allclose(vectors[user], expected, rtol=1e-6)
