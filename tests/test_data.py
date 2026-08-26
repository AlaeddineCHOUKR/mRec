"""Data-pipeline invariants, on a synthetic release built from scratch in the fixture."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from mrec.data.build import build
from mrec.data.sessions import load_sessions
from mrec.eval.popularity import train_popularity
from tests.conftest import CANARY_TRACK, USER_SESSIONS

KEPT_USERS = [u for u, n in USER_SESSIONS.items() if n >= 6]
DROPPED_USERS = [u for u, n in USER_SESSIONS.items() if n < 6]


@pytest.fixture
def built(raw_config):
    proc = build(raw_config, skip_fetch=True)
    return raw_config, proc, load_sessions(proc)


def test_item_idx_is_a_bijection_over_the_catalog(built):
    _, proc, sessions = built
    tracks = pl.read_parquet(proc / "tracks.parquet")
    assert tracks["item_idx"].to_list() == list(range(tracks.height))
    assert tracks["track_id"].to_list() == sorted(tracks["track_id"].to_list())
    assert tracks["track_id"].n_unique() == tracks.height == sessions.n_items


def test_embedding_rows_follow_item_idx_and_are_normalized(built):
    cfg, proc, _ = built
    svd = np.load(proc / "emb_svd.npy")
    audio = np.load(proc / "emb_audio.npy")
    tracks = pl.read_parquet(proc / "tracks.parquet")
    assert svd.shape[0] == audio.shape[0] == tracks.height
    assert np.allclose(np.linalg.norm(svd, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(audio, axis=1), 1.0)

    raw_tracks = pl.read_parquet(f"{cfg.extract_dir}/track_embeddings")
    row = raw_tracks.filter(pl.col("track_id") == 5)
    expected = np.array([d["item"] for d in row["svd"][0]["list"]], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    item_idx = tracks.filter(pl.col("track_id") == 5)["item_idx"][0]
    assert np.allclose(svd[item_idx], expected, atol=1e-6)


def test_min_sessions_filter_keeps_and_drops_the_right_users(built):
    _, _, sessions = built
    assert sorted(sessions.user_id.tolist()) == sorted(KEPT_USERS)
    assert all(u not in sessions.user_id.tolist() for u in DROPPED_USERS)
    assert min(sessions.n_user_sessions(u) for u in range(sessions.n_users)) >= 6


def test_sessions_are_ordered_by_time_and_events_within_them_too(built):
    _, _, sessions = built
    for u in range(sessions.n_users):
        first_ts = [sessions.session_ts(s)[0] for s in sessions.user_session_ids(u)]
        assert first_ts == sorted(first_ts)
        for s in sessions.user_session_ids(u):
            ts = sessions.session_ts(s)
            assert list(ts) == sorted(ts)


def test_held_out_sessions_are_disjoint_from_training_and_cover_the_tail(built):
    _, _, sessions = built
    for u in range(sessions.n_users):
        train = set(sessions.train_session_ids(u).tolist())
        valid = set(sessions.held_out_session_ids(u, "valid").tolist())
        test = set(sessions.held_out_session_ids(u, "test").tolist())
        assert train & (valid | test) == set()
        assert valid & test == set()
        assert train | valid | test == set(sessions.user_session_ids(u).tolist())


def test_history_never_contains_a_track_seen_only_in_a_held_out_session(built):
    """The canary track appears once, in the last session of the first user."""
    _, proc, sessions = built
    tracks = pl.read_parquet(proc / "tracks.parquet")
    canary_idx = tracks.filter(pl.col("track_id") == CANARY_TRACK)["item_idx"][0]
    for u in range(sessions.n_users):
        assert canary_idx not in sessions.history(u)
    last_user_sessions = sessions.user_session_ids(0)
    assert any(canary_idx in sessions.session_items(s) for s in last_user_sessions)


def test_popularity_is_computed_from_training_sessions_only(built):
    _, proc, sessions = built
    tracks = pl.read_parquet(proc / "tracks.parquet")
    canary_idx = tracks.filter(pl.col("track_id") == CANARY_TRACK)["item_idx"][0]
    popularity = train_popularity(sessions)
    assert popularity[canary_idx] == 0.0
    assert popularity.max() <= 1.0


def test_event_count_matches_the_raw_rows_of_kept_users(built):
    cfg, _, sessions = built
    raw_events = pl.read_parquet(f"{cfg.extract_dir}/user_sessions")
    expected = raw_events.filter(pl.col("user_id").is_in(KEPT_USERS)).height
    assert sessions.n_events == expected
    assert sessions.session_ptr[-1] == sessions.n_events


def test_rebuild_is_idempotent(built):
    cfg, proc, sessions = built
    before = {n: np.array(getattr(sessions, n)) for n in ("event_item", "event_ts", "session_ptr")}
    rebuilt = load_sessions(build(cfg, skip_fetch=True))
    for name, array in before.items():
        assert np.array_equal(array, getattr(rebuilt, name))


def test_manifest_records_counts_and_filter_parameters(built):
    cfg, proc, sessions = built
    manifest = json.loads((proc / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["config"]["min_sessions"] == cfg.min_sessions
    assert manifest["counts"]["n_users"] == sessions.n_users
    assert manifest["counts"]["n_events"] == sessions.n_events
    assert manifest["counts"]["n_users_raw"] == len(USER_SESSIONS)
    assert manifest["split"]["valid_indices"] and manifest["split"]["test_indices"]
