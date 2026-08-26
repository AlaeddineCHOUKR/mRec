"""Invariants on the real `data/proc/` build. Skipped when it has not been built."""

from __future__ import annotations

import json
import re
import subprocess

import numpy as np
import polars as pl
import pytest

from mrec.data.sessions import load_sessions
from mrec.utils.paths import PROC_DIR, REPO_ROOT

pytestmark = pytest.mark.slow

if not (PROC_DIR / "manifest.json").exists():
    pytest.skip("data/proc not built", allow_module_level=True)


@pytest.fixture(scope="module")
def sessions():
    return load_sessions(PROC_DIR)


def test_manifest_is_complete(sessions):
    manifest = json.loads((PROC_DIR / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["source_checksums"]["md5"] == "da77fe05f0bee8d655c18f924e52a13c"
    assert manifest["config"]["min_sessions"] == 300
    assert manifest["counts"]["n_users"] == sessions.n_users
    # The build stamps the commit it ran at. It must be a real commit in this
    # repository, but not necessarily HEAD: `data/proc/` is built once and every
    # later commit would otherwise fail this test until someone rebuilt 25M events.
    recorded = manifest["git_commit"]
    assert recorded is not None, "a built manifest records the commit that produced it"
    assert re.fullmatch(r"[0-9a-f]{40}", recorded), recorded
    resolved = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-t", recorded],
        capture_output=True,
        text=True,
    )
    assert resolved.stdout.strip() == "commit", f"{recorded} is not a commit in this repo"


def test_every_kept_user_clears_the_filter_with_training_left(sessions):
    counts = np.diff(sessions.user_ptr)
    assert counts.min() >= 300
    assert (counts - sessions.split.n_held_out).min() > 0


def test_sessions_hold_exactly_ten_events(sessions):
    """The release caps each session at the first 10 tracks, so |T| <= 10 = k throughout."""
    assert np.unique(np.diff(sessions.session_ptr)).tolist() == [10]


def test_events_and_sessions_are_time_ordered(sessions):
    rng = np.random.default_rng(0)
    for u in rng.integers(0, sessions.n_users, size=50):
        session_ids = sessions.user_session_ids(int(u))
        first_ts = [sessions.session_ts(s)[0] for s in session_ids]
        assert first_ts == sorted(first_ts)
        for s in session_ids[:20]:
            assert list(sessions.session_ts(s)) == sorted(sessions.session_ts(s))


def test_held_out_sessions_never_reach_training(sessions):
    rng = np.random.default_rng(1)
    for u in rng.integers(0, sessions.n_users, size=200):
        u = int(u)
        train = set(sessions.train_session_ids(u).tolist())
        held_out = set(sessions.held_out_session_ids(u, "valid").tolist()) | set(
            sessions.held_out_session_ids(u, "test").tolist()
        )
        assert len(held_out) == 20
        assert train & held_out == set()
        assert max(train) < min(held_out)


def test_catalog_is_a_bijection_and_embeddings_are_normalized(sessions):
    tracks = pl.read_parquet(PROC_DIR / "tracks.parquet")
    assert tracks["item_idx"].to_list() == list(range(50_000))
    assert tracks["track_id"].n_unique() == 50_000
    svd = np.load(PROC_DIR / "emb_svd.npy", mmap_mode="r")
    audio = np.load(PROC_DIR / "emb_audio.npy", mmap_mode="r")
    assert svd.shape == (50_000, 128)
    assert audio.shape == (50_000, 1024)
    assert np.allclose(np.linalg.norm(svd[:1000], axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(audio[:1000], axis=1), 1.0, atol=1e-5)


def test_item_indices_stay_inside_the_catalog(sessions):
    sample = np.asarray(sessions.event_item[:1_000_000])
    assert sample.min() >= 0
    assert sample.max() < sessions.n_items
