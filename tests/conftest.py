"""Synthetic fixtures. No test needs the 9.8 GB release."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mrec.data.build import DataConfig

N_TRACKS = 12
EMB_DIM = 4
# appears only in the final (held-out) session of the first user: a leakage canary
CANARY_TRACK = N_TRACKS
# user -> number of sessions; the last user falls below `min_sessions` in the fixture config.
# The kept users need enough history for a training window to exist past the held-out tail.
USER_SESSIONS = {10: 14, 20: 12, 30: 10, 40: 3}


def _write_tracks(path):
    rng = np.random.default_rng(0)
    svd = rng.normal(size=(N_TRACKS, EMB_DIM))
    audio = rng.normal(size=(N_TRACKS, EMB_DIM))
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            # deliberately not in track_id order: item_idx must not depend on row order
            "track_id": list(range(N_TRACKS, 0, -1)),
            "art_id": [1 + (t % 3) for t in range(N_TRACKS, 0, -1)],
            "svd": [{"list": [{"item": float(v)} for v in row]} for row in svd],
            "audio": [[float(v) for v in row] for row in audio],
        }
    ).write_parquet(path / "svd_audio_000")


def _write_sessions(path):
    rng = np.random.default_rng(1)
    rows = []
    session_id = 1000
    for user_id, n_sessions in USER_SESSIONS.items():
        for s in range(n_sessions):
            session_id += 1
            base_ts = 1_700_000_000 + s * 3600 + user_id
            tracks = list(rng.choice(np.arange(1, N_TRACKS), size=3, replace=False))
            tracks.append(tracks[0])  # a repeat inside the session
            if user_id == 10 and s == n_sessions - 1:
                tracks.append(CANARY_TRACK)
            for i, track_id in enumerate(tracks):
                rows.append(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "track_id": int(track_id),
                        "ts": base_ts + i,
                    }
                )
    path.mkdir(parents=True, exist_ok=True)
    # shuffled on disk: ordering must come from the pipeline, not from the input file
    pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=2).write_parquet(
        path / "sessions_000"
    )


@pytest.fixture
def raw_config(tmp_path) -> DataConfig:
    """A tiny extracted dataset plus the config that builds it."""
    extract_dir = tmp_path / "raw" / "deezer-recsys25"
    _write_sessions(extract_dir / "user_sessions")
    _write_tracks(extract_dir / "track_embeddings")
    return DataConfig(
        min_sessions=6,
        n_valid=2,
        n_test=2,
        raw_dir=str(tmp_path / "raw"),
        extract_dir=str(extract_dir),
        proc_dir=str(tmp_path / "proc"),
    )
