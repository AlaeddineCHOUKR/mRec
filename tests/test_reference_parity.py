"""Parity against `deezer/recsys25-reacta`, the implementation the protocol is copied from.

The reference is vendored (gitignored) by `git clone` into `vendor/`; the test skips
when it is absent. It targets numpy 1.x, so `np.asfarray` — removed in numpy 2 — is
shimmed for the duration of the test. Nothing else about it is adapted: a
disagreement is a finding about our port, not something to paper over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mrec.eval.metrics import as_item_set, ndcg_at_k, recall_at_k, repr_at_k

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "recsys25-reacta"
K = 10


@pytest.fixture(scope="module")
def reference():
    if not (VENDOR / "au2actr" / "eval" / "metrics" / "ndcg.py").exists():
        pytest.skip("reference implementation not vendored")
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    sys.path.insert(0, str(VENDOR))
    try:
        from au2actr.eval.metrics.ndcg import NDCG
        from au2actr.eval.metrics.recall import RECALL
        from au2actr.eval.metrics.repr import REPR
    finally:
        sys.path.remove(str(VENDOR))
    return {"ndcg": NDCG, "recall": RECALL, "repr": REPR}


def _cases(n_cases: int = 200):
    rng = np.random.default_rng(0)
    for _ in range(n_cases):
        ranked = rng.choice(60, size=K, replace=False)
        targets = as_item_set(rng.choice(60, size=int(rng.integers(1, 20)), replace=True))
        history = as_item_set(rng.choice(60, size=int(rng.integers(0, 30)), replace=True))
        yield ranked, targets, history


def test_ndcg_and_recall_match_the_reference(reference):
    for ranked, targets, history in _cases():
        reco = {1: [list(ranked)]}
        ref_items = {1: [set(targets.tolist())]}
        kwargs = {"user_tracks": {1: set(history.tolist())}, "consumption_mode": "all"}
        assert ndcg_at_k(ranked, targets, K) == pytest.approx(
            reference["ndcg"](k=K, **kwargs).eval(reco, ref_items)
        )
        assert recall_at_k(ranked, targets, K) == pytest.approx(
            reference["recall"](k=K, **kwargs).eval(reco, ref_items)
        )


def test_rep_and_exp_variants_match_the_reference(reference):
    for ranked, targets, history in _cases():
        reco = {1: [list(ranked)]}
        ref_items = {1: [set(targets.tolist())]}
        known = np.isin(targets, history)
        for mode, subset in (("rep", targets[known]), ("exp", targets[~known])):
            kwargs = {"user_tracks": {1: set(history.tolist())}, "consumption_mode": mode}
            ref_ndcg = reference["ndcg"](k=K, **kwargs).eval(reco, ref_items)
            if len(subset) == 0:
                # the reference skips the session, then averages an empty list to 0.
                assert ref_ndcg == 0.0
                continue
            assert ndcg_at_k(ranked, subset, K) == pytest.approx(ref_ndcg)
            assert recall_at_k(ranked, subset, K) == pytest.approx(
                reference["recall"](k=K, **kwargs).eval(reco, ref_items)
            )


def test_repr_matches_the_reference(reference):
    for ranked, targets, history in _cases():
        reco = {1: [list(ranked)]}
        ref_items = {1: [set(targets.tolist())]}
        kwargs = {"user_tracks": {1: set(history.tolist())}, "consumption_mode": "all"}
        assert repr_at_k(ranked, history, K) == pytest.approx(
            reference["repr"](k=K, **kwargs).eval(reco, ref_items)
        )
