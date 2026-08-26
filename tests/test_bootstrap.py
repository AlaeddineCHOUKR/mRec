"""Confidence intervals and paired significance."""

from __future__ import annotations

import numpy as np
import pytest

from mrec.eval.bootstrap import bootstrap_ci, paired_bootstrap
from mrec.eval.protocol import EVAL_SEEDS, sample_eval_users


def test_constant_values_give_a_zero_width_interval():
    mean, lo, hi = bootstrap_ci(np.full(5, 0.42), seed=0)
    assert (mean, lo, hi) == pytest.approx((0.42, 0.42, 0.42))


def test_interval_is_deterministic_given_the_seed():
    values = np.linspace(0.1, 0.9, 25)
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    assert bootstrap_ci(values, seed=7) != bootstrap_ci(values, seed=8)


def test_interval_brackets_the_mean():
    values = np.array([0.31, 0.35, 0.29, 0.4, 0.33])
    mean, lo, hi = bootstrap_ci(values, seed=1013)
    assert lo < mean < hi


def test_paired_bootstrap_of_a_series_against_itself_is_never_significant():
    values = np.linspace(0, 1, 50)
    diff, p = paired_bootstrap(values, values, seed=0)
    assert diff == 0.0
    assert p == pytest.approx(1.0)


def test_paired_bootstrap_detects_a_uniform_shift():
    values = np.linspace(0, 1, 200)
    diff, p = paired_bootstrap(values + 0.05, values, seed=0)
    assert diff == pytest.approx(0.05)
    assert p < 0.001


def test_paired_bootstrap_requires_aligned_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4), seed=0)


def test_eval_cohorts_differ_across_the_protocol_seeds():
    users = np.arange(10_000)
    cohorts = [set(sample_eval_users(users, 3000, seed)) for seed in EVAL_SEEDS]
    assert all(len(c) == 3000 for c in cohorts)
    assert cohorts[0] != cohorts[1]
    assert set(sample_eval_users(users, 3000, EVAL_SEEDS[0])) == cohorts[0]
