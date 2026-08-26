"""Hand-computed metric fixtures. Every expected value is written out, not derived
by re-implementing the metric in the test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrec.eval.metrics import (
    as_item_set,
    dcg,
    ndcg_at_k,
    pop_at_k,
    recall_at_k,
    repr_at_k,
)

K = 10
LOG2 = math.log2


def test_recall_denominator_is_the_target_count_not_min_k():
    """|T| = 15 > k: a perfect list still scores 10/15. This cap is intended."""
    targets = as_item_set(range(15))
    ranked = np.arange(10)
    assert recall_at_k(ranked, targets, K) == pytest.approx(10 / 15)


def test_ndcg_is_one_when_the_list_is_ideal_and_targets_outnumber_k():
    """IDCG runs over min(|T|, k), so a full top-k of hits is a perfect NDCG."""
    targets = as_item_set(range(15))
    assert ndcg_at_k(np.arange(10), targets, K) == pytest.approx(1.0)


def test_single_target_at_first_and_last_rank():
    ranked = np.arange(100, 110)
    assert ndcg_at_k(ranked, as_item_set([100]), K) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, as_item_set([109]), K) == pytest.approx(1 / LOG2(11))
    assert recall_at_k(ranked, as_item_set([109]), K) == pytest.approx(1.0)


def test_two_hits_at_ranks_two_and_five_with_three_targets():
    ranked = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    targets = as_item_set([1, 4, 99])  # ranks 2 and 5; the third target is unreachable
    expected_dcg = 1 / LOG2(3) + 1 / LOG2(6)
    expected_idcg = 1 / LOG2(2) + 1 / LOG2(3) + 1 / LOG2(4)
    assert ndcg_at_k(ranked, targets, K) == pytest.approx(expected_dcg / expected_idcg)
    assert recall_at_k(ranked, targets, K) == pytest.approx(2 / 3)


def test_items_ranked_beyond_k_do_not_count():
    ranked = np.arange(20)
    assert recall_at_k(ranked, as_item_set([10]), K) == 0.0
    assert ndcg_at_k(ranked, as_item_set([10]), K) == 0.0


def test_duplicate_tracks_in_a_session_count_once():
    """A held-out session repeating a track must not inflate the recall denominator."""
    targets = as_item_set([3, 3, 3, 7])
    assert len(targets) == 2
    assert recall_at_k(np.arange(10), targets, K) == pytest.approx(2 / 2)


def test_empty_target_set_is_an_error_not_a_zero():
    """Callers must skip the session; scoring it zero would bias the rep/exp split."""
    empty = as_item_set([])
    with pytest.raises(ValueError):
        recall_at_k(np.arange(10), empty, K)
    with pytest.raises(ValueError):
        ndcg_at_k(np.arange(10), empty, K)


def test_repr_is_the_share_of_the_list_the_user_already_knows():
    ranked = np.arange(10)
    history = as_item_set([0, 1, 2, 500])
    assert repr_at_k(ranked, history, K) == pytest.approx(3 / 10)
    assert repr_at_k(ranked, as_item_set([]), K) == 0.0


def test_repr_ignores_targets_and_keeps_k_in_the_denominator():
    ranked = np.arange(10)
    assert repr_at_k(ranked, as_item_set([0]), K) == pytest.approx(1 / 10)


def test_pop_is_the_mean_training_popularity_of_the_list():
    popularity = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
    ranked = np.arange(10)
    assert pop_at_k(ranked, popularity, K) == pytest.approx(1.5 / 10)
    tail_swapped = np.array([10, *range(1, 10)])
    assert pop_at_k(tail_swapped, popularity, K) > pop_at_k(ranked, popularity, K)


def test_dcg_uses_the_standard_log2_discount():
    assert dcg(np.array([1.0, 0.0, 1.0])) == pytest.approx(1 / LOG2(2) + 1 / LOG2(4))


def test_malformed_ranked_lists_fail_loudly():
    targets = as_item_set([1])
    with pytest.raises(ValueError):
        recall_at_k(np.arange(5), targets, K)  # shorter than k
    with pytest.raises(ValueError):
        recall_at_k(np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9]), targets, K)  # duplicate in top-k
