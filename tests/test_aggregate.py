"""Rep/exp skipping and macro aggregation — where a metric quietly becomes wrong."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mrec.eval.aggregate import macro, metrics_rows, score_session, score_sessions, split_targets
from mrec.eval.metrics import as_item_set

K = 10
RANKED = np.arange(10)


def test_split_targets_partitions_on_history():
    targets = as_item_set([1, 2, 3, 4])
    history = as_item_set([2, 4, 99])
    parts = split_targets(targets, history)
    assert list(parts["rep"]) == [2, 4]
    assert list(parts["exp"]) == [1, 3]
    assert len(parts["all"]) == 4


def test_empty_rep_set_is_skipped_not_scored_zero():
    """Everything in this session is new to the user: `rep` is unscorable, `exp` is not."""
    scores = score_session(RANKED, as_item_set([1, 2]), as_item_set([500]), K)
    assert scores[f"ndcg_rep@{K}"] is None
    assert scores[f"recall_rep@{K}"] is None
    assert scores[f"ndcg_exp@{K}"] is not None
    assert scores[f"ndcg_all@{K}"] is not None


def test_empty_exp_set_is_skipped_not_scored_zero():
    scores = score_session(RANKED, as_item_set([1, 2]), as_item_set([1, 2]), K)
    assert scores[f"ndcg_exp@{K}"] is None
    assert scores[f"recall_exp@{K}"] is None
    assert scores[f"ndcg_rep@{K}"] is not None


def test_user_mean_ignores_skipped_sessions():
    """User 1 has two sessions; only the second has repeat targets."""
    history = as_item_set([1])
    records = [
        (1, 0, RANKED, as_item_set([500]), history),  # exp only, and unreachable
        (1, 1, RANKED, as_item_set([1]), history),  # rep only, hit at rank 2
    ]
    scores = score_sessions(records, k=K)
    values = macro(scores)
    assert values[f"ndcg_rep@{K}"] == pytest.approx(1 / np.log2(3))
    assert values[f"ndcg_exp@{K}"] == pytest.approx(0.0)


def test_user_with_no_scorable_session_drops_out_of_that_variant():
    """User 2 never repeats, so the `rep` macro must be user 1's number alone."""
    records = [
        (1, 0, RANKED, as_item_set([1]), as_item_set([1])),
        (2, 0, RANKED, as_item_set([2]), as_item_set([500])),
    ]
    values = macro(score_sessions(records, k=K))
    assert values[f"ndcg_rep@{K}"] == pytest.approx(1 / np.log2(3))
    assert values[f"ndcg_exp@{K}"] == pytest.approx(1 / np.log2(4))


def test_single_held_out_session_user_scores_that_session():
    records = [(7, 0, RANKED, as_item_set([0]), as_item_set([]))]
    values = macro(score_sessions(records, k=K))
    assert values[f"ndcg_all@{K}"] == pytest.approx(1.0)
    assert values[f"repr@{K}"] == pytest.approx(0.0)


def test_aggregation_is_macro_over_users_not_micro_over_sessions():
    """A heavy user with five sessions must not outweigh a light user with one."""
    heavy = [(1, s, RANKED, as_item_set([500 + s]), as_item_set([])) for s in range(5)]
    light = [(2, 0, RANKED, as_item_set([0]), as_item_set([]))]
    scores = score_sessions(heavy + light, k=K)
    values = macro(scores)
    micro = scores.filter(pl.col("metric") == f"ndcg_all@{K}")["value"].mean()
    assert values[f"ndcg_all@{K}"] == pytest.approx(0.5)
    assert micro == pytest.approx(1 / 6)


def test_popularity_metric_is_optional_and_emitted_when_given():
    popularity = np.full(600, 0.25)
    records = [(1, 0, RANKED, as_item_set([0]), as_item_set([]))]
    assert f"pop@{K}" not in score_sessions(records, k=K)["metric"].to_list()
    values = macro(score_sessions(records, k=K, popularity=popularity))
    assert values[f"pop@{K}"] == pytest.approx(0.25)


def test_metrics_rows_have_the_committed_csv_shape():
    records = [(1, 0, RANKED, as_item_set([0]), as_item_set([]))]
    rows = metrics_rows(score_sessions(records, k=K), seed=1013, k=K)
    assert rows.columns == ["seed", "metric", "k", "value"]
    assert set(rows["seed"].to_list()) == {1013}


def test_repratio_gt_is_the_share_of_the_truth_the_user_already_knew():
    """The published tables' RepRatio-GT, from which RepBias is derived."""
    scores = score_session(RANKED, as_item_set([1, 2, 3, 4]), as_item_set([2, 4]), K)
    assert scores["repratio_gt"] == pytest.approx(0.5)
    all_new = score_session(RANKED, as_item_set([1, 2]), as_item_set([500]), K)
    assert all_new["repratio_gt"] == pytest.approx(0.0)


def test_rep_bias_is_derived_per_seed_from_repr_and_repratio_gt():
    from mrec.eval.runner import summarize

    records = [
        (1, 0, RANKED, as_item_set([1, 500]), as_item_set([1])),
        (2, 0, RANKED, as_item_set([2]), as_item_set([2])),
    ]
    values = macro(score_sessions(records, k=K))
    metrics = pl.DataFrame(
        {
            "seed": [1013] * len(values),
            "metric": list(values),
            "k": [K] * len(values),
            "value": list(values.values()),
        }
    )
    summary = summarize(metrics)
    row = summary.filter(pl.col("metric") == "rep_bias")
    assert row.height == 1
    assert row["mean"][0] == pytest.approx(values[f"repr@{K}"] - values["repratio_gt"])
