import numpy as np

from mrec.data.split import protocol_split


def test_split_matches_the_reference_draw():
    """The exact indices `np.random.seed(0)` produces in the reference implementation."""
    split = protocol_split()
    assert sorted(split.valid_indices) == [-19, -18, -16, -14, -12, -10, -7, -3, -2, -1]
    assert sorted(split.test_indices) == [-20, -17, -15, -13, -11, -9, -8, -6, -5, -4]


def test_valid_and_test_are_disjoint_and_cover_the_last_twenty():
    split = protocol_split()
    valid, test = set(split.valid_indices), set(split.test_indices)
    assert len(valid) == 10 and len(test) == 10
    assert valid & test == set()
    assert valid | test == set(range(-20, 0))


def test_split_is_not_the_last_ten_then_the_ten_before():
    """Validation and test interleave; copying the naive reading would void comparability."""
    assert set(protocol_split().test_indices) != set(range(-10, 0))


def test_draw_is_reproducible_and_leaves_no_global_state_assumption():
    np.random.seed(12345)
    first = protocol_split()
    np.random.seed(999)
    assert protocol_split() == first
