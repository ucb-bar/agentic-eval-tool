"""Tests for aet.core.metrics."""
import pytest
from aet.core.metrics import (
    mean_std, fmt, coerce_na,
    welch_ttest, confidence_interval, effect_size,
    jaccard_similarity, sequence_edit_distance,
)


# ---------------------------------------------------------------------------
# mean_std
# ---------------------------------------------------------------------------

def test_mean_std_empty():
    assert mean_std([]) == (None, None)


def test_mean_std_single():
    assert mean_std([5.0]) == (5.0, None)


def test_mean_std_none():
    assert mean_std([None, None]) == (None, None)


def test_mean_std_mixed():
    mean, std = mean_std([1.0, None, 3.0])
    assert mean == 2.0


def test_mean_std_two_values():
    mean, std = mean_std([2.0, 4.0])
    assert mean == 3.0
    assert std is not None
    assert std > 0


def test_mean_std_all_same():
    mean, std = mean_std([7.0, 7.0, 7.0])
    assert mean == 7.0
    assert std == 0.0


def test_mean_std_integers():
    mean, std = mean_std([1, 2, 3])
    assert mean == 2.0


def test_mean_std_ignores_non_numeric():
    mean, std = mean_std(["a", "b", 3.0])
    assert mean == 3.0
    assert std is None


def test_mean_std_returns_rounded():
    """mean_std rounds to 4 decimal places."""
    mean, std = mean_std([1.0, 2.0, 3.0])
    # Check that result has at most 4 decimal places
    assert mean == round(mean, 4)


# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------

def test_fmt_na():
    assert fmt(None, None) == "NA"


def test_fmt_single():
    assert fmt(1.5, None) == "1.5"


def test_fmt_with_std():
    result = fmt(2.0, 0.5)
    assert "2.0" in result
    assert "0.5" in result
    assert "±" in result


def test_fmt_zero_mean():
    assert fmt(0.0, None) == "0.0"


def test_fmt_zero_std():
    result = fmt(3.14, 0.0)
    assert "3.14" in result


# ---------------------------------------------------------------------------
# coerce_na
# ---------------------------------------------------------------------------

def test_coerce_na_none():
    assert coerce_na(None) == "NA"


def test_coerce_na_int():
    assert coerce_na(3) == "3"


def test_coerce_na_float():
    assert coerce_na(1.5) == "1.5"


def test_coerce_na_string():
    assert coerce_na("hello") == "hello"


def test_coerce_na_zero():
    assert coerce_na(0) == "0"


def test_coerce_na_false():
    assert coerce_na(False) == "False"


# ---------------------------------------------------------------------------
# welch_ttest
# ---------------------------------------------------------------------------

def test_welch_ttest_clearly_different():
    t, p = welch_ttest([1.0, 1.0, 1.0], [10.0, 10.0, 10.0])
    assert p is not None
    assert p < 0.01


def test_welch_ttest_same_distribution():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    t, p = welch_ttest(vals, vals)
    assert p is not None
    assert p > 0.05


def test_welch_ttest_small_sample():
    assert welch_ttest([1.0], [2.0]) == (None, None)


def test_welch_ttest_two_values_each():
    t, p = welch_ttest([1.0, 2.0], [10.0, 11.0])
    assert t is not None and p is not None
    assert p < 0.05


def test_welch_ttest_filters_none():
    t, p = welch_ttest([1.0, None, 1.0], [10.0, 10.0, None])
    assert t is not None and p is not None


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------

def test_confidence_interval_contains_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lower, upper = confidence_interval(values)
    mean = sum(values) / len(values)
    assert lower is not None and upper is not None
    assert lower < mean < upper


def test_confidence_interval_single_value():
    assert confidence_interval([5.0]) == (None, None)


def test_confidence_interval_empty():
    assert confidence_interval([]) == (None, None)


# ---------------------------------------------------------------------------
# effect_size
# ---------------------------------------------------------------------------

def test_effect_size_clearly_different():
    d = effect_size([1.0, 1.5, 0.5, 1.2, 0.8], [10.0, 10.5, 9.5, 10.2, 9.8])
    assert d is not None
    assert abs(d) > 2.0


def test_effect_size_single_group():
    assert effect_size([1.0], [10.0]) is None


def test_effect_size_zero_variance():
    assert effect_size([5.0, 5.0], [5.0, 5.0]) is None


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------

def test_jaccard_similarity_identical():
    assert jaccard_similarity(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_jaccard_similarity_disjoint():
    assert jaccard_similarity(["A", "B"], ["C", "D"]) == 0.0


def test_jaccard_similarity_partial_overlap():
    result = jaccard_similarity(["A", "B"], ["B", "C"])
    assert abs(result - 1 / 3) < 1e-9


def test_jaccard_similarity_both_empty():
    assert jaccard_similarity([], []) == 1.0


def test_jaccard_similarity_one_empty():
    assert jaccard_similarity(["A"], []) == 0.0


# ---------------------------------------------------------------------------
# sequence_edit_distance
# ---------------------------------------------------------------------------

def test_sequence_edit_distance_identical():
    assert sequence_edit_distance(["A", "B", "C"], ["A", "B", "C"]) == 0


def test_sequence_edit_distance_one_insertion():
    assert sequence_edit_distance(["A", "B"], ["A", "X", "B"]) == 1


def test_sequence_edit_distance_transposition():
    assert sequence_edit_distance(["A", "B"], ["B", "A"]) == 2


def test_sequence_edit_distance_empty():
    assert sequence_edit_distance([], ["A", "B"]) == 2
    assert sequence_edit_distance(["A"], []) == 1
