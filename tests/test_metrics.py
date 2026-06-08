"""Tests for aet.core.metrics."""
import pytest
from aet.core.metrics import mean_std, fmt, coerce_na


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
