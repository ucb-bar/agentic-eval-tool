"""Tests for aet.core.metrics."""
from aet.core.metrics import (
    mean_std, fmt, coerce_na,
    welch_ttest, confidence_interval, effect_size,
    jaccard_similarity, sequence_edit_distance,
    paired_deltas, wilcoxon_signed_rank, paired_bootstrap_ci, kaplan_meier,
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


# ---------------------------------------------------------------------------
# paired comparison + survival
# ---------------------------------------------------------------------------

def test_paired_deltas_aligns_by_index():
    assert paired_deltas([10, 20, 30], [7, 15, 28]) == ([3, 5, 2], 0)


def test_paired_deltas_reports_what_it_dropped():
    """A comparison over 2 of 3 seeds and one over 3 of 3 are different measurements."""
    deltas, dropped = paired_deltas([10, None, 30], [7, 15, 28])
    assert (deltas, dropped) == ([3, 2], 1)


def test_paired_deltas_refuses_misaligned_input():
    """Silently zipping to the shorter list would pair seed 0 with seed 0 and then stop, reporting
    a comparison over a subset nobody chose."""
    try:
        paired_deltas([1, 2, 3], [1, 2])
    except ValueError as e:
        assert "align" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_paired_deltas_rejects_bools():
    """bool is an int in Python; a True/False column silently averaging into a duration is worse
    than a dropped pair."""
    deltas, dropped = paired_deltas([True, 20], [1, 15])
    assert (deltas, dropped) == ([5], 1)


def test_wilcoxon_flags_that_pilot_n_cannot_reach_significance():
    """At n=3 the smallest two-sided p is 0.25, so 'p > 0.05' is arithmetic, not evidence. Without
    this flag a pilot could report a non-significant result as an absence of effect."""
    r = wilcoxon_signed_rank([10, 20, 30], [7, 15, 28])
    assert r["p_value"] == 0.25
    assert r["underpowered"] is True
    assert r["min_achievable_p"] == 0.25
    assert r["n_nonzero"] == 3


def test_wilcoxon_is_not_underpowered_once_n_is_large_enough():
    a = list(range(1, 13))
    b = [x - 1 for x in a]
    r = wilcoxon_signed_rank(a, b)
    assert r["underpowered"] is False
    assert r["p_value"] is not None and r["p_value"] < 0.05


def test_wilcoxon_reports_median_delta_and_direction():
    r = wilcoxon_signed_rank([10, 20, 30], [7, 15, 28])
    assert r["median_delta"] == 3, "positive = the first arm was larger"


def test_wilcoxon_handles_all_ties():
    r = wilcoxon_signed_rank([5, 5, 5], [5, 5, 5])
    assert r["n_nonzero"] == 0
    assert r["p_value"] is None, "no non-zero pair means no test, not p=1"
    assert r["median_delta"] == 0


def test_paired_bootstrap_is_deterministic():
    """A CI that moves between runs of the same analysis is not reportable."""
    a, b = [10, 20, 30, 40], [7, 15, 28, 39]
    assert paired_bootstrap_ci(a, b, seed=0) == paired_bootstrap_ci(a, b, seed=0)


def test_paired_bootstrap_brackets_the_mean_delta():
    r = paired_bootstrap_ci([10, 20, 30, 40], [7, 15, 28, 39], seed=0)
    assert r["lower"] <= r["mean_delta"] <= r["upper"]
    assert r["n_pairs"] == 4


def test_paired_bootstrap_declines_a_ci_for_one_pair():
    r = paired_bootstrap_ci([10], [7])
    assert r["mean_delta"] == 3
    assert r["lower"] is None and r["upper"] is None


def test_kaplan_meier_keeps_censored_runs_in_the_risk_set():
    """The whole reason it exists: a run that hit its budget without succeeding is neither a
    failure nor a slow success. Dropping it biases toward the lucky runs; scoring it at the cap
    biases the other way."""
    km = kaplan_meier([100, 200, 300, 300], [1, 1, 0, 0])
    assert km["n_events"] == 2 and km["n_censored"] == 2
    assert km["at_risk"] == [4, 3], "the censored runs were at risk until their cap"
    assert km["survival"] == [0.75, 0.5]


def test_kaplan_meier_median_is_none_when_never_reached():
    """'More than half never got there' is not a median, and reporting the largest duration as one
    would assert a completion that did not happen."""
    km = kaplan_meier([100, 500, 500], [1, 0, 0])
    assert km["survival"] == [round(2 / 3, 6)]
    assert km["median"] is None


def test_kaplan_meier_all_censored_has_no_steps():
    km = kaplan_meier([300, 300], [0, 0])
    assert km["times"] == [] and km["survival"] == []
    assert km["n_censored"] == 2 and km["median"] is None


def test_kaplan_meier_refuses_misaligned_input():
    try:
        kaplan_meier([1, 2, 3], [1, 1])
    except ValueError as e:
        assert "align" in str(e)
    else:
        raise AssertionError("expected ValueError")
