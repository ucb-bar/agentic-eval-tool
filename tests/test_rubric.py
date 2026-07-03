"""Tests for aet.core.rubric and log_rubric_score."""
import json
import logging
import pytest

from aet.core.rubric import RubricCriterion, compute_weighted_score, validate_rubric
from aet.tracking.run_logger import EvalRunLogger

_logger = logging.getLogger(__name__)


def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )


# ---------------------------------------------------------------------------
# compute_weighted_score
# ---------------------------------------------------------------------------

def test_compute_weighted_score_equal_weights():
    criteria = [RubricCriterion("a", 0.5), RubricCriterion("b", 0.5)]
    result = compute_weighted_score({"a": 1.0, "b": 0.0}, criteria)
    assert abs(result - 0.5) < 1e-9


def test_compute_weighted_score_unequal_weights():
    criteria = [RubricCriterion("correctness", 0.7), RubricCriterion("style", 0.3)]
    assert abs(compute_weighted_score({"correctness": 1.0, "style": 1.0}, criteria) - 1.0) < 1e-9
    assert abs(compute_weighted_score({"correctness": 0.0, "style": 1.0}, criteria) - 0.3) < 1e-9


def test_compute_weighted_score_missing_criterion_counts_zero():
    criteria = [RubricCriterion("a", 0.6), RubricCriterion("b", 0.4)]
    result = compute_weighted_score({"a": 1.0}, criteria)
    assert abs(result - 0.6) < 1e-9


def test_compute_weighted_score_all_zero():
    criteria = [RubricCriterion("x", 0.5), RubricCriterion("y", 0.5)]
    assert compute_weighted_score({}, criteria) == 0.0


# ---------------------------------------------------------------------------
# validate_rubric
# ---------------------------------------------------------------------------

def test_validate_rubric_raises_on_bad_sum():
    criteria = [RubricCriterion("x", 0.4), RubricCriterion("y", 0.4)]
    with pytest.raises(ValueError, match="sum"):
        validate_rubric(criteria)


def test_validate_rubric_passes_on_valid():
    criteria = [RubricCriterion("x", 0.6), RubricCriterion("y", 0.4)]
    validate_rubric(criteria)  # no exception


def test_rubric_criterion_rejects_invalid_weight():
    with pytest.raises(ValueError):
        RubricCriterion("bad", 1.5)


# ---------------------------------------------------------------------------
# log_rubric_score
# ---------------------------------------------------------------------------

def test_log_rubric_score_writes_metric(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_rubric_score("correctness", score=0.8, weight=0.3, explanation="mostly correct")
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    records = {json.loads(l)["name"]: json.loads(l)["value"] for l in lines}
    assert "rubric.correctness.score" in records
    assert "rubric.correctness.weight" in records
    assert abs(records["rubric.correctness.score"] - 0.8) < 1e-9
    assert abs(records["rubric.correctness.weight"] - 0.3) < 1e-9


def test_log_rubric_score_writes_event(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_rubric_score("completeness", score=1.0, weight=0.25)
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    rubric_events = [e for e in events if e["event"] == "aet.rubric.criterion"]
    assert len(rubric_events) == 1
    payload = rubric_events[0]["payload"]
    assert payload["criterion"] == "completeness"
    assert abs(payload["score"] - 1.0) < 1e-9
    assert abs(payload["weight"] - 0.25) < 1e-9
