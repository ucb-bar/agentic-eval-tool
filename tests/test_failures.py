"""Tests for aet.core.failures."""
import json
import pytest
from aet.core.failures import FailureCategory, FailureRecord


def test_failure_category_values():
    assert FailureCategory.SYNTAX_ERROR == "syntax_error"
    assert FailureCategory.SIMULATION_MISMATCH == "simulation_mismatch"


def test_failure_category_is_str():
    assert isinstance(FailureCategory.TOOL_CRASH, str)
    assert json.dumps({"cat": FailureCategory.TOOL_CRASH}) == '{"cat": "tool_crash"}'


def test_failure_record_defaults():
    rec = FailureRecord(category=FailureCategory.UNKNOWN, detail="something")
    assert rec.iteration is None
    assert rec.file is None
    assert rec.line is None
    assert len(rec.failure_id) == 32


def test_failure_record_to_dict():
    rec = FailureRecord(
        category=FailureCategory.SYNTAX_ERROR,
        detail="parse error",
        iteration=2,
        file="PE_MAC.sv",
        line=42,
    )
    d = rec.to_dict()
    assert d["category"] == "syntax_error"
    assert d["detail"] == "parse error"
    assert d["iteration"] == 2
    assert d["file"] == "PE_MAC.sv"
    assert json.dumps(d)  # must be JSON-serializable


def test_failure_record_from_dict_round_trip():
    rec = FailureRecord(
        category=FailureCategory.ELABORATION_ERROR,
        detail="undeclared signal",
        iteration=5,
    )
    d = rec.to_dict()
    rec2 = FailureRecord.from_dict(d)
    assert rec2.category == FailureCategory.ELABORATION_ERROR
    assert rec2.detail == "undeclared signal"
    assert rec2.iteration == 5


def test_failure_category_exhaustive():
    expected = {
        "syntax_error", "elaboration_error", "simulation_mismatch",
        "testbench_timeout", "tool_crash", "taint", "runner_crash", "unknown",
    }
    assert {c.value for c in FailureCategory} == expected


def test_failure_record_unique_ids():
    ids = {FailureRecord(FailureCategory.UNKNOWN, "x").failure_id for _ in range(20)}
    assert len(ids) == 20
