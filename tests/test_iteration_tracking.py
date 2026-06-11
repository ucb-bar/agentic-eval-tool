"""Tests for log_iteration_result and IterationTracker behavioral metrics."""
import json
import sys
import pytest
from pathlib import Path

# Make bench/ importable for IterationTracker
_ABC = Path(__file__).resolve().parent.parent.parent.parent / "mvp-lhwir" / "third_party" / "abc-testing"
if _ABC.exists():
    sys.path.insert(0, str(_ABC / "bench"))

from aet.tracking.run_logger import EvalRunLogger


def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="iter_r1", run_path=tmp_path, tracking_mode="local",
        target="debug-sv/trisc-sc/01", method="opus/xhigh",
        seed=0, project="abc-testing", suite="hardware_benchmark",
    )


def _events(tmp_path):
    p = tmp_path / "logs" / "events.jsonl"
    return [json.loads(l) for l in p.read_text().strip().splitlines()]


def test_log_iteration_result_event(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_iteration_result(0, oracle_output="%Error: syntax", passed=False,
                                failure_category="syntax_error", tok_in=3000)
    logger.finish("fail")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "iter.result")
    assert ev["payload"]["iteration"] == 0
    assert ev["payload"]["passed"] is False
    assert ev["payload"]["failure_category"] == "syntax_error"
    assert ev["payload"]["tok_in"] == 3000
    assert ev["stage"] == "eval"
    assert ev["actor"] == "oracle"


def test_log_iteration_result_pass(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_iteration_result(3, passed=True, failure_category=None, tok_in=12000)
    logger.finish("pass")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "iter.result")
    assert ev["payload"]["passed"] is True
    assert ev["payload"]["failure_category"] is None


def test_log_iteration_result_sequence(tmp_path):
    logger = _make_logger(tmp_path)
    for i in range(4):
        logger.log_iteration_result(i, passed=(i == 3))
    logger.finish("pass")
    evs = [e for e in _events(tmp_path) if e["event"] == "iter.result"]
    assert len(evs) == 4
    seqs = [e["sequence"] for e in evs]
    assert seqs == sorted(seqs)


@pytest.mark.skipif(not _ABC.exists(), reason="abc-testing repo not present")
def test_iteration_tracker_basic():
    from transcript_parser import IterationTracker

    class _TC:
        def __init__(self, name, inp, result="", turn_index=0):
            self.name = name
            self.tool_use_id = f"tu_{name}_{turn_index}"
            self.input = inp
            self.result = result
            self.turn_index = turn_index
            self.is_error = False
            self.duration_s = 0.0

    tool_calls = [
        _TC("Read", {"file_path": "spec/SPEC.md"}, turn_index=0),
        _TC("Write", {"file_path": "rtl/trisc_sc.sv", "content": "..."}, turn_index=1),
        _TC("Bash", {"command": "./run.sh"}, result="%Error: parse error", turn_index=2),
        _TC("Edit", {"file_path": "rtl/trisc_sc.sv"}, turn_index=3),
        _TC("Bash", {"command": "./run.sh"}, result="PASS: all tests passed.", turn_index=4),
    ]
    tracker = IterationTracker("trisc_sc.sv")
    tracker.infer(tool_calls)

    assert tracker.total_oracle_runs == 2
    assert tracker.oracle_without_edit_count == 0  # both had preceding writes
    assert tracker.unique_dut_edits == 2
    assert tracker.read_count == 1
    assert tracker.first_pass_iter == 1
    assert tracker.iterations[0]["oracle_passed"] is False
    assert tracker.iterations[1]["oracle_passed"] is True


@pytest.mark.skipif(not _ABC.exists(), reason="abc-testing repo not present")
def test_iteration_tracker_stall():
    from transcript_parser import IterationTracker

    class _TC:
        def __init__(self, name, inp, result="", turn_index=0):
            self.name = name
            self.tool_use_id = f"tu_{name}_{turn_index}"
            self.input = inp
            self.result = result
            self.turn_index = turn_index
            self.is_error = False
            self.duration_s = 0.0

    tool_calls = [
        _TC("Write", {"file_path": "rtl/trisc_sc.sv"}, turn_index=0),
        _TC("Bash", {"command": "./run.sh"}, result="FAIL: mismatch", turn_index=1),
        # stall: oracle without preceding write
        _TC("Bash", {"command": "./run.sh"}, result="FAIL: mismatch", turn_index=2),
    ]
    tracker = IterationTracker("trisc_sc.sv")
    tracker.infer(tool_calls)

    assert tracker.oracle_without_edit_count == 1
    assert tracker.stall_rate == pytest.approx(0.5)
    assert tracker.iterations[1]["stalled"] is True


@pytest.mark.skipif(not _ABC.exists(), reason="abc-testing repo not present")
def test_iteration_tracker_behavioral_summary_keys():
    from transcript_parser import IterationTracker
    tracker = IterationTracker("dut.sv")
    tracker.infer([])
    summary = tracker.behavioral_summary()
    expected_keys = {
        "agent.total_oracle_runs", "agent.oracle_without_edit_count",
        "agent.stall_rate", "agent.unique_dut_edits", "agent.read_count",
        "agent.write_count", "agent.read_to_write_ratio",
        "agent.max_consecutive_no_edit_turns",
        "agent.first_elaboration_iter", "agent.first_pass_iter",
        "agent.tok_in_at_first_elaboration", "agent.tok_in_at_first_pass",
    }
    assert expected_keys == set(summary.keys())


@pytest.mark.skipif(not _ABC.exists(), reason="abc-testing repo not present")
def test_iteration_tracker_first_elaboration():
    from transcript_parser import IterationTracker

    class _TC:
        def __init__(self, name, inp, result="", turn_index=0):
            self.name = name
            self.tool_use_id = f"tu_{name}_{turn_index}"
            self.input = inp
            self.result = result
            self.turn_index = turn_index
            self.is_error = False
            self.duration_s = 0.0

    tool_calls = [
        _TC("Write", {"file_path": "rtl/dut.sv"}, turn_index=0),
        _TC("Bash", {"command": "./run.sh"}, result="%Error-PARSE: syntax error at line 5", turn_index=1),
        _TC("Edit", {"file_path": "rtl/dut.sv"}, turn_index=2),
        _TC("Bash", {"command": "./run.sh"}, result="FAIL: mismatch at cycle 10", turn_index=3),
        _TC("Edit", {"file_path": "rtl/dut.sv"}, turn_index=4),
        _TC("Bash", {"command": "./run.sh"}, result="PASS: all tests passed.", turn_index=5),
    ]
    tracker = IterationTracker("dut.sv")
    tracker.infer(tool_calls)

    # iter 0: syntax_error → not elaboration
    # iter 1: mismatch → first elaboration
    # iter 2: pass → first pass
    assert tracker.first_elaboration_iter == 1
    assert tracker.first_pass_iter == 2
