"""Tests for new hardware-benchmark EvalRunLogger methods."""
import json
import logging
import pytest
from pathlib import Path
from aet.tracking.run_logger import EvalRunLogger

_logger = logging.getLogger(__name__)


def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="hw_r1", run_path=tmp_path, tracking_mode="local",
        target="debug-sv/trisc-sc/01-easy", method="claude-opus-4-8/xhigh",
        seed=1, project="abc-testing", suite="hardware_benchmark",
    )


def _events(tmp_path):
    p = tmp_path / "logs" / "events.jsonl"
    return [json.loads(l) for l in p.read_text().strip().splitlines()]


def _metrics(tmp_path):
    p = tmp_path / "logs" / "metrics.jsonl"
    return {json.loads(l)["name"]: json.loads(l)["value"]
            for l in p.read_text().strip().splitlines()}


def test_log_run_start_emits_event(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_start(benchmark="abc-testing", variant="debug-sv/trisc-sc/01",
                         tool_tier="claude/xhigh", repo_initial_commit="abc123")
    logger.finish("pass")
    evs = _events(tmp_path)
    start_ev = next(e for e in evs if e["event"] == "run.start")
    assert start_ev["payload"]["benchmark"] == "abc-testing"
    assert start_ev["stage"] == "setup"
    assert start_ev["actor"] == "harness"


def test_log_run_end_emits_event_and_metric(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_end("pass", wall_time_s=123.4)
    logger.finish("pass")
    evs = _events(tmp_path)
    end_ev = next(e for e in evs if e["event"] == "run.end")
    assert end_ev["payload"]["wall_time_s"] == 123.4
    m = _metrics(tmp_path)
    assert m["run.wall_time_s"] == 123.4


def test_log_tool_start_and_end(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_tool_start("Bash", "tu_001", input_summary="ls -la", turn=1)
    logger.log_tool_end("Bash", "tu_001", result_summary="file.sv", duration_s=0.4)
    logger.finish("pass")
    evs = _events(tmp_path)
    starts = [e for e in evs if e["event"] == "tool.start"]
    ends = [e for e in evs if e["event"] == "tool.end"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0]["payload"]["tool"] == "Bash"
    assert ends[0]["payload"]["duration_s"] == 0.4


def test_log_file_diff(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_file_diff("rtl/PE.sv", "write", sha256_before=None,
                         sha256_after="deadbeef", iteration=3)
    logger.finish("pass")
    evs = _events(tmp_path)
    diff_ev = next(e for e in evs if e["event"] == "file.diff")
    assert diff_ev["payload"]["path"] == "rtl/PE.sv"
    assert diff_ev["payload"]["sha256_after"] == "deadbeef"
    assert diff_ev["stage"] == "agent_action"


def test_log_eval_score_writes_metrics(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_score(
        testbench_pass=True,
        localization_recall=0.75,
        localization_precision=0.80,
        regression_count=2,
        tainted=False,
    )
    logger.finish("pass")
    m = _metrics(tmp_path)
    assert m["hw.testbench_pass"] == 1
    assert m["hw.localization_recall"] == 0.75
    assert m["hw.localization_precision"] == 0.80
    assert m["hw.regression_count"] == 2


def test_log_eval_score_event(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_score(testbench_pass=False, localization_recall=0.0,
                          localization_precision=0.0)
    logger.finish("pass")
    evs = _events(tmp_path)
    score_ev = next(e for e in evs if e["event"] == "eval.score")
    assert score_ev["stage"] == "eval"
    assert score_ev["payload"]["testbench_pass"] is False


def test_log_synth_end(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_synth_end("syntax_error", iteration=2,
                         verilator_output="%Error: parse error",
                         failure_category="syntax_error")
    logger.finish("pass")
    evs = _events(tmp_path)
    synth_ev = next(e for e in evs if e["event"] == "synth.end")
    assert synth_ev["payload"]["status"] == "syntax_error"
    assert synth_ev["payload"]["failure_category"] == "syntax_error"


def test_log_llm_response(tmp_path):
    logger = _make_logger(tmp_path)
    eid = logger.log_prompt_sent("/tmp/prompt.txt")
    logger.log_llm_response("claude-opus-4-8", turn=1,
                             tok_in=5000, tok_out=200, prompt_event_id=eid)
    logger.finish("pass")
    evs = _events(tmp_path)
    resp = next(e for e in evs if e["event"] == "llm.response")
    assert resp["payload"]["tok_in"] == 5000
    assert resp["input_refs"] == [eid]


def test_write_run_record(tmp_path):
    logger = _make_logger(tmp_path)
    path = logger.write_run_record({"benchmark": "abc-testing", "variant": "debug-sv"})
    logger.finish("pass")
    assert path.exists()
    rec = json.loads(path.read_text())
    assert rec["schema_version"] == "1.1"
    assert rec["benchmark"] == "abc-testing"
    assert rec["run_id"] == "hw_r1"


def test_write_summary_metrics(tmp_path):
    logger = _make_logger(tmp_path)
    path = logger.write_summary_metrics({
        "hw.testbench_pass": 1,
        "hw.localization_recall": 0.9,
    })
    logger.finish("pass")
    assert path.exists()
    s = json.loads(path.read_text())
    assert s["hw.testbench_pass"] == 1
    assert s["run_id"] == "hw_r1"


def test_milestone_record_elaboration(tmp_path):
    logger = _make_logger(tmp_path)
    logger.record_elaboration(3)
    logger.finish("pass")
    evs = _events(tmp_path)
    ms = next(e for e in evs if e["event"] == "eval.milestone")
    assert ms["payload"]["milestone"] == "first_elaboration"
    assert ms["payload"]["iteration"] == 3
    m = _metrics(tmp_path)
    assert m["hw.first_elaboration_iter"] == 3


def test_milestone_record_public_pass(tmp_path):
    logger = _make_logger(tmp_path)
    logger.record_public_pass(7)
    logger.finish("pass")
    m = _metrics(tmp_path)
    assert m["hw.first_public_pass_iter"] == 7


def test_log_run_abort(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_abort("wall_timeout", detail="exceeded 60min budget")
    logger.finish("abort")
    evs = _events(tmp_path)
    abort_ev = next(e for e in evs if e["event"] == "run.abort")
    assert abort_ev["payload"]["reason"] == "wall_timeout"
