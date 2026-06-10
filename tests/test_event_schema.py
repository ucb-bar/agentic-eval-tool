"""Tests for enriched event schema (event_id, sequence, stage, actor, refs)."""
import json
import logging
import pytest
from aet.tracking.run_logger import EvalRunLogger

_logger = logging.getLogger(__name__)


def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="r1", run_path=tmp_path, tracking_mode="local",
        target="t", method="m", seed=1, project="p", suite="s",
    )


def _events(tmp_path):
    lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    return [json.loads(l) for l in lines]


def test_event_has_event_id(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_event("test.event", {"x": 1})
    logger.finish("pass")
    evs = _events(tmp_path)
    assert all("event_id" in e for e in evs)
    assert all(isinstance(e["event_id"], str) and len(e["event_id"]) == 32 for e in evs)


def test_event_has_sequence(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_event("a")
    logger.log_event("b")
    logger.log_event("c")
    logger.finish("pass")
    evs = _events(tmp_path)
    seqs = [e["sequence"] for e in evs]
    assert seqs == sorted(seqs)
    assert seqs[0] >= 1
    assert seqs[-1] - seqs[0] == len(seqs) - 1


def test_rich_event_has_stage_and_actor(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_start(benchmark="abc", variant="debug-sv/dut1")
    logger.finish("pass")
    evs = _events(tmp_path)
    run_start = next(e for e in evs if e["event"] == "run.start")
    assert run_start["stage"] == "setup"
    assert run_start["actor"] == "harness"


def test_rich_event_output_refs(tmp_path):
    logger = _make_logger(tmp_path)
    eid = logger.log_prompt_sent("/tmp/prompt.txt", prompt_hash="abc123")
    logger.finish("pass")
    evs = _events(tmp_path)
    prompt_ev = next(e for e in evs if e["event"] == "prompt.sent")
    assert "/tmp/prompt.txt" in prompt_ev.get("output_refs", [])


def test_event_ids_are_unique(tmp_path):
    logger = _make_logger(tmp_path)
    for i in range(10):
        logger.log_event(f"ev.{i}")
    logger.finish("pass")
    evs = _events(tmp_path)
    ids = [e["event_id"] for e in evs]
    assert len(ids) == len(set(ids))


def test_plain_log_event_also_gets_event_id(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_event("plain.event", {"k": "v"})
    logger.finish("pass")
    evs = _events(tmp_path)
    plain = next(e for e in evs if e["event"] == "plain.event")
    assert "event_id" in plain
    assert "sequence" in plain


def test_sequence_is_monotonic_across_types(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_start()
    logger.log_event("mid.event")
    logger.log_run_end("pass")
    logger.finish("pass")
    evs = _events(tmp_path)
    seqs = [e["sequence"] for e in evs]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
