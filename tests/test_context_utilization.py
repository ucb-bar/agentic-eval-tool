"""Tests for context window utilization tracking."""
import json
import logging
from dataclasses import dataclass, field

from aet.tracking.run_logger import EvalRunLogger

_logger = logging.getLogger(__name__)

_CONTEXT_LIMITS = {"claude": 200_000}


@dataclass
class _TurnUsage:
    turn: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = "claude-sonnet-4-6"
    finish_reasons: list = field(default_factory=list)
    reasoning_text: str = ""

    @property
    def total_input_tokens(self):
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


def _compute_context_pct(tu: _TurnUsage) -> float:
    limit = next(
        (v for k, v in _CONTEXT_LIMITS.items() if (tu.model or "").startswith(k)),
        200_000,
    )
    return round(tu.total_input_tokens / limit * 100, 2)


def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="r1", run_path=tmp_path, tracking_mode="local",
        target="t", method="m", seed=1, project="p", suite="s",
    )


def test_context_pct_calculation():
    tu = _TurnUsage(turn=0, input_tokens=100_000, output_tokens=100)
    assert _compute_context_pct(tu) == 50.0


def test_context_pct_under_threshold(tmp_path):
    logger = _make_logger(tmp_path)
    turn_usage = [_TurnUsage(turn=i, input_tokens=50_000, output_tokens=100) for i in range(3)]
    max_pct = max(_compute_context_pct(tu) for tu in turn_usage)
    logger.log_metric("aet.context.max_pct_used", round(max_pct, 2))
    if max_pct > 80:
        logger.log_event("aet.context.high_utilization_warning", {"max_pct": max_pct, "threshold": 80})
    logger.finish(status="pass")

    events_path = tmp_path / "logs" / "events.jsonl"
    events = [json.loads(l) for l in events_path.read_text().strip().splitlines()]
    warnings = [e for e in events if e["event"] == "aet.context.high_utilization_warning"]
    assert len(warnings) == 0


def test_context_pct_over_threshold(tmp_path):
    logger = _make_logger(tmp_path)
    turn_usage = [_TurnUsage(turn=0, input_tokens=170_000, output_tokens=100)]
    max_pct = _compute_context_pct(turn_usage[0])
    logger.log_metric("aet.context.max_pct_used", round(max_pct, 2))
    if max_pct > 80:
        logger.log_event("aet.context.high_utilization_warning", {"max_pct": max_pct, "threshold": 80})
    logger.finish(status="pass")

    events_path = tmp_path / "logs" / "events.jsonl"
    events = [json.loads(l) for l in events_path.read_text().strip().splitlines()]
    warnings = [e for e in events if e["event"] == "aet.context.high_utilization_warning"]
    assert len(warnings) == 1
    assert warnings[0]["payload"]["max_pct"] == 85.0


def test_context_max_pct_metric(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_metric("aet.context.max_pct_used", 42.5)
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    records = {json.loads(l)["name"]: json.loads(l)["value"] for l in lines}
    assert "aet.context.max_pct_used" in records
    assert records["aet.context.max_pct_used"] == 42.5
