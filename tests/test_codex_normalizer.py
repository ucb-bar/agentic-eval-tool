"""Codex JSONL normalizer: structural dispatch, lossless unknowns, token-subset semantics, resume.

Ground-truthed against the two REAL codex-cli 0.147.0 captures in tests/fixtures/codex/.
"""
from pathlib import Path

from aet.trajectory.codex import (
    CodexNormalizer, CodexTurnUsage, normalize_text, UNPARSED_TAG,
)

FIX = Path(__file__).parent / "fixtures" / "codex"


def _text(name):
    return (FIX / name).read_text()


# --------------------------------------------------------------- real captures (ground truth)
def test_real_simple_agent_message():
    run = normalize_text(_text("real_simple_agent_message.jsonl"))
    assert run.thread_id == "01a01160-7e52-7153-a3f1-a3ee492ab99e"
    assert run.agent_messages == ["PONG"]
    assert len(run.turns) == 1
    u = run.turns[0]
    assert u.input_tokens == 17713
    assert u.cached_input_tokens == 9984
    assert u.cache_write_input_tokens == 0
    assert u.output_tokens == 6
    assert u.reasoning_output_tokens == 0
    assert u.uncached_input_tokens == 17713 - 9984 - 0
    assert not run.unparsed_lines and not run.unknown_events


def test_real_tool_and_filechange():
    run = normalize_text(_text("real_tool_and_filechange.jsonl"))
    assert run.thread_id == "01a01161-68a0-71e3-9afb-6ec7a839f084"
    # two agent messages, one command_execution + one file_change tool span
    assert run.agent_messages[-1] == "FINISHED"
    kinds = sorted(t.kind for t in run.tools)
    assert kinds == ["command_execution", "file_change"]
    cmd = [t for t in run.tools if t.kind == "command_execution"][0]
    assert cmd.exit_code == 0 and cmd.status == "completed"
    assert cmd.command and "hello.txt" in cmd.command   # structured field, not parsed
    fc = [t for t in run.tools if t.kind == "file_change"][0]
    assert fc.changes and fc.changes[0]["kind"] == "add"
    assert run.turns[0].input_tokens == 53494


# --------------------------------------------------------------- structural dispatch / lossless
def test_unparsed_line_kept_not_crash():
    run = normalize_text(_text("synthetic_full.jsonl"))
    assert len(run.unparsed_lines) == 1
    assert run.unparsed_lines[0].startswith(UNPARSED_TAG)
    assert "not json" in run.unparsed_lines[0]


def test_unknown_event_kept_losslessly():
    text = ('{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"some.future.event","payload":{"x":1}}\n'
            '{"type":"item.completed","item":{"id":"i","type":"brand_new_item","blob":42}}\n')
    run = normalize_text(text)
    assert len(run.unknown_events) == 2          # unknown domain + unknown item.type
    assert run.unknown_events[0]["type"] == "some.future.event"
    # unknown events were parsed → counted as normalized, never dropped
    assert run.normalized_event_count == 3


def test_turn_failed_and_error_recorded():
    run = normalize_text(_text("synthetic_full.jsonl"))
    assert len(run.errors) == 1                  # the turn.failed line


def test_raw_vs_normalized_reconcile():
    run = normalize_text(_text("synthetic_full.jsonl"))
    # every raw line is dispatched JSON or a kept unparsed line
    assert run.normalized_event_count + len(run.unparsed_lines) == run.raw_event_count


# --------------------------------------------------------------- token subset semantics
def test_null_cache_write_preserved_not_zero():
    run = normalize_text(_text("synthetic_full.jsonl"))
    # turn 2 has cache_write_input_tokens: null → stays None (unknown ≠ 0)
    assert run.turns[1].cache_write_input_tokens is None


def test_totals_absent_bucket_is_none_not_zero():
    # a stream whose single turn never reports reasoning → totals.reasoning stays None
    text = ('{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":10}}\n')
    run = normalize_text(text)
    tot = run.totals()
    assert tot["input_tokens"] == 100
    assert tot["reasoning_output_tokens"] is None
    assert tot["cached_input_tokens"] is None
    assert tot["uncached_input_tokens"] == 100    # no cache reported → all uncached


def test_uncached_none_when_input_unknown():
    u = CodexTurnUsage(input_tokens=None, cached_input_tokens=5)
    assert u.uncached_input_tokens is None


def test_totals_partial_bucket_sums_reported_only():
    run = normalize_text(_text("synthetic_full.jsonl"))
    tot = run.totals()
    assert tot["input_tokens"] == 22000
    assert tot["cache_write_input_tokens"] == 1000   # only turn 1 reported it
    assert tot["uncached_input_tokens"] == 22000 - 13000 - 1000


# --------------------------------------------------------------- streaming == batch
def test_line_by_line_equals_batch():
    text = _text("synthetic_full.jsonl")
    batch = normalize_text(text)
    norm = CodexNormalizer()
    for ln in text.splitlines():
        norm.feed_line(ln)
    assert norm.result().to_dict() == batch.to_dict()


# --------------------------------------------------------------- resume (same-thread continuation)
def test_resume_same_thread_continues():
    norm = CodexNormalizer()
    norm.feed_text(_text("synthetic_full.jsonl"))
    tid = norm.result().thread_id
    norm.feed_text(_text("synthetic_resume.jsonl"))
    run = norm.result()
    assert run.thread_id == tid                  # resume stream shares the thread id
    assert len(run.turns) == 3                   # 2 original + 1 resumed
