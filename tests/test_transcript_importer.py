"""Generic transcript importer: one-or-many Claude Code *.jsonl → one RunTrajectory.

Covers the two on-disk shapes the ecosystem produces:
  * CLI stream-json (has a `result` event → billed cost, exact);
  * desktop session log (no `result` event → provisional list-price cost) with re-emitted
    assistant messages that must be counted once.
Plus: multiple session files order by timestamp and concatenate as rounds; a terminal pass/fail
is recorded as the last round's verdict + a single end-of-run milestone; no signal → degrades.
"""
import json

from aet.trajectory.importers.transcript import import_transcript


def _asst(mid, iso, *, out=30, cache_read=50, tool=None):
    content = [{"type": "thinking", "thinking": "plan"}]
    if tool:
        content.append({"type": "tool_use", "id": f"{mid}_t", "name": tool, "input": {"command": "ls"}})
    return {"timestamp": iso, "type": "assistant", "message": {
        "id": mid, "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "end_turn",
        "content": content,
        "usage": {"input_tokens": 100, "output_tokens": out,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": cache_read}}}


def _result(iso, cost, session):
    return {"timestamp": iso, "type": "result", "subtype": "success",
            "total_cost_usd": cost, "duration_ms": 20000, "num_turns": 2,
            "result": "done", "session_id": session}


def _cli_stream(session, minute, cost) -> list[dict]:
    """A CLI stream-json round with a terminal result event (authoritative cost)."""
    return [
        {"timestamp": f"2026-06-20T16:{minute:02d}:00.000Z", "type": "system", "session_id": session},
        _asst("m1", f"2026-06-20T16:{minute:02d}:00.000Z", tool="Read"),
        _asst("m2", f"2026-06-20T16:{minute:02d}:20.000Z"),
        _result(f"2026-06-20T16:{minute:02d}:20.000Z", cost, session),
    ]


def test_cli_stream_json_exact_billed_cost(tmp_path):
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in _cli_stream("s", 0, 0.07)) + "\n")
    traj = import_transcript(f)
    assert traj.num_rounds == 1
    assert traj.source == "import:transcript"
    assert abs(traj.final_cost_usd - 0.07) < 1e-9      # billed, exact
    assert not traj.provisional                         # a result event was seen
    assert traj.milestones == []                        # no pass signal → degrades to empty
    assert any(b.category in ("read", "think", "bash") for b in traj.bands)


def test_desktop_session_log_provisional_and_dedup(tmp_path):
    # no result event → provisional cost; the same assistant message id re-emitted 3× counts once
    a = _asst("dup", "2026-06-20T16:00:00.000Z", out=40, cache_read=1000)
    events = [{"timestamp": "2026-06-20T16:00:00.000Z", "type": "queue-operation"},
              a, a, a,
              _asst("m2", "2026-06-20T16:00:30.000Z", out=10, cache_read=2000)]
    f = tmp_path / "session.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    traj = import_transcript(f)
    assert traj.num_rounds == 1
    assert traj.provisional                                       # no billed number
    assert all(p.provisional_cost for p in traj.points)
    assert traj.final_cost_usd > 0                                # list-price estimate, not zero
    assert traj.final_output_tokens == 50                         # 40 + 10, dup counted once (not 130)


def test_multiple_session_files_order_and_concatenate(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    # write out of order on disk; importer must order by first embedded timestamp
    (d / "b_later.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _cli_stream("s1", 10, 0.02)) + "\n")
    (d / "a_earlier.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _cli_stream("s0", 0, 0.03)) + "\n")
    traj = import_transcript(d, label="arm")
    assert traj.run_id == "arm"
    assert traj.num_rounds == 2
    assert abs(traj.final_cost_usd - 0.05) < 1e-9                 # both counted
    assert traj.rounds[1].t_start_s == traj.rounds[0].t_end_s     # concatenated axis
    # earlier session (session_id s0) sorted first
    assert traj.rounds[0].session_id == "s0"


def test_terminal_pass_records_verdict_and_milestone(tmp_path):
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in _cli_stream("s", 0, 0.05)) + "\n")
    traj = import_transcript(f, pass_bool=True, n_total=3)
    assert traj.rounds[-1].n_passed == 3 and traj.rounds[-1].n_total == 3
    assert traj.final_tests() == 3 and traj.tests_total() == 3
    assert len(traj.milestones) == 1 and traj.milestones[0].t_s == traj.duration_s
    # a FAIL records 0-passing
    traj_fail = import_transcript(f, pass_bool=False, n_total=3)
    assert traj_fail.rounds[-1].n_passed == 0 and traj_fail.final_tests() == 0


def test_single_file_multiple_invocations_split(tmp_path):
    """One file concatenating two CLI invocations (two result events) → two rounds, both costs."""
    f = tmp_path / "transcript.jsonl"
    events = _cli_stream("sA", 0, 0.02) + _cli_stream("sB", 10, 0.02)
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    traj = import_transcript(f)
    assert traj.num_rounds == 2
    assert abs(traj.final_cost_usd - 0.04) < 1e-9
