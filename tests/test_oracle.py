"""Oracle-progression extraction: mine each testbench invocation → a tests-passing climb."""
import json

from aet.trajectory.oracle import parse_oracle_result, extract_oracle_progression
from aet.trajectory.importers.transcript import import_transcript


def test_parse_pass_with_cases():
    r = parse_oracle_result("*** PASSED *** vpu replay: 182 cases\nRESULT: PASS")
    assert r == (182, 182, True, True)          # explicit count


def test_parse_fail_uses_hint_total():
    r = parse_oracle_result("*** FAILED *** vpu replay: 5 lane mismatches\nRESULT: FAIL",
                            n_total_hint=182)
    assert r == (0, 182, False, False)          # 0 passing / hinted total, not explicit


def test_parse_no_verdict_is_none():
    assert parse_oracle_result("just some build log, nothing conclusive") is None


def _asst_bash(mid, tid, cmd, ts):
    return {"timestamp": ts, "type": "assistant", "message": {
        "id": mid, "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd}}],
        "usage": {"input_tokens": 100, "output_tokens": 10,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}


def _tool_result(tid, text, ts):
    return {"timestamp": ts, "type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": [{"type": "text", "text": text}]}]}}


def test_extract_progression_climbs_and_rejects_other_testbenches():
    # fail@run.sh, then a foreign sub-test (607 cases — must be rejected), then pass@run.sh
    seg = [
        (0.0, json.dumps({"timestamp": "t", "type": "system", "session_id": "s"})),
        (0.0, json.dumps(_asst_bash("m1", "t1", "./run.sh", "t"))),
        (60.0, json.dumps(_tool_result("t1", "*** FAILED *** vpu replay: 3 lane mismatches\nRESULT: FAIL", "t"))),
        (120.0, json.dumps(_asst_bash("m2", "t2", "python3 tools/run_test.py sub.sv SubTestbench.sv", "t"))),
        (130.0, json.dumps(_tool_result("t2", "*** PASSED *** sub replay: 607 cases", "t"))),
        (180.0, json.dumps(_asst_bash("m3", "t3", "bash run.sh", "t"))),
        (200.0, json.dumps(_tool_result("t3", "*** PASSED *** vpu replay: 182 cases\nRESULT: PASS", "t"))),
    ]
    # markers = run.sh only; hint 182 rejects the 607-case foreign testbench
    reads = extract_oracle_progression([(ts, ln) for ts, ln in seg],
                                       markers=("run.sh",), n_total_hint=182)
    assert [(r.n_passed, r.n_total) for r in reads] == [(0, 182), (182, 182)]
    assert reads[0].t_s == 60.0 and reads[1].t_s == 200.0      # at each invocation's result time


def test_import_transcript_oracle_markers_build_climb(tmp_path):
    events = [
        {"timestamp": "2026-06-20T16:00:00.000Z", "type": "system", "session_id": "s"},
        _asst_bash("m1", "t1", "./run.sh", "2026-06-20T16:00:10.000Z"),
        _tool_result("t1", "*** FAILED *** vpu replay: 4 lane mismatches", "2026-06-20T16:01:00.000Z"),
        _asst_bash("m2", "t2", "./run.sh", "2026-06-20T16:02:00.000Z"),
        _tool_result("t2", "*** PASSED *** vpu replay: 182 cases", "2026-06-20T16:03:00.000Z"),
        {"timestamp": "2026-06-20T16:03:05.000Z", "type": "result", "subtype": "success",
         "total_cost_usd": 1.5, "duration_ms": 185000, "num_turns": 2, "session_id": "s"},
    ]
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    traj = import_transcript(f, run_id="vpu", oracle_markers=["run.sh"],
                             n_passed=182, n_total=182)
    counts = [(m.n_passed, m.n_total) for m in traj.milestones]
    assert counts == [(0, 182), (182, 182)]           # the climb, not a single terminal dot
    assert all(m.source == "oracle_log" for m in traj.milestones)
    assert traj.final_tests() == 182 and traj.tests_total() == 182


def test_no_oracle_calls_falls_back_to_terminal(tmp_path):
    # a transcript with no run.sh invocations → single terminal milestone (graceful degradation)
    events = [
        {"timestamp": "2026-06-20T16:00:00.000Z", "type": "system", "session_id": "s"},
        _asst_bash("m1", "t1", "ls -la", "2026-06-20T16:00:10.000Z"),
        _tool_result("t1", "some files", "2026-06-20T16:00:20.000Z"),
        {"timestamp": "2026-06-20T16:00:30.000Z", "type": "result", "subtype": "success",
         "total_cost_usd": 0.2, "duration_ms": 30000, "num_turns": 1, "session_id": "s"},
    ]
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    traj = import_transcript(f, run_id="x", oracle_markers=["run.sh"], n_passed=8, n_total=8)
    assert len(traj.milestones) == 1 and traj.milestones[0].source == "terminal_verdict"
    assert (traj.milestones[0].n_passed, traj.milestones[0].n_total) == (8, 8)
