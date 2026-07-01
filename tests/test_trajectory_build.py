"""append_round: build a RunTrajectory from timestamped Claude stream results."""
import json

from aet.tracking.claude_stream import parse_timestamped_stream
from aet.trajectory.build import append_round
from aet.trajectory.classify import ActivityClassifier, capsule_bench_config
from aet.trajectory.model import RunTrajectory


def _round_events(*, session, t0, cost, verilator_wait=30.0):
    """A 2-turn round: (thinking + verilator Bash) then a plain-text closing turn."""
    lines = [
        (t0, {"type": "system", "subtype": "init", "session_id": session}),
        (t0, {"type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-opus-4-8",
            "stop_reason": "tool_use", "content": [
                {"type": "thinking", "thinking": "planning..."},
                {"type": "tool_use", "id": "tc1", "name": "Bash",
                 "input": {"command": "run.py --sim verilator"}}],
            "usage": {"input_tokens": 100, "output_tokens": 30,
                      "cache_creation_input_tokens": 200, "cache_read_input_tokens": 50}}}),
        (t0 + verilator_wait, {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tc1",
             "content": [{"type": "text", "text": "ok"}]}]}}),
        (t0 + verilator_wait + 2, {"type": "assistant", "message": {
            "id": "m2", "role": "assistant", "model": "claude-opus-4-8",
            "stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}],
            "usage": {"input_tokens": 40, "output_tokens": 10,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 300}}}),
        (t0 + verilator_wait + 3, {"type": "result", "subtype": "success",
            "total_cost_usd": cost, "duration_ms": int((verilator_wait + 3) * 1000),
            "num_turns": 2, "result": "Done.", "session_id": session}),
    ]
    return [(ts, json.dumps(e)) for ts, e in lines]


def _parse(events):
    # re-base timestamps to 0 the way an importer/stream would
    base = events[0][0]
    return parse_timestamped_stream([(ts - base, line) for ts, line in events])


def test_single_round_points_and_totals():
    traj = RunTrajectory(run_id="t", source="test")
    clf = ActivityClassifier(capsule_bench_config())
    result = _parse(_round_events(session="s1", t0=1000.0, cost=0.02))
    append_round(traj, result, classifier=clf, verdict={"n_passed": 13, "n_capsules": 20})

    assert traj.num_rounds == 1
    assert len(traj.points) == 2                    # one point per turn
    assert traj.points[-1].cum_input_tokens == 140  # 100 + 40 (raw input, excl. cache)
    assert traj.points[-1].cum_output_tokens == 40
    assert traj.points[-1].cum_cache_tokens == 550  # 250 + 300
    assert abs(traj.final_cost_usd - 0.02) < 1e-9
    assert abs(traj.points[-1].cum_cost_usd - 0.02) < 1e-9   # sums to billed at round end
    assert traj.points[-1].provisional_cost is False
    assert traj.rounds[0].n_passed == 13 and traj.rounds[0].n_total == 20


def test_verilator_bash_is_a_long_tool_band():
    traj = RunTrajectory()
    clf = ActivityClassifier(capsule_bench_config())
    append_round(traj, _parse(_round_events(session="s", t0=0.0, cost=0.02, verilator_wait=40.0)),
                 classifier=clf)
    tool_bands = [b for b in traj.bands if b.category == "tool"]
    assert tool_bands, "verilator Bash should classify as a long tool-wait band"
    # the band spans the real ~40s wait (turn1 → turn2), not a nominal width
    assert tool_bands[0].duration_s >= 39.0
    # the closing plain-text turn is bash, not think (no thinking block)
    assert traj.bands[-1].category == "bash"


def test_two_rounds_concatenate_on_one_axis():
    traj = RunTrajectory()
    clf = ActivityClassifier(capsule_bench_config())
    append_round(traj, _parse(_round_events(session="s1", t0=0.0, cost=0.02)),
                 classifier=clf, verdict={"n_passed": 13, "n_capsules": 20})
    d1 = traj.duration_s
    append_round(traj, _parse(_round_events(session="s2", t0=9999.0, cost=0.03)),
                 classifier=clf, verdict={"n_passed": 17, "n_capsules": 20})
    assert traj.num_rounds == 2
    assert traj.rounds[1].t_start_s == d1                 # round 2 starts where round 1 ended
    assert traj.points[-1].t_s >= d1                       # later points on the shared axis
    assert abs(traj.final_cost_usd - 0.05) < 1e-9          # 0.02 + 0.03
    # cumulative token + spend series never decrease across the whole run
    for key in ("cum_input_tokens", "cum_output_tokens", "cum_cache_tokens", "cum_cost_usd"):
        vals = [getattr(p, key) for p in traj.points]
        assert all(vals[i] >= vals[i - 1] for i in range(1, len(vals)))


def test_missing_cost_is_provisional():
    traj = RunTrajectory()
    clf = ActivityClassifier(capsule_bench_config())
    append_round(traj, _parse(_round_events(session="s", t0=0.0, cost=0.0)), classifier=clf)
    assert traj.provisional is True
    assert all(p.provisional_cost for p in traj.points)
    assert traj.final_cost_usd > 0.0                      # provisional estimate from list price
