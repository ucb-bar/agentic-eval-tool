"""Sandboxed agent runner — dummy-agent path (no real claude / bwrap), watchdog, and sandbox argv.

The dummy agent is a shell command that ``cat``s a canned stream-json transcript, so the whole
record → materialize → resume path runs in milliseconds. Rate-limit verdicts are injected so the
wait→resume and weekly→unfinished branches are exercised without a real 5-hour wait.
"""
import json
import shlex
from pathlib import Path

from aet.isolation import SandboxSpec, bwrap_argv
from aet.ratelimit import RateLimitState
from aet.runner import run_agent, resume_run, STATUS_COMPLETED, STATUS_UNFINISHED
from aet.trajectory.model import RunTrajectory


def _canned_transcript(session="sess1", cost=0.05) -> str:
    events = [
        {"type": "system", "subtype": "init", "session_id": session},
        {"type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "plan"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
            "usage": {"input_tokens": 100, "output_tokens": 30,
                      "cache_creation_input_tokens": 200, "cache_read_input_tokens": 50}}},
        {"type": "result", "subtype": "success", "total_cost_usd": cost,
         "duration_ms": 20000, "num_turns": 1, "result": "done", "session_id": session},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _dummy_cmd(tmp_path: Path, session="sess1") -> str:
    canned = tmp_path / "canned.jsonl"
    canned.write_text(_canned_transcript(session))
    return f"cat {shlex.quote(str(canned))}"


def test_dummy_run_materializes_recoverable_trajectory(tmp_path):
    ws = tmp_path / "ws"
    res = run_agent("do the thing", ws, into=tmp_path / "run", sandbox="none",
                    agent_cmd=_dummy_cmd(tmp_path), label="dummy")
    assert res.status == STATUS_COMPLETED
    assert res.session_id == "sess1"
    assert res.trajectory_path.is_file()
    # the materialized run round-trips through the canonical loader
    traj = RunTrajectory.from_run_dir(res.run_dir)
    assert traj.num_rounds == 1
    assert abs(traj.final_cost_usd - 0.05) < 1e-9
    assert (res.run_dir / "run_manifest.yaml").is_file()
    assert (res.run_dir / "TASK.md").read_text() == "do the thing"


def test_watchdog_waits_to_reset_then_resumes(tmp_path):
    slept = []
    # attempt 1 → five-hour rejection (resets_at 1000, now 900 → wait 130); attempt 2 → completes
    states = [RateLimitState(rejected=True, saw_rejection=True, limit_type="five_hour",
                             resets_at=1000),
              RateLimitState(rejected=False)]
    res = run_agent("t", tmp_path / "ws", into=tmp_path / "run", sandbox="none",
                    agent_cmd=_dummy_cmd(tmp_path), label="wd",
                    inject_states=states, sleep=lambda s: slept.append(s), now=lambda: 900.0)
    assert res.status == STATUS_COMPLETED
    assert res.attempts == 2               # burned attempt + the resumed one
    assert res.rate_limit_waits == 1
    assert slept == [130.0]                # (1000 + 30 jitter) - 900


def test_watchdog_polls_when_reset_epoch_missing(tmp_path):
    slept = []
    states = [RateLimitState(rejected=True, saw_rejection=True, limit_type="five_hour",
                             resets_at=None),
              RateLimitState(rejected=False)]
    res = run_agent("t", tmp_path / "ws", into=tmp_path / "run", sandbox="none",
                    agent_cmd=_dummy_cmd(tmp_path), inject_states=states,
                    poll_seconds=1200, sleep=lambda s: slept.append(s), now=lambda: 0.0)
    assert res.status == STATUS_COMPLETED
    assert slept == [1200]                 # one 20-min poll tick, not a blind guess


def test_weekly_limit_writes_unfinished_and_status(tmp_path):
    states = [RateLimitState(rejected=True, saw_rejection=True, limit_type="weekly",
                             resets_at=1_900_000_000)]
    res = run_agent("t", tmp_path / "ws", into=tmp_path / "run", sandbox="none",
                    agent_cmd=_dummy_cmd(tmp_path, session="wk"), inject_states=states,
                    sleep=lambda s: (_ for _ in ()).throw(AssertionError("must not sleep on weekly")))
    assert res.status == STATUS_UNFINISHED
    assert res.limit_type == "weekly"
    unfinished = res.run_dir / "UNFINISHED.md"
    assert unfinished.is_file()
    body = unfinished.read_text()
    assert "aet run --resume" in body and "weekly" in body
    # manifest carries the resumable status + resume metadata
    from aet.core.run_manifest import RunManifest
    m = RunManifest.load(res.run_dir / "run_manifest.yaml")
    assert m.status == STATUS_UNFINISHED
    assert m.metadata.get("session_id") == "wk"
    assert m.metadata.get("resume_cmd", "").startswith("aet run --resume")


def test_wait_budget_exhaustion_gives_up(tmp_path):
    # every attempt is a five-hour rejection; with max_rate_limit_waits=2 it gives up after 2 waits
    states = [RateLimitState(rejected=True, saw_rejection=True, limit_type="five_hour",
                             resets_at=None)]
    res = run_agent("t", tmp_path / "ws", into=tmp_path / "run", sandbox="none",
                    agent_cmd=_dummy_cmd(tmp_path), inject_states=states,
                    max_rate_limit_waits=2, poll_seconds=1, sleep=lambda s: None, now=lambda: 0.0)
    assert res.status == STATUS_UNFINISHED
    assert res.rate_limit_waits == 2


def test_resume_run_reads_session_from_manifest(tmp_path):
    # first: a weekly stop that records session + TASK; then resume_run completes it
    states = [RateLimitState(rejected=True, saw_rejection=True, limit_type="weekly", resets_at=1)]
    run_agent("original task", tmp_path / "ws", into=tmp_path / "run", sandbox="none",
              agent_cmd=_dummy_cmd(tmp_path, session="resume-me"), inject_states=states,
              sleep=lambda s: None)
    res = resume_run(tmp_path / "run", sandbox="none", agent_cmd=_dummy_cmd(tmp_path))
    assert res.status == STATUS_COMPLETED


def test_bwrap_argv_deny_after_allow_ordering(tmp_path):
    # pure-string sandbox check: workspace bound rw, allow ro-bound, deny tmpfs-masked AFTER allow
    ws = tmp_path / "ws"
    ws.mkdir()
    allowed = tmp_path / "tools"
    allowed.mkdir()
    denied = tmp_path / "tools" / "answers"
    denied.mkdir()
    argv = bwrap_argv(SandboxSpec(workspace=ws, allow=[allowed], deny=[denied]))
    assert "--bind" in argv and str(ws) in argv          # workspace writable
    assert "--ro-bind" in argv                            # allow bound read-only
    joined = " ".join(argv)
    # deny mask (tmpfs on the answers dir) comes AFTER the allow ro-bind of its parent
    assert joined.index(f"--ro-bind {allowed} {allowed}") < joined.index(f"--tmpfs {denied}")


def test_unsandboxed_real_agent_refused(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="refusing to run unsandboxed"):
        run_agent("t", tmp_path / "ws", sandbox="none")   # no agent_cmd, no allow_unsandboxed
