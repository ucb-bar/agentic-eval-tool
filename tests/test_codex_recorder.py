"""Live CodexTrajectoryRecorder: streaming line-by-line mirrors the batch importer."""
from pathlib import Path

from aet.trajectory.codex_recorder import CodexTrajectoryRecorder
from aet.trajectory.importers.codex import import_codex

FIX = Path(__file__).parent / "fixtures" / "codex"


def test_streaming_matches_batch_import():
    text = (FIX / "synthetic_full.jsonl").read_text()
    rec = CodexTrajectoryRecorder(run_id="r", model="gpt-5-codex")
    for ln in text.splitlines():
        rec.feed_line(ln)                         # no timestamps → pseudo-time, like the importer
    live = rec.trajectory()
    batch = import_codex(FIX / "synthetic_full.jsonl", model="gpt-5-codex")
    batch.run_id = "r"
    batch.source = live.source
    assert live.to_dict() == batch.to_dict()


def test_feed_timestamped_places_points_on_wall_time():
    rec = CodexTrajectoryRecorder(run_id="r", model="gpt-5-codex")
    rec.feed_timestamped((0.0, '{"type":"thread.started","thread_id":"t"}'))
    rec.feed_timestamped({"t_s": 1.0, "line": '{"type":"turn.started"}'})
    rec.feed_timestamped((5.5, '{"type":"turn.completed","usage":'
                               '{"input_tokens":100,"cached_input_tokens":10,"output_tokens":5}}'))
    traj = rec.trajectory()
    assert traj.points[-1].t_s == 5.5             # real offset, not pseudo-time
    assert traj.final_input_tokens == 90          # uncached = 100 - 10


def test_on_update_callback_fires_and_fail_open():
    seen = {"n": 0}

    def _cb(traj):
        seen["n"] += 1
        raise RuntimeError("callback boom")       # must be swallowed (fail-open)

    rec = CodexTrajectoryRecorder(run_id="r", on_update=_cb)
    rec.feed_line('{"type":"thread.started","thread_id":"t"}')
    rec.feed_line("this is not json")             # unparsed: no normalized event → no callback
    assert seen["n"] == 1                          # only the parsed line triggered the callback


def test_feed_malformed_never_raises():
    rec = CodexTrajectoryRecorder()
    rec.feed_line("{ broken json")
    rec.feed_timestamped(("bad-ts", "{ also broken"))
    assert len(rec.run.unparsed_lines) == 2
