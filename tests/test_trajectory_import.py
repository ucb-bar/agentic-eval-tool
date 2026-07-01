"""capsule-bench importer: transcripts + qa verdicts + self-check milestones → RunTrajectory."""
import json
from pathlib import Path

from aet.trajectory.importers.capsule_bench import import_run


def _transcript(session: str, iso_start: str, verilator: bool) -> list[dict]:
    """A 2-turn round; timestamps are ISO with a trailing Z (real transcript shape)."""
    cmd = "run.py --sim verilator" if verilator else "ls -la"
    # crude ISO stepping: reuse the same date, bump seconds via the string tail
    base = iso_start
    def at(sec: str) -> str:
        return base[:-7] + sec + "Z"
    return [
        {"timestamp": at("00.000"), "type": "system", "subtype": "init", "session_id": session},
        {"timestamp": at("00.000"), "type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "plan"},
                        {"type": "tool_use", "id": "tc1", "name": "Bash", "input": {"command": cmd}}],
            "usage": {"input_tokens": 100, "output_tokens": 30,
                      "cache_creation_input_tokens": 200, "cache_read_input_tokens": 50}}},
        {"timestamp": at("30.000"), "type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tc1",
             "content": [{"type": "text", "text": "ok"}]}]}},
        {"timestamp": at("32.000"), "type": "assistant", "message": {
            "id": "m2", "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Done."}],
            "usage": {"input_tokens": 40, "output_tokens": 10,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 300}}},
        {"timestamp": at("33.000"), "type": "result", "subtype": "success",
         "total_cost_usd": 0.02, "duration_ms": 33000, "num_turns": 2,
         "result": "Done.", "session_id": session},
    ]


def _make_run(tmp_path: Path, *, circt_name=False) -> Path:
    name = "merlincirct_test" if circt_name else "rb_test"
    run = tmp_path / name
    (run / "rounds").mkdir(parents=True)
    (run / "qa_history").mkdir(parents=True)
    for k, iso in enumerate(("2026-06-20T16:00:00.000Z", "2026-06-20T16:05:00.000Z")):
        lines = _transcript(f"sess{k}", iso, verilator=True)
        (run / "rounds" / f"round_{k:02d}.transcript.jsonl").write_text(
            "\n".join(json.dumps(e) for e in lines) + "\n")
    (run / "qa_history" / "verdict_round_00.json").write_text(json.dumps({"n_passed": 13, "n_capsules": 20}))
    (run / "qa_history" / "verdict_round_01.json").write_text(json.dumps({"n_passed": 17, "n_capsules": 20}))
    # self-check log: 13 → 17 → 20 all-scope, plus noise rows that must be filtered out
    rows = [
        {"wall_offset_s": 100.0, "capsules": "A2_single", "n_passed": 1, "n_capsules": 1},   # not all-scope
        {"wall_offset_s": 200.0, "capsules": "all", "n_passed": 5, "n_capsules": 10},          # <20 capsules
        {"wall_offset_s": 300.0, "capsules": "all", "n_passed": 13, "n_capsules": 20},
        {"wall_offset_s": 450.0, "capsules": "all", "n_passed": 13, "n_capsules": 20},         # not increasing
        {"wall_offset_s": 600.0, "capsules": "all", "n_passed": 17, "n_capsules": 20},
        {"wall_offset_s": 900.0, "capsules": "all", "n_passed": 20, "n_capsules": 20},
    ]
    (run / "selfcheck_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_import_produces_milestones_13_17_20(tmp_path):
    traj = import_run(_make_run(tmp_path))
    counts = [m.n_passed for m in sorted(traj.milestones, key=lambda m: m.t_s)]
    assert counts == [13, 17, 20]                       # noise rows filtered, strictly increasing
    assert all(m.n_total == 20 for m in traj.milestones)
    assert all(0.0 <= m.t_s <= traj.duration_s for m in traj.milestones)


def test_import_per_round_verdicts_and_rounds(tmp_path):
    traj = import_run(_make_run(tmp_path))
    assert traj.num_rounds == 2
    assert [r.n_passed for r in traj.rounds] == [13, 17]
    assert traj.rounds[1].t_start_s == traj.rounds[0].t_end_s   # concatenated axis


def test_import_token_and_cost_series_monotonic(tmp_path):
    traj = import_run(_make_run(tmp_path))
    assert len(traj.points) == 4                        # 2 rounds × 2 turns
    for key in ("cum_total_tokens", "cum_cost_usd"):
        vals = [getattr(p, key) for p in traj.points]
        assert all(vals[i] >= vals[i - 1] for i in range(1, len(vals)))
    assert abs(traj.final_cost_usd - 0.04) < 1e-9       # 0.02 + 0.02


def test_import_verilator_bands_and_circt_autodetect(tmp_path):
    # name contains "circt" → circt classifier auto-enabled; verilator still classifies as tool
    traj = import_run(_make_run(tmp_path, circt_name=True))
    assert any(b.category == "tool" for b in traj.bands)
    assert "circt" in traj.source or traj.classifier_config["long_wait_rules"]
    # circt run has the extra RTL-facts rule
    assert len(traj.classifier_config["long_wait_rules"]) == 2


def test_import_splits_multiple_result_events_in_one_file(tmp_path):
    """A transcript file that concatenates two invocations (two result events) must count both
    costs and become two rounds — not silently keep only the last result's cost."""
    run = tmp_path / "rb_multi"
    (run / "rounds").mkdir(parents=True)
    seg_a = _transcript("sessA", "2026-06-20T16:00:00.000Z", verilator=True)   # result cost 0.02
    seg_b = _transcript("sessB", "2026-06-20T16:10:00.000Z", verilator=False)  # result cost 0.02
    (run / "rounds" / "round_00.transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in seg_a + seg_b) + "\n")

    traj = import_run(run)
    assert traj.num_rounds == 2                       # one file → two invocations → two rounds
    assert abs(traj.final_cost_usd - 0.04) < 1e-9     # both result costs counted


def test_import_survives_missing_qa_and_selfcheck(tmp_path):
    run = _make_run(tmp_path)
    (run / "selfcheck_log.jsonl").unlink()
    import shutil
    shutil.rmtree(run / "qa_history")
    traj = import_run(run)
    assert traj.num_rounds == 2
    assert traj.milestones == []
    assert all(r.n_passed is None for r in traj.rounds)   # no verdicts → None, not a crash
