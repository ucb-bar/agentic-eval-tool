"""Native recording: emit a trajectory through EvalRunLogger and reconstruct it from logs/."""
from pathlib import Path

from aet.tracking.run_logger import EvalRunLogger
from aet.trajectory.importers.capsule_bench import import_run
from aet.trajectory.model import RunTrajectory
from aet.trajectory.recording import emit_trajectory, materialize_run, trajectory_from_logs
from tests.test_trajectory_import import _make_run


def _logger(run_path: Path) -> EvalRunLogger:
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    (run_path / "metrics").mkdir(parents=True, exist_ok=True)
    return EvalRunLogger.start(project="aet", suite="trajectory", target="x",
                               method="test", seed=0, run_id=run_path.name,
                               run_path=run_path, tracking_mode="local")


def test_emit_then_reconstruct_from_logs(tmp_path):
    traj = import_run(_make_run(tmp_path))
    run_path = tmp_path / "run"
    emit_trajectory(traj, _logger(run_path), run_path)

    # canonical logs were written
    assert (run_path / "logs" / "metrics.jsonl").is_file()
    assert (run_path / "logs" / "events.jsonl").is_file()
    assert (run_path / "metrics" / "trajectory.json").is_file()

    back = trajectory_from_logs(run_path)
    assert len(back.points) == len(traj.points)
    assert len(back.rounds) == len(traj.rounds)
    assert len(back.milestones) == len(traj.milestones)
    assert [m.n_passed for m in back.milestones] == [m.n_passed for m in traj.milestones]
    assert [r.n_passed for r in back.rounds] == [r.n_passed for r in traj.rounds]
    assert abs(back.final_cost_usd - traj.final_cost_usd) < 1e-6
    # the cumulative curves survive the round-trip through step-metrics
    assert abs(back.points[-1].cum_total_tokens - traj.points[-1].cum_total_tokens) < 1.0


def test_materialize_then_from_run_dir_is_full_fidelity(tmp_path):
    traj = import_run(_make_run(tmp_path))
    run_path = materialize_run(traj, tmp_path / "aetrun")
    assert (run_path / "run_manifest.yaml").is_file()

    # from_run_dir prefers the trajectory.json fast-path → bands survive too
    back = RunTrajectory.from_run_dir(run_path)
    assert len(back.bands) == len(traj.bands) and len(back.bands) > 0
    assert back.to_dict() == traj.to_dict()


def test_from_run_dir_falls_back_to_logs(tmp_path):
    traj = import_run(_make_run(tmp_path))
    run_path = materialize_run(traj, tmp_path / "aetrun2")
    (run_path / "metrics" / "trajectory.json").unlink()   # force the logs/ reconstruction path

    back = RunTrajectory.from_run_dir(run_path)
    assert len(back.points) == len(traj.points)
    assert len(back.rounds) == len(traj.rounds)
    assert len(back.milestones) == len(traj.milestones)


def test_cache_read_and_creation_survive_the_logs_round_trip(tmp_path):
    """The split, not just the sum.

    A cache read is billed at ~0.1x the input rate and a cache write at ~1.25x — a factor of 12.5
    apart — so a trajectory that only round-trips their sum cannot answer where a run's cache spend
    went, and ``plot_trajectory(split_cache=True)`` draws two lines flat at zero. The fixture run
    has a cold first turn (200 written, 50 read) and a warm second one (0 written, 300 read), so the
    two series are unequal and neither equals the total: a fix that logged the sum twice would fail
    here.
    """
    traj = import_run(_make_run(tmp_path))
    run_path = tmp_path / "run_split"
    emit_trajectory(traj, _logger(run_path), run_path)

    back = trajectory_from_logs(run_path)

    assert [p.cum_cache_read_tokens for p in back.points] == \
        [p.cum_cache_read_tokens for p in traj.points]
    assert [p.cum_cache_creation_tokens for p in back.points] == \
        [p.cum_cache_creation_tokens for p in traj.points]
    # the sum stays consistent with its parts at every point
    for p in back.points:
        assert abs(p.cum_cache_read_tokens + p.cum_cache_creation_tokens
                   - p.cum_cache_tokens) < 1e-6
    # and the split is real structure, not the sum duplicated
    last = back.points[-1]
    assert last.cum_cache_read_tokens != last.cum_cache_creation_tokens
    assert last.cum_cache_read_tokens < last.cum_cache_tokens

    # run-level totals carry the split too
    assert back.final_cache_read_tokens == traj.final_cache_read_tokens
    assert back.final_cache_creation_tokens == traj.final_cache_creation_tokens


def test_a_log_without_the_split_metrics_reconstructs_at_zero(tmp_path):
    """Backward compatibility: a run recorded before this change must still load.

    Its two split series come back at zero — which is what ``aet plot --split-cache`` warns about
    rather than drawing.
    """
    import json

    traj = import_run(_make_run(tmp_path))
    run_path = tmp_path / "run_legacy"
    emit_trajectory(traj, _logger(run_path), run_path)

    metrics = run_path / "logs" / "metrics.jsonl"
    kept = [ln for ln in metrics.read_text().splitlines()
            if ln.strip() and json.loads(ln).get("name") not in (
                "aet.traj.cum_cache_read_tokens", "aet.traj.cum_cache_creation_tokens")]
    metrics.write_text("\n".join(kept) + "\n")
    params = run_path / "logs" / "params.json"
    p = json.loads(params.read_text())
    for k in ("final_cache_read_tokens", "final_cache_creation_tokens"):
        p.get("aet.traj.summary", {}).pop(k, None)
    params.write_text(json.dumps(p))

    back = trajectory_from_logs(run_path)
    assert len(back.points) == len(traj.points)          # still loads
    assert all(p.cum_cache_read_tokens == 0.0 for p in back.points)
    assert all(p.cum_cache_creation_tokens == 0.0 for p in back.points)
    # the sum is unaffected — the old field is still there
    assert back.points[-1].cum_cache_tokens == traj.points[-1].cum_cache_tokens
