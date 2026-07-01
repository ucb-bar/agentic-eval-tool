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
