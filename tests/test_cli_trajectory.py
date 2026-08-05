"""CLI smoke: `aet import` writes a parseable trajectory.json."""
import subprocess
import sys

from aet.trajectory.model import RunTrajectory
from tests.test_trajectory_import import _make_run


def _run_cli(*argv) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "aet.cli.main", *argv],
                          capture_output=True, text=True)


def test_cli_import_writes_trajectory(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "traj.json"
    cp = _run_cli("import", "--source", "capsule-bench", "--raw", str(run), "--out", str(out))
    assert cp.returncode == 0, cp.stderr
    assert out.is_file()
    traj = RunTrajectory.from_json(out)
    assert traj.num_rounds == 2
    assert [m.n_passed for m in sorted(traj.milestones, key=lambda m: m.t_s)] == [13, 17, 20]
    assert "milestones" in cp.stdout or "imported" in cp.stdout


def test_cli_import_unknown_source_errors(tmp_path):
    run = _make_run(tmp_path)
    cp = _run_cli("import", "--source", "nope", "--raw", str(run))
    assert cp.returncode != 0
    assert "unknown import source" in (cp.stderr + cp.stdout)


def test_cli_plot_split_cache_renders_the_two_series(tmp_path):
    """``--split-cache`` is reachable from the CLI and draws both cache lines.

    The kwarg existed on ``plot_trajectory`` and both series existed on ``TrajectoryPoint``, but
    no flag ever set it, so this path was unreachable from ``aet plot``.
    """
    from aet.trajectory.importers.capsule_bench import import_run
    from aet.trajectory.recording import materialize_run

    traj = import_run(_make_run(tmp_path))
    run = materialize_run(traj, tmp_path / "aetrun")
    (run / "metrics" / "trajectory.json").unlink()   # force the logs/ reconstruction path

    out = tmp_path / "fig.png"
    cp = _run_cli("plot", str(run), "--kind", "trajectory", "--split-cache", "--out", str(out))
    assert cp.returncode == 0, cp.stderr
    assert out.is_file() and out.stat().st_size > 0
    # the run *does* have a split, so no missing-data warning
    assert "no cache read/creation split" not in cp.stderr


def test_cli_plot_split_cache_warns_when_the_split_is_missing(tmp_path):
    """A log recorded before the split existed must say so, not draw two lines flat at zero."""
    import json

    from aet.trajectory.importers.capsule_bench import import_run
    from aet.trajectory.recording import materialize_run

    traj = import_run(_make_run(tmp_path))
    run = materialize_run(traj, tmp_path / "aetrun_legacy")
    (run / "metrics" / "trajectory.json").unlink()
    metrics = run / "logs" / "metrics.jsonl"
    metrics.write_text("\n".join(
        ln for ln in metrics.read_text().splitlines()
        if ln.strip() and json.loads(ln).get("name") not in (
            "aet.traj.cum_cache_read_tokens", "aet.traj.cum_cache_creation_tokens")) + "\n")

    cp = _run_cli("plot", str(run), "--kind", "trajectory", "--split-cache",
                  "--out", str(tmp_path / "fig2.png"))
    assert cp.returncode == 0, cp.stderr
    assert "no cache read/creation split" in cp.stderr
