"""CLI smoke: `aet import` writes a parseable trajectory.json."""
import json
import subprocess
import sys
from pathlib import Path

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
