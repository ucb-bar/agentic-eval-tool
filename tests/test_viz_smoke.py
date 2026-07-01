"""Viz smoke: plots render to non-empty images. Skipped when the [viz] extra is absent."""
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

import matplotlib
matplotlib.use("Agg")

from aet.trajectory.importers.capsule_bench import import_run
from aet.viz.trajectory_plot import plot_trajectory, plot_comparison, activity_share
from tests.test_trajectory_import import _make_run


def test_plot_trajectory_renders_png(tmp_path):
    traj = import_run(_make_run(tmp_path))
    fig = plot_trajectory(traj)
    out = tmp_path / "traj.png"
    fig.savefig(out, dpi=80)
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_comparison_renders(tmp_path):
    t1 = import_run(_make_run(tmp_path / "a"))
    t2 = import_run(_make_run(tmp_path / "b", circt_name=True))
    fig = plot_comparison([t1, t2], labels=["raw", "circt"])
    out = tmp_path / "cmp.png"
    fig.savefig(out, dpi=80)
    assert out.is_file() and out.stat().st_size > 1000


def test_activity_share_normalizes(tmp_path):
    traj = import_run(_make_run(tmp_path))
    g, sh = activity_share(traj, ngrid=100, win=5)
    assert len(g) == 100
    # shares across the lanes sum to ~1 wherever there is any activity
    import numpy as np
    total = sum(sh[a] for a in sh)
    assert np.allclose(total, 1.0, atol=1e-6)
