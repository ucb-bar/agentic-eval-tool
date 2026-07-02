"""Smoke tests for the presentation comparison figures (skip cleanly without the [viz] extra).

Uses tiny synthetic trajectories (no external data) so it runs anywhere: each figure must save a
non-empty PNG, rate_series must be finite/non-negative with no spurious spikes, and the N-arm
series_styles must be distinct.
"""
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

import matplotlib
matplotlib.use("Agg")

from aet.trajectory.model import (
    RunTrajectory, TrajectoryPoint, ActivityBand, TestMilestone, RoundBoundary,
)


def _arm(run_id, *, dur_s, final_cost, passes, provisional=False):
    pts, bands = [], []
    n = 12
    for i in range(n):
        t = dur_s * i / (n - 1)
        pts.append(TrajectoryPoint(
            t_s=t, cum_input_tokens=1000 * i, cum_output_tokens=300 * i,
            cum_cache_tokens=5000 * i, cum_cost_usd=final_cost * i / (n - 1),
            round_index=0 if i < n // 2 else 1, provisional_cost=provisional))
        cat = ["think", "read", "write", "bash", "tool"][i % 5]
        bands.append(ActivityBand(t, dur_s * (i + 1) / (n - 1), cat, weight=1.0))
    ms = [TestMilestone(dur_s * 0.5, passes[0], 20, source="selfcheck_log"),
          TestMilestone(dur_s * 0.9, passes[1], 20, source="selfcheck_log")] if passes else []
    rounds = [RoundBoundary(0, 0.0, dur_s * 0.5, n_passed=passes[0] if passes else None, n_total=20),
              RoundBoundary(1, dur_s * 0.5, dur_s, n_passed=passes[1] if passes else None, n_total=20)]
    return RunTrajectory(
        run_id=run_id, source="test", duration_s=dur_s, num_rounds=2, provisional=provisional,
        points=pts, bands=bands, milestones=ms, rounds=rounds,
        final_cost_usd=final_cost, final_input_tokens=1000 * (n - 1),
        final_output_tokens=300 * (n - 1), final_cache_tokens=5000 * (n - 1))


@pytest.fixture
def trajs():
    return [_arm("arm-A", dur_s=2100, final_cost=17, passes=(0, 0)),
            _arm("arm-B", dur_s=5200, final_cost=26, passes=(17, 20)),
            _arm("arm-C", dur_s=5150, final_cost=36, passes=(16, 20), provisional=True)]


def test_rate_series_finite_nonneg_no_spike(trajs):
    import numpy as np
    from aet.viz.comparison import rate_series
    for series in ("input", "output"):
        t, r = rate_series(trajs[1], series)
        assert len(t) == len(r)
        assert np.all(np.isfinite(r)) and np.all(r >= 0)
        assert float(r.max()) < 1e7          # no divide-by-tiny-step spike


def test_series_styles_distinct():
    from aet.viz.style import series_styles
    styles = series_styles(4)
    assert len({s[0] for s in styles}) == 4   # 4 distinct colours
    assert len({s[1] for s in styles}) == 4   # 4 distinct markers


def test_rate_panels_saves_png(tmp_path, trajs):
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels(trajs, ["A", "B", "C"])
    out = tmp_path / "rate.png"
    fig.savefig(out, dpi=80)
    assert out.stat().st_size > 5000


def test_cost_vs_time_saves_png(tmp_path, trajs):
    from aet.viz.comparison import plot_cost_vs_time
    fig = plot_cost_vs_time(trajs, ["A", "B", "C"])
    out = tmp_path / "cost.png"
    fig.savefig(out, dpi=80)
    assert out.stat().st_size > 5000


def test_tests_facets_saves_png(tmp_path, trajs):
    from aet.viz.comparison import plot_tests_facets
    fig = plot_tests_facets(trajs, ["A", "B", "C"])
    out = tmp_path / "facets.png"
    fig.savefig(out, dpi=80)
    assert out.stat().st_size > 5000


def test_rate_panels_short_run_scale_bar_does_not_explode(tmp_path):
    # a minute-scale run mixed with a near-zero (crashed) run: the fixed ruler must clip to each
    # panel so bbox_inches='tight' can't blow the canvas to millions of px (regression guard)
    from aet.viz.comparison import plot_rate_panels
    short = _arm("short", dur_s=8, final_cost=0.5, passes=(1, 1))     # 0.13 min
    normal = _arm("normal", dur_s=300, final_cost=2, passes=(0, 1))   # 5 min
    fig = plot_rate_panels([short, normal], ["short", "normal"])
    w, h = fig.get_size_inches()
    assert w < 60 and h < 80                         # bounded canvas, not runaway
    out = tmp_path / "short.png"
    fig.savefig(out, dpi=80, bbox_inches="tight")    # the mode that previously exploded
    assert out.stat().st_size > 3000


def test_tests_facets_degrades_without_milestones(tmp_path):
    # a run with no milestones and no round verdicts → flat lane, still renders
    from aet.viz.comparison import plot_tests_facets
    flat = _arm("flat", dur_s=1000, final_cost=5, passes=None)
    flat.milestones = []
    for r in flat.rounds:
        r.n_passed = None
    fig = plot_tests_facets([flat], ["flat"])
    out = tmp_path / "flat.png"
    fig.savefig(out, dpi=80)
    assert out.stat().st_size > 3000
