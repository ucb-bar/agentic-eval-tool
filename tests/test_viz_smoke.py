"""Viz smoke: plots render to non-empty images. Skipped when the [viz] extra is absent."""
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def test_labels_do_not_round_a_real_measurement_to_zero():
    """"0 min" for a 26-second run and "0M" for 151,000 tokens both read as findings."""
    from aet.viz import style as S

    assert S.fmt_duration(26.0) == "26 s"
    assert S.fmt_duration(89.0) == "89 s"
    assert S.fmt_duration(600.0) == "10 min"

    assert S.fmt_tokens(151_392) == "151k"
    assert S.fmt_tokens(940) == "940"
    assert S.fmt_tokens(2_400_000) == "2.4M"


# --------------------------------------------------------------------- rate panels
def _bare_traj(run_id="chia-like", *, cache_read=0.0, cache_creation=0.0, bands=None,
               cache_total=None):
    """A trajectory shaped like chia's aet sink emits one: points, no tools, no tests.

    ``cache_total`` defaults to read+creation; pass it explicitly to model a source that
    recorded the sum but not the split, which is what the pre-fix sink produced."""
    from aet.trajectory.model import RunTrajectory, TrajectoryPoint
    total = (cache_read + cache_creation) if cache_total is None else cache_total
    pts = [TrajectoryPoint(t_s=float(i) * 10.0,
                           cum_input_tokens=10.0 * (i + 1), cum_output_tokens=80.0 * (i + 1),
                           cum_cache_tokens=total * (i + 1),
                           cum_cache_read_tokens=cache_read * (i + 1),
                           cum_cache_creation_tokens=cache_creation * (i + 1),
                           cum_cost_usd=0.01 * (i + 1))
            for i in range(8)]
    return RunTrajectory(run_id=run_id, duration_s=70.0, num_rounds=1, points=pts,
                         bands=list(bands or []),
                         final_input_tokens=80.0, final_output_tokens=640.0,
                         final_cache_tokens=total * 8, final_cost_usd=0.08)


def _texts(fig):
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts)


def test_no_test_record_means_no_score_on_the_chip():
    """The defect this was written for: the chip read 'final 0/20' for a run that scored nothing."""
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels([_bare_traj()], ["chia-cold"])
    blob = _texts(fig)
    assert "/20" not in blob and "final" not in blob
    assert "0.08" in blob or "$0.08" in blob        # the things it DID measure still render
    plt.close(fig)


def test_the_legend_only_names_activities_something_drew():
    """A five-entry activity key over a blank axis asserted five measurements that never happened."""
    from aet.viz.comparison import plot_rate_panels
    from aet.viz.style import ACT_LAB
    fig = plot_rate_panels([_bare_traj()], ["chia-cold"])
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert not (labels & set(ACT_LAB.values()))     # no lane is advertised
    assert "rate input" in labels and "rate output" in labels
    plt.close(fig)


def test_a_run_with_bands_still_gets_its_lanes():
    """The complement — the fix must not suppress a legend that has something behind it."""
    from aet.trajectory.model import ActivityBand
    from aet.viz.comparison import plot_rate_panels
    from aet.viz.style import ACT_LAB
    traj = _bare_traj(bands=[ActivityBand(t0_s=0.0, t1_s=20.0, category="read"),
                             ActivityBand(t0_s=20.0, t1_s=50.0, category="bash")])
    fig = plot_rate_panels([traj], ["with-tools"])
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert ACT_LAB["read"] in labels and ACT_LAB["bash"] in labels
    assert ACT_LAB["think"] not in labels           # only what was drawn
    plt.close(fig)


def test_the_cache_class_is_on_the_rate_axis():
    """It was not. On the reference chia run that left 151,392 of 152,088 tokens off a panel
    labelled 'token rate'."""
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels([_bare_traj(cache_read=15000.0, cache_creation=3500.0)], ["cold"])
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert "rate cache" in labels
    plt.close(fig)


def test_split_cache_separates_the_two_classes_on_the_rate_axis():
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels([_bare_traj(cache_read=15000.0, cache_creation=3500.0)], ["cold"],
                           split_cache=True)
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert "rate cache read" in labels and "rate cache write" in labels
    assert "rate cache" not in labels
    plt.close(fig)


def test_split_cache_falls_back_when_the_source_recorded_only_the_sum():
    """Asking for a split a run does not carry must draw the sum, not two lines flat at zero."""
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels([_bare_traj(cache_read=0.0, cache_creation=0.0,
                                       cache_total=18000.0)], ["nosplit"], split_cache=True)
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert "rate cache" in labels and "rate cache read" not in labels
    plt.close(fig)


def test_a_zero_rate_is_omitted_not_drawn_at_the_log_floor():
    """The warm reference arm writes no cache at all.

    ``clip(r, 1, None)`` is what keeps the log axis drawable, but it renders a true zero as an
    apparent 1 tok/min — a rate the run never had. The series must be absent, not floored.
    """
    from aet.viz.comparison import plot_rate_panels
    warm = _bare_traj(cache_read=18000.0, cache_creation=0.0)
    fig = plot_rate_panels([warm], ["chia-warm"], split_cache=True)
    twins = [ax for ax in fig.axes if ax.get_ylabel().startswith("token rate")]
    assert twins, "no rate axis found"
    drawn = [ln for ax in twins for ln in ax.get_lines()]
    assert len(drawn) == 3            # input, output, cache read — not cache write
    for ln in drawn:
        assert (ln.get_ydata() > 1).any()
    plt.close(fig)


def test_no_milestone_legend_entry_without_milestones():
    from aet.viz.comparison import plot_rate_panels
    fig = plot_rate_panels([_bare_traj()], ["chia-cold"])
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert "test-pass milestone" not in labels
    plt.close(fig)


def test_the_rate_legend_names_only_lines_that_were_drawn():
    """A run too short to differentiate has no rate at all — rate_series needs four points.
    A key listing four series over an empty rate axis describes a different run."""
    from aet.trajectory.model import RunTrajectory, TrajectoryPoint
    from aet.viz.comparison import plot_rate_panels
    short = RunTrajectory(run_id="short", duration_s=20.0, num_rounds=1,
                          points=[TrajectoryPoint(t_s=float(i) * 10.0,
                                                  cum_input_tokens=10.0 * (i + 1),
                                                  cum_output_tokens=20.0 * (i + 1))
                                  for i in range(3)])
    fig = plot_rate_panels([short], ["short"])
    labels = {t.get_text() for lg in fig.legends for t in lg.get_texts()}
    assert not any(l.startswith("rate ") for l in labels)
    plt.close(fig)
