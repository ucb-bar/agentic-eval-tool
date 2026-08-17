from aet.trajectory.model import ActivityBand, InferenceRecord, RunTrajectory
from aet.viz.comparison import plot_agent_profiles


def test_agent_profile_figure_renders(tmp_path):
    traj = RunTrajectory(
        run_id="run",
        duration_s=2,
        inferences=[InferenceRecord(
            request_id="r", t_start_s=0, t_end_s=2, agent_id="root",
            input_tokens=10, output_tokens=5, cache_read_tokens=20,
            context_window_tokens=100, estimated_context_tokens=30,
        )],
        bands=[ActivityBand(t0_s=0.5, t1_s=1.0, category="bash")],
    )
    fig = plot_agent_profiles([traj], ["run"])
    assert len(fig.axes) == 4
    assert {tick.get_text() for tick in fig.axes[3].get_xticklabels()} == {"model", "bash"}
    out = tmp_path / "profile.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0
