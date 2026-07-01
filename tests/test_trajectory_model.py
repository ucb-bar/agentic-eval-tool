"""RunTrajectory data-model: construction, derived views, and round-trip serialization."""
from aet.trajectory.model import (
    RunTrajectory, TrajectoryPoint, ActivityBand, TestMilestone, RoundBoundary,
)


def _sample() -> RunTrajectory:
    return RunTrajectory(
        run_id="rb_test", source="import:capsule-bench", model="claude-opus-4-8",
        duration_s=600.0, num_rounds=2,
        points=[
            TrajectoryPoint(t_s=0.0, cum_input_tokens=10, cum_output_tokens=5,
                            cum_cache_tokens=100, cum_cost_usd=0.1, round_index=0),
            TrajectoryPoint(t_s=300.0, cum_input_tokens=30, cum_output_tokens=15,
                            cum_cache_tokens=300, cum_cost_usd=0.5, round_index=1),
        ],
        bands=[ActivityBand(0.0, 30.0, "think"),
               ActivityBand(30.0, 90.0, "tool", tool_name="Bash", weight=28.0)],
        milestones=[TestMilestone(120.0, 13, 20, source="selfcheck_log"),
                    TestMilestone(400.0, 17, 20, source="selfcheck_log")],
        rounds=[RoundBoundary(0, 0.0, 300.0, cost_usd=0.1, n_passed=13, n_total=20),
                RoundBoundary(1, 300.0, 600.0, cost_usd=0.4, n_passed=17, n_total=20)],
        final_cost_usd=0.5, final_input_tokens=30, final_output_tokens=15,
        final_cache_tokens=300, classifier_config={"weights": {"tool": 28.0}},
    )


def test_cum_total_tokens_property():
    p = TrajectoryPoint(t_s=0.0, cum_input_tokens=10, cum_output_tokens=5, cum_cache_tokens=100)
    assert p.cum_total_tokens == 115


def test_band_and_round_durations():
    assert ActivityBand(30.0, 90.0, "tool").duration_s == 60.0
    assert RoundBoundary(0, 0.0, 300.0).duration_s == 300.0
    # never negative even if endpoints are swapped
    assert ActivityBand(90.0, 30.0, "tool").duration_s == 0.0


def test_dict_round_trip_is_lossless():
    t = _sample()
    back = RunTrajectory.from_dict(t.to_dict())
    assert back.to_dict() == t.to_dict()
    assert back.run_id == "rb_test"
    assert len(back.points) == 2 and len(back.milestones) == 2 and len(back.rounds) == 2


def test_json_round_trip(tmp_path):
    t = _sample()
    p = t.to_json(tmp_path / "metrics" / "trajectory.json")
    assert p.is_file()
    back = RunTrajectory.from_json(p)
    assert back.to_dict() == t.to_dict()


def test_from_run_dir_prefers_fast_path(tmp_path):
    t = _sample()
    t.to_json(tmp_path / "metrics" / "trajectory.json")
    back = RunTrajectory.from_run_dir(tmp_path)
    assert back.run_id == "rb_test"


def test_token_series_in_minutes_and_monotonic():
    t = _sample()
    s = t.token_series()
    assert s["t"] == [0.0, 5.0]                 # seconds → minutes
    assert s["total"] == [115.0, 345.0]         # input+output+cache, cumulative
    assert s["spend"] == [0.1, 0.5]
    for key in ("input", "output", "cache", "total", "spend"):
        vals = s[key]
        assert all(vals[i] >= vals[i - 1] for i in range(1, len(vals)))


def test_milestone_series_sorted_pairs():
    t = _sample()
    assert t.milestone_series() == [(2.0, 13), (400.0 / 60.0, 17)]
