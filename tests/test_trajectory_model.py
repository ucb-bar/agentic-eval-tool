"""RunTrajectory data-model: construction, derived views, and round-trip serialization."""
from aet.trajectory.model import (
    CHECKPOINT_KINDS, ActivityBand, Checkpoint, InferenceRecord, RoundBoundary,
    RunTrajectory, TestMilestone, TrajectoryPoint,
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


def test_v12_trajectory_loads_without_inferences():
    data = _sample().to_dict()
    data["schema_version"] = "1.2"
    data.pop("inferences")
    loaded = RunTrajectory.from_dict(data)
    assert loaded.schema_version == "1.2"
    assert loaded.inferences == []


def test_inference_subset_arithmetic_and_agent_rollup():
    record = InferenceRecord(
        request_id="r", t_start_s=0, t_end_s=2, agent_id="child", parent_agent_id="root",
        input_tokens=10, cache_read_tokens=80, cache_write_tokens=10,
        output_tokens=20, reasoning_tokens=5, context_window_tokens=200,
        estimated_context_tokens=100,
    )
    traj = RunTrajectory(inferences=[record])
    assert record.billed_input_tokens == 100
    assert record.cache_hit_ratio == 0.8
    assert record.context_occupancy_ratio == 0.5
    rollup = traj.per_agent_rollup()["child"]
    assert rollup["reasoning_tokens"] == 5
    assert rollup["activity_share"] == 1.0


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


def test_tests_steps_from_milestones_monotonic():
    t = _sample()   # milestones 13@120s, 17@400s; duration 600s
    xs, ys = t.tests_steps()
    assert xs[0] == 0.0 and ys[0] == 0
    assert ys == sorted(ys)                       # non-decreasing
    assert ys[-1] == t.final_tests() == 17        # ends at the best reached
    assert xs[-1] == t.duration_s / 60.0          # padded to run end
    assert t.tests_total() == 20


def test_tests_steps_falls_back_to_round_verdicts():
    t = RunTrajectory(
        run_id="x", duration_s=600.0, num_rounds=2,
        rounds=[RoundBoundary(0, 0.0, 300.0, n_passed=5, n_total=20),
                RoundBoundary(1, 300.0, 600.0, n_passed=9, n_total=20)],
    )
    xs, ys = t.tests_steps()
    assert ys[-1] == 9 and t.final_tests() == 9
    assert t.tests_total() == 20


def test_tests_steps_empty_progression_degrades():
    t = RunTrajectory(run_id="x", duration_s=120.0)   # no milestones, no round verdicts
    xs, ys = t.tests_steps()
    assert ys == [0, 0] and xs == [0.0, 2.0]
    assert t.final_tests() == 0


def test_a_run_with_no_test_record_has_no_denominator():
    """``None``, not 20.

    A run from a source that scores nothing (an LLM-call trajectory, say) used to report a
    denominator of 20, and the figure chip rendered "final 0/20" — a score against a suite size
    nothing had measured. "No tests recorded" and "0 of 20 passed" are different facts.
    """
    assert RunTrajectory(run_id="x", duration_s=120.0).tests_total() is None


def test_a_zero_n_total_is_still_no_record():
    """A verdict carrying n_total=0 must not be mistaken for a recorded suite of size 0."""
    t = RunTrajectory(run_id="x", duration_s=60.0, num_rounds=1,
                      rounds=[RoundBoundary(0, 0.0, 60.0, n_passed=0, n_total=0)])
    assert t.tests_total() is None


# --------------------------------------------------------------------------- checkpoints (v1.1)


class TestCheckpoints:
    """One-shot progress landmarks, distinct from the pass/total milestone axis.

    Time-to-first-parse and time-to-first-elaboration cannot be expressed as `n_passed/n_total`
    without abusing `scope`, and `EvalRunLogger.record_elaboration` records an iteration index
    rather than a `t_s` — so neither reached the trajectory before this.
    """

    def _t(self) -> RunTrajectory:
        return RunTrajectory(
            run_id="cp",
            checkpoints=[
                Checkpoint(t_s=40.0, kind="first_parse", source="build_log"),
                Checkpoint(t_s=12.0, kind="first_parse", source="build_log"),
                Checkpoint(t_s=90.0, kind="first_module_elab", scope="atlas.mxu0"),
            ],
        )

    def test_time_to_returns_the_first_crossing(self):
        assert self._t().time_to("first_parse") == 12.0, "not the last, and not the one appended first"

    def test_time_to_is_none_when_never_reached(self):
        """Substituting the run duration would turn 'never got there' into 'got there at the very
        end' — the difference between a censored observation and a slow one."""
        assert self._t().time_to("public_all") is None

    def test_time_to_is_scoped(self):
        t = self._t()
        assert t.time_to("first_module_elab", scope="atlas.mxu0") == 90.0
        assert t.time_to("first_module_elab") is None, "the default scope did not reach it"

    def test_the_ladder_keeps_unreached_rungs(self):
        ladder = self._t().checkpoint_ladder()
        assert [k for k, _ in ladder] == list(CHECKPOINT_KINDS), "order is the comparison"
        assert dict(ladder)["first_parse"] == 12.0
        assert dict(ladder)["full_elab"] is None
        assert len(ladder) == len(CHECKPOINT_KINDS), "a stalled run and an unmeasured one differ"

    def test_checkpoints_round_trip(self):
        t = self._t()
        back = RunTrajectory.from_dict(t.to_dict())
        assert [(c.kind, c.t_s, c.scope) for c in back.checkpoints] == [
            (c.kind, c.t_s, c.scope) for c in t.checkpoints
        ]

    def test_a_v1_0_trajectory_still_loads(self):
        """Every trajectory written before this field existed has no `checkpoints` key."""
        d = _sample().to_dict()
        d.pop("checkpoints")
        d["schema_version"] = "1.0"
        back = RunTrajectory.from_dict(d)
        assert back.checkpoints == []
        assert back.milestone_series(), "the rest of the trajectory is untouched"
