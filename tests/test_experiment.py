"""Arms, the arm×seed grid, and the fairness check that has to be able to fail.

Before this package, "arm" was a plot label. The claim that two conditions differed in exactly one
declared way rested on whoever wrote the driver script having got it right — not a control, because
nobody could re-derive it afterwards. These tests are mostly about the failure direction: a fairness
check that only ever passes is indistinguishable from no check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aet.experiment import Arm, ArmMatrix, compare_arms, diff_arms, pair_by_seed
from aet.experiment.matrix import format_matrix

_HELD = dict(model="m", model_version="v1", agent="claude", max_tokens=100,
             max_cost_usd=5.0, public_suite="p", hidden_eval="h")


def _ladder() -> list[Arm]:
    a1 = Arm(id="a1", treatment={"rep": "markdown", "scaffold": "none", "qa": "none"},
             granted=("bundle",), **_HELD)
    a2 = a1.with_(id="a2", treatment={"rep": "ir", "scaffold": "none", "qa": "none"},
                  granted=("ir",))
    a3 = a2.with_(id="a3", treatment={"rep": "ir", "scaffold": "compiled", "qa": "none"},
                  granted=("ir", "scaffold"))
    a4 = a3.with_(id="a4", treatment={"rep": "ir", "scaffold": "compiled", "qa": "full"},
                  tools=("qa",))
    return [a1, a2, a3, a4]


class TestArm:
    def test_fingerprint_excludes_identity_and_notes(self):
        a = Arm(id="x", label="X", notes={"why": "free text"}, **_HELD)
        fp = a.fingerprint()
        assert "id" not in fp and "label" not in fp and "notes" not in fp
        assert fp["model"] == "m"

    def test_a_declared_treatment_difference_is_declared(self):
        a1, a2, *_ = _ladder()
        diffs = diff_arms(a1, a2)
        assert {d.field for d in diffs} == {"treatment", "granted"}
        assert all(d.declared for d in diffs)

    def test_a_held_constant_difference_is_never_declared(self):
        """The realistic mistake: a debugging token bump left on one arm."""
        a1, a2, *_ = _ladder()
        diffs = diff_arms(a1, a2.with_(max_tokens=999))
        tok = [d for d in diffs if d.field == "max_tokens"]
        assert tok and tok[0].undeclared

    def test_a_treatment_field_differing_without_a_treatment_declaration_is_undeclared(self):
        """Two arms with the same declared treatment that nonetheless grant different files are not
        expressing a treatment — they are expressing a mistake."""
        a1 = _ladder()[0]
        sneaky = a1.with_(id="a1b", granted=("bundle", "an_extra_thing"))
        diffs = diff_arms(a1, sneaky)
        assert [d.field for d in diffs] == ["granted"]
        assert diffs[0].undeclared

    def test_sandbox_kwargs_carry_the_isolation_flags(self):
        a = _ladder()[0]
        kw = a.sandbox_kwargs(Path("/tmp/ws"))
        assert kw["unshare_net"] and kw["clearenv"] and kw["mask_git"]
        assert kw["allow"] == [Path("bundle")]


class TestMatrix:
    def test_a_clean_four_arm_ladder_validates(self):
        m = ArmMatrix(arms=_ladder(), seeds=[0, 1, 2])
        assert m.validate() == [], format_matrix(m)
        assert m.valid

    def test_each_step_changes_exactly_one_dimension(self):
        m = ArmMatrix(arms=_ladder(), seeds=[0, 1, 2])
        assert [changed for _, _, changed in m.ladder_steps()] == [
            ("rep",), ("scaffold",), ("qa",)]

    def test_an_undeclared_difference_fails_validation(self):
        a1, a2, a3, a4 = _ladder()
        m = ArmMatrix(arms=[a1, a2, a3.with_(max_tokens=999), a4], seeds=[0])
        codes = {f.code for f in m.validate()}
        assert "ARM001" in codes

    def test_two_dimensions_moving_at_once_fails_validation(self):
        """A delta across two changes isolates neither."""
        a1, a2, a3, _ = _ladder()
        a4 = a3.with_(id="a4", treatment={"rep": "ir", "scaffold": "templated", "qa": "full"})
        m = ArmMatrix(arms=[a1, a2, a3, a4], seeds=[0])
        assert any(f.code == "ARM002" for f in m.validate())

    def test_an_ablation_is_exempt_from_the_ladder(self):
        """An ablation is supposed to differ by more than one dimension; that is what it is for."""
        arms = _ladder()
        ref = arms[2].with_(id="a3_ref",
                            treatment={"rep": "ir", "scaffold": "reference_skeleton", "qa": "none"},
                            granted=("ir", "scaffold", "sidecars"), max_tokens=100)
        m = ArmMatrix(arms=arms + [ref], seeds=[0], ablations=("a3_ref",))
        assert m.validate() == [], format_matrix(m)

    def test_the_grid_is_every_arm_crossed_with_every_seed(self):
        m = ArmMatrix(arms=_ladder(), seeds=[0, 1, 2])
        assert len(m.cells()) == 12
        assert {c.seed for c in m.cells()} == {0, 1, 2}

    def test_execution_order_is_shuffled_but_reproducible(self):
        """An unreproducible order cannot be audited, which is exactly what a reviewer would want
        to check when a result looks load-dependent."""
        m = ArmMatrix(arms=_ladder(), seeds=[0, 1, 2])
        a = m.execution_order(seed=7)
        assert a == m.execution_order(seed=7)
        assert [c.arm for c in a] != [c.arm for c in m.cells()], "not shuffled at all"
        assert sorted((c.arm, c.seed) for c in a) == sorted((c.arm, c.seed) for c in m.cells())

    def test_duplicate_seeds_and_arms_are_rejected(self):
        assert any(f.code == "ARM003" for f in ArmMatrix(arms=_ladder(), seeds=[0, 0]).validate())
        assert any(f.code == "ARM003" for f in ArmMatrix(arms=_ladder(), seeds=[]).validate())

    def test_run_ids_are_stable_and_sortable(self):
        m = ArmMatrix(arms=_ladder(), seeds=[0, 1, 10])
        ids = [c.run_id for c in m.cells() if c.arm == "a1"]
        assert ids == ["a1_seed000", "a1_seed001", "a1_seed010"]


class TestPaired:
    def test_pairing_keeps_only_shared_seeds_and_counts_the_rest(self):
        seeds, a, b, dropped = pair_by_seed({0: 10, 1: 20, 2: 30}, {0: 7, 1: 15})
        assert (seeds, a, b, dropped) == ([0, 1], [10, 20], [7, 15], 1)

    def test_compare_reports_the_paired_delta_with_an_interval(self):
        c = compare_arms("cost_usd", "a1", "a2", {0: 10.0, 1: 20.0, 2: 30.0},
                         {0: 7.0, 1: 15.0, 2: 28.0})
        assert c.n_paired == 3
        assert c.mean_delta == pytest.approx(10 / 3, rel=1e-3)
        assert c.ci_lower is not None and c.ci_lower <= c.mean_delta <= c.ci_upper

    def test_compare_carries_the_underpowered_flag_through(self):
        """At n=3 no result can reach p<0.05. A summary that omitted that would invite the reader
        to read a non-significant result as an absence of effect."""
        c = compare_arms("cost_usd", "a1", "a2", {0: 10.0, 1: 20.0, 2: 30.0},
                         {0: 7.0, 1: 15.0, 2: 28.0})
        assert c.underpowered is True
        assert c.min_achievable_p == 0.25
        assert "UNDERPOWERED" in c.summary()

    def test_compare_reports_unpaired_seeds(self):
        c = compare_arms("m", "a", "b", {0: 1.0, 1: 2.0, 5: 9.0}, {0: 1.0, 1: 1.0})
        assert c.n_paired == 2 and c.n_unpaired == 1

    def test_survival_is_attached_not_folded_into_the_mean(self):
        """A mean over censored durations is not a mean of anything."""
        c = compare_arms(
            "time_to_success_s", "a1", "a2",
            {0: 100.0, 1: 200.0, 2: 300.0}, {0: 90.0, 1: 180.0, 2: 300.0},
            censored_a={2: True}, censored_b={2: True})
        assert c.survival is not None
        assert c.survival["a1"]["n_censored"] == 1
        assert c.survival["a1"]["median"] == 200.0
        assert c.mean_delta is not None, "the point estimate is still reported, separately"

    def test_no_shared_seeds_is_reported_not_crashed(self):
        c = compare_arms("m", "a", "b", {0: 1.0}, {1: 2.0})
        assert c.n_paired == 0
        assert "no paired seeds" in c.summary()
