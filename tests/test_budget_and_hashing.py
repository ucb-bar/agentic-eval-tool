"""Budget enforcement and immutable input hashes.

Both existed on paper. `aet spend --budget-usd` is post-hoc — by the time the number exists the
money is gone, and an arm that overran its cap is no longer comparable to one that did not.
`RunSpec.spec_version_hash` and friends were declared and never populated, which is worse than
absent: a declared-and-unread field reads like a control while providing none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aet.core.budget import (
    Budget,
    BudgetExceeded,
    BudgetGuard,
    BudgetUnknown,
    Usage,
    preflight,
)
from aet.core.hashing import (
    InputsDiffer,
    assert_comparable,
    compare_inputs,
    hash_inputs,
    sha256_dir,
    sha256_file,
)


class TestBudgetGuard:
    def test_under_budget_is_silent(self):
        g = BudgetGuard(Budget(max_tokens=100, max_cost_usd=1.0))
        assert g.check(Usage(tokens=50, cost_usd=0.5)) is None
        assert g.clean

    def test_a_breach_raises_with_the_numbers_attached(self):
        g = BudgetGuard(Budget(max_tokens=100), run_id="r1")
        with pytest.raises(BudgetExceeded) as e:
            g.check(Usage(tokens=150))
        assert e.value.kind == "tokens"
        assert (e.value.limit, e.value.actual) == (100.0, 150.0)
        assert "r1" in str(e.value)

    def test_a_breach_can_be_recorded_without_raising(self):
        g = BudgetGuard(Budget(max_cost_usd=1.0))
        assert g.check(Usage(cost_usd=2.0), raise_on_breach=False) == "cost_usd"
        assert not g.clean and len(g.breaches) == 1

    def test_an_unpriced_model_is_not_treated_as_free(self):
        """`cost_usd=None` means the price table has no entry, not that the run cost nothing.
        Raising here would kill a run for a pricing gap; treating None as 0 would authorise an
        unbounded one. Skipping, and refusing at pre-flight instead, is the only honest option."""
        g = BudgetGuard(Budget(max_cost_usd=1.0))
        assert g.check(Usage(tokens=1, cost_usd=None)) is None

    def test_an_unbounded_budget_never_breaches(self):
        g = BudgetGuard(Budget())
        assert not Budget().is_bounded()
        assert g.check(Usage(tokens=10**9, cost_usd=10**9)) is None


class TestPreflight:
    def test_a_grid_that_fits_is_authorised_with_its_arithmetic(self):
        out = preflight(12, Budget(max_cost_usd=5.0),
                        total_cost_cap_usd=200.0, already_spent_usd=52.74)
        assert out["authorised"]
        assert out["estimated_total_usd"] == 60.0
        assert out["headroom_usd"] == pytest.approx(87.26)

    def test_a_grid_that_does_not_fit_is_refused_before_anything_is_spent(self):
        """A refusal here costs nothing. The same refusal after cell 9 of 12 has wasted nine runs
        and left a grid that cannot be analysed."""
        with pytest.raises(BudgetExceeded) as e:
            preflight(60, Budget(max_cost_usd=5.0),
                      total_cost_cap_usd=200.0, already_spent_usd=52.74)
        assert "300" in str(e.value) and "147.26" in str(e.value)

    def test_an_unpriceable_grid_is_a_different_refusal(self):
        """'Costs more than the cap' and 'nobody can say what this costs' call for different
        decisions; one flag bypassing both would teach the operator to bypass both."""
        with pytest.raises(BudgetUnknown):
            preflight(12, Budget(), total_cost_cap_usd=200.0)

    def test_no_cap_authorises_but_says_so(self):
        out = preflight(12, Budget(), total_cost_cap_usd=None)
        assert out["authorised"] and out["note"]

    def test_the_estimate_defaults_to_the_worst_case(self):
        """Authorising on an optimistic estimate and discovering the truth at cell 9 is how a
        budget gets blown."""
        worst = preflight(10, Budget(max_cost_usd=5.0), total_cost_cap_usd=1000.0)
        assert worst["per_run_usd"] == 5.0
        actual = preflight(10, Budget(max_cost_usd=5.0), total_cost_cap_usd=1000.0,
                           estimated_cost_per_run=1.0)
        assert actual["estimated_total_usd"] == 10.0


class TestHashing:
    @pytest.fixture
    def tree(self, tmp_path):
        d = tmp_path / "bundle"
        (d / "sub").mkdir(parents=True)
        (d / "a.md").write_text("alpha\n")
        (d / "sub" / "b.md").write_text("beta\n")
        return d

    def test_a_tree_hashes_deterministically(self, tree):
        assert sha256_dir(tree) == sha256_dir(tree)

    def test_editing_one_byte_changes_the_hash(self, tree):
        before = sha256_dir(tree)
        (tree / "a.md").write_text("alphb\n")
        assert sha256_dir(tree) != before

    def test_renaming_changes_the_hash(self, tree):
        before = sha256_dir(tree)
        (tree / "a.md").rename(tree / "z.md")
        assert sha256_dir(tree) != before

    def test_vcs_churn_does_not(self, tree):
        before = sha256_dir(tree)
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        assert sha256_dir(tree) == before

    def test_a_missing_input_hashes_to_none_and_keeps_its_key(self, tree):
        """Dropping the key would make a run that lacked an input compare EQUAL to one that had it."""
        h = hash_inputs({"bundle": tree, "scaffold": tree / "nope"})
        assert h["bundle"] and h["scaffold"] is None
        assert set(h) == {"bundle", "scaffold"}

    def test_absent_is_not_equal_to_present(self, tree):
        a = hash_inputs({"x": tree})
        b = hash_inputs({"x": tree / "nope"})
        assert compare_inputs(a, b) == ["x"]

    def test_sha256_file_streams_and_matches_content(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"z" * (3 << 20))       # larger than one read chunk
        import hashlib
        assert sha256_file(f) == hashlib.sha256(b"z" * (3 << 20)).hexdigest()


class TestAssertComparable:
    def test_identical_inputs_are_comparable(self, tmp_path):
        d = tmp_path / "b"
        d.mkdir()
        (d / "x").write_text("1")
        h = hash_inputs({"bundle": d})
        assert_comparable({"r1": h, "r2": dict(h)})

    def test_divergent_inputs_are_refused(self, tmp_path):
        """The function that turns recorded hashes into a control. Without it they are a field
        somebody might read."""
        d = tmp_path / "b"
        d.mkdir()
        (d / "x").write_text("1")
        h1 = hash_inputs({"bundle": d})
        (d / "x").write_text("2")
        h2 = hash_inputs({"bundle": d})
        with pytest.raises(InputsDiffer) as e:
            assert_comparable({"r1": h1, "r2": h2})
        assert "bundle" in str(e.value)

    def test_inputs_that_are_supposed_to_differ_can_be_declared(self, tmp_path):
        """In an arm comparison the scaffold differs by design; `ignore` is how that gets declared
        rather than assumed."""
        d = tmp_path / "b"
        d.mkdir()
        (d / "x").write_text("1")
        h1 = hash_inputs({"bundle": d, "scaffold": d})
        (d / "x").write_text("2")
        h2 = hash_inputs({"bundle": d, "scaffold": d})
        with pytest.raises(InputsDiffer):
            assert_comparable({"r1": h1, "r2": h2})
        assert_comparable({"r1": h1, "r2": h2}, ignore=["bundle", "scaffold"])

    def test_a_single_run_is_trivially_comparable(self):
        assert_comparable({"r1": {"bundle": "abc"}})


class TestManifestIntegration:
    def test_input_hashes_reach_the_manifest_and_survive_a_roundtrip(self, tmp_path):
        from aet.core.run_manifest import RunManifest
        from aet.core.run_spec import RunSpec

        d = tmp_path / "bundle"
        d.mkdir()
        (d / "x").write_text("1")
        spec = RunSpec(project="p", suite="s", method="a1", seed=0,
                       input_hashes=hash_inputs({"bundle": d}))
        m = RunManifest.create(spec, "run_1", "abc123")
        assert m.input_hashes["bundle"]

        f = tmp_path / "manifest.yaml"
        m.dump(f)
        assert RunManifest.load(f).input_hashes == m.input_hashes
