"""Tests for aet.suites.targetgen.architecture_rules."""
import pytest

from aet.suites.targetgen.architecture_rules import (
    check_all,
    check_r1,
    check_r2,
    check_r3,
    _check_dialect_plan_rules,
)


def _make_manifest(target="gemmini", **kwargs):
    base = {
        "target": target,
        "method": "v0",
        "seed": 1,
        "schema_version": "1.0",
        "git_hash_at_init": "unknown",
    }
    base.update(kwargs)
    return base


class TestCheckR1:
    def test_r1_pass_when_dir_exists(self, tmp_path):
        (tmp_path / "generated" / "gemmini-mlir").mkdir(parents=True)
        manifest = _make_manifest(target="gemmini")
        result = check_r1(tmp_path, manifest)
        assert result["passed"] is True
        assert result["rule_id"] == "R1"

    def test_r1_fail_when_dir_missing(self, tmp_path):
        manifest = _make_manifest(target="gemmini")
        result = check_r1(tmp_path, manifest)
        assert result["passed"] is False
        assert result["severity"] == "error"


class TestCheckR2:
    def test_r2_fail_when_xdsl_absent(self, tmp_path):
        (tmp_path / "generated" / "gemmini-mlir").mkdir(parents=True)
        manifest = _make_manifest(target="gemmini")
        result = check_r2(tmp_path, manifest)
        assert result["passed"] is False

    def test_r2_pass_when_xdsl_nonempty(self, tmp_path):
        xdsl_dir = tmp_path / "generated" / "gemmini-mlir" / "xdsl"
        xdsl_dir.mkdir(parents=True)
        (xdsl_dir / "ops.py").write_text("# ops")
        manifest = _make_manifest(target="gemmini")
        result = check_r2(tmp_path, manifest)
        assert result["passed"] is True


class TestCheckR3:
    def test_r3_pass_when_no_tablegen_files(self, tmp_path):
        (tmp_path / "generated" / "gemmini-mlir").mkdir(parents=True)
        manifest = _make_manifest(target="gemmini", promotion_flag=False)
        result = check_r3(tmp_path, manifest)
        assert result["passed"] is True

    def test_r3_fail_when_td_file_present(self, tmp_path):
        gen_dir = tmp_path / "generated" / "gemmini-mlir"
        gen_dir.mkdir(parents=True)
        (gen_dir / "ops.td").write_text("// tablegen")
        manifest = _make_manifest(target="gemmini", promotion_flag=False)
        result = check_r3(tmp_path, manifest)
        assert result["passed"] is False
        assert result["severity"] == "error"

    def test_r3_pass_when_promotion_flag_set(self, tmp_path):
        gen_dir = tmp_path / "generated" / "gemmini-mlir"
        gen_dir.mkdir(parents=True)
        (gen_dir / "ops.td").write_text("// tablegen")
        manifest = _make_manifest(target="gemmini", promotion_flag=True)
        result = check_r3(tmp_path, manifest)
        assert result["passed"] is True


class TestCheckAll:
    def test_returns_list(self, tmp_path):
        manifest = _make_manifest()
        repo_root = tmp_path
        results = check_all(tmp_path, manifest, repo_root)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_no_crash_on_empty_manifest(self, tmp_path):
        """check_all with minimal manifest must not crash."""
        manifest = {"target": "gemmini"}
        try:
            results = check_all(tmp_path, manifest, tmp_path)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"check_all crashed: {e}")

    def test_all_results_have_required_keys(self, tmp_path):
        manifest = _make_manifest()
        results = check_all(tmp_path, manifest, tmp_path)
        for r in results:
            assert "rule_id" in r
            assert "passed" in r
            assert "severity" in r
            assert "message" in r

    def test_r1_included_in_results(self, tmp_path):
        manifest = _make_manifest()
        results = check_all(tmp_path, manifest, tmp_path)
        rule_ids = [r["rule_id"] for r in results]
        assert "R1" in rule_ids

    def test_dialect_plan_rules_skipped_when_no_plan(self, tmp_path):
        """Without dialect_plan.yaml, R5-R10 are all info/pass."""
        manifest = _make_manifest()
        results = _check_dialect_plan_rules(tmp_path, manifest)
        assert all(r["passed"] for r in results)
