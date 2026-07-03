"""Test that TargetGenSuite.validate on an empty run returns partial/fail, never crashes."""
import pytest

from aet.suites import get_suite
from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths


def _make_targetgen_run(tmp_path, target="gemmini", method="v0", seed=1):
    spec = RunSpec(
        project="p",
        suite="targetgen",
        method=method,
        seed=seed,
        target=target,
        project_root=tmp_path,
    )
    run_id = f"2099-01-01_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    return spec, paths


class TestTargetGenValidateEmpty:
    def test_validate_empty_run(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        report = suite.validate(spec, paths, logger=None)
        assert report["overall"] in ("fail", "partial", "pass")
        assert "validators" in report

    def test_validate_returns_dict(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        report = suite.validate(spec, paths, logger=None)
        assert isinstance(report, dict)

    def test_validate_has_schema_version(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        report = suite.validate(spec, paths, logger=None)
        assert report.get("schema_version") == "1.0"

    def test_validate_writes_report_json(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        suite.validate(spec, paths, logger=None)
        assert (paths.run_path / "validation_report.json").exists()

    def test_validate_never_raises(self, tmp_path):
        """validate must not raise even with empty/uninitialized run dir."""
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        # Don't call init_run — run dir doesn't exist yet
        # validate should still not crash
        try:
            report = suite.validate(spec, paths, logger=None)
            assert isinstance(report, dict)
        except Exception as e:
            pytest.fail(f"validate raised unexpectedly: {e}")

    def test_validate_total_errors_is_int(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        report = suite.validate(spec, paths, logger=None)
        assert isinstance(report.get("total_errors"), int)

    def test_validate_all_expected_validators_present(self, tmp_path):
        spec, paths = _make_targetgen_run(tmp_path)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        report = suite.validate(spec, paths, logger=None)
        expected = {"schema", "evidence", "xdsl", "passes",
                    "dialect_design", "runtime_mock", "merlin_integration"}
        assert expected == set(report["validators"].keys())
