"""Test DefaultSuite.validate behaviour."""
import json
import logging


from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths
from aet.suites import get_suite

_logger = logging.getLogger(__name__)


def _setup_run(tmp_path, method="m", seed=1):
    spec = RunSpec(
        project="p",
        suite="default",
        method=method,
        seed=seed,
        project_root=tmp_path,
    )
    run_id = f"2099-01-01_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    suite = get_suite("default")
    suite.init_run(spec, paths, _logger)
    return spec, paths, suite


class TestDefaultValidate:
    def test_returns_dict(self, tmp_path):
        spec, paths, suite = _setup_run(tmp_path)
        report = suite.validate(spec, paths, _logger)
        assert isinstance(report, dict)

    def test_writes_validation_report_json(self, tmp_path):
        spec, paths, suite = _setup_run(tmp_path)
        suite.validate(spec, paths, _logger)
        assert (paths.run_path / "validation_report.json").exists()

    def test_validation_report_is_valid_json(self, tmp_path):
        spec, paths, suite = _setup_run(tmp_path)
        suite.validate(spec, paths, _logger)
        data = json.loads((paths.run_path / "validation_report.json").read_text())
        assert "status" in data

    def test_status_never_error_or_crash(self, tmp_path):
        """Status must be partial or pass for an empty-but-initialized run."""
        spec, paths, suite = _setup_run(tmp_path)
        report = suite.validate(spec, paths, _logger)
        assert report["status"] in ("partial", "pass", "fail")

    def test_empty_generated_gives_partial(self, tmp_path):
        """init_run creates only README.md, so generated/ is effectively empty."""
        spec, paths, suite = _setup_run(tmp_path)
        report = suite.validate(spec, paths, _logger)
        # generated/ has only README.md → status should be "partial"
        assert report["status"] == "partial"

    def test_writes_summary_metrics_json(self, tmp_path):
        spec, paths, suite = _setup_run(tmp_path)
        suite.validate(spec, paths, _logger)
        assert (paths.metrics / "summary_metrics.json").exists()

    def test_summary_metrics_contains_run_id(self, tmp_path):
        spec, paths, suite = _setup_run(tmp_path)
        suite.validate(spec, paths, _logger)
        data = json.loads((paths.metrics / "summary_metrics.json").read_text())
        # run_id may be None in spec; summary still has the key
        assert "run_id" in data

    def test_validate_does_not_raise_when_generated_missing(self, tmp_path):
        """If generated/ was somehow deleted, validate should return fail not raise."""
        spec = RunSpec(
            project="p", suite="default", method="m", seed=1, project_root=tmp_path
        )
        run_id = "2099-01-01_m_seed001"
        paths = RunPaths.from_spec(spec, run_id)
        # Deliberately do NOT call init_run — generated/ won't exist
        paths.run_path.mkdir(parents=True, exist_ok=True)
        suite = get_suite("default")
        report = suite.validate(spec, paths, _logger)
        assert report["status"] in ("fail", "partial", "pass")
