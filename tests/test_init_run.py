"""Test that init_run creates expected directory structures."""
import logging


from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths
from aet.suites import get_suite

_logger = logging.getLogger(__name__)


def _make_spec_and_paths(tmp_path, suite_name, method="m", seed=1, target=None):
    spec = RunSpec(
        project="p",
        suite=suite_name,
        method=method,
        seed=seed,
        target=target,
        project_root=tmp_path,
    )
    run_id = f"2099-01-01_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    return spec, paths


class TestDefaultInitRun:
    def test_creates_generated_dir(self, tmp_path):
        spec, paths = _make_spec_and_paths(tmp_path, "default")
        suite = get_suite("default")
        suite.init_run(spec, paths, _logger)
        assert paths.generated.exists()

    def test_creates_readme_in_generated(self, tmp_path):
        spec, paths = _make_spec_and_paths(tmp_path, "default")
        suite = get_suite("default")
        suite.init_run(spec, paths, _logger)
        assert (paths.generated / "README.md").exists()

    def test_logs_dir_path_property(self, tmp_path):
        spec, paths = _make_spec_and_paths(tmp_path, "default")
        # RunPaths.logs is a property — just verify it resolves under run_path
        assert paths.logs == paths.run_path / "logs"

    def test_metrics_dir_path_property(self, tmp_path):
        spec, paths = _make_spec_and_paths(tmp_path, "default")
        assert paths.metrics == paths.run_path / "metrics"


class TestTargetgenInitRun:
    def test_creates_target_mlir_dir(self, tmp_path):
        spec, paths = _make_spec_and_paths(
            tmp_path, "targetgen", target="gemmini"
        )
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert (paths.generated / "gemmini-mlir").exists()

    def test_creates_contracts_dir(self, tmp_path):
        spec, paths = _make_spec_and_paths(
            tmp_path, "targetgen", target="gemmini"
        )
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert paths.contracts.exists()

    def test_creates_logs_dir(self, tmp_path):
        spec, paths = _make_spec_and_paths(
            tmp_path, "targetgen", target="gemmini"
        )
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert paths.logs.exists()

    def test_creates_metrics_dir(self, tmp_path):
        spec, paths = _make_spec_and_paths(
            tmp_path, "targetgen", target="gemmini"
        )
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert paths.metrics.exists()

    def test_readme_in_generated_mlir(self, tmp_path):
        spec, paths = _make_spec_and_paths(
            tmp_path, "targetgen", target="gemmini"
        )
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert (paths.generated / "gemmini-mlir" / "README.md").exists()

    def test_unknown_target_defaults_to_unknown(self, tmp_path):
        spec, paths = _make_spec_and_paths(tmp_path, "targetgen", target=None)
        suite = get_suite("targetgen")
        suite.init_run(spec, paths, logger=None)
        assert (paths.generated / "unknown-mlir").exists()
