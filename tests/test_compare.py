"""Test DefaultSuite.compare writes metrics.csv."""
import csv
import json
import logging
import pytest
from pathlib import Path

from aet.suites import get_suite
from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths

_logger = logging.getLogger(__name__)


def _create_run(tmp_path, suite_name, method, seed):
    spec = RunSpec(
        project="p",
        suite=suite_name,
        method=method,
        seed=seed,
        project_root=tmp_path,
    )
    run_id = f"2099-01-01_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    return spec, paths


class TestDefaultCompare:
    def test_compare_writes_metrics_csv(self, tmp_path):
        suite = get_suite("default")
        run_paths = []
        for seed in [1, 2]:
            spec, paths = _create_run(tmp_path, "default", "m", seed)
            suite.init_run(spec, paths, _logger)
            suite.validate(spec, paths, _logger)
            run_paths.append(paths.run_path)

        report_dir = tmp_path / "reports" / "default"
        report_dir.mkdir(parents=True, exist_ok=True)
        suite.compare(run_paths, report_dir, _logger)

        assert (report_dir / "metrics.csv").exists()

    def test_compare_writes_summary_md(self, tmp_path):
        suite = get_suite("default")
        run_paths = []
        for seed in [1, 2]:
            spec, paths = _create_run(tmp_path, "default", "m", seed)
            suite.init_run(spec, paths, _logger)
            suite.validate(spec, paths, _logger)
            run_paths.append(paths.run_path)

        report_dir = tmp_path / "reports" / "default"
        report_dir.mkdir(parents=True, exist_ok=True)
        suite.compare(run_paths, report_dir, _logger)

        assert (report_dir / "summary.md").exists()

    def test_compare_metrics_csv_has_rows(self, tmp_path):
        suite = get_suite("default")
        run_paths = []
        for seed in [1, 2]:
            spec, paths = _create_run(tmp_path, "default", "m", seed)
            suite.init_run(spec, paths, _logger)
            suite.validate(spec, paths, _logger)
            run_paths.append(paths.run_path)

        report_dir = tmp_path / "reports" / "default"
        report_dir.mkdir(parents=True, exist_ok=True)
        suite.compare(run_paths, report_dir, _logger)

        with open(report_dir / "metrics.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2

    def test_compare_empty_run_list_does_not_crash(self, tmp_path):
        suite = get_suite("default")
        report_dir = tmp_path / "reports" / "default"
        report_dir.mkdir(parents=True, exist_ok=True)
        # Should not raise even with empty list
        suite.compare([], report_dir, _logger)
        assert (report_dir / "metrics.csv").exists()

    def test_compare_uses_runs_from_directory(self, tmp_path):
        """Simulate running compare via the directory enumeration pattern."""
        suite = get_suite("default")
        for seed in [1, 2, 3]:
            spec, paths = _create_run(tmp_path, "default", "m", seed)
            suite.init_run(spec, paths, _logger)
            suite.validate(spec, paths, _logger)

        runs_root = tmp_path / "runs" / "default"
        run_paths = [p for p in runs_root.iterdir() if p.is_dir()]
        report_dir = tmp_path / "reports" / "default"
        report_dir.mkdir(parents=True, exist_ok=True)
        suite.compare(run_paths, report_dir, _logger)

        with open(report_dir / "metrics.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
