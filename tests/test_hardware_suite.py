"""Tests for HardwareBenchmarkSuite."""
import json
import logging
from aet.suites import get_suite
from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths

_logger = logging.getLogger(__name__)


def _make_run(tmp_path, method="m", seed=1):
    spec = RunSpec(project="p", suite="hardware_benchmark", method=method,
                   seed=seed, project_root=tmp_path)
    run_id = f"hw_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    suite = get_suite("hardware_benchmark")
    suite.init_run(spec, paths, _logger)
    return spec, paths, suite


def _inject_score(paths, testbench_pass=1, recall=0.8, precision=0.9,
                  tainted=0, wall_time=42.0):
    score = {
        "testbench_pass": testbench_pass,
        "localization_recall": recall,
        "localization_precision": precision,
        "tainted": tainted,
        "wall_time_seconds": wall_time,
        "tok_in": 10000,
        "tok_out": 500,
    }
    hw_dir = paths.run_path / "hw_benchmark"
    hw_dir.mkdir(parents=True, exist_ok=True)
    (hw_dir / "score.json").write_text(json.dumps(score))
    return score


def test_get_suite_hardware_benchmark():
    suite = get_suite("hardware_benchmark")
    assert suite is not None


def test_get_suite_hardware_benchmark_hyphen():
    suite = get_suite("hardware-benchmark")
    assert suite is not None


def test_init_run_creates_dirs(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    assert (paths.run_path / "hw_benchmark").exists()
    assert (paths.run_path / "metrics").exists()
    assert (paths.run_path / "logs").exists()


def test_init_run_creates_readme(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    readme = paths.run_path / "hw_benchmark" / "README.md"
    assert readme.exists()
    assert "hardware_benchmark" in readme.read_text().lower() or "Hardware" in readme.read_text()


def test_validate_pass_with_score_json(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    _inject_score(paths)
    # validate needs summary_metrics.json too for full pass
    (paths.run_path / "metrics" / "summary_metrics.json").write_text(
        json.dumps({"run_id": spec.run_id})
    )
    report = suite.validate(spec, paths, _logger)
    assert report["status"] == "pass"
    assert report["errors"] == []


def test_validate_fail_missing_score(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    report = suite.validate(spec, paths, _logger)
    assert report["status"] == "fail"
    assert any("score.json" in e for e in report["errors"])


def test_validate_partial_missing_metrics(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    _inject_score(paths)
    # no summary_metrics.json
    report = suite.validate(spec, paths, _logger)
    assert report["status"] == "partial"


def test_validate_writes_validation_report(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    _inject_score(paths)
    suite.validate(spec, paths, _logger)
    assert (paths.run_path / "validation_report.json").exists()


def test_collect_metrics_from_summary(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    summary = {"run_id": spec.run_id, "hw.testbench_pass": 1, "hw.localization_recall": 0.9}
    (paths.run_path / "metrics" / "summary_metrics.json").write_text(json.dumps(summary))
    result = suite.collect_metrics(spec, paths, _logger)
    assert result["hw.testbench_pass"] == 1


def test_collect_metrics_fallback_to_score(tmp_path):
    spec, paths, suite = _make_run(tmp_path)
    _inject_score(paths, recall=0.75)
    result = suite.collect_metrics(spec, paths, _logger)
    assert result.get("hw.localization_recall") == 0.75


def test_compare_writes_metrics_csv(tmp_path):
    suite = get_suite("hardware_benchmark")
    run_paths = []
    for seed in [1, 2, 3]:
        spec, paths, _ = _make_run(tmp_path, seed=seed)
        score = _inject_score(paths, recall=0.5 + seed * 0.1)
        (paths.run_path / "metrics" / "summary_metrics.json").write_text(
            json.dumps({"run_id": spec.run_id, "method": "m",
                        "hw.testbench_pass": 1, "hw.localization_recall": score["localization_recall"],
                        "hw.localization_precision": 0.9, "run.wall_time_s": 30.0})
        )
        run_paths.append(paths.run_path)

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    suite.compare(run_paths, report_dir, _logger)
    assert (report_dir / "metrics.csv").exists()
    assert (report_dir / "hw_summary.md").exists()


def test_compare_empty_run_list_does_not_crash(tmp_path):
    suite = get_suite("hardware_benchmark")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    suite.compare([], report_dir, _logger)
    assert (report_dir / "metrics.csv").exists()
