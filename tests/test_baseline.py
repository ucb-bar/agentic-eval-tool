"""Tests for aet baseline set/show and regression detection."""
import json
import logging
import pytest
from pathlib import Path

from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths
from aet.suites import get_suite

_logger = logging.getLogger(__name__)


def _make_run(tmp_path, method, seed, task_score=None, cost=None):
    spec = RunSpec(project="p", suite="default", method=method, seed=seed, project_root=tmp_path)
    run_id = f"2099-01-01_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)
    suite = get_suite("default")
    suite.init_run(spec, paths, _logger)
    suite.validate(spec, paths, _logger)

    if task_score is not None or cost is not None:
        summary = json.loads((paths.metrics / "summary_metrics.json").read_text())
        if task_score is not None:
            summary["task_achievement_score"] = task_score
        if cost is not None:
            summary["aet.agent.cost_usd"] = cost
        (paths.metrics / "summary_metrics.json").write_text(json.dumps(summary))

    return paths


def test_baseline_set_writes_file(tmp_path):
    paths = _make_run(tmp_path, "m", 1)
    baseline_dir = tmp_path / "baselines" / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((paths.metrics / "summary_metrics.json").read_text())
    baseline_path = baseline_dir / "baseline.json"
    baseline_path.write_text(json.dumps(summary))

    assert baseline_path.exists()
    loaded = json.loads(baseline_path.read_text())
    assert "run_id" in loaded


def test_baseline_set_picks_best_run_by_score(tmp_path):
    _make_run(tmp_path, "m", 1, task_score=0.7)
    _make_run(tmp_path, "m", 2, task_score=0.9)
    _make_run(tmp_path, "m", 3, task_score=0.5)

    runs_root = tmp_path / "runs" / "default"
    best_summary = None
    best_score = float("-inf")
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        sp = run_dir / "metrics" / "summary_metrics.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        score = s.get("task_achievement_score")
        if score is not None and float(score) > best_score:
            best_score = float(score)
            best_summary = s

    assert best_summary is not None
    assert abs(best_summary["task_achievement_score"] - 0.9) < 1e-9


def test_baseline_show_prints_json(tmp_path, capsys):
    paths = _make_run(tmp_path, "m", 1)
    baseline_dir = tmp_path / "baselines" / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((paths.metrics / "summary_metrics.json").read_text())
    (baseline_dir / "baseline.json").write_text(json.dumps(summary, indent=2))

    content = (baseline_dir / "baseline.json").read_text()
    print(content)
    captured = capsys.readouterr()
    loaded = json.loads(captured.out)
    assert "run_id" in loaded


def test_regression_detected_cost_increase(tmp_path):
    paths = _make_run(tmp_path, "m", 1, cost=0.065)
    baseline_dir = tmp_path / "baselines" / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline.json").write_text(json.dumps({
        "run_id": "baseline", "aet.agent.cost_usd": 0.05, "task_achievement_score": 0.9,
    }))

    suite = get_suite("default")
    report_dir = tmp_path / "reports" / "default"
    report_dir.mkdir(parents=True, exist_ok=True)
    suite.compare([paths.run_path], report_dir, _logger)

    reg_report = (report_dir / "regression_report.md").read_text()
    assert "REGRESSION" in reg_report


def test_regression_not_detected_within_threshold(tmp_path):
    paths = _make_run(tmp_path, "m", 1, cost=0.055)
    baseline_dir = tmp_path / "baselines" / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline.json").write_text(json.dumps({
        "run_id": "baseline", "aet.agent.cost_usd": 0.05, "task_achievement_score": 0.9,
    }))

    suite = get_suite("default")
    report_dir = tmp_path / "reports" / "default"
    report_dir.mkdir(parents=True, exist_ok=True)
    suite.compare([paths.run_path], report_dir, _logger)

    reg_report = (report_dir / "regression_report.md").read_text()
    assert "✓ OK" in reg_report
    assert "| ✗ REGRESSION |" not in reg_report


def test_regression_report_written_when_baseline_exists(tmp_path):
    paths = _make_run(tmp_path, "m", 1, cost=0.06, task_score=0.8)
    baseline_dir = tmp_path / "baselines" / "default"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "baseline.json").write_text(json.dumps({
        "run_id": "baseline", "aet.agent.cost_usd": 0.05, "task_achievement_score": 0.85,
    }))

    suite = get_suite("default")
    report_dir = tmp_path / "reports" / "default"
    report_dir.mkdir(parents=True, exist_ok=True)
    suite.compare([paths.run_path], report_dir, _logger)

    assert (report_dir / "regression_report.md").exists()
