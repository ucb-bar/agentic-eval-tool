"""Tests for examples/claude_code_eval.py dry-run mode."""
import json
import sys
from pathlib import Path

import pytest

# Make sure the examples directory is importable
_EXAMPLES = Path(__file__).parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from claude_code_eval import run_claude_code_eval


def test_dry_run_returns_pass_or_partial(tmp_path):
    report = run_claude_code_eval(
        project_root=tmp_path,
        tracking_mode="local",
        dry_run=True,
    )
    assert report["status"] in ("pass", "partial"), f"unexpected status: {report['status']}"


def test_dry_run_creates_claude_output_md(tmp_path):
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True)
    run_id = next(
        (p for p in (tmp_path / "runs" / "default").iterdir() if p.is_dir()),
        None,
    )
    assert run_id is not None
    output_file = run_id / "generated" / "claude_output.md"
    assert output_file.exists(), f"claude_output.md not found under {run_id}"
    content = output_file.read_text()
    assert "Dry-run" in content, "expected dry-run marker in output file"


def test_dry_run_writes_validation_report(tmp_path):
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True)
    run_dirs = list((tmp_path / "runs" / "default").iterdir())
    assert run_dirs
    report_path = run_dirs[0] / "validation_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "status" in report


def test_dry_run_writes_local_logs(tmp_path):
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True)
    run_dirs = list((tmp_path / "runs" / "default").iterdir())
    assert run_dirs
    logs_dir = run_dirs[0] / "logs"
    assert logs_dir.exists()
    log_files = list(logs_dir.iterdir())
    assert log_files, "no log files written"


def test_dry_run_logs_task_length_param(tmp_path):
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True)
    run_dirs = list((tmp_path / "runs" / "default").iterdir())
    params_file = run_dirs[0] / "logs" / "params.json"
    assert params_file.exists()
    params = json.loads(params_file.read_text())
    assert "task_length_chars" in params
    assert "dry_run" in params


def test_dry_run_different_seeds_different_run_ids(tmp_path):
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True, seed=1)
    run_claude_code_eval(project_root=tmp_path, tracking_mode="local", dry_run=True, seed=2)
    run_dirs = sorted((tmp_path / "runs" / "default").iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name


def test_dry_run_with_custom_task(tmp_path):
    report = run_claude_code_eval(
        project_root=tmp_path,
        tracking_mode="local",
        dry_run=True,
        task="What is 2+2?",
    )
    assert report["status"] in ("pass", "partial")


def test_dry_run_no_mlflow_no_crash(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    report = run_claude_code_eval(
        project_root=tmp_path,
        tracking_mode="local",
        dry_run=True,
    )
    assert report["status"] in ("pass", "partial")
