"""Tests for per-validator step metrics emitted by TargetGenSuite.validate()."""
import json
from pathlib import Path

import pytest

from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths
from aet.suites.targetgen.suite import TargetGenSuite
from aet.tracking.run_logger import EvalRunLogger


def _make_minimal_run(tmp_path: Path) -> tuple[RunSpec, RunPaths, EvalRunLogger]:
    project_root = tmp_path / "project"
    run_id = "2099-01-01_test_seed001"
    spec = RunSpec(
        project="test",
        suite="targetgen",
        method="test",
        seed=1,
        run_id=run_id,
        project_root=project_root,
        target="testchip",
    )
    paths = RunPaths.from_spec(spec, run_id)
    paths.run_path.mkdir(parents=True, exist_ok=True)
    paths.generated.mkdir(parents=True, exist_ok=True)
    paths.contracts.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.metrics.mkdir(parents=True, exist_ok=True)

    # Minimal manifest so validators don't crash on missing file
    (paths.run_path / "run_manifest.yaml").write_text(
        "schema_version: '1.0'\n"
        "run_id: 2099-01-01_test_seed001\n"
        "suite: targetgen\n"
        "method: test\n"
        "seed: 1\n"
        "target: testchip\n"
        "git_hash: deadbeef\n"
        "created_at: '2099-01-01T00:00:00+00:00'\n"
    )

    logger = EvalRunLogger.start(
        project=spec.project, suite=spec.suite,
        target=spec.target or "", method=spec.method, seed=spec.seed,
        run_id=run_id, run_path=paths.run_path,
        tracking_mode="local",
    )
    return spec, paths, logger


def _read_metrics(run_path: Path) -> list[dict]:
    metrics_file = run_path / "logs" / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    return [json.loads(l) for l in metrics_file.read_text().splitlines() if l.strip()]


# ── step metrics are emitted per validator ─────────────────────────────────────

def test_validate_emits_step_metrics_per_validator(tmp_path):
    spec, paths, logger = _make_minimal_run(tmp_path)
    suite = TargetGenSuite()
    suite.validate(spec, paths, logger)

    metrics = _read_metrics(paths.run_path)
    step_metrics = [m for m in metrics if "step" in m]
    assert len(step_metrics) >= 7, f"expected ≥7 step metrics, got {len(step_metrics)}"


def test_validate_emits_passed_metric_for_each_validator(tmp_path):
    spec, paths, logger = _make_minimal_run(tmp_path)
    suite = TargetGenSuite()
    suite.validate(spec, paths, logger)

    metrics = _read_metrics(paths.run_path)
    passed_names = {m["name"] for m in metrics if "validator." in m.get("name", "") and ".passed" in m.get("name", "")}
    expected_validators = {"schema", "evidence", "xdsl", "passes", "dialect_design", "runtime_mock", "merlin_integration"}
    for v in expected_validators:
        assert f"validator.{v}.passed" in passed_names, f"missing step metric for validator {v}"


def test_validate_emits_cumulative_errors_series(tmp_path):
    spec, paths, logger = _make_minimal_run(tmp_path)
    suite = TargetGenSuite()
    suite.validate(spec, paths, logger)

    metrics = _read_metrics(paths.run_path)
    cumulative = [m for m in metrics if m.get("name") == "cumulative_errors"]
    assert len(cumulative) >= 7, f"expected ≥7 cumulative_errors entries, got {len(cumulative)}"


def test_cumulative_errors_monotonically_nondecreasing(tmp_path):
    spec, paths, logger = _make_minimal_run(tmp_path)
    suite = TargetGenSuite()
    suite.validate(spec, paths, logger)

    metrics = _read_metrics(paths.run_path)
    cumulative = sorted(
        [m for m in metrics if m.get("name") == "cumulative_errors"],
        key=lambda m: m.get("step", 0),
    )
    values = [m["value"] for m in cumulative]
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1], (
            f"cumulative_errors decreased at step {i}: {values[i - 1]} → {values[i]}"
        )


def test_validate_step_indices_match_validator_order(tmp_path):
    spec, paths, logger = _make_minimal_run(tmp_path)
    suite = TargetGenSuite()
    suite.validate(spec, paths, logger)

    metrics = _read_metrics(paths.run_path)
    validator_order = ["schema", "evidence", "xdsl", "passes", "dialect_design", "runtime_mock", "merlin_integration"]
    for expected_step, name in enumerate(validator_order):
        metric_name = f"validator.{name}.passed"
        entry = next((m for m in metrics if m.get("name") == metric_name), None)
        assert entry is not None, f"no entry for {metric_name}"
        assert entry["step"] == expected_step, (
            f"{metric_name}: expected step={expected_step}, got {entry['step']}"
        )
