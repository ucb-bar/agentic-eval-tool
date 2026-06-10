"""Tests for MLflow nested runs and log_metric_step."""
import json
from pathlib import Path

import pytest

from aet.tracking.types import TrackingConfig
from aet.tracking.run_logger import EvalRunLogger


# ── TrackingConfig carries parent_run_id ──────────────────────────────────────

def test_tracking_config_parent_run_id_default():
    cfg = TrackingConfig(target="t", method="m", seed=0)
    assert cfg.parent_run_id is None


def test_tracking_config_parent_run_id_set():
    cfg = TrackingConfig(target="t", method="m", seed=0, parent_run_id="abc123")
    assert cfg.parent_run_id == "abc123"


# ── EvalRunLogger.start() accepts parent_run_id ───────────────────────────────

def test_run_logger_start_accepts_parent_run_id(tmp_path):
    logger = EvalRunLogger.start(
        project="p", suite="s", target="t", method="m", seed=0,
        run_id="r1", run_path=tmp_path / "r1",
        tracking_mode="local",
        parent_run_id="sweep_parent_42",
    )
    assert logger._config.parent_run_id == "sweep_parent_42"


# ── log_metric_step writes step field to local events.jsonl ───────────────────

def test_log_metric_step_writes_to_local(tmp_path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    logger = EvalRunLogger.start(
        project="p", suite="s", target="t", method="m", seed=0,
        run_id="r1", run_path=run_path,
        tracking_mode="local",
    )
    logger.log_metric_step("validator.schema.passed", 1.0, step=0)
    logger.log_metric_step("validator.evidence.passed", 0.0, step=1)
    logger.log_metric_step("cumulative_errors", 1.0, step=1)

    metrics_file = run_path / "logs" / "metrics.jsonl"
    assert metrics_file.exists()
    lines = [json.loads(l) for l in metrics_file.read_text().splitlines() if l.strip()]
    names = [l["name"] for l in lines]
    assert "validator.schema.passed" in names
    assert "validator.evidence.passed" in names
    assert "cumulative_errors" in names

    # step field is present in each entry
    step_entries = [l for l in lines if "step" in l]
    assert len(step_entries) >= 3

    # check step values
    schema_entry = next(l for l in lines if l["name"] == "validator.schema.passed")
    assert schema_entry["step"] == 0
    evidence_entry = next(l for l in lines if l["name"] == "validator.evidence.passed")
    assert evidence_entry["step"] == 1


# ── log_metric_step is safe when MLflow unavailable ───────────────────────────

def test_log_metric_step_no_crash_without_mlflow(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)
    run_path = tmp_path / "run"
    run_path.mkdir()
    logger = EvalRunLogger.start(
        project="p", suite="s", target="t", method="m", seed=0,
        run_id="r1", run_path=run_path,
        tracking_mode="local",
        parent_run_id="parent_xyz",
    )
    logger.log_metric_step("some.metric", 0.5, step=3)


# ── MLflowBackend handles parent_run_id gracefully when mlflow absent ─────────

def test_mlflow_backend_parent_run_id_no_mlflow(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)
    from aet.tracking.local_backend import LocalBackend
    from aet.tracking.mlflow_backend import MLflowBackend

    cfg = TrackingConfig(
        mode="mlflow", target="t", method="m", seed=0,
        run_path=tmp_path, run_id="r1",
        parent_run_id="p999",
    )
    local = LocalBackend(cfg)
    backend = MLflowBackend(cfg, local)
    assert not backend._enabled
    backend.log_step_metric("x", 1.0, step=0)


# ── mlflow_run_url returns None in local mode ─────────────────────────────────

def test_mlflow_run_url_none_in_local_mode(tmp_path):
    logger = EvalRunLogger.start(
        project="p", suite="s", target="t", method="m", seed=0,
        run_id="r1", run_path=tmp_path / "r1",
        tracking_mode="local",
    )
    assert logger.mlflow_run_url is None
