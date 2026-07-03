"""Test that mlflow mode degrades gracefully when mlflow is unreachable."""

from aet.tracking import EvalRunLogger


def test_mlflow_graceful_no_install(tmp_path):
    """Should not crash even if mlflow is not installed or server is unreachable."""
    logger = EvalRunLogger.start(
        run_id="test_run",
        run_path=tmp_path,
        tracking_mode="mlflow",
        mlflow_tracking_uri="http://localhost:19999",  # definitely not running
        target="test",
        method="smoke",
        seed=0,
        project="p",
        suite="s",
    )
    logger.log_event("test.event", {})
    logger.finish(status="pass")

    # Local backend still wrote events regardless of mlflow failure
    assert (tmp_path / "logs" / "events.jsonl").exists()


def test_mlflow_graceful_param_and_metric(tmp_path):
    """log_param and log_metric must not crash in mlflow mode with dead server."""
    logger = EvalRunLogger.start(
        run_id="test_run",
        run_path=tmp_path,
        tracking_mode="mlflow",
        mlflow_tracking_uri="http://localhost:19999",
        target="t",
        method="m",
        seed=0,
        project="p",
        suite="s",
    )
    logger.log_param("key", "val")
    logger.log_metric("loss", 0.5)
    logger.finish(status="fail")

    # Local files must exist
    assert (tmp_path / "logs" / "params.json").exists()
    assert (tmp_path / "logs" / "metrics.jsonl").exists()


def test_mlflow_mode_falls_back_to_local_events(tmp_path):
    """Even in mlflow mode, local events.jsonl is always written."""
    logger = EvalRunLogger.start(
        run_id="gr_run",
        run_path=tmp_path,
        tracking_mode="mlflow",
        mlflow_tracking_uri="http://localhost:19999",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_event("fallback.check", {"ok": True})
    logger.finish(status="pass")

    import json
    lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    names = [e["event"] for e in events]
    assert "fallback.check" in names
