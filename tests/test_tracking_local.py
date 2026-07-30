"""Test EvalRunLogger in local mode."""
import json

from aet.tracking import EvalRunLogger
from aet.tracking.claude_stream import ModelUsage


def test_local_events(tmp_path):
    logger = EvalRunLogger.start(
        run_id="test_run",
        run_path=tmp_path,
        tracking_mode="local",
        target="test",
        method="smoke",
        seed=0,
        project="test_project",
        suite="default",
    )
    logger.log_event("test.event", {"key": "value"})
    logger.log_param("my_param", "my_value")
    logger.log_metric("my_metric", 1.23)
    logger.finish(status="pass")

    assert (tmp_path / "logs" / "events.jsonl").exists()
    assert (tmp_path / "logs" / "params.json").exists()
    assert (tmp_path / "logs" / "metrics.jsonl").exists()


def test_events_jsonl_contains_event(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_event("my.event", {"x": 42})
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    names = [e["event"] for e in events]
    assert "my.event" in names


def test_params_json_contains_param(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_param("alpha", "beta")
    logger.finish(status="pass")

    params = json.loads((tmp_path / "logs" / "params.json").read_text())
    assert params["alpha"] == "beta"


def test_metrics_jsonl_contains_metric(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_metric("accuracy", 0.99)
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    records = [json.loads(l) for l in lines]
    names = [r["name"] for r in records]
    assert "accuracy" in names


def test_finish_writes_run_finished_event(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    names = [e["event"] for e in events]
    assert "run.finished" in names


def test_log_params_batch(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_params({"k1": "v1", "k2": 2})
    logger.finish(status="pass")

    params = json.loads((tmp_path / "logs" / "params.json").read_text())
    assert params["k1"] == "v1"
    assert params["k2"] == 2


def test_log_metrics_batch(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    logger.log_metrics({"loss": 0.5, "acc": 0.9}, prefix="train")
    logger.finish(status="pass")

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    names = {json.loads(l)["name"] for l in lines}
    assert "train.loss" in names
    assert "train.acc" in names


def test_logs_dir_property(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    assert logger.logs_dir == tmp_path / "logs"
    logger.finish(status="pass")


def test_mode_property(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    assert logger.mode == "local"
    logger.finish(status="pass")


def test_log_model_usage_writes_per_model_metrics_and_event(tmp_path):
    logger = EvalRunLogger.start(
        run_id="r1",
        run_path=tmp_path,
        tracking_mode="local",
        target="t",
        method="m",
        seed=1,
        project="p",
        suite="s",
    )
    usage = [
        ModelUsage(model="claude-opus-4-8", input_tokens=100, output_tokens=30,
                   cache_read_input_tokens=50, cache_creation_input_tokens=200, cost_usd=0.42),
        ModelUsage(model="claude-haiku-4-5", input_tokens=10, output_tokens=5,
                   cache_read_input_tokens=0, cache_creation_input_tokens=0, cost_usd=0.01,
                   web_search_requests=2),
    ]
    logger.log_model_usage(usage)
    logger.log_model_usage(None)   # null-safe no-op
    logger.finish(status="pass")

    # per_model.* metrics land in metrics.jsonl under the sanitized model name
    m_lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    metrics = {json.loads(l)["name"]: json.loads(l)["value"] for l in m_lines}
    assert metrics["per_model.claude_opus_4_8.cost_usd"] == 0.42
    assert metrics["per_model.claude_opus_4_8.input_tokens_raw"] == 100
    assert metrics["per_model.claude_opus_4_8.output_tokens"] == 30
    assert metrics["per_model.claude_opus_4_8.cache_read_tokens"] == 50
    assert metrics["per_model.claude_opus_4_8.cache_creation_tokens"] == 200
    assert metrics["per_model.claude_opus_4_8.total_billed_tokens"] == 380  # 100+30+50+200
    assert metrics["per_model.claude_haiku_4_5.cost_usd"] == 0.01

    # one structured gen_ai.model.usage event per ModelUsage
    e_lines = (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    usage_events = [json.loads(l) for l in e_lines if json.loads(l)["event"] == "gen_ai.model.usage"]
    assert len(usage_events) == 2
    by_model = {e["payload"]["model"]: e["payload"] for e in usage_events}
    assert by_model["claude-opus-4-8"]["cache_creation_tokens"] == 200
    assert by_model["claude-haiku-4-5"]["web_search_requests"] == 2
