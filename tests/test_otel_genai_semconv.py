"""Tests for OTel GenAI semconv span structure.

Uses InMemorySpanExporter to verify that aet emits the correct span hierarchy
and attributes without requiring a running collector.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_otel_backend_with_exporter(tmp_path, suite="default", method="test", seed=1):
    """Return (OtelBackend, InMemorySpanExporter) wired together."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from aet.tracking.types import TrackingConfig
    from aet.tracking.otel_backend import OtelBackend

    config = TrackingConfig(
        mode="full",
        run_id="test-run-001",
        project="p",
        suite=suite,
        target="gemmini",
        method=method,
        seed=seed,
        run_path=tmp_path,
        enable_openllmetry=False,  # don't instrument anthropic in unit tests
    )

    class _FakeLocal:
        def warn(self, msg): pass

    backend = OtelBackend.__new__(OtelBackend)
    backend._enabled = True
    backend._tracer = provider.get_tracer("aet")
    backend._meter = None
    backend._op_duration = None
    backend._token_usage = None
    backend._validation_duration = None
    backend._local = _FakeLocal()
    backend._config = config
    backend._trace_id = None
    backend._provider = provider

    return backend, exporter


# ---------------------------------------------------------------------------
# semconv constants
# ---------------------------------------------------------------------------

def test_semconv_constants_defined():
    from aet.tracking import semconv
    assert semconv.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert semconv.GEN_AI_WORKFLOW_NAME == "gen_ai.workflow.name"
    assert semconv.GEN_AI_TOOL_NAME == "gen_ai.tool.name"
    assert semconv.GEN_AI_EVAL_RESULT_EVENT == "gen_ai.evaluation.result"
    assert semconv.AET_SUITE == "aet.suite"
    assert semconv.AET_VALIDATOR_NAME == "aet.validator.name"
    assert semconv.OP_INVOKE_WORKFLOW == "invoke_workflow"
    assert semconv.OP_EXECUTE_TOOL == "execute_tool"


# ---------------------------------------------------------------------------
# workflow span
# ---------------------------------------------------------------------------

def test_workflow_span_has_genai_attributes(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_workflow_span(
        "default-eval", suite="default", method="m", seed=1, target="gemmini", run_id="r1"
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "invoke_workflow default-eval"
    assert s.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert s.attributes["gen_ai.workflow.name"] == "default-eval"
    assert s.attributes["aet.suite"] == "default"
    assert s.attributes["aet.method"] == "m"
    assert s.attributes["aet.seed"] == 1
    assert s.attributes["aet.target"] == "gemmini"
    assert s.attributes["aet.run_id"] == "r1"


# ---------------------------------------------------------------------------
# tool span
# ---------------------------------------------------------------------------

def test_tool_span_has_execute_tool_attributes(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_tool_span("validate_schema", validator_name="schema"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "execute_tool validate_schema"
    assert s.attributes["gen_ai.operation.name"] == "execute_tool"
    assert s.attributes["gen_ai.tool.name"] == "validate_schema"
    assert s.attributes["aet.validator.name"] == "schema"


def test_multiple_tool_spans(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    validators = ["schema", "evidence", "xdsl", "passes"]
    for v in validators:
        with backend.start_tool_span(f"validate_{v}", validator_name=v):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == len(validators)
    names = {s.name for s in spans}
    assert "execute_tool validate_schema" in names
    assert "execute_tool validate_passes" in names


# ---------------------------------------------------------------------------
# agent span
# ---------------------------------------------------------------------------

def test_agent_span_has_invoke_agent_attributes(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_agent_span("claude-code", model="claude-sonnet-4-6", provider="anthropic"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "invoke_agent claude-code"
    assert s.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert s.attributes["gen_ai.agent.name"] == "claude-code"
    assert s.attributes["gen_ai.provider.name"] == "anthropic"
    assert s.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# evaluation event
# ---------------------------------------------------------------------------

def test_evaluation_result_event_on_workflow_span(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_workflow_span("targetgen-eval", suite="targetgen"):
        backend.log_evaluation_event("validation_overall", 1.0, "pass")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    events = s.events
    assert len(events) == 1
    ev = events[0]
    assert ev.name == "gen_ai.evaluation.result"
    assert ev.attributes["gen_ai.evaluation.name"] == "validation_overall"
    assert ev.attributes["gen_ai.evaluation.score.value"] == 1.0
    assert ev.attributes["gen_ai.evaluation.score.label"] == "pass"


def test_evaluation_result_fail_score(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_workflow_span("targetgen-eval", suite="targetgen"):
        backend.log_evaluation_event("validation_overall", 0.0, "fail")

    spans = exporter.get_finished_spans()
    s = spans[0]
    ev = s.events[0]
    assert ev.attributes["gen_ai.evaluation.score.value"] == 0.0
    assert ev.attributes["gen_ai.evaluation.score.label"] == "fail"


# ---------------------------------------------------------------------------
# span nesting
# ---------------------------------------------------------------------------

def test_tool_spans_nested_inside_workflow(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    backend, exporter = _make_otel_backend_with_exporter(tmp_path)

    with backend.start_workflow_span("default-eval", suite="default"):
        with backend.start_tool_span("validate_schema", validator_name="schema"):
            pass
        with backend.start_tool_span("validate_passes", validator_name="passes"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    workflow = next(s for s in spans if "invoke_workflow" in s.name)
    tools = [s for s in spans if "execute_tool" in s.name]
    assert len(tools) == 2

    for t in tools:
        assert t.parent is not None
        assert t.parent.span_id == workflow.context.span_id


# ---------------------------------------------------------------------------
# run_logger delegation
# ---------------------------------------------------------------------------

def test_run_logger_start_run_span_delegates_to_otel(tmp_path):
    """Verify run_logger delegates to otel backend via its span factories."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from aet.tracking.types import TrackingConfig
    from aet.tracking.otel_backend import OtelBackend
    from aet.tracking.run_logger import EvalRunLogger

    run_path = tmp_path / "runs" / "default" / "r1"
    run_path.mkdir(parents=True, exist_ok=True)

    config = TrackingConfig(
        mode="full", run_id="r1", project="p", suite="default",
        target="gemmini", method="m", seed=1, run_path=run_path,
        enable_openllmetry=False,
    )

    class _FakeLocal:
        def warn(self, msg): pass
        def log_event(self, *a, **kw): pass

    fake_local = _FakeLocal()
    backend = OtelBackend.__new__(OtelBackend)
    backend._enabled = True
    backend._tracer = provider.get_tracer("aet")
    backend._meter = None
    backend._op_duration = None
    backend._token_usage = None
    backend._validation_duration = None
    backend._local = fake_local
    backend._config = config
    backend._trace_id = None
    backend._provider = provider

    logger = EvalRunLogger.__new__(EvalRunLogger)
    from aet.tracking.local_backend import LocalBackend
    logger._config = config
    logger._local = LocalBackend(config)
    logger._mlflow = None
    logger._otel = backend

    with logger.start_run_span("default-eval"):
        with logger.start_tool_span("validate_schema", validator_name="schema"):
            pass

    spans = exporter.get_finished_spans()
    workflow_spans = [s for s in spans if "invoke_workflow" in s.name]
    assert len(workflow_spans) == 1
    assert workflow_spans[0].attributes["gen_ai.operation.name"] == "invoke_workflow"
    tool_spans = [s for s in spans if "execute_tool" in s.name]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["aet.validator.name"] == "schema"


# ---------------------------------------------------------------------------
# nullcontext fallback (no otel backend)
# ---------------------------------------------------------------------------

def test_run_logger_graceful_without_otel(tmp_path):
    from aet.tracking.run_logger import EvalRunLogger

    (tmp_path / "runs" / "default" / "r1").mkdir(parents=True, exist_ok=True)
    logger = EvalRunLogger.start(
        project="p", suite="default", target="",
        method="m", seed=1, run_id="r1",
        run_path=tmp_path / "runs" / "default" / "r1",
        tracking_mode="local",  # no OTel
    )

    # All these must not raise
    with logger.start_run_span("default-eval"):
        with logger.start_tool_span("validate_schema"):
            pass
        with logger.start_agent_span("claude-code"):
            pass
        logger.log_evaluation_result("validation_overall", 1.0, "pass")
        logger.record_llm_call(0.5, 100, 50)
    tp = logger.get_traceparent_for_subprocess()
    assert tp is None
