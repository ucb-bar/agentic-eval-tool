"""OpenTelemetry backend — optional. Returns nullcontext spans when not available."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any

from aet.tracking.types import TrackingConfig


class OtelBackend:
    def __init__(self, config: TrackingConfig, local) -> None:
        self._enabled = False
        self._tracer = None
        self._meter = None
        self._op_duration = None
        self._token_usage = None
        self._validation_duration = None
        self._local = local
        self._config = config
        self._trace_id: str | None = None
        self._provider = None

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:
            local.warn(
                "opentelemetry-sdk not installed; OTel tracing disabled. "
                "Install with: uv pip install 'aet[tracking]'"
            )
            return

        self._setup(config, local, trace)

    def _setup(self, config: TrackingConfig, local, trace_module) -> None:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

            resource = Resource.create({
                "service.name": config.service_name,
                "aet.target": config.target,
                "aet.method": config.method,
                "aet.seed": str(config.seed),
                "aet.run_id": config.run_id,
                "aet.suite": config.suite,
            })
            provider = TracerProvider(resource=resource)
            self._provider = provider

            if config.otel_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    exporter = OTLPSpanExporter(endpoint=config.otel_endpoint.rstrip("/") + "/v1/traces")
                except Exception as e:
                    local.warn(f"OTel OTLP exporter setup failed ({e}); using console exporter")
                    exporter = ConsoleSpanExporter()
            else:
                exporter = ConsoleSpanExporter()

            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace_module.set_tracer_provider(provider)
            self._tracer = trace_module.get_tracer("aet")
            self._enabled = True

            self._setup_metrics(config, local)
            self._setup_openllmetry(config, local, provider)

        except Exception as e:
            local.warn(f"OTel setup failed ({e}); OTel tracing disabled")
            self._enabled = False

    def _setup_metrics(self, config: TrackingConfig, local) -> None:
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({
                "service.name": config.service_name,
                "aet.run_id": config.run_id,
            })

            if config.otel_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                    metric_exporter = OTLPMetricExporter(
                        endpoint=config.otel_endpoint.rstrip("/") + "/v1/metrics"
                    )
                except Exception:
                    metric_exporter = ConsoleMetricExporter()
            else:
                metric_exporter = ConsoleMetricExporter()

            reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter("aet")

            self._op_duration = self._meter.create_histogram(
                "gen_ai.client.operation.duration",
                unit="s",
                description="GenAI operation duration",
            )
            self._token_usage = self._meter.create_histogram(
                "gen_ai.client.token.usage",
                unit="{token}",
                description="Input and output token usage",
            )
            self._validation_duration = self._meter.create_histogram(
                "aet.validation.duration",
                unit="s",
                description="Per-validator execution time",
            )
        except Exception as e:
            local.warn(f"OTel metrics setup failed ({e}); metrics disabled")

    def _setup_openllmetry(self, config: TrackingConfig, local, provider) -> None:
        if not config.enable_openllmetry:
            return
        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            AnthropicInstrumentor().instrument(tracer_provider=provider)
        except ImportError:
            local.warn(
                "openllmetry anthropic instrumentation not installed; "
                "Anthropic SDK calls not auto-traced. "
                "Install with: uv pip install opentelemetry-instrumentation-anthropic"
            )
        except Exception as e:
            local.warn(f"OpenLLMetry Anthropic instrumentor failed ({e})")

    # ------------------------------------------------------------------
    # Trace context helpers

    def _current_traceparent(self) -> str | None:
        """Return W3C traceparent string for the active span, or None."""
        try:
            from opentelemetry import trace
            ctx = trace.get_current_span().get_span_context()
            if ctx.is_valid:
                trace_id = format(ctx.trace_id, "032x")
                span_id = format(ctx.span_id, "016x")
                return f"00-{trace_id}-{span_id}-01"
        except Exception:
            pass
        return None

    def _inject_traceparent(self, env: dict) -> dict:
        """Inject TRACEPARENT into an env dict for subprocess launch."""
        tp = self._current_traceparent()
        if tp:
            env["TRACEPARENT"] = tp
        return env

    # ------------------------------------------------------------------
    # Span factories

    def start_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Return a context manager — real OTel span or nullcontext."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        @contextmanager
        def _span_cm():
            with self._tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                try:
                    from opentelemetry import trace
                    ctx = trace.get_current_span().get_span_context()
                    if ctx.is_valid:
                        self._trace_id = format(ctx.trace_id, "032x")
                except Exception:
                    pass
                yield span

        return _span_cm()

    def start_workflow_span(
        self,
        workflow_name: str,
        suite: str = "",
        method: str = "",
        seed: int = 0,
        target: str = "",
        run_id: str = "",
    ):
        """Root span for a full eval run — gen_ai.operation.name=invoke_workflow."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_WORKFLOW_NAME,
            AET_SUITE, AET_METHOD, AET_SEED, AET_TARGET, AET_RUN_ID,
            OP_INVOKE_WORKFLOW,
        )
        attrs = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_WORKFLOW,
            GEN_AI_WORKFLOW_NAME: workflow_name,
            AET_SUITE: suite,
            AET_METHOD: method,
            AET_SEED: seed,
            AET_TARGET: target,
            AET_RUN_ID: run_id,
        }

        @contextmanager
        def _span_cm():
            span_name = f"invoke_workflow {workflow_name}"
            with self._tracer.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    from opentelemetry import trace
                    ctx = trace.get_current_span().get_span_context()
                    if ctx.is_valid:
                        self._trace_id = format(ctx.trace_id, "032x")
                except Exception:
                    pass
                yield span

        return _span_cm()

    def start_agent_span(self, agent_name: str, model: str = "", provider: str = "anthropic"):
        """Span for an agent invocation — gen_ai.operation.name=invoke_agent.

        If config.claude_code_trace_parent is set, sets it as W3C context so this
        span becomes a child of the outer Claude Code span.
        """
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_AGENT_NAME, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MODEL,
            OP_INVOKE_AGENT,
        )
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            GEN_AI_AGENT_NAME: agent_name,
            GEN_AI_PROVIDER_NAME: provider,
        }
        if model:
            attrs[GEN_AI_REQUEST_MODEL] = model

        @contextmanager
        def _span_cm():
            context = None
            tp = self._config.claude_code_trace_parent
            if tp:
                try:
                    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
                    carrier = {"traceparent": tp}
                    context = TraceContextTextMapPropagator().extract(carrier=carrier)
                except Exception:
                    pass

            span_name = f"invoke_agent {agent_name}"
            with self._tracer.start_as_current_span(span_name, context=context, attributes=attrs) as span:
                yield span

        return _span_cm()

    def start_tool_span(self, tool_name: str, validator_name: str | None = None):
        """Span for a tool/validator execution — gen_ai.operation.name=execute_tool."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_TOOL_NAME, AET_VALIDATOR_NAME,
            OP_EXECUTE_TOOL,
        )
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL,
            GEN_AI_TOOL_NAME: tool_name,
        }
        if validator_name:
            attrs[AET_VALIDATOR_NAME] = validator_name

        @contextmanager
        def _span_cm():
            t0 = time.monotonic()
            span_name = f"execute_tool {tool_name}"
            with self._tracer.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    yield span
                finally:
                    elapsed = time.monotonic() - t0
                    if self._validation_duration and validator_name:
                        self._validation_duration.record(
                            elapsed,
                            attributes={"aet.validator.name": validator_name},
                        )

        return _span_cm()

    # ------------------------------------------------------------------
    # Span events

    def log_evaluation_event(self, name: str, score: float, label: str) -> None:
        """Add gen_ai.evaluation.result event to the currently active span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            from aet.tracking.semconv import (
                GEN_AI_EVAL_RESULT_EVENT, GEN_AI_EVAL_NAME,
                GEN_AI_EVAL_SCORE_VALUE, GEN_AI_EVAL_SCORE_LABEL,
            )
            span = trace.get_current_span()
            span.add_event(
                GEN_AI_EVAL_RESULT_EVENT,
                attributes={
                    GEN_AI_EVAL_NAME: name,
                    GEN_AI_EVAL_SCORE_VALUE: score,
                    GEN_AI_EVAL_SCORE_LABEL: label,
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Metrics

    def record_metric(
        self,
        op_duration: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        operation: str = "chat",
        model: str = "",
    ) -> None:
        """Record LLM call metrics to OTel histograms."""
        if not self._enabled:
            return
        attrs: dict[str, Any] = {"gen_ai.operation.name": operation}
        if model:
            attrs["gen_ai.request.model"] = model
        try:
            if op_duration is not None and self._op_duration:
                self._op_duration.record(op_duration, attributes=attrs)
            if input_tokens is not None and self._token_usage:
                self._token_usage.record(
                    input_tokens,
                    attributes={**attrs, "gen_ai.token.type": "input"},
                )
            if output_tokens is not None and self._token_usage:
                self._token_usage.record(
                    output_tokens,
                    attributes={**attrs, "gen_ai.token.type": "output"},
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    def get_traceparent_for_subprocess(self) -> str | None:
        """Return TRACEPARENT value for the currently active span (use before subprocess launch)."""
        return self._current_traceparent()
