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
        self._ttft = None
        self._validation_duration = None
        self._local = local
        self._config = config
        self._trace_id: str | None = None
        self._provider = None
        self._meter_provider = None
        self._openllmetry_instrumentors: list = []

        try:
            # availability probe: import the SDK pieces just to fail fast with a friendly hint
            # if opentelemetry-sdk is absent (only `trace` is used here; _setup re-imports the rest)
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource  # noqa: F401
            from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter  # noqa: F401,E501
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
            self._meter_provider = meter_provider
            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter("aet")

            self._op_duration = self._meter.create_histogram(
                "gen_ai.client.operation.duration",
                unit="s",
                description="GenAI operation duration",
                explicit_bucket_boundaries_advisory=[
                    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,
                    1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
                ],
            )
            self._token_usage = self._meter.create_histogram(
                "gen_ai.client.token.usage",
                unit="{token}",
                description="Number of input and output tokens used",
                explicit_bucket_boundaries_advisory=[
                    1, 4, 16, 64, 256, 1024, 4096, 16384,
                    65536, 262144, 1048576, 4194304, 16777216, 67108864,
                ],
            )
            self._ttft = self._meter.create_histogram(
                "gen_ai.client.operation.time_to_first_chunk",
                unit="s",
                description="Time to first chunk in streaming response",
                explicit_bucket_boundaries_advisory=[
                    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,
                    1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
                ],
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
        mp = self._meter_provider
        activated: list[str] = []

        # Anthropic SDK — captures response.id, tool definitions, reasoning tokens, TTFT
        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            inst = AnthropicInstrumentor(
                enrich_token_usage=False,   # we record token histograms ourselves
                use_legacy_attributes=False, # modern gen_ai.* semconv
            )
            if not inst.is_instrumented_by_opentelemetry:
                kwargs: dict = {"tracer_provider": provider}
                if mp:
                    kwargs["meter_provider"] = mp
                inst.instrument(**kwargs)
                self._openllmetry_instrumentors.append(inst)
                activated.append("anthropic")
        except ImportError:
            local.warn(
                "opentelemetry-instrumentation-anthropic not installed; "
                "install with: uv pip install opentelemetry-instrumentation-anthropic"
            )
        except Exception as e:
            local.warn(f"OpenLLMetry Anthropic instrumentor failed: {e}")

        # OpenAI SDK — same treatment
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
            inst_oa = OpenAIInstrumentor(use_legacy_attributes=False)
            if not inst_oa.is_instrumented_by_opentelemetry:
                kwargs_oa: dict = {"tracer_provider": provider}
                if mp:
                    kwargs_oa["meter_provider"] = mp
                inst_oa.instrument(**kwargs_oa)
                self._openllmetry_instrumentors.append(inst_oa)
                activated.append("openai")
        except ImportError:
            pass  # openai SDK not installed — silent, don't warn
        except Exception as e:
            local.warn(f"OpenLLMetry OpenAI instrumentor failed: {e}")

        if activated:
            local.warn(f"[OpenLLMetry] auto-instrumented: {', '.join(activated)}")

    def uninstrument_openllmetry(self) -> None:
        """Uninstrument all OpenLLMetry instrumentors (call on cleanup)."""
        for inst in self._openllmetry_instrumentors:
            try:
                inst.uninstrument()
            except Exception:
                pass
        self._openllmetry_instrumentors.clear()

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

    def start_agent_span(
        self,
        agent_name: str,
        model: str = "",
        provider: str = "anthropic",
        server_address: str = "",
        output_type: str = "text",
    ):
        """Span for an agent invocation — gen_ai.operation.name=invoke_agent."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_AGENT_NAME, GEN_AI_PROVIDER_NAME,
            GEN_AI_REQUEST_MODEL, GEN_AI_OUTPUT_TYPE, GEN_AI_REQUEST_STREAM,
            OP_INVOKE_AGENT, SERVER_ADDRESS_ANTHROPIC, PROVIDER_ANTHROPIC,
        )
        if not server_address and provider == PROVIDER_ANTHROPIC:
            server_address = SERVER_ADDRESS_ANTHROPIC
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            GEN_AI_AGENT_NAME: agent_name,
            GEN_AI_PROVIDER_NAME: provider,
            GEN_AI_OUTPUT_TYPE: output_type,
            GEN_AI_REQUEST_STREAM: True,
        }
        if model:
            attrs[GEN_AI_REQUEST_MODEL] = model
        if server_address:
            attrs["server.address"] = server_address

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

    def start_inference_span(
        self,
        model: str,
        provider: str = "anthropic",
        server_address: str = "",
        operation: str = "chat",
        stream: bool = False,
    ):
        """Span for a direct LLM inference call (Anthropic SDK, OpenAI, Bedrock).

        Use this when calling the API directly from Python, not via the claude CLI.
        gen_ai.operation.name = "chat" | "text_completion" | "generate_content"
        """
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MODEL,
            GEN_AI_REQUEST_STREAM, SERVER_ADDRESS_ANTHROPIC, PROVIDER_ANTHROPIC,
        )
        if not server_address and provider == PROVIDER_ANTHROPIC:
            server_address = SERVER_ADDRESS_ANTHROPIC
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: operation,
            GEN_AI_PROVIDER_NAME: provider,
            GEN_AI_REQUEST_MODEL: model,
            GEN_AI_REQUEST_STREAM: stream,
        }
        if server_address:
            attrs["server.address"] = server_address

        @contextmanager
        def _span_cm():
            span_name = f"{operation} {model}"
            with self._tracer.start_as_current_span(span_name, attributes=attrs) as span:
                yield span

        return _span_cm()

    def start_tool_span(self, tool_name: str, validator_name: str | None = None,
                        provider: str = "anthropic"):
        """Span for a tool/validator execution — gen_ai.operation.name=execute_tool."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        from aet.tracking.semconv import (
            GEN_AI_OPERATION_NAME, GEN_AI_PROVIDER_NAME, GEN_AI_TOOL_NAME,
            AET_VALIDATOR_NAME, OP_EXECUTE_TOOL,
        )
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL,
            GEN_AI_PROVIDER_NAME: provider,
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
    # Span attribute helpers

    def set_span_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the currently active span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute(key, value)
        except Exception:
            pass

    def set_span_attributes(self, attrs: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                for k, v in attrs.items():
                    if v is not None:
                        span.set_attribute(k, v)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Span events

    def log_evaluation_event(self, name: str, score: float, label: str,
                             explanation: str = "") -> None:
        """Add gen_ai.evaluation.result event to the currently active span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            from aet.tracking.semconv import (
                GEN_AI_EVAL_RESULT_EVENT, GEN_AI_EVAL_NAME,
                GEN_AI_EVAL_SCORE_VALUE, GEN_AI_EVAL_SCORE_LABEL, GEN_AI_EVAL_EXPLANATION,
            )
            span = trace.get_current_span()
            attrs: dict[str, Any] = {
                GEN_AI_EVAL_NAME: name,
                GEN_AI_EVAL_SCORE_VALUE: score,
                GEN_AI_EVAL_SCORE_LABEL: label,
            }
            if explanation:
                attrs[GEN_AI_EVAL_EXPLANATION] = explanation
            span.add_event(GEN_AI_EVAL_RESULT_EVENT, attributes=attrs)
        except Exception:
            pass

    def log_inference_details_event(
        self,
        operation: str,
        provider: str,
        model: str = "",
        conversation_id: str = "",
        output_type: str = "text",
        stream: bool = True,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        finish_reasons: list[str] | None = None,
        response_id: str = "",
        ttft_s: float = 0.0,
    ) -> None:
        """Emit gen_ai.client.inference.operation.details event (opt-in details per spec)."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            from aet.tracking.semconv import (
                GEN_AI_INFERENCE_DETAILS_EVENT,
                GEN_AI_OPERATION_NAME, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MODEL,
                GEN_AI_CONVERSATION_ID, GEN_AI_OUTPUT_TYPE, GEN_AI_REQUEST_STREAM,
                GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS,
                GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_RESPONSE_ID,
                GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
            )
            span = trace.get_current_span()
            if not span.is_recording():
                return
            attrs: dict[str, Any] = {
                GEN_AI_OPERATION_NAME: operation,
                GEN_AI_PROVIDER_NAME: provider,
                GEN_AI_OUTPUT_TYPE: output_type,
                GEN_AI_REQUEST_STREAM: stream,
            }
            if model:
                attrs[GEN_AI_REQUEST_MODEL] = model
            if conversation_id:
                attrs[GEN_AI_CONVERSATION_ID] = conversation_id
            if input_tokens:
                attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
            if output_tokens:
                attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
            if cache_read_tokens:
                attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_tokens
            if cache_creation_tokens:
                attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_creation_tokens
            if finish_reasons:
                attrs[GEN_AI_RESPONSE_FINISH_REASONS] = finish_reasons
            if response_id:
                attrs[GEN_AI_RESPONSE_ID] = response_id
            if ttft_s:
                attrs[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = ttft_s
            span.add_event(GEN_AI_INFERENCE_DETAILS_EVENT, attributes=attrs)
        except Exception:
            pass

    def log_exception_event(self, exc_type: str, message: str, stacktrace: str = "") -> None:
        """Emit gen_ai.client.operation.exception event on the current span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            from aet.tracking.semconv import GEN_AI_OPERATION_EXCEPTION_EVENT
            span = trace.get_current_span()
            if not span.is_recording():
                return
            attrs: dict[str, Any] = {
                "exception.type": exc_type,
                "exception.message": message,
            }
            if stacktrace:
                attrs["exception.stacktrace"] = stacktrace
            span.add_event(GEN_AI_OPERATION_EXCEPTION_EVENT, attributes=attrs)
        except Exception:
            pass

    def log_prompt_event(self, content: str, role: str = "user") -> None:
        """Add a gen_ai.user.message or gen_ai.system event to the current span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                event_name = "gen_ai.system" if role == "system" else "gen_ai.user.message"
                span.add_event(event_name, {"content": content, "role": role})
        except Exception:
            pass

    def log_completion_event(self, content: str) -> None:
        """Add a gen_ai.assistant.message event to the current span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span.add_event("gen_ai.assistant.message", {"content": content})
        except Exception:
            pass

    def log_tool_call_event(
        self,
        tool_name: str,
        tool_call_id: str,
        input_summary: str,
        result_summary: str,
        is_error: bool,
    ) -> None:
        """Add a gen_ai.tool.call event to the current span."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span.add_event("gen_ai.tool.call", {
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.tool.call.id": tool_call_id,
                    "gen_ai.tool.input": input_summary,
                    "gen_ai.tool.result": result_summary,
                    "gen_ai.tool.is_error": is_error,
                })
        except Exception:
            pass

    def emit_tool_call_spans(
        self,
        tool_calls: list,
        turn_usage: list,
        t0_ns: int,
        stream_duration_s: float,
    ) -> None:
        """Create child spans for every tool call AND every LLM inference turn.

        Must be called while the parent invoke_agent span is still active.
        Uses t0_ns (real wall-clock ns when the first stream event arrived) plus
        per-tool start_offset_s to place bars accurately in the SigNoz waterfall,
        giving the alternating [inference → tool → inference → tool] pattern.
        """
        if not self._enabled or not self._tracer:
            return
        try:
            from opentelemetry import context as otel_context
            parent_ctx = otel_context.get_current()

            def _emit(name: str, start_offset_s: float, duration_s: float, attrs: dict) -> None:
                start_ns = t0_ns + int(start_offset_s * 1_000_000_000)
                dur_ns = max(int(duration_s * 1_000_000_000), 1_000_000)
                span = self._tracer.start_span(name, context=parent_ctx,
                                               start_time=start_ns, attributes=attrs)
                span.end(end_time=start_ns + dur_ns)

            # ── Inference (think) spans — one per LLM turn ────────────────
            # A turn spans from assistant-message arrival to the end of its last tool result.
            # We approximate: turn ends at the start of the first tool call of the NEXT turn.
            # For the last turn, it spans to stream_duration_s.
            by_turn: dict[int, list] = {}
            for tc in tool_calls:
                by_turn.setdefault(tc.turn_index, []).append(tc)

            for i, tu in enumerate(turn_usage):
                if not hasattr(tu, "start_offset_s"):
                    continue
                # End of inference = start of first tool call in this turn (or stream end)
                this_turn_tools = sorted(by_turn.get(i + 1, []), key=lambda x: x.start_offset_s)
                if this_turn_tools:
                    infer_end = this_turn_tools[0].start_offset_s
                else:
                    infer_end = stream_duration_s
                infer_dur = max(infer_end - tu.start_offset_s, 0.001)
                _context_limit = 200_000
                _model_name = tu.model or ""
                if _model_name.startswith("claude"):
                    _context_limit = 200_000
                _context_pct = round(
                    (tu.input_tokens + getattr(tu, "cache_read_input_tokens", 0)
                     + getattr(tu, "cache_creation_input_tokens", 0))
                    / _context_limit * 100, 2
                )
                _emit(
                    f"inference turn {tu.turn}",
                    tu.start_offset_s,
                    infer_dur,
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": tu.model,
                        "gen_ai.usage.input_tokens": tu.input_tokens,
                        "gen_ai.usage.output_tokens": tu.output_tokens,
                        "gen_ai.usage.cache_read.input_tokens": tu.cache_read_input_tokens,
                        "gen_ai.usage.cache_creation.input_tokens": tu.cache_creation_input_tokens,
                        "aet.turn.index": tu.turn,
                        "aet.turn.reasoning_text": (tu.reasoning_text or "")[:200],
                        "aet.turn.context_pct_used": _context_pct,
                    },
                )

            # ── Tool call spans ───────────────────────────────────────────
            for tc in tool_calls:
                tool_type = "MCP" if tc.is_mcp else ("SKILL" if tc.name == "Skill" else "tool")
                attrs: dict[str, Any] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tc.name,
                    "gen_ai.tool.call.id": tc.tool_use_id,
                    "gen_ai.tool.is_error": str(tc.is_error).lower(),
                    "aet.tool.type": tool_type,
                    "aet.tool.duration_s": tc.duration_s,
                    "aet.tool.turn_index": tc.turn_index,
                }
                if tc.file_paths:
                    attrs["aet.tool.file_paths"] = ", ".join(tc.file_paths[:5])
                if tc.reasoning_before:
                    attrs["aet.tool.reasoning_before"] = tc.reasoning_before[:200]
                _emit(f"execute_tool {tc.name}", tc.start_offset_s, tc.duration_s, attrs)

        except Exception:
            pass

    def log_ttft(self, ttft_s: float, provider: str = "anthropic", model: str = "",
                 operation: str = "invoke_agent") -> None:
        """Record time-to-first-chunk as span attribute and histogram."""
        if not self._enabled:
            return
        self.set_span_attribute("gen_ai.response.time_to_first_chunk", ttft_s)
        try:
            if self._ttft:
                attrs: dict[str, Any] = {
                    "gen_ai.operation.name": operation,
                    "gen_ai.provider.name": provider,
                }
                if model:
                    attrs["gen_ai.request.model"] = model
                self._ttft.record(ttft_s, attributes=attrs)
        except Exception:
            pass

    def log_finish_reasons(self, reasons: list[str]) -> None:
        """Set gen_ai.response.finish_reasons on the current span."""
        if not self._enabled or not reasons:
            return
        self.set_span_attribute("gen_ai.response.finish_reasons", reasons)

    def log_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        model: str = "",
        provider: str = "anthropic",
    ) -> None:
        """Set gen_ai.usage.* attributes on the current span and record to histograms."""
        if not self._enabled:
            return
        try:
            from opentelemetry import trace
            from aet.tracking.semconv import (
                GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS,
                GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            )
            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                if cache_creation_tokens:
                    span.set_attribute(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens)
                if cache_read_tokens:
                    span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens)
        except Exception:
            pass
        # Also record to histograms (spec: gen_ai.provider.name required on token.usage)
        hist_attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": provider,
        }
        if model:
            hist_attrs["gen_ai.request.model"] = model
        try:
            if self._token_usage:
                self._token_usage.record(input_tokens, attributes={**hist_attrs, "gen_ai.token.type": "input"})
                self._token_usage.record(output_tokens, attributes={**hist_attrs, "gen_ai.token.type": "output"})
                if cache_read_tokens:
                    self._token_usage.record(cache_read_tokens, attributes={**hist_attrs, "gen_ai.token.type": "cache_read"})
                if cache_creation_tokens:
                    self._token_usage.record(cache_creation_tokens, attributes={**hist_attrs, "gen_ai.token.type": "cache_creation"})
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
        provider: str = "anthropic",
    ) -> None:
        """Record LLM call metrics to OTel histograms."""
        if not self._enabled:
            return
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider,
        }
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
