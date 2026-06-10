# Claude Code + OpenTelemetry

Claude Code exports OTel spans when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Combined with
aet's GenAI semconv instrumentation you get end-to-end traces: workflow → agent session →
every tool call → LLM turn → token cost → validation score.

## What Claude Code emits

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, Claude Code emits:

- **Session span** — covers the full `claude` process lifetime
- **`execute_tool Bash/Read/Edit/...`** — one span per tool call with duration
- **`chat {model}`** — one span per LLM turn (via OpenLLMetry on Claude Code's side)
  - `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
  - `gen_ai.request.model`, `gen_ai.response.model`
- **`gen_ai.conversation.id`** — links turns across a session

## Environment variables

```bash
# Required: point Claude Code at your collector (or Jaeger direct)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Recommended: tag all Claude Code spans with the current aet run
export OTEL_SERVICE_NAME=claude-code
export OTEL_RESOURCE_ATTRIBUTES="aet.run_id=<run_id>,aet.suite=targetgen,aet.target=gemmini"
```

## Linking Claude Code spans into the aet trace

aet can create an `invoke_agent claude-code` span and inject its trace context into the
Claude Code subprocess environment so all Claude Code spans appear **nested under** the
aet agent span in Jaeger.

### In your eval script

```python
import os
import subprocess

# logger is an EvalRunLogger with tracking_mode="full" and otel_endpoint set
with logger.start_agent_span("claude-code", model="claude-sonnet-4-6") as span:
    env = dict(os.environ)
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    env["OTEL_SERVICE_NAME"] = "claude-code"
    env["OTEL_RESOURCE_ATTRIBUTES"] = f"aet.run_id={run_id}"

    # Inject TRACEPARENT so Claude Code spans are children of this agent span
    tp = logger.get_traceparent_for_subprocess()
    if tp:
        env["TRACEPARENT"] = tp

    result = subprocess.run(
        ["claude", "--print", "--no-verbose", prompt],
        env=env, capture_output=True, text=True,
    )
```

### What you'll see in Jaeger

```
invoke_workflow targetgen-eval         ← aet run span
  └── invoke_agent claude-code         ← created by your eval script
        └── [claude session span]      ← emitted by Claude Code
              ├── execute_tool Bash    ← Claude ran a shell command
              ├── execute_tool Read    ← Claude read a file
              ├── chat claude-sonnet-4-6   ← LLM call (tokens, model)
              └── execute_tool Edit   ← Claude edited a file
  ├── execute_tool validate_schema     ← aet validator span
  └── gen_ai.evaluation.result event  ← pass/fail score
```

## Full local setup

```bash
# 1. Start Jaeger (receives OTLP directly, no collector needed)
docker compose -f observability/docker-compose.jaeger.yml up -d

# 2. Run aet with full tracking and OTel
aet run-suite \
  --suite targetgen --target gemmini \
  --methods v0_naive_claude --seeds 1 \
  --tracking full \
  --otel-endpoint http://localhost:4318

# 3. Open http://localhost:16686 → select service "aet"
#    Drill into a trace to see the full span tree
```

For metrics (token usage over time, validator durations), use the full stack:

```bash
docker compose -f observability/docker-compose.observability.yml up -d
# Prometheus at http://localhost:9090
# Jaeger at http://localhost:16686
```

## Correlating with MLflow

The `otel_trace_id` is stored in `run_manifest.yaml` under `observability.opentelemetry.trace_id`
and in `validation_report.json`. Use it to jump from an MLflow run to the corresponding
Jaeger trace:

```
http://localhost:16686/trace/<otel_trace_id>
```

## Metrics available in Prometheus

| Metric | Labels | Description |
|---|---|---|
| `aet_gen_ai_client_operation_duration_bucket` | `gen_ai.operation.name`, `gen_ai.request.model` | LLM call latency histogram |
| `aet_gen_ai_client_token_usage_bucket` | `gen_ai.token.type=input\|output` | Token usage histogram |
| `aet_validation_duration_bucket` | `aet.validator.name` | Per-validator wall-clock time |
