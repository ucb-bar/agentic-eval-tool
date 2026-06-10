# OTel GenAI Semantic Conventions in aet

aet emits OpenTelemetry spans aligned with the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (status: Development, 2025).

## Span Hierarchy

```
invoke_workflow {suite}-eval          [INTERNAL]
  gen_ai.workflow.name = {suite}-eval
  gen_ai.operation.name = invoke_workflow
  aet.suite, aet.method, aet.seed, aet.target, aet.run_id

  ├── invoke_agent claude-code         [INTERNAL]  (when using claude CLI)
  │   gen_ai.operation.name = invoke_agent
  │   gen_ai.agent.name = claude-code
  │   gen_ai.provider.name = anthropic
  │   gen_ai.request.model = claude-sonnet-4-6 (if set)
  │
  │   └── chat claude-sonnet-4-6       [CLIENT]  (auto by OpenLLMetry)
  │       gen_ai.usage.input_tokens = N
  │       gen_ai.usage.output_tokens = N
  │
  ├── execute_tool validate_schema     [INTERNAL]
  │   gen_ai.operation.name = execute_tool
  │   gen_ai.tool.name = validate_schema
  │   aet.validator.name = schema
  │
  ├── execute_tool validate_passes     [INTERNAL]
  │   gen_ai.tool.name = validate_passes
  │   aet.validator.name = passes
  │
  ├── ... (one span per validator)
  │
  └── gen_ai.evaluation.result  [event on workflow span]
      gen_ai.evaluation.name = validation_overall
      gen_ai.evaluation.score.value = 0.0 | 1.0
      gen_ai.evaluation.score.label = pass | partial | fail
```

## Span Naming Convention

| `gen_ai.operation.name` | Span name pattern | SpanKind |
|---|---|---|
| `invoke_workflow` | `invoke_workflow {suite}-eval` | INTERNAL |
| `invoke_agent` | `invoke_agent {agent_name}` | INTERNAL |
| `execute_tool` | `execute_tool {tool_name}` | INTERNAL |
| `chat` | `chat {model}` | CLIENT (OpenLLMetry) |

## Attribute Reference

### Standard GenAI attributes

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.operation.name` | string | Operation category (see table above) |
| `gen_ai.workflow.name` | string | Name of the eval workflow |
| `gen_ai.agent.name` | string | Agent identifier (e.g. `claude-code`) |
| `gen_ai.provider.name` | string | LLM provider (e.g. `anthropic`) |
| `gen_ai.request.model` | string | Model identifier |
| `gen_ai.tool.name` | string | Tool/validator name |
| `gen_ai.usage.input_tokens` | int | Input tokens consumed |
| `gen_ai.usage.output_tokens` | int | Output tokens consumed |
| `gen_ai.evaluation.name` | string | Evaluation metric name |
| `gen_ai.evaluation.score.value` | float | Numeric score (0.0–1.0) |
| `gen_ai.evaluation.score.label` | string | Human-readable label |

### aet-specific attributes

| Attribute | Type | Description |
|---|---|---|
| `aet.suite` | string | Suite name (e.g. `targetgen`) |
| `aet.method` | string | Method name (e.g. `v0_naive_claude`) |
| `aet.seed` | int | Random seed |
| `aet.target` | string | Hardware target (e.g. `gemmini`) |
| `aet.run_id` | string | Unique run identifier |
| `aet.validator.name` | string | Short validator key (e.g. `schema`) |
| `aet.run.total_errors` | int | Errors after full validation |
| `aet.run.total_warnings` | int | Warnings after full validation |

### Compilation/MLIR pass attributes

For projects instrumenting MLIR compilation passes:

| Attribute | Type | Description |
|---|---|---|
| `aet.compilation.pass_name` | string | Name of the MLIR pass |
| `aet.compilation.dialect_from` | string | Input dialect (e.g. `linalg`) |
| `aet.compilation.dialect_to` | string | Output dialect (e.g. `affine`) |

Use `logger.start_tool_span("pass_name")` with these attributes to instrument compilation stages.

## OTel Metrics

aet emits three histograms when `--tracking full` and `--otel-endpoint` are set:

| Metric | Unit | Description |
|---|---|---|
| `gen_ai.client.operation.duration` | `s` | Duration of LLM operations |
| `gen_ai.client.token.usage` | `{token}` | Input/output token counts |
| `aet.validation.duration` | `s` | Per-validator execution time |

Metrics are scraped by Prometheus from the OTel Collector at port 8889.

## Quick Start

```bash
# 1. Start the observability stack
docker compose -f observability/docker-compose.observability.yml up -d

# 2. Run aet with full tracking
aet run-suite \
  --suite targetgen --target gemmini \
  --methods v0_naive_claude --seeds 1 \
  --tracking full \
  --otel-endpoint http://localhost:4318

# 3. Open Jaeger UI at http://localhost:16686
#    Search for service: aet
```

Or use the minimal Jaeger-only setup (no OTel Collector):

```bash
docker compose -f observability/docker-compose.jaeger.yml up -d
# Jaeger receives OTLP directly on port 4318
aet ... --otel-endpoint http://localhost:4318
```

## OpenLLMetry Auto-instrumentation

When `opentelemetry-instrumentation-anthropic` is installed (included in `aet[tracking]`),
the Anthropic SDK is automatically instrumented. Every `anthropic.Anthropic().messages.create()`
call inside an aet run emits a `chat {model}` span with token usage — without any code changes
in the eval scripts themselves.

To disable: set `enable_openllmetry=False` in `TrackingConfig` or use the Python API directly.

## Instrumenting MLIR Passes

```python
with logger.start_tool_span("lower_linalg_to_affine"):
    # ... run your MLIR pass ...
    result = run_pass(module)
```

The span automatically records wall-clock duration to `aet.validation.duration`.
