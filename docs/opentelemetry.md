# OpenTelemetry Integration

## Overview

`aet` optionally emits OpenTelemetry (OTel) traces during validation. Each validator in a suite runs inside a span. This gives timing and pass/fail visibility per-validator across runs.

OTel is strictly optional. No OTel infrastructure is required for local operation or paper reporting.

## Enabling OTel

Use `--tracking full` or `--tracking debug`:

```
aet validate runs/targetgen/2024-01-15_agent_v1_seed001 \
  --tracking full \
  --otel-endpoint http://localhost:4318/v1/traces
```

```
aet run-suite \
  --suite targetgen \
  --methods agent_v1 \
  --seeds 1,2,3 \
  --tracking full \
  --otel-endpoint http://localhost:4318/v1/traces \
  --mlflow-tracking-uri http://localhost:5000
```

`debug` mode is identical to `full` but logs every backend call to stderr:

```
aet validate ... --tracking debug --otel-endpoint http://localhost:4318/v1/traces
```

## Tracking Modes with OTel

| Mode | OTel active |
|------|-------------|
| `local` | No |
| `mlflow` | No |
| `full` | Yes |
| `debug` | Yes (verbose) |

## What Gets Traced

A span is emitted for each of the following during `validate`:

| Span name | Description |
|-----------|-------------|
| `aet.validate` | Root span for the entire validate call |
| `aet.validator.schema` | Schema validation |
| `aet.validator.evidence` | Evidence/coverage validation |
| `aet.validator.xdsl` | xDSL artifact validation |
| `aet.validator.passes` | Pass test validation |
| `aet.validator.design` | Dialect design validation |
| `aet.validator.runtime_mock` | Runtime mock match validation |
| `aet.validator.merlin_integration` | Merlin integration check |
| `aet.architecture_rules` | Architecture rules check (R1–R10) |

Each span carries attributes:

- `aet.run_id` — run identifier
- `aet.suite` — suite name
- `aet.target` — target hardware (if set)
- `aet.method` — method name
- `aet.seed` — seed value
- `aet.validator.passed` — boolean result for validator spans
- `aet.validator.error_count` — number of errors for validator spans

## OTel Endpoint

The `--otel-endpoint` flag sets the OTLP HTTP endpoint for trace export. The default OTLP HTTP port is `4318`. Standard endpoints:

| Service | Endpoint |
|---------|----------|
| SigNoz (local) | `http://localhost:4318/v1/traces` |
| Jaeger (OTLP) | `http://localhost:4318/v1/traces` |
| OTel Collector | `http://localhost:4318/v1/traces` |
| Honeycomb | `https://api.honeycomb.io/v1/traces` |

## Installation

OTel support is included in the `tracking` extra:

```
pip install 'aet[tracking]'
```

## Graceful Degradation

If the OTel endpoint is unreachable or the `opentelemetry-sdk` package is not installed, the harness:

1. Catches the exception.
2. Writes a warning to `logs/tracking_warnings.jsonl`.
3. Continues without tracing.

No span data is required for run correctness. The `otel_trace_id` column in `summary_metrics.json` will be `null` if OTel export failed or was not enabled.

## No Required Service

`aet` does not require any specific OTel backend. Any OTLP-compatible collector works. See `docs/signoz_optional.md` for a self-hosted viewer option.
