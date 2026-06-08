# SigNoz (Optional OTel Viewer)

## What SigNoz Is

SigNoz is an open-source, self-hosted observability platform. It provides a web UI for viewing OpenTelemetry traces, metrics, and logs. In the context of `aet`, it is an optional viewer for the OTel traces that `aet` already emits when `--tracking full` or `--tracking debug` is used.

SigNoz is **not**:
- A Python dependency of `aet`.
- Required for paper reporting.
- Required for local operation.
- Required for MLflow tracking.

It is simply a convenient UI over traces you are already emitting.

## Installation

SigNoz is deployed as a set of Docker containers. It is not installed via `pip`. See the official SigNoz self-host documentation:

https://signoz.io/docs/install/self-host/docker/

A reference compose file is provided at the root of this repo for convenience:

```
docker-compose.signoz.optional.yml
```

This file is marked optional — it is not part of any default `docker compose up` invocation.

## Quickstart

```
# Start SigNoz
docker compose -f docker-compose.signoz.optional.yml up -d

# Run aet with OTel traces pointing at SigNoz's OTLP collector
aet validate runs/targetgen/2024-01-15_agent_v1_seed001 \
  --tracking full \
  --otel-endpoint http://localhost:4318/v1/traces
```

The SigNoz UI is available at `http://localhost:3301` by default.

## What You See

Once traces are flowing, the SigNoz UI shows:

- A trace per `aet validate` call.
- Child spans for each validator (`schema`, `evidence`, `xdsl`, `passes`, `design`, `runtime_mock`, `merlin_integration`).
- Span attributes: `aet.run_id`, `aet.suite`, `aet.target`, `aet.method`, `aet.seed`, `aet.validator.passed`, `aet.validator.error_count`.
- Wall-clock timing for each validator.

This is useful for identifying slow validators across a sweep and for correlating trace IDs with MLflow run IDs (both are stored in `summary_metrics.json`).

## Relationship to Other Backends

```
aet validate (--tracking full)
  |
  +-- local backend    --> runs/*/metrics/*.json  (always, canonical)
  +-- MLflow backend   --> MLflow server           (optional, paper-facing)
  +-- OTel backend     --> OTLP collector          (optional, any backend)
                              |
                              +-- SigNoz UI        (optional viewer)
                              +-- Jaeger UI        (alternative viewer)
                              +-- Honeycomb        (alternative viewer)
```

SigNoz sits at the end of the OTel pipeline. Switching to a different OTel viewer requires only changing the viewer, not `aet` itself.

## Ports (default SigNoz)

| Port | Service |
|------|---------|
| `3301` | SigNoz web UI |
| `4317` | OTLP gRPC collector |
| `4318` | OTLP HTTP collector (use this with `--otel-endpoint`) |
