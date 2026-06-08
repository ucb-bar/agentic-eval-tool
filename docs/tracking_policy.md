# Tracking Policy

## Canonical Record: Local Files

Local JSON/YAML files are the canonical reproducibility record. Every run writes:

- `run_manifest.yaml` — configuration and identity at init time
- `metrics/summary_metrics.json` — aggregated validation results
- `metrics/*.json` — per-validator detail
- `logs/tracking_warnings.jsonl` — any backend failures, written by the harness

These files are written unconditionally. No network dependency, no optional package, and no backend failure can prevent them from being created.

## MLflow: Official Paper-Facing Tracker

MLflow is the official experiment tracker for paper reporting. When enabled, every run logs:

- Parameters: method, seed, suite, target, budget, model, dtype, substrate
- Metrics: all numeric columns from `summary_metrics.json`
- Artifacts: `run_manifest.yaml`, `metrics/summary_metrics.json`

MLflow is optional. The harness does not require it for local operation.

## OpenTelemetry: Optional Trace Transport

OpenTelemetry (OTel) is the optional distributed tracing layer. When enabled, each validator emits a span with its result. Spans are sent to the configured OTel collector endpoint.

OTel is strictly optional. If the endpoint is unreachable the harness degrades gracefully: the span is dropped, a warning is written to `logs/tracking_warnings.jsonl`, and the run continues normally.

## SigNoz: Optional Self-Hosted Viewer

SigNoz is an optional self-hosted UI for viewing OTel traces. It is not a Python dependency and is not required for paper reporting or local operation.

To use SigNoz, deploy it separately (see `docker-compose.signoz.optional.yml` and the official SigNoz self-host docs). Configure aet to send traces to the SigNoz OTel collector endpoint. SigNoz then provides a UI over traces that aet already emits.

## Tracking Modes

| Mode | What runs |
|------|-----------|
| `local` | Local JSON/YAML only. Default. No network required. |
| `mlflow` | Local + MLflow. Requires `aet[tracking]` and a running MLflow server. |
| `full` | Local + MLflow + OpenTelemetry. Requires `aet[tracking]` and an OTel collector. |
| `debug` | Same as `full`, with verbose logging to stderr for each backend call. |

Set the mode with `--tracking <mode>` on any command that accepts tracking flags.

## Backend Failure Policy

All backend failures are caught by the harness. Failures are:

1. Logged to `logs/tracking_warnings.jsonl` in the run directory with an ISO timestamp and the full exception message.
2. Printed to stderr at warning level.
3. Never re-raised. The run continues.

This applies to: MLflow connection errors, OTel send failures, serialization errors, and import errors for optional packages.

The local backend never fails silently. If it cannot write a file it raises immediately, because the local record is the canonical reproducibility record and a silent failure there would corrupt the run.
