# MLflow Tracking

## Overview

MLflow is the official paper-facing experiment tracker for `aet`. It is optional. Local JSON/YAML files are always written regardless of whether MLflow is enabled.

## Installation

MLflow support requires the `tracking` extra:

```
pip install 'aet[tracking]'
```

This installs `mlflow` and `opentelemetry-sdk` as optional dependencies.

## Enabling MLflow

Pass `--tracking mlflow` and `--mlflow-tracking-uri` to any command that accepts tracking flags:

```
aet init-run \
  --suite targetgen \
  --method agent_v1 \
  --seed 1 \
  --target gemmini \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --experiment-name targetgen-gemmini
```

```
aet validate runs/targetgen/2024-01-15_agent_v1_seed001 \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --experiment-name targetgen-gemmini
```

```
aet run-suite \
  --suite targetgen \
  --methods agent_v1,baseline \
  --seeds 1,2,3 \
  --target gemmini \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000
```

## Quickstart: Local MLflow Server

```yaml
# docker-compose.mlflow.yml
version: "3.8"
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
volumes:
  mlflow-data:
```

```
docker compose -f docker-compose.mlflow.yml up -d
```

The MLflow UI is then available at `http://localhost:5000`.

## What Gets Logged

### Parameters (logged at run start)

| Parameter | Source |
|-----------|--------|
| `method` | `--method` |
| `seed` | `--seed` |
| `suite` | `--suite` |
| `target` | `--target` |
| `budget` | `--budget` |
| `model` | `--model` |
| `dtype` | `--dtype` |
| `substrate` | `--substrate` |
| `is_smoke_test` | `--smoke` / `--no-smoke` |
| `git_hash_at_init` | `git rev-parse HEAD` at init time |

### Metrics (logged at validate time)

All numeric columns from `metrics/summary_metrics.json`, including:

- `schema_valid`, `xdsl_files`, `xdsl_op_estimate`
- `pass_tests_pass`, `pass_tests_total`
- `evidence_coverage`, `unsupported_claim_rate`
- `arch_rules_passed`, `arch_rules_failed`
- `observed_cost_usd`, `estimated_cost_usd`
- `tokens_input`, `tokens_output`
- `wall_clock_seconds`, `time_to_first_validation_s`
- `agent_turns`, `tool_calls`

### Artifacts (logged at validate time)

- `run_manifest.yaml`
- `metrics/summary_metrics.json`

## Experiment Naming

Use `--experiment-name` to group runs. If not specified, runs go to the MLflow default experiment. Recommended convention:

```
<suite>-<target>-<date>
# e.g. targetgen-gemmini-2024-01
```

## Failure Behavior

If the MLflow server is unreachable or any MLflow call fails, the harness:

1. Catches the exception.
2. Writes a warning to `logs/tracking_warnings.jsonl`.
3. Continues the run normally.

The `mlflow_run_id` column in `summary_metrics.json` will be `null` if MLflow logging failed.
