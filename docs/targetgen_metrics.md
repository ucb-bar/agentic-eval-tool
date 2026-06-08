# TargetGen Suite Metrics

This document describes every column in `metrics/summary_metrics.json` and in the `reports/targetgen/metrics.csv` produced by `aet compare`.

## Column Reference

### Identity Columns

These three columns are always first and identify which project, suite, and target the run belongs to.

| Column | Type | Description |
|--------|------|-------------|
| `project` | string | Project name (directory name of `--project-root`) |
| `suite` | string | Suite name. Always `targetgen` for this suite. |
| `target` | string | Hardware target name (e.g. `gemmini`, `saturn`) |

### Core Run Identity

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Unique run identifier. Format: `<date>_<method>_seed<NNN>` |
| `method` | string | Method name passed to `--method` |
| `seed` | int | Random seed passed to `--seed` |
| `is_smoke_test` | bool | True if this was a smoke test run (`--smoke` / `--no-smoke`) |
| `budget` | string | Budget identifier (e.g. `cheap_smoke`, `paper_budget`) |

### Validation Quality

| Column | Type | Description |
|--------|------|-------------|
| `promotion_flag` | bool | True if TableGen/C++ generation is permitted (set in manifest) |
| `overall_status` | string | Aggregate validation status: `pass`, `fail`, or `error` |
| `schema_valid` | bool | Whether `run_manifest.yaml` passed schema validation |
| `evidence_score` | float | Fraction of ops in `dialect_plan.yaml` with evidence populated |
| `xdsl_valid` | bool | Whether `generated/<target>-mlir/xdsl/` is non-empty |
| `passes_pass_rate` | float | Fraction of positive test files present (executor not yet wired) |
| `design_ops_coverage` | float | Fraction of ops satisfying all dialect design constraints |
| `runtime_mock_match` | bool | Whether the runtime mock matches the expected interface |
| `merlin_integration_score` | float | 1.0 if no Merlin core files were modified; 0.0 otherwise |

### Git and Tracking Metadata

| Column | Type | Description |
|--------|------|-------------|
| `git_hash_at_init` | string | `git rev-parse HEAD` at `init-run` time |
| `tracking_mode` | string | Tracking mode used: `local`, `mlflow`, `full`, or `debug` |
| `mlflow_run_id` | string or null | MLflow run ID, if MLflow tracking was active |
| `otel_trace_id` | string or null | OTel trace ID, if OTel tracing was active |

### Cost and Effort

| Column | Type | Description |
|--------|------|-------------|
| `observed_cost_usd` | float or null | Actual API cost in USD, if reported by the agent harness |
| `estimated_cost_usd` | float or null | Estimated cost in USD, based on token counts and model pricing |
| `cost_source` | string or null | How cost was determined: `observed`, `estimated`, or null |
| `tokens_input` | int or null | Total input tokens consumed by the agent |
| `tokens_output` | int or null | Total output tokens produced by the agent |
| `token_source` | string or null | Source of token counts: `agent_report`, `estimated`, or null |
| `wall_clock_seconds` | float or null | Total wall-clock time for the run in seconds |
| `time_to_first_validation_s` | float or null | Seconds from run start to first validator passing |
| `agent_turns` | int or null | Number of agent turns (model calls) |
| `tool_calls` | int or null | Total number of tool calls made by the agent |

## Notes on Null Values

Effort columns (`observed_cost_usd`, `estimated_cost_usd`, `tokens_input`, `tokens_output`, `wall_clock_seconds`, `time_to_first_validation_s`, `agent_turns`, `tool_calls`) are null unless the agent harness writes them into the run directory before `aet validate` is called. The `aet compare` command computes mean±std only for non-null, non-string columns.

## Injecting Effort Metrics

An agent harness can inject effort metrics by writing a JSON file to `metrics/effort_metrics.json` before calling `aet validate`. The validate command reads this file if present and merges the values into `summary_metrics.json`. Expected keys match the column names above.

## Architecture Rules Columns (Aggregate)

The architecture rules checker (R1–R10) contributes two aggregate columns to the summary. Per-rule results are in `metrics/arch_rules.json`.

| Column | Type | Description |
|--------|------|-------------|
| `arch_rules_passed` | int | Number of architecture rules that passed (out of 10) |
| `arch_rules_failed` | int | Number of architecture rules that failed |
