# aet — Agentic Eval Tool

**Repo-agnostic research evaluation harness for UC Berkeley BAR / SLICE.**

## What it is

`aet` is a lightweight, repo-agnostic harness for running, tracking, and comparing
research evaluation suites. It targets compiler, runtime, and agentic code-generation
experiments — structured benchmarks where you want reproducible runs, artifact
validation, and side-by-side method comparisons. TargetGen (MLIR dialect generation
from hardware specs) is the first fully-supported suite.

## Quickstart

```bash
pip install aet
# or with optional deps:
pip install "aet[tracking]"

# Initialize a project:
aet init-project --template targetgen --project-root ./my-evals
cd my-evals

# Start a run:
aet init-run --suite targetgen --target gemmini --method v0_naive_claude --seed 1

# Validate:
aet validate runs/targetgen/2026-06-08_v0_naive_claude_seed001/

# Compare:
aet compare --suite targetgen

# Run a sweep:
aet run-suite --suite targetgen --target gemmini \
    --methods v0_naive_claude,v2_schema_generator --seeds 1,2,3
```

## Suites

| Suite       | Description                                         | Use case                                    |
|-------------|-----------------------------------------------------|---------------------------------------------|
| `default`   | Generic pass/fail artifact evaluation               | Any project; minimal config required        |
| `targetgen` | MLIR dialect + lowering generation from HW specs    | TargetGen / compiler dialect research       |

## Tracking modes

| Mode      | What it does                                          | When to use                        |
|-----------|-------------------------------------------------------|------------------------------------|
| local     | JSON run manifests in `runs/`; always on              | Every run, no setup needed         |
| mlflow    | Logs params, metrics, artifacts to an MLflow server   | Experiment dashboards, sweep grids |
| OTel      | Emits spans/metrics via OpenTelemetry SDK             | Distributed tracing, CI pipelines  |
| SigNoz    | OTel-compatible viewer (receiver, not an SDK dep)     | Self-hosted observability UI       |

## Optional dependencies

| Extra        | Packages included                          | Use case                              |
|--------------|--------------------------------------------|---------------------------------------|
| `[tracking]` | `mlflow`, `opentelemetry-sdk`              | MLflow + OTel experiment tracking     |
| `[ray]`      | `ray[default]`                             | Parallel sweep execution via Ray      |
| `[dev]`      | `pytest`, `ruff`, `mypy`, `pre-commit`     | Development and CI                    |
| `[all]`      | All of the above                           | Full installation                     |

Only `pyyaml` is required at install time.

## Project structure

After `aet init-project --template targetgen --project-root ./my-evals`:

```
my-evals/
  configs/          # Suite and target configs (YAML)
  datasets/         # Input specs and golden reference files
  methods/          # Method definitions (prompt templates, scripts)
  runs/             # Run artifacts and manifests (written by aet)
  reports/          # Comparison summaries (written by aet compare)
  observability/    # OTel collector config, SigNoz compose file
```

## Repo-agnostic design

`aet` ships as an installed package that provides the harness, validators, and CLI.
It has no opinions about your compiler or model code. Project-specific data — golden
files, method configs, datasets, and target specs — lives entirely in your project
directory, initialized via `aet init-project`. This means the same `aet` install can
drive evaluations across unrelated repositories without any per-repo monkey-patching.

## License

Apache-2.0. Copyright UC Berkeley BAR / SLICE.  
Future home: https://github.com/ucb-bar/agentic-eval-tool
