# aet Philosophy

## Why aet Exists

Research projects at UC Berkeley BAR/SLICE regularly need to answer the question: "does this agent/method actually work, repeatably, across seeds?" The answer requires running experiments, collecting metrics, and comparing results in a way that survives hand-offs between contributors and across paper revisions.

`aet` (agentic-eval-tool) is the shared harness for this. It is opinionated about a small set of things and silent about everything else.

## Core Principles

### Reproducibility First

Every run writes a `run_manifest.yaml` at init time. The manifest captures the method, seed, git hash, suite, and all configuration that was live when the run began. This file is the canonical record. Validation, comparison, and paper tables all derive from it.

Local JSON/YAML files are written unconditionally. No network dependency can prevent the run record from being written.

### Repo-Agnostic

`aet` does not know about any specific research project. It operates on a directory convention (`runs/`, `contracts/`, `generated/`, `metrics/`) and a manifest schema. Any project that follows the convention can use any suite. The `--project-root` flag points `aet` at any repo.

### Suite Plugin System

Evaluation logic lives in suites. A suite implements three methods: `init_run`, `validate`, and `compare`. Built-in suites (`default`, `targetgen`) are shipped with the package. External suites can be registered via Python package entry points. Running a different suite requires only `--suite <name>` — no changes to the harness.

### Graceful Degradation

Optional backends (MLflow, OpenTelemetry) are caught-and-logged, never fatal. If an MLflow server is unreachable, the run continues and a warning is written to `logs/tracking_warnings.jsonl`. The local backend always writes its records regardless of what the optional backends do.

The same principle applies to optional Python dependencies. `aet` has no required dependencies beyond the standard library and PyYAML. MLflow, Ray, and OpenTelemetry packages are optional extras.

## What aet Is Not

- Not a training framework.
- Not a job scheduler (Ray integration is planned but not yet active).
- Not a dashboard (SigNoz is an optional viewer, not a dependency).
- Not a replacement for your project's own test suite.

`aet` handles the harness. Your project handles the actual work.
