# TargetGen Suite

## Purpose

The `targetgen` suite evaluates an agent's ability to generate a new compiler target for the Merlin research compiler. Specifically, it validates whether an agent working in the `oscar-merlin` repository has correctly scaffolded a new hardware target following the xDSL-first workflow.

## What It Validates

The suite runs 7 validators plus the architecture rules checker:

| Validator | ID | What it checks |
|-----------|----|----------------|
| Schema | `schema` | `run_manifest.yaml` is present and conforms to the required schema |
| xDSL artifacts | `xdsl` | `generated/<target>-mlir/xdsl/` exists and contains `.py` files with op definitions |
| Evidence | `evidence` | `contracts/dialect_plan.yaml` has per-op evidence fields populated |
| Pass tests | `passes` | `datasets/<target>/tests/positive/` and `negative/` contain `.mlir` test files |
| Dialect design | `design` | `contracts/dialect_plan.yaml` satisfies op-level design constraints |
| Runtime mock | `runtime_mock` | Runtime mock in `generated/<target>-mlir/` matches expected interface |
| Merlin integration | `merlin_integration` | Merlin core files have not been modified since `git_hash_at_init` |

After all validators, the architecture rules checker (R1–R10) runs against the run directory. See `docs/targetgen_architecture_rules.md` for the full rule table.

## What It Generates

Each run directory has the following structure after `init-run` and the agent's work:

```
runs/targetgen/<run_id>/
  run_manifest.yaml          # written by init-run
  logs/
    tracking_warnings.jsonl  # backend failures (if any)
  metrics/
    summary_metrics.json     # aggregate result
    schema_metrics.json
    xdsl_metrics.json
    evidence_metrics.json
    pass_metrics.json
    design_metrics.json
    effort_metrics.json
  generated/
    <target>-mlir/           # agent writes here
      xdsl/                  # xDSL Python op definitions
      ...
  contracts/
    dialect_plan.yaml        # agent writes this
  patches/                   # optional: agent patches to Merlin core
  artifacts/                 # optional: extra artifacts
```

After `validate`, the metrics directory is populated by the suite. After `compare`, a CSV and comparison report are written to `reports/targetgen/`.

## The `contracts/` Directory

The key file an agent must produce is `contracts/dialect_plan.yaml`. This file declares:

- The ops being added to the target dialect.
- Evidence for each op (e.g., references to hardware documentation or benchmarks).
- Verifier coverage (what properties the verifier checks).
- Lowering exits (how each op eventually becomes lower-level IR).
- Whether unsupported cases are handled explicitly.

The architecture rules R5–R10 all check fields within this file. If `contracts/dialect_plan.yaml` is absent, R5–R10 pass with an informational message (empty run is valid for a smoke test).

## Run Lifecycle

1. **`aet init-run`** — Creates the directory structure, writes `run_manifest.yaml`, records `git_hash_at_init`.
2. **Agent work** — The agent reads `run_manifest.yaml`, writes to `generated/<target>-mlir/`, writes `contracts/dialect_plan.yaml`, and optionally writes `patches/`.
3. **`aet validate`** — Runs all 7 validators and the architecture rules checker. Writes `metrics/`.
4. **`aet compare`** — Aggregates all runs for the suite into `reports/targetgen/metrics.csv` and a comparison report.

## Smoke Tests vs Full Runs

By default, `aet init-run` marks runs as smoke tests (`is_smoke_test: true`). Smoke tests use `--budget cheap_smoke` and are excluded from paper comparison tables with `aet compare --no-smoke`.

For paper ablations:

```
aet run-suite \
  --suite targetgen \
  --target gemmini \
  --methods agent_v1,agent_v2 \
  --seeds 1,2,3 \
  --no-smoke \
  --budget paper_budget \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000
```
