# TargetGen Evaluation Example: Gemmini Target

This example shows how to use `aet` to evaluate an agent that is generating a new compiler target (`gemmini`) for the `oscar-merlin` project.

## Prerequisites

```
pip install 'aet[tracking]'   # MLflow + OTel support; base aet works without this
```

## 1. Initialize the Project

Run once in the `oscar-merlin` repository root to scaffold the `aet` project structure:

```
cd /path/to/oscar-merlin
aet init-project --template targetgen
```

This creates:

```
aet.project.yaml            # project metadata
runs/                       # run directories written here
reports/                    # compare output written here
datasets/gemmini/tests/     # put .mlir test files here
```

## 2. Initialize a Run

Before launching the agent, initialize a run directory. This records the current git hash and method configuration:

```
aet init-run \
  --suite targetgen \
  --method agent_v1 \
  --seed 1 \
  --target gemmini \
  --no-smoke \
  --budget paper_budget
```

Output: the path to the created run directory, e.g.:

```
runs/targetgen/2024-01-15_agent_v1_seed001
```

The agent reads `run_manifest.yaml` from this directory to understand what to build.

## 3. Run the Agent

Point your agent at the run directory. The agent should:

- Write xDSL op definitions to `runs/targetgen/<run_id>/generated/gemmini-mlir/xdsl/`
- Write `runs/targetgen/<run_id>/contracts/dialect_plan.yaml`
- Not modify files under `merlin/` (R4 will catch this)

## 4. Validate

After the agent finishes, validate the outputs:

```
aet validate runs/targetgen/2024-01-15_agent_v1_seed001 \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --experiment-name targetgen-gemmini
```

Output:

```
[aet] Validation complete: status=pass, total_errors=0
```

Detailed results are in `runs/targetgen/2024-01-15_agent_v1_seed001/metrics/`.

## 5. Compare Across Runs

After running multiple methods and seeds, aggregate results:

```
aet compare \
  --suite targetgen \
  --no-smoke \
  --output-dir reports/targetgen
```

This writes `reports/targetgen/metrics.csv` with one row per run and all columns from `docs/targetgen_metrics.md`.

## 6. Full Sweep (run-suite)

To run init-run + validate for all combinations in one command:

```
aet run-suite \
  --suite targetgen \
  --target gemmini \
  --methods agent_v1,agent_v2,baseline \
  --seeds 1,2,3 \
  --no-smoke \
  --budget paper_budget \
  --tracking mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --experiment-name targetgen-gemmini-sweep
```

This runs 9 combinations (3 methods × 3 seeds) sequentially and then calls `compare` automatically.

## Directory Layout After a Full Sweep

```
oscar-merlin/
  runs/targetgen/
    2024-01-15_agent_v1_seed001/
      run_manifest.yaml
      generated/gemmini-mlir/xdsl/
      contracts/dialect_plan.yaml
      metrics/summary_metrics.json
    2024-01-15_agent_v1_seed002/
    ...
  reports/targetgen/
    metrics.csv
    comparison_report.json
```

## See Also

- `docs/targetgen_suite.md` — suite overview and validator list
- `docs/targetgen_metrics.md` — all CSV columns
- `docs/targetgen_architecture_rules.md` — R1–R10 rule table
- `docs/mlflow_tracking.md` — MLflow setup
