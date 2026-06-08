# targetgen-evals

Evaluation harness for LLM-assisted MLIR dialect generation (TargetGen).

## Quickstart

```bash
# 1. Freeze the Gemmini source snapshot (do this once before any method run)
#    Copy chipyard/generators/gemmini/ subset into datasets/gemmini/source_snapshot/
#    then set `frozen: true` in datasets/gemmini/dataset_manifest.yaml.

# 2. Run the smoke-test budget to confirm harness mechanics
aet init-run --target gemmini --method v0_naive_claude --seed 1 \
  --budget configs/budgets/cheap_smoke.yaml

# 3. Validate the run
aet validate runs/targetgen/<run_id>/

# 4. Compare results across methods
aet compare --target gemmini
```

## Methods

| Method | Description |
|---|---|
| `v0_naive_claude` | Unconstrained LLM generation — comparison ceiling |
| `v1_schema_only` | Schema planning only, no code generation |
| `v2_schema_generator` | Schema planning + deterministic code generator |
| `v3_evidence_graph` | Evidence-first planning from source snapshot |
| `v4_rtl_tools` | RTL analysis tools + schema planning |
| `v5_kernel_miner` | Bottom-up design from kernel corpus |
| `v6_full` | Full pipeline: RTL + kernels + evidence + schema + generator |

## Observability

See `observability/README.md` for tracking modes (local, MLflow, OTel).

## Dataset

See `datasets/gemmini/dataset_manifest.yaml` for dataset contents and curation instructions.
