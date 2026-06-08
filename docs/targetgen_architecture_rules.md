# TargetGen Architecture Rules

The architecture rules checker runs after all validators during `aet validate`. It evaluates structural correctness of the generated output against the Merlin xDSL-first design workflow.

Each rule produces a result with fields: `rule_id`, `name`, `passed`, `severity`, `message`.

Aggregate counts (`arch_rules_passed`, `arch_rules_failed`) appear in `metrics/summary_metrics.json`. Per-rule detail is in `metrics/arch_rules.json`.

## Rule Table

| Rule | Name | Severity | Description |
|------|------|----------|-------------|
| R1 | `generated-repo-naming` | error | The generated output directory must be named `generated/<target>-mlir/`. All agent-produced artifacts must live under this path. |
| R2 | `xdsl-before-tablegen` | warning | The `generated/<target>-mlir/xdsl/` directory must exist and be non-empty before any TableGen or C++ promotion is attempted. xDSL artifacts are required first. |
| R3 | `no-premature-tablegen` | error | `.td` and `.cpp` files must not appear in `generated/<target>-mlir/` unless `promotion_flag` is set in the manifest. TableGen/C++ generation is only permitted after explicit promotion. |
| R4 | `merlin-core-immutable` | error | Merlin core files (under `merlin/`) must not be modified since `git_hash_at_init`. A target addition should not require changes to the Merlin core. If `git_hash_at_init` is unknown, this rule emits a warning. |
| R5 | `op-evidence` | error | Every op declared in `contracts/dialect_plan.yaml` must have an `evidence` field populated. Evidence links each op to hardware documentation or benchmarks that justify its semantics. |
| R6 | `op-verifier-coverage` | error | Every op in `contracts/dialect_plan.yaml` must have a `verifier` field. The verifier specifies what properties the op verifier will check. |
| R7 | `op-lowering-exit` | error | Every op in `contracts/dialect_plan.yaml` must have at least one `lowering_exits` entry. This documents how the op is eventually lowered to a more concrete IR. |
| R8 | `no-scheduling-in-semantics` | error | Ops must not have a `scheduling_policy` field in their semantic definition in `dialect_plan.yaml`. Scheduling policy belongs in the Schedule dialect, not in op semantics. |
| R9 | `no-runtime-in-types` | error | Ops must not have `runtime_launch_in_type` set. Runtime launch details (e.g. kernel launch parameters) must not be encoded in pure types. They belong in the Runtime dialect. |
| R10 | `unsupported-fails-early` | error | Any op with `has_unsupported_cases: true` in `dialect_plan.yaml` must also have an `unsupported_handling` field. Unsupported inputs must fail explicitly and early rather than silently producing incorrect output. |

## Behavior When `contracts/dialect_plan.yaml` Is Absent

Rules R5–R10 all read from `contracts/dialect_plan.yaml`. If this file does not exist, all six rules pass with severity `info` and the message:

```
dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
```

This is the expected state for a fresh `init-run` before any agent work. Rules R1–R4 still run.

## `promotion_flag`

R3 checks for the `promotion_flag` field in `run_manifest.yaml`. This flag is set explicitly when a run is intended to produce TableGen/C++ output. It is not set by default. To enable TableGen generation in an agent run, add `promotion_flag: true` to the manifest before the agent runs.

## Severity

| Severity | Meaning |
|----------|---------|
| `error` | Rule failure is a hard violation. Contributes to `arch_rules_failed`. |
| `warning` | Rule failure indicates a potential issue but may be expected in some configurations. Contributes to `arch_rules_failed`. |
| `info` | Rule passed or was skipped for a known-good reason. Contributes to `arch_rules_passed`. |
