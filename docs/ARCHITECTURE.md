# Architecture

`aet` has one spine — the **`RunTrajectory`** — and everything else produces or consumes it.

```
 transcript.jsonl ─┐
 desktop session ──┤─▶ import_transcript ─┐
 capsule-bench run ┘   (importers/)        │
 live claude run ─────▶ TrajectoryStream ──┼─▶  RunTrajectory  ──▶ emit_trajectory ──▶ run dir/
 aet run (sandboxed) ─▶ runner.py ─────────┘   (trajectory/model)   (recording.py)    metrics/…
                                               │
                                               ├──▶ aet.viz (figures: comparison.py, trajectory_plot.py)
                                               └──▶ aet.tracking (EvalRunLogger → local/mlflow/otel)
```

## The spine — `aet.trajectory`
- **`model.py`** — `RunTrajectory` (+ `TrajectoryPoint`, `ActivityBand`, `TestMilestone`,
  `RoundBoundary`). Pure stdlib, append-only, self-describing (carries its `classifier_config`). One
  object is built identically by a completed-run importer and a live stream.
- **`build.py::append_round`** — extends a trajectory with one agent invocation (token/cost points +
  activity bands), on a single active-wall time axis. The one code path.
- **`classify.py`** — maps a tool call to an activity category via an `ActivityConfig`
  (`LongWaitRule`s). Project rules are **config factories** (`capsule_bench_config`,
  `spec_to_rtl_config`), never harness source.
- **`importers/`** — one module per on-disk layout, registered in `IMPORTER_REGISTRY`. `transcript`
  is the generic default; `capsule-bench` handles a full QA-loop layout.
- **`oracle.py`** — mines a tests-passing *climb* from the agent's own testbench invocations in the
  transcript (see [ADR-0001](adr/0001-transcript-oracle-mining.md)).
- **`stream.py`** / **`recording.py`** — live incremental build; emit/materialize a canonical run.

## Consumers
- **`aet.viz`** — figures behind the `[viz]` extra. `comparison.py` has the three presentation
  figures (`plot_rate_panels`, `plot_cost_vs_time`, `plot_tests_facets`); `style.py` is the house
  style (`use_house_style`). Consumes only the data-model — never a raw transcript.
- **`aet.tracking`** — `EvalRunLogger` facade fanning out to local JSON / MLflow / OTel backends.
- **`aet.viz`/reports** read; nothing writes back to the trajectory.

## Running agents
- **`runner.py`** — a sandboxed, recorded single invocation; streams stdout → transcript → trajectory
  → materialized run. **`ratelimit.py`** — five-hour/weekly detection + wake-timing
  ([ADR-0002](adr/0002-session-log-provisional-cost.md) is about cost, not this). **`isolation/`** —
  deny-by-default bwrap sandbox + post-run audit + ledger ([ADR-0003](adr/0003-bwrap-over-docker.md)).

## CLI
`cli/main.py` is only the argparse table + dispatch; handlers live in `cli/commands/{lifecycle,
reporting,trajectory}.py` with shared helpers in `cli/_common.py`. Adding a verb = add a handler +
register it in `main()`.

For "how to extend", see [AGENTS.md](../AGENTS.md).
