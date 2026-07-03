# Trajectories — record an agentic run

A **`RunTrajectory`** is the canonical record of what an agent did over time: cumulative tokens
(input/output/cache), cumulative cost, an activity timeline (thinking / reading / writing / bash /
long tool-waits), and test-pass milestones. It is pure-stdlib and is what every figure and report
consumes.

## Get one

```python
from aet.trajectory.importers.transcript import import_transcript

traj = import_transcript("run/transcript.jsonl", run_id="run-a")
print(traj.num_rounds, traj.final_cost_usd, traj.provisional)
print(traj.token_series()["spend"])      # cumulative $ over time (minutes on "t")
```

- Handles CLI `stream-json` (billed cost) and desktop session logs (provisional cost — see
  [ADR-0002](adr/0002-session-log-provisional-cost.md)).
- Pass a directory of `*.jsonl` to combine many sessions (ordered by first timestamp).
- `pass_bool=`/`n_passed=`,`n_total=` records a terminal verdict; `oracle_markers=["run.sh"]` mines a
  tests-passing *climb* from the agent's own testbench runs (see
  [ADR-0001](adr/0001-transcript-oracle-mining.md)).

## Plot it

```python
from aet.viz.comparison import plot_rate_panels, plot_cost_vs_time, plot_tests_facets
plot_cost_vs_time([traj_a, traj_b], ["a", "b"]).savefig("cost.png")
```

or from the CLI: `aet plot-sessions a/transcript.jsonl b/transcript.jsonl --out plots/`.

## Persist it

`emit_trajectory(traj, logger, run_dir)` writes `metrics/trajectory.json` (+ logs) so the run is
queryable via `aet runs` / `aet show` / `aet plot`. `RunTrajectory.from_run_dir(run_dir)` reads it
back.

See the [API reference](reference/trajectory.md) for the full data-model.
