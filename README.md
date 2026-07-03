# aet — Agentic Eval Tool

**Repo-agnostic harness to run, record, and *visualize* agentic coding runs.** UC Berkeley BAR / SLICE.

`aet` turns an agent session (Claude Code / any `stream-json` transcript) into a canonical
**trajectory** — cumulative tokens, cost, an activity timeline, and test-pass milestones over time —
then compares and **plots** many runs in a consistent house style. It also runs agents in a
deny-by-default sandbox, survives the Claude 5-hour usage limit unattended, and drives structured
evaluation suites. Nothing about your compiler/model/project is baked in; project data lives in your
repo, `aet` ships as an installed package.

## Install

```bash
pip install aet            # core (pure-stdlib + pyyaml)
pip install "aet[viz]"     # + matplotlib/numpy for the figures
pip install "aet[all]"     # everything (tracking, ray, viz, docs, dev)
```

## Two things it does

### 1. Record → plot an agentic run

Point it at one or many Claude Code transcripts and get the comparison figures — no setup:

```bash
# one step: raw sessions → figures
aet plot-sessions run_a/transcript.jsonl run_b/transcript.jsonl --out plots/

# or: ingest into a canonical trajectory, then plot any kind
aet import --source transcript --raw run_a/transcript.jsonl --into runs/run_a
aet plot runs/run_a --comparison runs/run_b --kind rate-panels --out fig.png
```

Figure kinds (`--kind`): `rate-panels` (per-arm token-rate panels on their own time scale, activity
bands, test-pass milestones), `cost-vs-time` (cumulative spend per arm), `tests-facets` (tests
passing over time, one lane per arm), plus `trajectory`/`comparison`. `.png` output also writes `.svg`.

Cost is the **billed** number when the transcript has a `result` event, otherwise a flagged list-price
estimate. See [docs/trajectory.md](docs/trajectory.md).

### 2. Run an agent — sandboxed, recorded, rate-limit-resilient

```bash
aet run --task TASK.md --workspace ./ws --sandbox bwrap \
        --allow /path/to/inputs --deny /path/to/answers
aet run --resume runs/<run>       # continue a run that hit the 5-hour limit
```

`aet run` launches a single agent invocation inside a deny-by-default
[bubblewrap sandbox](docs/isolation.md), streams it to a recorded trajectory, and materializes a
canonical run. On the five-hour usage limit it checkpoints, waits to the exact reset (or polls), and
resumes; on the weekly limit it writes `UNFINISHED.md` + a resumable status. `aet monitor --attach
<transcript>` gives a live view of an in-flight run.

### (also) Structured evaluation suites

The original harness — initialize a project, run method×seed sweeps, validate artifacts, and compare
with statistics/regression detection:

```bash
aet init-project --template targetgen --project-root ./my-evals
aet run-suite --suite targetgen --methods v0_naive,v2_schema --seeds 1,2,3
aet compare --suite targetgen --plots
```

## CLI

`aet <command> --help` is the source of truth (argparse-derived, always current):

| Command | Purpose |
|---|---|
| `import` / `plot` / `plot-sessions` | ingest transcripts → trajectory → figures |
| `run` / `resume` / `monitor` | sandboxed recorded agent runs + live view |
| `init-project` / `init-run` / `validate` | evaluation-suite lifecycle |
| `run-suite` / `compare` / `baseline` | sweeps, comparison reports, regression baselines |
| `runs` / `show` | list / inspect recorded runs |

## Subsystems

| Package | What |
|---|---|
| `aet.trajectory` | the canonical `RunTrajectory` data-model + importers (`transcript`, `capsule-bench`), live streaming, and the [oracle-progression](docs/adr/0001-transcript-oracle-mining.md) miner |
| `aet.viz` | house-style figures (behind the `[viz]` extra) |
| `aet.isolation` | deny-by-default bwrap sandbox + post-run audit + file-access ledger |
| `aet.runner` / `aet.ratelimit` | the sandboxed runner + five-hour/weekly watchdog |
| `aet.tracking` | `EvalRunLogger` → local JSON / MLflow / OpenTelemetry backends |
| `aet.suites` | pluggable evaluation suites (`default`, and the bundled-example `targetgen`) |

## Extras

| Extra | Adds | For |
|---|---|---|
| `[viz]` | matplotlib, numpy | the figures (`plot`, `plot-sessions`, `compare --plots`) |
| `[tracking]` | mlflow, opentelemetry | MLflow dashboards + OTel tracing |
| `[ray]` | ray | parallel sweep execution |
| `[docs]` | mkdocs, mkdocstrings | build the docs site (`mkdocs serve`) |
| `[dev]` | pytest, ruff | development |
| `[all]` | all of the above | |

Only `pyyaml` is required at install time; `import`/`monitor` work without `[viz]`.

## Docs & contributing

- Full docs (guides + auto-generated API reference): `pip install "aet[docs]" && mkdocs serve`.
- Architecture map + "how to add an importer / suite / figure": [AGENTS.md](AGENTS.md) and
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- Design rationale: [docs/adr/](docs/adr/).

## License

Apache-2.0. Copyright UC Berkeley BAR / SLICE. Home: https://github.com/ucb-bar/agentic-eval-tool
