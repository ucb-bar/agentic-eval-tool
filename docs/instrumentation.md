# Instrument & plot an agent run

End-to-end, using **only the installed `aet` package** — no repo checkout, no OTel collector, no
external services. You bring a Claude Code invocation; `aet` captures it and renders the trajectory
figures (cumulative tokens/cost with cache, token-rate panels with activity bands, test-pass climb).

There are two capture paths. Pick by how much fidelity you need:

| path | infra | what you get | use when |
|---|---|---|---|
| **A — full-fidelity (OTel)** | one localhost process (`aet otel-sink`) | real **per-turn** input / output / **cache-read** / cost / duration + tool spans, no interpolation | you want the accurate cumulative-cache line, real activity bands, per-turn cost |
| **B — transcript-only** | none | tokens/cost/timeline reconstructed from the `stream-json` transcript | you just have a saved transcript, or can't run a sidecar process |

Both produce the same `RunTrajectory` and feed the same `aet plot`.

## Install

No clone required — install the package (with the plotting extra) straight from GitHub:

```bash
pip install "aet[viz] @ git+https://github.com/ucb-bar/agentic-eval-tool.git"
```

`aet` core is pure-stdlib; the `[viz]` extra adds matplotlib + numpy for the figures. Verify:

```bash
aet --help          # subcommands include: otel-sink, import, plot, plot-sessions
```

## Path A — full-fidelity capture (OTel)

Claude Code, with telemetry enabled, exports one `claude_code.api_request` record **per API turn**
carrying the real `input_tokens / output_tokens / cache_read_tokens / cache_creation_tokens /
cost_usd / duration_ms` and a real timestamp — plus tool/prompt events and cost/token metrics.
`aet otel-sink` is a tiny stdlib OTLP/HTTP receiver that appends each envelope to a JSONL file;
`aet import --source otel` reconstructs the exact per-turn trajectory from it.

### 1. Start the sink (one per run)

```bash
aet otel-sink --port 4317 --out ./otel_logs.jsonl &
```

It listens on `127.0.0.1:4317` and writes one line per received OTLP envelope. Leave it running for
the duration of the agent invocation, then stop it (`kill %1`) so it flushes.

### 2. Run the agent pointed at the sink

Set these on the agent's environment, then invoke Claude Code however you normally do:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
export OTEL_LOGS_EXPORT_INTERVAL=1000 OTEL_METRIC_EXPORT_INTERVAL=1000   # flush ~1s

claude --print --model claude-opus-4-8 \
    --output-format stream-json --verbose \
    < prompt.md > transcript.jsonl
```

Saving the `stream-json` transcript alongside `otel_logs.jsonl` is worth it: the OTel importer uses
it (if present as a sibling `transcript.jsonl`) to map tool calls → commands for precise tool-wait
classification and to densify the sub-turn thinking stream. It is optional — OTel alone is enough.

### 3. Import → trajectory

```bash
aet import --source otel --raw ./otel_logs.jsonl --run-id run-a \
    --pass --n-total 32 --out ./trajectory.json
```

`--pass` / `--fail` + `--n-total N` records a terminal verdict (drawn as the test-pass milestone);
omit them if you have no verdict. The command prints the reconstructed totals (tokens incl. cache,
billed cost, duration) so you can eyeball that capture worked.

### 4. Plot

```bash
aet plot ./trajectory.json --kind trajectory   --out fig_cumulative.png   # cumulative lines + cache + spend
aet plot ./trajectory.json --kind rate-panels  --out fig_rates.png        # token-rate + activity bands
```

That is the whole loop. As a copy-paste wrapper around any single Claude Code call:

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT=$(mktemp -d)
aet otel-sink --port 4317 --out "$OUT/otel_logs.jsonl" & SINK=$!
sleep 1
CLAUDE_CODE_ENABLE_TELEMETRY=1 OTEL_LOGS_EXPORTER=otlp OTEL_METRICS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=http/json OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  claude --print --output-format stream-json --verbose < prompt.md > "$OUT/transcript.jsonl"
sleep 2; kill "$SINK"                       # let the CLI flush its final export
aet import --source otel --raw "$OUT/otel_logs.jsonl" --out "$OUT/trajectory.json"
aet plot "$OUT/trajectory.json" --kind trajectory --out "$OUT/fig.png"
echo "figures + trajectory in $OUT"
```

To compare several runs on aligned panels, import each to its own `trajectory.json` and pass the
rest as `--comparison`:

```bash
aet plot a/trajectory.json --kind comparison --comparison b/trajectory.json c/trajectory.json \
    --out compare.png
```

## Path B — transcript-only (zero infra)

If you already have (or can save) a Claude Code `stream-json` transcript, skip the sink entirely:

```bash
claude --print --output-format stream-json --verbose < prompt.md > transcript.jsonl

aet import --source transcript --raw transcript.jsonl --label run-a \
    --pass --n-total 32 --out trajectory.json
aet plot trajectory.json --kind trajectory --out fig.png
```

Or render figures for several transcripts in one step:

```bash
aet plot-sessions run-a.jsonl run-b.jsonl --out ./figs
```

Transcript-only cost is billed-accurate when the transcript is CLI `stream-json` (it carries the
`result` event's `total_cost_usd`); desktop session logs are marked *provisional* (see
[ADR-0002](adr/0002-session-log-provisional-cost.md)). The cache-read line is only fully truthful on
the OTel path — Path A is the one to use when cache/per-turn cost accuracy matters.

## Figure kinds

`aet plot <trajectory-or-run> --kind <kind>`:

| kind | shows |
|---|---|
| `trajectory` | one run: cumulative input / output / **cache-read** / total token lines (log axis) + cumulative spend + real activity bands |
| `comparison` | several runs stacked on aligned panels (`--comparison ...`) — the same cumulative view per arm |
| `rate-panels` | per-run token-rate panels · own time scale · activity-share bands (incl. long tool-waits) · cumulative spend |
| `cost-vs-time` | one cumulative-spend line per run |
| `tests-facets` | one pass/fail (k/N) lane per run over its wall time |

`--linear-tokens` switches the token axis off log; `--no-spend` drops the spend twin axis;
`--dpi` sets resolution (a `.png` also writes a matching `.svg`).

## Notes

- **One sink per run.** The sink appends; point each run at its own `--out` file (or a fresh port)
  so trajectories don't cross-contaminate.
- **Localhost only.** `aet otel-sink` binds `127.0.0.1` by default. If the agent runs in a container
  or sandbox, make sure it can reach the sink's host/port (a shared network namespace suffices;
  see [Isolation](isolation.md)).
- **Programmatic use.** Everything above is also a library: `from aet.trajectory.importers.otel
  import import_otel` and `from aet.viz.trajectory_plot import plot_trajectory, plot_comparison`.
  See the [Trajectories guide](trajectory.md) and the API reference.
