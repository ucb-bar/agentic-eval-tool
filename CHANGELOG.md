# Changelog

All notable changes to `aet` are recorded here.

## [Unreleased]

### Added
- **Tests-passing CLIMB mined from oracle invocations (`aet.trajectory.oracle`)** — reconstructs the
  tests-over-time progression from the transcript itself: each `./run.sh` (testbench) invocation the
  agent runs becomes a `k/N` milestone at its wall time. Retroactive (works on existing runs), no
  harness change. Rejects a different testbench by suite-size mismatch.
  `import_transcript(oracle_markers=…)` + `classify.spec_to_rtl_config()` (verilator/`run.sh` →
  tool-wait band) reproduce the reference `rate-panels` figure for abc-testing spec-to-rtl.
- **`aet plot-sessions`** — point at raw Claude session transcripts (files or dirs) and render the
  comparison figures in one step; no prior `aet import` needed.
- **`k/N` terminal grade + per-lane facet scaling** — `import_transcript(n_passed=, n_total=)` records
  a real fraction (e.g. `182/182 cases`), and `plot_tests_facets` scales each lane to its own suite
  size so heterogeneous arms aren't dwarfed.
- **Adaptive `$`-label precision** — figure cost labels show cents below \$10, so small-per-session
  sweeps don't collapse to `$2`.
- **Sandboxed agent runner (`aet run` / `aet.runner`)** — a single Claude Code invocation launched
  inside a deny-by-default `aet.isolation` bwrap sandbox, streamed to a transcript, recorded as a
  `RunTrajectory`, and materialized into a canonical aet run (manifest + `logs/` +
  `metrics/trajectory.json`) that `aet runs`/`aet show`/`aet plot` read directly. Closes
  *sandbox-run → record → plot* for any project. `--sandbox none` needs `--allow-unsandboxed`;
  `--agent-cmd` overrides the `claude` command for custom launchers / dummy runs.
- **Rate-limit watchdog + auto-resume (`aet.ratelimit`, `aet run --resume`)** — unattended runs
  survive the Claude five-hour usage limit: on a rejected-with-no-work invocation the runner
  checkpoints, waits to the exact `resetsAt` (or polls every ~20 min up to a 5h20m cap when the
  epoch is missing) and resumes the same session — never burning the attempt. On the *weekly* limit
  (or an exhausted wait budget) it stops honestly: writes `UNFINISHED.md` + sets manifest
  `status: rate_limited_unfinished` with a `resume_cmd`, surfaced by `aet runs`, so a person or
  another session picks it up with `aet run --resume <run>`. Daemon-free (checkpoint + relaunch),
  and fully testable (injectable spawn/`sleep`/`now`) with no real `claude`, `bwrap`, or 5-hour wait.
- **Generic transcript importer (`aet import --source transcript`)** — the repo-agnostic default:
  ingests one *or many* Claude Code `*.jsonl` files into one trajectory with zero project-specific
  code. Handles both on-disk shapes — CLI `stream-json` (billed cost, exact) and desktop/app session
  logs (no `result` event → provisional list-price cost) — orders multiple session files by first
  timestamp, and records an optional terminal `--pass/--fail` verdict + milestone (e.g. abc-testing's
  `functional_pass`). Verified on the recovered abc9/abc11 desktop sessions and the abc4 CLI arms.
- **Presentation comparison figures (`aet plot --kind`, `aet.viz.comparison`)** — the polished N-arm
  views, consuming only the data-model: `rate-panels` (per-arm token-rate panels, each on its own
  time scale with a below-axis fixed-duration ruler, activity bands, gold milestones, corner chip),
  `cost-vs-time` (one labeled cumulative-spend line per arm), and `tests-facets` (small-multiple
  tests-passing step-lanes; degrades gracefully to the final verdict when there's no over-time
  signal). `series_styles(n)` gives repo-agnostic per-arm colour/marker/dash identity. `--out` writes
  a `.png` and its sibling `.svg`; `compare --plots` renders the full set.
- **`claude_stream` robustness** — the parser now tolerates string content-blocks (desktop/app
  session logs carry a `user` message's `content` as a plain string), so real session logs import
  without crashing.
- **Agentic trajectory recording (`aet.trajectory`)** — canonical, repo-agnostic record of what an
  agent did over time: cumulative tokens (input/output/cache), cumulative cost, an activity timeline
  (thinking / reading / writing / bash / long tool-waits), and external-oracle test-pass milestones.
  `RunTrajectory` is pure-stdlib and built the same way from a completed run or a live stream via
  `append_round` (one code path). Timing comes from `claude_stream.parse_timestamped_stream` (real
  per-tool offsets, superseding the old within-round weighting). The activity classifier is pluggable
  config (`ActivityConfig`/`LongWaitRule`; the verilator/CIRCT long-wait rule is data via
  `capsule_bench_config`, never hardcoded), so the core stays generic.
  - **Native recording** — `EvalRunLogger.log_trajectory_point` / `log_test_milestone` /
    `log_round_boundary` emit the trajectory through the existing tracking primitives, so it is
    reconstructable from canonical `logs/` (`RunTrajectory.from_run_dir`), plus a
    `metrics/trajectory.json` fast-path artifact.
  - **Importer** — `aet import --source capsule-bench --raw <dir> [--into <run>]` ingests existing
    agentic runs (transcripts + qa verdicts + selfcheck log) into a canonical trajectory; `--into`
    materializes a full aet run so old data is queryable via `aet runs`/`aet show`/`aet plot`.
    Handles per-round `wall_offset_s` resets in the self-check log (cumulative-clock reconstruction).
  - **Live monitor** — `aet monitor --attach <transcript>` tails an in-flight `stream-json` transcript,
    updating the same data-model incrementally; cost is `~$…(provisional)` until the terminal result
    event, then flips to the billed number. Headless-first (one rewriting status line).
- **Visualization (`aet.viz`, optional `[viz]` extra)** — house-style trajectory plots consuming only
  the data-model. `aet plot <run|json> [--comparison …]` and `compare --plots` render per-run and
  stacked comparison figures (cumulative tokens on a log axis, spend twin-axis, activity-share
  background bands, gold test-pass milestones). matplotlib/numpy stay behind the extra with a friendly
  `pip install 'aet[viz]'` hint; `import`/`monitor` work without it.
- **`claude_stream` correctness (for full session-log transcripts)** — the parser now dedups
  re-emitted assistant messages by id (session logs emit the same message 2–3× with identical
  usage) so tokens are counted once, and consumers can split a transcript at each `result` event
  (a file may concatenate several invocations). Together these make imported token/cost totals
  match the authoritative per-model billing exactly. `TurnUsage.has_thinking` added.
- **Isolation & integrity (`aet.isolation`)** — reusable, project-agnostic filesystem isolation for
  agentic runs. `SandboxSpec`/`bwrap_argv`/`wrap_command` build a deny-by-default bubblewrap allow-list
  (agent sees only granted files + tools; answers, sibling runs, and other projects masked; per-file
  `/dev/null` masking; DNS + nested-session-env handling; permission-safe on locked dirs).
  `AuditPolicy`/`audit_run` is a post-run allow-list transcript check (hard cheats vs soft out-of-scope vs
  review-warnings). `file_access_ledger` enumerates every file the agent touched and what it did. See
  `docs/isolation.md`. Extracted from the gemmini agentic A/B harness.
- **Multi-run statistics** — `compare()` now writes `statistical_comparison.md` with Welch's
  t-test, 95% confidence intervals, and Cohen's d effect size for every key metric across
  methods. Significance markers (`***` / `**` / `*` / `ns`) included.
- **Structured rubric scoring** — `RubricCriterion` dataclass, `compute_weighted_score`, and
  `validate_rubric` in `aet.core.rubric`. `EvalRunLogger.log_rubric_score()` fans out to local
  JSONL and MLflow.
- **Trajectory similarity** — `jaccard_similarity` and `sequence_edit_distance` (Levenshtein)
  in `aet.core.metrics`. `compare()` writes `trajectory_similarity.md` pairwise Jaccard matrix
  when `tool_sequence` is present in run summaries.
- **Context window utilization** — `turn.context_pct_used` step metric, `aet.context.max_pct_used`
  summary metric, and `aet.context.high_utilization_warning` event (>80% threshold). OTel
  inference spans carry `aet.turn.context_pct_used` attribute.
- **Baseline / regression detection** — `aet baseline set/show` CLI subcommand stores a
  reference run's `summary_metrics.json` under `baselines/<suite>/baseline.json`. Subsequent
  `compare()` calls write `regression_report.md` flagging runs where cost >1.2× baseline or
  score <baseline−0.05.
- **`aet runs`** and **`aet show`** CLI subcommands for listing and inspecting recorded runs.

### Changed
- **Lint-clean + enforced**: repo is `ruff`-clean; `[tool.ruff]`/`[tool.pytest.ini_options]` pinned in
  `pyproject.toml`; the `all` extra now composes the other extras (single source of truth) + a new
  `[docs]` extra.
- **De-branded viz API**: `use_merlin_style()` → `use_house_style()` (deprecated alias kept).
- **CLI de-godded**: `cli/main.py` (1379 LOC) split into a thin argparse table + `cli/_common.py` +
  `cli/commands/{lifecycle,reporting,trajectory}.py`. No behavior change.
- **Logging**: tracking warnings emit via `logging` (silent by default) instead of raw `print`.

### Docs
- Rewritten `README.md` (the real record→plot / sandboxed-run surface), a root `AGENTS.md`
  (architecture map + "how to add X" recipes + Definition-of-Done), `docs/ARCHITECTURE.md`, ADRs under
  `docs/adr/`, and an auto-generated API site (mkdocstrings). `tests/test_docs.py` + `mkdocs build
  --strict` in CI keep docs from drifting.

## [0.1.0] — 2026-05-01

### Added
- Initial release: `default` and `targetgen` suites, `EvalRunLogger` with local / MLflow /
  OTel backends, `aet init-project`, `aet init-run`, `aet validate`, `aet compare`,
  `aet run-suite`.
- SigNoz observability stack (`docker-compose.observability.yml`) with OTel Collector,
  Jaeger, Prometheus, and Grafana.
- Per-turn token / cost / cache breakdown; per-tool-call timing spans with GenAI semconv
  attributes; OpenLLMetry auto-instrumentation.
- `ClaudeStreamResult` parser for Claude Code JSONL stream output.
- Ray backend for parallel sweep execution.
