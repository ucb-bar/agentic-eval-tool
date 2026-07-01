# Changelog

All notable changes to `aet` are recorded here.

## [Unreleased]

### Added
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
