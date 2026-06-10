# Changelog

All notable changes to `aet` are recorded here.

## [Unreleased]

### Added
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
