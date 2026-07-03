# AGENTS.md — orientation for agents (and humans) working *on* aet

Read this first. It is the map + the rules + the recipes. Keep it short; if it grows, move detail
into `docs/`. (This file is for working on `aet`; the `README.md` is for *using* it.)

## What aet is
A repo-agnostic package that turns agentic runs into a canonical **`RunTrajectory`** (tokens, cost,
activity, test milestones over time), then compares/plots them, and can run agents sandboxed +
rate-limit-resilient. Project specifics never live in `aet` — they come in as config/data.

## Architecture map (where things live)
```
src/aet/
  trajectory/     the spine. model.py (RunTrajectory) · build.py (append_round) ·
                  classify.py (activity classifier + config factories) · importers/ (registry) ·
                  oracle.py (mine tests-passing climb from a transcript) · stream.py (live) ·
                  recording.py (emit/materialize a run)
  viz/            figures (behind [viz]): style.py (house style) · comparison.py (the 3 presentation
                  figures) · trajectory_plot.py (single/stacked)
  tracking/       EvalRunLogger facade → local_backend / mlflow_backend / otel_backend
  isolation/      bwrap sandbox (sandbox.py) + audit.py + ledger.py
  runner.py       sandboxed recorded agent run  ·  ratelimit.py  five-hour/weekly watchdog
  suites/         evaluation suites (registry): default/, targetgen/ (bundled example, Merlin-specific)
  cli/            main.py (argparse table + dispatch) → _common.py + commands/{lifecycle,reporting,trajectory}
  core/           run spec/manifest/paths, hashing, errors, artifact store
```
Data flow: `transcript.jsonl → import_transcript → RunTrajectory → (emit_trajectory) run dir → aet.viz`.

## Invariants (don't break these)
- **Repo-agnostic core.** No project names/paths in `src/aet/**` except the clearly-labelled
  `suites/targetgen/` bundled example. Project rules come via config (e.g. `ActivityConfig`,
  `classify.spec_to_rtl_config()`), never hardcoded in the harness.
- **`RunTrajectory` is pure stdlib** (`trajectory/model.py`) — no numpy. Array math lives in `aet.viz`.
- **matplotlib/numpy only behind `[viz]`** — import them via `aet.viz.style`'s guarded import; core
  commands (`import`, `monitor`, `run`) must work without `[viz]`.
- **Extension points are registries**, not if/else chains (see recipes).
- **Tests live in `tests/`**, one `test_<area>.py` per module; run `pytest` (config in `pyproject`).
- **CLI help is the doc** — every subcommand has rich argparse `help=`; don't restate it in prose.

## How to add X (point at the seam; don't duplicate)
- **A transcript importer** → add `import_<x>(...) -> RunTrajectory` under
  `trajectory/importers/`, register it in `IMPORTER_REGISTRY` (`importers/__init__.py`). It becomes
  `aet import --source <x>` for free.
- **A figure kind** → add `plot_<x>(trajs, labels)` to `viz/comparison.py`, then wire the name into
  `cli/commands/trajectory.py` (`_cmd_plot`'s `--kind` + `_SESSION_KINDS`).
- **A suite** → subclass the suite base under `suites/<name>/`, register in `suites/__init__.py`'s
  `get_suite`. It becomes `aet init-project/run-suite --suite <name>`.
- **A tracking backend** → implement the backend, wire it in `tracking/run_logger.py`'s facade.
- **A classifier rule** (e.g. a new long-wait tool) → a `classify.ActivityConfig` factory
  (see `capsule_bench_config`, `spec_to_rtl_config`) — DATA, never harness source.

## Definition of Done (a change isn't done until)
1. `pytest` green and `ruff check src/ tests/` clean (config pinned in `pyproject.toml`).
2. Docstrings updated on any touched public symbol — explain **why**, not what.
3. `CHANGELOG.md` `[Unreleased]` gets an entry.
4. If you named a new CLI verb / subsystem, `tests/test_docs.py` still passes (it asserts every CLI
   subcommand named in the README resolves).
5. New public API is reachable from the docs (`mkdocs build --strict` stays green — it fails on a
   broken/renamed reference, which is the anti-drift backstop).

## Known follow-ups
- `tracking/run_logger.py` (~1150 LOC, one **cohesive** `EvalRunLogger` facade). Big but single-
  responsibility, so it is not a blocker. A mixin split is **not** recommended — mixins don't reduce
  the real coupling (every method needs the facade's `self._local/_mlflow/_otel` state) and add MRO
  indirection. If trimming: extract the pure report-**writers** (`write_summary_metrics`,
  `write_eval_report`, `write_metrics_structured`, `write_run_record`) into a `tracking/reports.py`
  of free functions — low-risk, reduces size *and* coupling.
- Ray executor: `execution/ray_backend.py` is a `NotImplementedError` skeleton + a heavy `[ray]`
  extra, now undocumented. Candidate for full removal (backend + extra + `--execution ray` choice +
  its test) so the package advertises only what runs.
