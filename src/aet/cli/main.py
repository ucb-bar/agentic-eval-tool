"""aet CLI — main entry point."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from aet.core.run_spec import RunSpec
from aet.core.run_manifest import RunManifest
from aet.core.run_paths import RunPaths
from aet.core.command_runner import git_head
from aet.core.file_utils import copy_template
from aet.core.errors import AetError, SuiteNotFoundError, RunAlreadyExistsError
from aet.tracking import EvalRunLogger
from aet.suites import get_suite


# ---------------------------------------------------------------------------
# Shared argument-group helpers
# ---------------------------------------------------------------------------

def _add_global_args(p: argparse.ArgumentParser) -> None:
    """Add --project-root and --execution flags."""
    p.add_argument(
        "--project-root",
        metavar="PATH",
        default=None,
        help="Root of the project (default: current working directory)",
    )
    p.add_argument(
        "--execution",
        choices=["local", "ray"],
        default="local",
        help="Execution backend (default: local)",
    )


def _add_tracking_args(p: argparse.ArgumentParser) -> None:
    """Add all tracking-related flags."""
    p.add_argument(
        "--tracking",
        metavar="MODE",
        dest="tracking_mode",
        choices=["local", "mlflow", "full", "debug"],
        default="local",
        help="Tracking mode (default: local)",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        metavar="URI",
        default=None,
        help="MLflow tracking URI",
    )
    p.add_argument(
        "--experiment-name",
        metavar="NAME",
        default=None,
        help="MLflow experiment name",
    )
    p.add_argument(
        "--otel-endpoint",
        metavar="URL",
        default=None,
        help="OpenTelemetry collector endpoint",
    )


def _resolve_project_root(args) -> Path:
    if args.project_root:
        return Path(args.project_root).resolve()
    return Path.cwd()


# ---------------------------------------------------------------------------
# init-project
# ---------------------------------------------------------------------------

def _cmd_init_project(args) -> None:
    project_root = _resolve_project_root(args)
    force = args.force
    _TEMPLATE_DIRS = {"default": "project", "targetgen": "targetgen_project"}
    template = args.template
    template_dir = Path(__file__).parent.parent / "templates" / _TEMPLATE_DIRS.get(template, template)
    if not template_dir.exists():
        print(
            f"[aet] Error: template {template!r} not found at {template_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    created = copy_template(template_dir, project_root, force=force)
    if created:
        print(f"[aet] Initialized project from template '{template}' at {project_root}")
        for path in created:
            print(f"  created: {path.relative_to(project_root)}")
    else:
        print(
            f"[aet] No files written (all already exist; use --force to overwrite)"
        )


# ---------------------------------------------------------------------------
# init-run helpers and command
# ---------------------------------------------------------------------------

def _do_init_run(args, method: str, seed: int, parent_run_id: str | None = None) -> Path:
    """Core init-run logic. Returns the run_path created."""
    project_root = _resolve_project_root(args)

    spec = RunSpec(
        project=project_root.name,
        suite=args.suite,
        method=method,
        seed=seed,
        project_root=project_root,
        tracking_mode=args.tracking_mode,
        target=getattr(args, "target", None),
        model=getattr(args, "model", None),
        dtype=getattr(args, "dtype", None),
        substrate=getattr(args, "substrate", None),
        execution=args.execution,
        is_smoke_test=args.smoke,
        budget=args.budget,
        force=args.force,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
    )

    run_id = f"{date.today().isoformat()}_{method}_seed{seed:03d}"
    paths = RunPaths.from_spec(spec, run_id)

    if paths.run_path.exists() and not args.force:
        raise RunAlreadyExistsError(
            f"Run directory already exists: {paths.run_path}\n"
            "Use --force to overwrite."
        )

    # Create all required directories
    for d in (
        paths.run_path,
        paths.logs,
        paths.metrics,
        paths.generated,
        paths.patches,
        paths.contracts,
        paths.artifacts_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    git_hash = git_head(project_root)

    manifest = RunManifest.create(spec, run_id, git_hash)
    manifest.dump(paths.run_path / "run_manifest.yaml")

    logger = EvalRunLogger.start(
        project=spec.project,
        suite=spec.suite,
        target=spec.target or "",
        method=method,
        seed=seed,
        run_id=run_id,
        run_path=paths.run_path,
        tracking_mode=args.tracking_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
        parent_run_id=parent_run_id,
    )

    suite = get_suite(args.suite)
    with logger.start_run_span(f"{spec.suite}-init"):
        suite.init_run(spec, paths, logger)
    logger.finish(status="initialized")

    return paths.run_path


def _cmd_init_run(args) -> None:
    run_path = _do_init_run(args, method=args.method, seed=args.seed)
    print(run_path)


# ---------------------------------------------------------------------------
# validate helpers and command
# ---------------------------------------------------------------------------

def _do_validate(run_path: Path, args, parent_run_id: str | None = None) -> dict:
    """Core validate logic. Returns the validation report dict."""
    project_root = _resolve_project_root(args)

    manifest = RunManifest.load(run_path / "run_manifest.yaml")

    spec = RunSpec(
        project=manifest.project,
        suite=manifest.suite,
        method=manifest.method,
        seed=manifest.seed,
        run_id=manifest.run_id,
        project_root=project_root,
        tracking_mode=args.tracking_mode,
        target=manifest.target,
        model=manifest.model,
        dtype=manifest.dtype,
        substrate=manifest.substrate,
        execution=args.execution,
        is_smoke_test=manifest.is_smoke_test,
        budget=manifest.budget,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
    )

    paths = RunPaths.from_run_dir(run_path, project_root)

    logger = EvalRunLogger.start(
        project=spec.project,
        suite=spec.suite,
        target=spec.target or "",
        method=spec.method,
        seed=spec.seed,
        run_id=spec.run_id or manifest.run_id,
        run_path=paths.run_path,
        tracking_mode=args.tracking_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
        parent_run_id=parent_run_id,
    )

    suite = get_suite(manifest.suite)
    with logger.start_run_span(f"{manifest.suite}-eval"):
        report = suite.validate(spec, paths, logger)
        try:
            suite.collect_metrics(spec, paths, logger)
        except Exception:
            pass
    final_status = report.get("overall") or report.get("status") or "unknown"
    logger.finish(status=final_status)

    return report


def _cmd_validate(args) -> None:
    run_path = Path(args.run_path).resolve()
    report = _do_validate(run_path, args)

    status = report.get("overall") or report.get("status") or "unknown"
    total_errors = report.get("total_errors", len(report.get("errors", [])))
    print(f"[aet] Validation complete: status={status}, total_errors={total_errors}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _cmd_compare(args) -> None:
    project_root = _resolve_project_root(args)
    suite_name = args.suite

    if args.output_dir:
        report_dir = Path(args.output_dir).resolve()
    else:
        report_dir = project_root / "reports" / suite_name

    runs_root = project_root / "runs" / suite_name
    if runs_root.exists():
        run_paths = [p for p in runs_root.glob("*") if p.is_dir()]
    else:
        run_paths = []

    # Optionally filter smoke-test runs
    if getattr(args, "no_smoke", False):
        filtered = []
        for rp in run_paths:
            manifest_path = rp / "run_manifest.yaml"
            if manifest_path.exists():
                try:
                    m = RunManifest.load(manifest_path)
                    if not m.is_smoke_test:
                        filtered.append(rp)
                except Exception:
                    filtered.append(rp)
            else:
                filtered.append(rp)
        run_paths = filtered

    report_dir.mkdir(parents=True, exist_ok=True)

    # Minimal logger for compare (no run-specific context needed)
    import logging
    logger = logging.getLogger("aet.compare")

    suite = get_suite(suite_name)
    suite.compare(run_paths, report_dir, logger)

    print(f"[aet] Compare complete: wrote report to {report_dir}")


# ---------------------------------------------------------------------------
# run-suite
# ---------------------------------------------------------------------------

def _cmd_run_suite(args) -> None:
    if args.execution == "ray":
        raise NotImplementedError(
            "Ray execution backend is not yet implemented. "
            "Use --execution local to run locally."
        )

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    combos = [(method, seed) for method in methods for seed in seeds]
    project_root = _resolve_project_root(args)

    # Create a sweep-level parent run in MLflow so all child runs nest under it.
    sweep_run_id = f"sweep_{date.today().isoformat()}"
    sweep_run_path = project_root / "runs" / args.suite / "_sweep"
    sweep_run_path.mkdir(parents=True, exist_ok=True)
    sweep_logger = EvalRunLogger.start(
        project=project_root.name,
        suite=args.suite,
        target=getattr(args, "target", None) or "",
        method="sweep",
        seed=0,
        run_id=sweep_run_id,
        run_path=sweep_run_path,
        tracking_mode=args.tracking_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
    )
    sweep_parent_id = sweep_logger.mlflow_run_id
    if sweep_parent_id:
        print(f"[aet] sweep parent run: {sweep_parent_id}")

    run_paths: list[Path] = []
    for method, seed in combos:
        print(f"[aet] init-run: suite={args.suite} method={method} seed={seed}")
        run_path = _do_init_run(args, method=method, seed=seed, parent_run_id=sweep_parent_id)
        run_paths.append(run_path)

        print(f"[aet] validate: {run_path}")
        report = _do_validate(run_path, args, parent_run_id=sweep_parent_id)
        status = report.get("overall") or report.get("status") or "unknown"
        total_errors = report.get("total_errors", len(report.get("errors", [])))
        print(f"  status={status}, total_errors={total_errors}")

    sweep_logger.finish(status="sweep_complete")
    print(f"[aet] All {len(combos)} run(s) complete. Running compare...")
    _cmd_compare(args)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def _baseline_dir(project_root: Path, suite: str) -> Path:
    return project_root / "baselines" / suite


def _cmd_baseline_set(args) -> None:
    import json as _json
    project_root = _resolve_project_root(args)
    suite = args.suite
    baseline_path = _baseline_dir(project_root, suite) / "baseline.json"

    runs_root = project_root / "runs" / suite
    if not runs_root.exists():
        print(f"[aet] Error: no runs directory for suite {suite}: {runs_root}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "run_id", None):
        run_dir = runs_root / args.run_id
        summary_path = run_dir / "metrics" / "summary_metrics.json"
        if not summary_path.exists():
            print(f"[aet] Error: summary_metrics.json not found at {summary_path}", file=sys.stderr)
            sys.exit(1)
        summary = _json.loads(summary_path.read_text())
    else:
        best_summary = None
        best_score = float("-inf")
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            sp = run_dir / "metrics" / "summary_metrics.json"
            if not sp.exists():
                continue
            try:
                s = _json.loads(sp.read_text())
                score = s.get("task_achievement_score")
                if score is not None and float(score) > best_score:
                    best_score = float(score)
                    best_summary = s
            except Exception:
                pass
        if best_summary is None:
            print(f"[aet] Error: no valid runs found for suite {suite}", file=sys.stderr)
            sys.exit(1)
        summary = best_summary

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(_json.dumps(summary, indent=2))
    print(f"[aet] Baseline set: {baseline_path}")


def _cmd_baseline_show(args) -> None:
    import json as _json
    project_root = _resolve_project_root(args)
    suite = args.suite
    baseline_path = _baseline_dir(project_root, suite) / "baseline.json"
    if not baseline_path.exists():
        print(f"[aet] No baseline found for suite {suite} at {baseline_path}", file=sys.stderr)
        sys.exit(1)
    print(baseline_path.read_text())


def _cmd_baseline(args) -> None:
    if not hasattr(args, "baseline_func"):
        args._baseline_parser.print_help()
        sys.exit(0)
    args.baseline_func(args)


# ---------------------------------------------------------------------------
# runs  (list all runs under a project)
# ---------------------------------------------------------------------------

def _cmd_runs(args) -> None:
    import json

    project_root = _resolve_project_root(args)
    runs_root = project_root / "runs"
    suite_filter = getattr(args, "suite", None)

    run_dirs: list[Path] = []
    if runs_root.exists():
        for suite_dir in sorted(runs_root.iterdir()):
            if not suite_dir.is_dir():
                continue
            if suite_filter and suite_dir.name != suite_filter:
                continue
            for run_dir in sorted(suite_dir.iterdir()):
                if run_dir.is_dir() and not run_dir.name.startswith("_"):
                    run_dirs.append(run_dir)

    if not run_dirs:
        print("[aet] No runs found" + (f" for suite '{suite_filter}'" if suite_filter else "") + ".")
        return

    fmt = getattr(args, "format", "table")

    rows = []
    for run_dir in run_dirs:
        suite = run_dir.parent.name
        run_id = run_dir.name
        status = "?"
        model = ""
        cost = None
        turns = None
        tokens_in = None
        trace_id = None
        mlflow_run = None

        report_path = run_dir / "validation_report.json"
        if report_path.exists():
            try:
                r = json.loads(report_path.read_text())
                status = r.get("status", "?")
            except Exception:
                pass

        params_path = run_dir / "logs" / "params.json"
        if params_path.exists():
            try:
                p = json.loads(params_path.read_text())
                model = p.get("gen_ai.response.model", "")
                mlflow_run = p.get("mlflow_run_id", "")
                trace_id = p.get("aet.otel_trace_id", "")
            except Exception:
                pass

        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            try:
                for line in metrics_path.read_text().splitlines():
                    m = json.loads(line)
                    if m["name"] == "aet.agent.cost_usd":
                        cost = m["value"]
                    elif m["name"] == "aet.agent.num_turns":
                        turns = int(m["value"])
                    elif m["name"] == "gen_ai.usage.input_tokens":
                        tokens_in = int(m["value"])
            except Exception:
                pass

        rows.append({
            "suite": suite, "run_id": run_id, "status": status,
            "model": model, "turns": turns, "tokens_in": tokens_in,
            "cost": cost, "trace_id": trace_id, "mlflow_run": mlflow_run,
        })

    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return

    STATUS_SYM = {"pass": "✓", "partial": "~", "error": "✗", "unknown": "?", "?": "?"}
    col_w = max(len(r["run_id"]) for r in rows)
    print(f"\n  {'':2}  {'suite':<14}  {'run_id':<{col_w}}  {'model':<24}  {'turns':>5}  {'tokens_in':>10}  {'cost':>10}")
    print(f"  {'':2}  {'-'*14}  {'-'*col_w}  {'-'*24}  {'-'*5}  {'-'*10}  {'-'*10}")
    for r in rows:
        sym = STATUS_SYM.get(r["status"], "?")
        cost_s = f"${r['cost']:.4f}" if r["cost"] is not None else "—"
        turns_s = str(r["turns"]) if r["turns"] is not None else "—"
        tokens_s = str(r["tokens_in"]) if r["tokens_in"] is not None else "—"
        model_s = (r["model"] or "—")[:24]
        print(f"  {sym:<2}  {r['suite']:<14}  {r['run_id']:<{col_w}}  {model_s:<24}  {turns_s:>5}  {tokens_s:>10}  {cost_s:>10}")
    print()


# ---------------------------------------------------------------------------
# show  (dump all data for a single run)
# ---------------------------------------------------------------------------

def _cmd_show(args) -> None:
    import json

    run_path = Path(args.run_path).resolve()
    if not run_path.exists():
        print(f"[aet] Run path not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    fmt = getattr(args, "format", "text")

    # ── load all local data ────────────────────────────────────────────────
    params: dict = {}
    params_path = run_path / "logs" / "params.json"
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
        except Exception:
            pass

    metrics: list[dict] = []
    metrics_path = run_path / "logs" / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except Exception:
                pass

    events: list[dict] = []
    events_path = run_path / "logs" / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                pass

    report: dict = {}
    report_path = run_path / "validation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass

    if fmt == "json":
        print(json.dumps({"params": params, "metrics": metrics,
                          "events": events, "report": report}, indent=2))
        return

    # ── text display ───────────────────────────────────────────────────────
    SEP = "─" * 60

    print(f"\n{SEP}")
    print(f"  Run: {run_path.name}   suite: {run_path.parent.name}")
    print(SEP)

    # params
    if params:
        print("\n  PARAMS")
        for k, v in params.items():
            print(f"    {k:<40} {v}")

    # metrics grouped
    if metrics:
        print("\n  METRICS")
        groups: dict[str, list] = {}
        for m in metrics:
            cat = m["name"].split(".")[0]
            groups.setdefault(cat, []).append(m)

        # cost block first
        for cat in ["aet", "gen_ai", "cost_usd", "input_tokens", "output_tokens",
                    "cache_read_tokens", "cache_creation_tokens",
                    "turn", "task_achievement_score", "validation_errors"]:
            if cat not in groups:
                continue
            print(f"\n    [{cat}]")
            for m in groups.pop(cat):
                step = f"  step={m['step']}" if m.get("step") is not None else ""
                val = m["value"]
                if isinstance(val, float) and "cost" in m["name"]:
                    val = f"${val:.6f}"
                print(f"      {m['name']:<52} {val}{step}")
        for cat, ms in groups.items():
            print(f"\n    [{cat}]")
            for m in ms:
                step = f"  step={m['step']}" if m.get("step") is not None else ""
                print(f"      {m['name']:<52} {m['value']}{step}")

    # events timeline
    if events:
        print("\n  EVENTS")
        for e in events:
            ts = e.get("ts", "")[:19].replace("T", " ")
            print(f"\n    {ts}  {e['event']}")
            for k, v in (e.get("payload") or {}).items():
                vs = str(v)
                if len(vs) > 120:
                    vs = vs[:120] + "…"
                print(f"      {k}: {vs}")

    # validation
    if report:
        print(f"\n  VALIDATION  →  {report.get('status', '?').upper()}")
        errs = report.get("errors", [])
        if errs:
            for e in errs:
                print(f"    ✗  {e}")
        else:
            print("    ✓  no errors")

    # links
    print("\n  LINKS")
    trace_id = params.get("aet.otel_trace_id", "")
    mlflow_run_id = params.get("mlflow_run_id", "")
    mlflow_uri = params.get("mlflow_tracking_uri", "")
    mlflow_exp = params.get("mlflow_experiment_name", "")
    if trace_id:
        print(f"    signoz   http://localhost:8080/trace/{trace_id}")
    if mlflow_run_id and mlflow_uri:
        print(f"    mlflow   {mlflow_uri}/#/experiments/1/runs/{mlflow_run_id}")
    session_id = params.get("gen_ai.conversation.id", "")
    if session_id:
        print(f"    replay   claude --resume {session_id}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aet",
        description="Agentic Eval Tool — research evaluation harness",
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # ------------------------------------------------------------------
    # init-project
    # ------------------------------------------------------------------
    p_init_project = subparsers.add_parser(
        "init-project",
        help="Initialize a new project from a template",
    )
    p_init_project.add_argument(
        "--template",
        choices=["default", "targetgen"],
        required=True,
        help="Template to use",
    )
    p_init_project.add_argument(
        "--project-root",
        metavar="PATH",
        default=None,
        help="Destination directory (default: current working directory)",
    )
    p_init_project.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing files",
    )
    p_init_project.set_defaults(func=_cmd_init_project)

    # ------------------------------------------------------------------
    # init-run
    # ------------------------------------------------------------------
    p_init_run = subparsers.add_parser(
        "init-run",
        help="Initialize a new evaluation run",
    )
    p_init_run.add_argument("--suite", required=True, help="Evaluation suite name")
    p_init_run.add_argument("--method", required=True, help="Method name")
    p_init_run.add_argument("--seed", required=True, type=int, help="Random seed")
    p_init_run.add_argument("--target", default=None, help="Target hardware/platform")
    p_init_run.add_argument("--model", default=None, help="Model identifier")
    p_init_run.add_argument("--dtype", default=None, help="Data type")
    p_init_run.add_argument("--substrate", default=None, help="Substrate identifier")
    p_init_run.add_argument(
        "--smoke", dest="smoke", action="store_true", default=True,
        help="Mark as smoke test run (default)",
    )
    p_init_run.add_argument(
        "--no-smoke", dest="smoke", action="store_false",
        help="Mark as a full (non-smoke) run",
    )
    p_init_run.add_argument(
        "--budget", default="cheap_smoke", help="Budget identifier (default: cheap_smoke)"
    )
    p_init_run.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing run directory",
    )
    _add_global_args(p_init_run)
    _add_tracking_args(p_init_run)
    p_init_run.set_defaults(func=_cmd_init_run)

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate outputs of a completed run",
    )
    p_validate.add_argument("run_path", metavar="RUN_PATH", help="Path to the run directory")
    _add_global_args(p_validate)
    _add_tracking_args(p_validate)
    p_validate.set_defaults(func=_cmd_validate)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    p_compare = subparsers.add_parser(
        "compare",
        help="Aggregate and compare multiple runs for a suite",
    )
    p_compare.add_argument("--suite", required=True, help="Suite name to compare")
    p_compare.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory for reports (default: <project-root>/reports/<suite>)",
    )
    p_compare.add_argument(
        "--no-smoke",
        dest="no_smoke",
        action="store_true",
        default=False,
        help="Exclude smoke-test runs from comparison",
    )
    _add_global_args(p_compare)
    _add_tracking_args(p_compare)
    p_compare.set_defaults(func=_cmd_compare)

    # ------------------------------------------------------------------
    # run-suite
    # ------------------------------------------------------------------
    p_run_suite = subparsers.add_parser(
        "run-suite",
        help="Run init-run + validate for all method/seed combos, then compare",
    )
    p_run_suite.add_argument("--suite", required=True, help="Suite name")
    p_run_suite.add_argument("--target", default=None, help="Target hardware/platform")
    p_run_suite.add_argument(
        "--methods",
        required=True,
        metavar="m1,m2,...",
        help="Comma-separated list of methods",
    )
    p_run_suite.add_argument(
        "--seeds",
        required=True,
        metavar="1,2,...",
        help="Comma-separated list of seeds (integers)",
    )
    # run-suite also needs init-run style args
    p_run_suite.add_argument("--model", default=None, help="Model identifier")
    p_run_suite.add_argument("--dtype", default=None, help="Data type")
    p_run_suite.add_argument("--substrate", default=None, help="Substrate identifier")
    p_run_suite.add_argument(
        "--smoke", dest="smoke", action="store_true", default=True,
        help="Mark runs as smoke tests (default)",
    )
    p_run_suite.add_argument(
        "--no-smoke", dest="smoke", action="store_false",
        help="Mark runs as full (non-smoke) runs",
    )
    p_run_suite.add_argument(
        "--budget", default="cheap_smoke", help="Budget identifier (default: cheap_smoke)"
    )
    p_run_suite.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing run directories",
    )
    p_run_suite.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory for compare reports (default: <project-root>/reports/<suite>)",
    )
    _add_global_args(p_run_suite)
    _add_tracking_args(p_run_suite)
    p_run_suite.set_defaults(func=_cmd_run_suite)

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    p_runs = subparsers.add_parser(
        "runs",
        help="List all runs in a project",
    )
    p_runs.add_argument("--suite", default=None, help="Filter by suite name")
    p_runs.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    _add_global_args(p_runs)
    p_runs.set_defaults(func=_cmd_runs)

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------
    p_baseline = subparsers.add_parser(
        "baseline",
        help="Set or show a baseline for regression detection",
    )
    p_baseline._baseline_parser = p_baseline
    baseline_sub = p_baseline.add_subparsers(dest="baseline_subcommand")

    p_baseline_set = baseline_sub.add_parser("set", help="Set the baseline for a suite")
    p_baseline_set.add_argument("--suite", required=True, help="Suite name")
    p_baseline_set.add_argument("--run-id", dest="run_id", default=None,
                                help="Run ID to use as baseline (omit to pick best)")
    _add_global_args(p_baseline_set)
    p_baseline_set.set_defaults(baseline_func=_cmd_baseline_set)

    p_baseline_show = baseline_sub.add_parser("show", help="Show the current baseline")
    p_baseline_show.add_argument("--suite", required=True, help="Suite name")
    _add_global_args(p_baseline_show)
    p_baseline_show.set_defaults(baseline_func=_cmd_baseline_show)

    p_baseline.set_defaults(func=_cmd_baseline)

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------
    p_show = subparsers.add_parser(
        "show",
        help="Show all captured data for a single run",
    )
    p_show.add_argument("run_path", metavar="RUN_PATH", help="Path to the run directory")
    p_show.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    p_show.set_defaults(func=_cmd_show)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    try:
        args.func(args)
    except (AetError, SuiteNotFoundError, RunAlreadyExistsError) as e:
        print(f"[aet] Error: {e}", file=sys.stderr)
        sys.exit(1)
