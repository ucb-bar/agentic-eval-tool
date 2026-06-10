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

def _do_init_run(args, method: str, seed: int) -> Path:
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

def _do_validate(run_path: Path, args) -> dict:
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

    run_paths: list[Path] = []
    for method, seed in combos:
        print(f"[aet] init-run: suite={args.suite} method={method} seed={seed}")
        run_path = _do_init_run(args, method=method, seed=seed)
        run_paths.append(run_path)

        print(f"[aet] validate: {run_path}")
        report = _do_validate(run_path, args)
        status = report.get("overall") or report.get("status") or "unknown"
        total_errors = report.get("total_errors", len(report.get("errors", [])))
        print(f"  status={status}, total_errors={total_errors}")

    print(f"[aet] All {len(combos)} run(s) complete. Running compare...")
    _cmd_compare(args)


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
