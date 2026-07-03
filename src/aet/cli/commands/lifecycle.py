"""Lifecycle commands: init-project, init-run, validate, run-suite."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from aet.core.run_spec import RunSpec
from aet.core.run_manifest import RunManifest
from aet.core.run_paths import RunPaths
from aet.core.command_runner import git_head
from aet.core.file_utils import copy_template
from aet.core.errors import RunAlreadyExistsError
from aet.tracking import EvalRunLogger
from aet.suites import get_suite

from aet.cli._common import (
    _resolve_project_root,
)
from aet.cli.commands.reporting import _cmd_compare


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
            "[aet] No files written (all already exist; use --force to overwrite)"
        )


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
