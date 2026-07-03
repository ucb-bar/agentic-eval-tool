"""Shared CLI helpers — arg groups, project-root resolution, figure/trajectory IO."""
from __future__ import annotations

import argparse
from pathlib import Path



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


def _load_trajectory(path: Path):
    """A trajectory.json file, or a run dir (fast-path artifact else logs/ reconstruction)."""
    from aet.trajectory.model import RunTrajectory
    path = Path(path)
    if path.is_file():
        return RunTrajectory.from_json(path)
    return RunTrajectory.from_run_dir(path)


def _save_fig(fig, out: Path, dpi: int) -> list[Path]:
    """Save a figure to ``out``; if the extension is .png/.svg, ALSO write the sibling format
    (dual output, matching the reference figure pipeline). Returns the paths written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    written = [out]
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if out.suffix.lower() == ".png":
        svg = out.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")
        written.append(svg)
    return written

