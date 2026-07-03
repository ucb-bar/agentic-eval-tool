"""Run-report writers — pure serialization, no backend/logger state.

These build the canonical JSON products of a run (`run_record.json`, `metrics/summary_metrics.json`,
`eval_report.json`, `metrics.json`) from plain values + a target directory. Kept as free functions
(not `EvalRunLogger` methods) so the report *shape* is testable and reusable in isolation, and the
logger facade stays a thin delegator. See :class:`aet.tracking.run_logger.EvalRunLogger`'s
``write_*`` methods.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _dump(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def write_run_record(run_path: Path, *, run_id: str, project: str, suite: str, target,
                     method: str, seed: int, mode: str, extra: dict | None = None) -> Path:
    """`run_record.json` at the run root — identity + provenance of the run."""
    record = {
        "schema_version": "1.1", "run_id": run_id, "project": project, "suite": suite,
        "target": target, "method": method, "seed": seed, "tracking_mode": mode,
        "created_at": _now(),
    }
    if extra:
        record.update(extra)
    return _dump(run_path / "run_record.json", record)


def write_summary_metrics(run_path: Path, *, run_id: str, project: str, suite: str, method: str,
                          seed: int, target, extra: dict | None = None) -> Path:
    """`metrics/summary_metrics.json` — the headline metrics for `aet runs`/`compare`."""
    summary = {
        "run_id": run_id, "project": project, "suite": suite, "method": method, "seed": seed,
        "target": target, "recorded_at": _now(),
    }
    if extra:
        summary.update(extra)
    return _dump(run_path / "metrics" / "summary_metrics.json", summary)


def write_eval_report(run_path: Path, *, run_id: str, tests: list[dict],
                      contracts: list[dict] | None = None, assertions: list[dict] | None = None,
                      coverage: list[dict] | None = None, extra: dict | None = None) -> Path:
    """`eval_report.json` — per-test / per-contract / per-assertion / coverage results."""
    report = {
        "schema_version": "1.0", "run_id": run_id, "generated_at": _now(),
        "tests": tests, "contracts": contracts or [], "assertions": assertions or [],
        "coverage": coverage or [],
    }
    if extra:
        report.update(extra)
    return _dump(run_path / "eval_report.json", report)


def write_metrics_structured(run_path: Path, *, run_id: str, cost: dict, quality: dict,
                             process: dict, extra: dict | None = None) -> Path:
    """`metrics.json` — the structured cost / quality / process breakdown."""
    metrics = {
        "schema_version": "1.0", "run_id": run_id, "generated_at": _now(),
        "cost": cost, "quality": quality, "process": process,
    }
    if extra:
        metrics.update(extra)
    return _dump(run_path / "metrics.json", metrics)
