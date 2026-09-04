"""Machine-readable exports derived from canonical inference records."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from aet.trajectory.model import RunTrajectory


def inference_rows(traj: RunTrajectory) -> list[dict]:
    """Return per-request rows with measured counters and labelled derivations."""
    rows = []
    for record in traj.inferences:
        row = asdict(record)
        duration = record.duration_s
        row.update({
            "duration_s": duration,
            "input_tokens_per_s": record.input_tokens / duration if duration else None,
            "output_tokens_per_s": record.output_tokens / duration if duration else None,
            "cache_read_tokens_per_s": record.cache_read_tokens / duration if duration else None,
            "cache_write_tokens_per_s": record.cache_write_tokens / duration if duration else None,
            "cache_hit_ratio": record.cache_hit_ratio,
            "context_occupancy_ratio": record.context_occupancy_ratio,
            "context_occupancy_provenance": (
                "derived_from_reported_tokens_and_configured_window"
                if record.context_occupancy_ratio is not None else "unavailable"
            ),
            "cache_ttl_provenance": (
                "inferred_from_idle_gap_and_cache_transition"
                if record.ttl_inference == "probable_expiry" else "not_inferred"
            ),
        })
        rows.append(row)
    return rows


def activity_rollup(traj: RunTrajectory) -> dict[str, float]:
    """Active seconds by model/tool category; no claim about unobserved wall time."""
    totals: dict[str, float] = {}
    for record in traj.inferences:
        # Reasoning tokens do not reveal reasoning wall-time. Attribute the
        # request span only to model activity unless a measured thinking band
        # exists independently.
        totals["model"] = totals.get("model", 0.0) + record.duration_s
    for band in traj.bands:
        totals[band.category] = totals.get(band.category, 0.0) + band.duration_s
    return totals


def export_agent_profiles(traj: RunTrajectory, stem: str | Path) -> tuple[Path, Path]:
    """Write per-request CSV and hierarchy/rollup JSON next to a profile figure."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".inferences.csv")
    json_path = stem.with_suffix(".agents.json")
    rows = inference_rows(traj)
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    json_path.write_text(json.dumps({
        "schema_version": traj.schema_version,
        "run_id": traj.run_id,
        "agents": traj.per_agent_rollup(),
        "activity_seconds": activity_rollup(traj),
        "provenance": {
            "tokens": "provider_reported",
            "activity_share": "derived_from_recorded_active_spans",
            "context_occupancy": "estimated_not_physical_kv_fullness",
            "ttl": "probable_only_when_labelled_probable_expiry",
        },
    }, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path
