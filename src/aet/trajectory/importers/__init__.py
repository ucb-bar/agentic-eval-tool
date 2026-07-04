"""Importers — ingest EXISTING agentic runs into a canonical :class:`RunTrajectory`.

Each importer knows one on-disk run layout and reproduces its trajectory using the canonical
parser + builder. Register new layouts in ``IMPORTER_REGISTRY``.
"""
from __future__ import annotations

from aet.core.errors import AetError
from aet.trajectory.importers.capsule_bench import import_run as _import_capsule_bench
from aet.trajectory.importers.otel import import_otel as _import_otel
from aet.trajectory.importers.transcript import import_transcript as _import_transcript

IMPORTER_REGISTRY = {
    "capsule-bench": _import_capsule_bench,
    "transcript": _import_transcript,
    # full-fidelity: an OTLP logs capture (otel_logs.jsonl from `aet otel-sink`) → real per-turn
    # tokens/cost/duration + cache, no interpolation. Tolerates the generic import kwargs it does
    # not use (classifier_config/circt/milestone_time) via **_ignored in its signature.
    "otel": _import_otel,
}


def get_importer(source: str):
    try:
        return IMPORTER_REGISTRY[source]
    except KeyError:
        known = ", ".join(sorted(IMPORTER_REGISTRY))
        raise AetError(f"unknown import source {source!r}; known sources: {known}")


__all__ = ["IMPORTER_REGISTRY", "get_importer"]
