"""aet.trajectory — canonical recording of what an agent did over time.

The data-model (:class:`RunTrajectory` and its parts) is pure-stdlib and dependency-free. It is
built the same way from a completed run (batch importer) and from a live stream, then consumed
by the native recorder, the ``aet monitor`` live view, and the ``aet.viz`` plots.
"""
from __future__ import annotations

from aet.trajectory.model import (
    RunTrajectory,
    TrajectoryPoint,
    ActivityBand,
    TestMilestone,
    RoundBoundary,
    SCHEMA_VERSION,
)
from aet.trajectory.classify import (
    ActivityClassifier,
    ActivityConfig,
    LongWaitRule,
    capsule_bench_config,
    DEFAULT_WEIGHTS,
)
from aet.trajectory.pricing import PriceTable

__all__ = [
    "RunTrajectory",
    "TrajectoryPoint",
    "ActivityBand",
    "TestMilestone",
    "RoundBoundary",
    "SCHEMA_VERSION",
    "ActivityClassifier",
    "ActivityConfig",
    "LongWaitRule",
    "capsule_bench_config",
    "DEFAULT_WEIGHTS",
    "PriceTable",
]
