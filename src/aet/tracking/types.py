"""Shared types for the aet tracking layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACKING_MODES = ("local", "mlflow", "full", "debug")


@dataclass
class TrackingConfig:
    mode: str = "local"
    run_id: str = ""
    project: str = ""
    suite: str = ""
    target: str = ""
    method: str = ""
    seed: int = 0
    run_path: Path = field(default_factory=Path)
    mlflow_tracking_uri: str | None = None
    experiment_name: str | None = None
    otel_endpoint: str | None = None
    service_name: str = "aet"
