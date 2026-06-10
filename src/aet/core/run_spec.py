from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RunSpec:
    project: str
    suite: str
    method: str
    seed: int
    run_id: str | None = None
    project_root: Path = field(default_factory=Path)
    tracking_mode: str = "local"
    target: str | None = None
    model: str | None = None
    dtype: str | None = None
    substrate: str | None = None
    execution: str = "local"
    is_smoke_test: bool = True
    budget: str = "cheap_smoke"
    promotion_flag: bool = False
    force: bool = False
    mlflow_tracking_uri: str | None = None
    experiment_name: str | None = None
    otel_endpoint: str | None = None
    extra: dict = field(default_factory=dict)
    # Hardware benchmark extensions
    benchmark: str | None = None
    variant: str | None = None
    tool_tier: str | None = None
    spec_version_hash: str | None = None
    tb_version_hash: str | None = None
    repo_initial_commit: str | None = None
    repo_final_commit: str | None = None
