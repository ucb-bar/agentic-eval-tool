from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from aet.core.yaml_utils import load_yaml, dump_yaml

@dataclass
class RunManifest:
    schema_version: str = "1.0"
    project: str = ""
    suite: str = ""
    method: str = ""
    seed: int = 0
    run_id: str = ""
    target: str | None = None
    model: str | None = None
    dtype: str | None = None
    substrate: str | None = None
    git_hash_at_init: str = "unknown"
    is_smoke_test: bool = True
    budget: str = "cheap_smoke"
    promotion_flag: bool = False
    created_at: str = ""
    status: str = "initialized"
    tracking: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    #: {input_name: sha256 | None} for every immutable input this run was built from. Written at
    #: init and never updated — a hash recomputed after the run would describe the wrong thing.
    #: `aet.core.hashing.assert_comparable` refuses to aggregate runs whose values disagree, which
    #: is what makes this a control rather than a field somebody might read.
    input_hashes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None or k in (
            "target", "model", "dtype", "substrate"
        )}

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        data = load_yaml(path)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def dump(self, path: Path) -> None:
        dump_yaml(self.to_dict(), path)

    @classmethod
    def create(cls, spec, run_id: str, git_hash: str) -> "RunManifest":
        now = datetime.now(tz=timezone.utc).isoformat()
        obs = {
            "tracking_mode": spec.tracking_mode,
            "mlflow": {
                "enabled": spec.tracking_mode != "local",
                "tracking_uri": spec.mlflow_tracking_uri,
                "experiment_name": spec.experiment_name,
                "run_id": None,
            },
            "opentelemetry": {
                "enabled": spec.tracking_mode in ("full", "debug"),
                "endpoint": spec.otel_endpoint,
                "service_name": "aet",
                "trace_id": None,
            },
            "signoz": {
                "enabled": False,
                "endpoint": None,
            },
            "capture_policy": {
                "capture_prompts": True,
                "capture_outputs": True,
                "capture_tool_results": True,
                "redact_secrets": True,
                "store_raw_content": "local_only",
            },
        }
        return cls(
            project=spec.project,
            suite=spec.suite,
            method=spec.method,
            seed=spec.seed,
            run_id=run_id,
            target=spec.target,
            model=spec.model,
            dtype=spec.dtype,
            substrate=spec.substrate,
            git_hash_at_init=git_hash,
            is_smoke_test=spec.is_smoke_test,
            budget=spec.budget,
            promotion_flag=spec.promotion_flag,
            created_at=now,
            status="initialized",
            input_hashes=dict(getattr(spec, "input_hashes", None) or {}),
            tracking={"execution_backend": spec.execution},
            execution={"backend": spec.execution},
            observability=obs,
        )
