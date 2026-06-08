from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from aet.core.run_spec import RunSpec

@dataclass
class RunPaths:
    project_root: Path
    run_path: Path

    @property
    def runs_root(self) -> Path:
        return self.project_root / "runs"

    @property
    def logs(self) -> Path:
        return self.run_path / "logs"

    @property
    def metrics(self) -> Path:
        return self.run_path / "metrics"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_path / "artifacts"

    @property
    def generated(self) -> Path:
        return self.run_path / "generated"

    @property
    def patches(self) -> Path:
        return self.run_path / "patches"

    @property
    def contracts(self) -> Path:
        return self.run_path / "contracts"

    @classmethod
    def from_spec(cls, spec: RunSpec, run_id: str) -> "RunPaths":
        run_path = spec.project_root / "runs" / spec.suite / run_id
        return cls(project_root=spec.project_root, run_path=run_path)

    @classmethod
    def from_run_dir(cls, run_dir: Path, project_root: Path) -> "RunPaths":
        return cls(project_root=project_root, run_path=run_dir)
