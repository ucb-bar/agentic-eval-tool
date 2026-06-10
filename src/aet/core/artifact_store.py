"""Artifact store with content-hashing and origin tracking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from aet.core.hashing import sha256_file


class ArtifactOrigin(str, Enum):
    AGENT_WRITTEN  = "agent_written"
    HARNESS_COPIED = "harness_copied"
    ORACLE_OUTPUT  = "oracle_output"
    USER_PROVIDED  = "user_provided"
    GENERATED      = "generated"


@dataclass
class ArtifactRecord:
    path: str
    sha256: str | None
    origin: ArtifactOrigin
    size_bytes: int | None
    recorded_at: str
    protected: bool = False
    run_id: str = ""
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"] = self.origin.value
        return d


class ArtifactStore:
    """Per-run artifact registry backed by artifact_manifest.json."""

    def __init__(self, run_dir: Path, run_id: str = "") -> None:
        self._run_dir = Path(run_dir)
        self._run_id = run_id
        self._manifest_path = self._run_dir / "artifact_manifest.json"
        self._records: list[ArtifactRecord] = []
        if self._manifest_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._manifest_path.read_text())
            for entry in data.get("artifacts", []):
                entry = dict(entry)
                entry["origin"] = ArtifactOrigin(entry.get("origin", "harness_copied"))
                self._records.append(ArtifactRecord(**entry))
        except Exception:
            self._records = []

    def record(
        self,
        path: Path,
        origin: ArtifactOrigin,
        protected: bool = False,
        tags: dict | None = None,
    ) -> ArtifactRecord:
        """Hash path, create an ArtifactRecord, and append to artifact_manifest.json."""
        p = Path(path)
        sha = sha256_file(p)
        size = p.stat().st_size if p.exists() else None
        rec = ArtifactRecord(
            path=str(p),
            sha256=sha,
            origin=origin,
            size_bytes=size,
            recorded_at=datetime.now(tz=timezone.utc).isoformat(),
            protected=protected,
            run_id=self._run_id,
            tags=tags or {},
        )
        self._records.append(rec)
        self._flush()
        return rec

    def _flush(self) -> None:
        self._manifest_path.write_text(
            json.dumps({
                "schema_version": "1.0",
                "run_id": self._run_id,
                "artifacts": [r.to_dict() for r in self._records],
            }, indent=2)
        )

    def find_by_origin(self, origin: ArtifactOrigin) -> list[ArtifactRecord]:
        return [r for r in self._records if r.origin == origin]

    def find_by_sha256(self, sha256: str) -> list[ArtifactRecord]:
        return [r for r in self._records if r.sha256 == sha256]

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def __len__(self) -> int:
        return len(self._records)
