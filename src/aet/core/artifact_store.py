"""Content-addressed artifact registry for benchmark runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class ArtifactOrigin(str, Enum):
    AGENT_WRITTEN = "agent_written"
    HARNESS_COPIED = "harness_copied"
    ORACLE_OUTPUT = "oracle_output"
    USER_PROVIDED = "user_provided"
    GENERATED = "generated"
    # Spec-required origins
    AUTHORED = "authored"
    COMPILER_GENERATED = "compiler_generated"
    TEST_GENERATED = "test_generated"
    MANUAL_PATCH = "manual_patch"
    ORACLE_GENERATED = "oracle_generated"
    PROTECTED_EVALUATOR = "protected_evaluator"


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
    # Spec-required fields
    kind: str = ""               # rtl | tb | log | trace | prompt | response | diff | synth | ppa
    created_at_iteration: int | None = None
    input_refs: list = field(default_factory=list)
    line_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "origin": self.origin.value if isinstance(self.origin, ArtifactOrigin) else self.origin,
            "size_bytes": self.size_bytes,
            "recorded_at": self.recorded_at,
            "protected": self.protected,
            "run_id": self.run_id,
            "tags": self.tags,
            "kind": self.kind,
            "created_at_iteration": self.created_at_iteration,
            "input_refs": self.input_refs,
            "line_count": self.line_count,
        }


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class ArtifactStore:
    """Per-run content-addressed artifact registry backed by artifact_manifest.json."""

    def __init__(self, run_dir: Path, run_id: str = "") -> None:
        self._run_dir = Path(run_dir)
        self._run_id = run_id
        self._records: list[ArtifactRecord] = []
        self._load()

    def _load(self) -> None:
        p = self.manifest_path
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for r in data.get("artifacts", []):
                    try:
                        origin = ArtifactOrigin(r.get("origin", "generated"))
                    except ValueError:
                        origin = ArtifactOrigin.GENERATED
                    self._records.append(ArtifactRecord(
                        path=r.get("path", ""),
                        sha256=r.get("sha256"),
                        origin=origin,
                        size_bytes=r.get("size_bytes"),
                        recorded_at=r.get("recorded_at", _now()),
                        protected=r.get("protected", False),
                        run_id=r.get("run_id", self._run_id),
                        tags=r.get("tags", {}),
                        kind=r.get("kind", ""),
                        created_at_iteration=r.get("created_at_iteration"),
                        input_refs=r.get("input_refs", []),
                        line_count=r.get("line_count"),
                    ))
            except Exception:
                pass

    def record(
        self,
        path,
        origin: ArtifactOrigin,
        protected: bool = False,
        tags: dict | None = None,
        kind: str = "",
        created_at_iteration: int | None = None,
        input_refs: list | None = None,
    ) -> ArtifactRecord:
        from aet.core.hashing import sha256_file
        p = Path(path)
        sha = sha256_file(p) if p.is_file() else None
        size = p.stat().st_size if p.is_file() else None
        line_count = None
        if p.is_file():
            try:
                line_count = sum(1 for _ in p.open("rb"))
            except Exception:
                pass
        rec = ArtifactRecord(
            path=str(p),
            sha256=sha,
            origin=origin,
            size_bytes=size,
            recorded_at=_now(),
            protected=protected,
            run_id=self._run_id,
            tags=tags or {},
            kind=kind,
            created_at_iteration=created_at_iteration,
            input_refs=input_refs or [],
            line_count=line_count,
        )
        self._records.append(rec)
        self._flush()
        return rec

    def _flush(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps({"artifacts": [r.to_dict() for r in self._records]}, indent=2)
        )

    def find_by_origin(self, origin: ArtifactOrigin) -> list[ArtifactRecord]:
        return [r for r in self._records if r.origin == origin]

    def find_by_sha256(self, sha256: str) -> list[ArtifactRecord]:
        return [r for r in self._records if r.sha256 == sha256]

    def find_by_kind(self, kind: str) -> list[ArtifactRecord]:
        return [r for r in self._records if r.kind == kind]

    def find_protected(self) -> list[ArtifactRecord]:
        return [r for r in self._records if r.protected]

    @property
    def manifest_path(self) -> Path:
        return self._run_dir / "artifact_manifest.json"

    def __len__(self) -> int:
        return len(self._records)
