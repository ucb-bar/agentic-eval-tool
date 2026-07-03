import json
from datetime import datetime, timezone
from pathlib import Path

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def log_artifact_entry(artifacts_jsonl: Path, path: Path, artifact_path: str | None = None) -> None:
    p = Path(path)
    record = {
        "ts": _now(),
        "path": str(p),
        "artifact_path": artifact_path,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
    }
    with open(artifacts_jsonl, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
