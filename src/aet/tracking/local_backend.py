"""Local file backend — always active, pure stdlib."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aet.tracking.types import TrackingConfig

# Library diagnostics go through logging (silent by default; apps opt in with logging.basicConfig).
# The warning is ALSO persisted to logs/tracking_warnings.jsonl for the run record.
log = logging.getLogger("aet.tracking")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


class LocalBackend:
    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        logs_dir = config.run_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self._events_path = logs_dir / "events.jsonl"
        self._params_path = logs_dir / "params.json"
        self._metrics_path = logs_dir / "metrics.jsonl"
        self._artifacts_path = logs_dir / "artifacts.jsonl"
        self._warnings_path = logs_dir / "tracking_warnings.jsonl"
        self._seq: int = 0

        # Initialise params.json if absent
        if not self._params_path.exists():
            self._params_path.write_text("{}\n")

    # ------------------------------------------------------------------
    def warn(self, message: str) -> None:
        log.warning(message)
        _append_jsonl(self._warnings_path, {"ts": _now(), "message": message})

    # ------------------------------------------------------------------
    def log_param(self, name: str, value: Any) -> None:
        try:
            params = json.loads(self._params_path.read_text())
        except Exception:
            params = {}
        params[name] = value
        self._params_path.write_text(json.dumps(params, indent=2, default=str) + "\n")

    def log_params(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    # ------------------------------------------------------------------
    def log_metric(
        self,
        name: str,
        value: int | float | bool | None,
        step: int | None = None,
        source: str | None = None,
    ) -> None:
        _append_jsonl(
            self._metrics_path,
            {
                "ts": _now(),
                "run_id": self._config.run_id,
                "name": name,
                "value": value,
                "step": step,
                "source": source,
            },
        )

    def log_metrics(self, metrics: dict[str, Any], prefix: str | None = None) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)) or v is None:
                name = f"{prefix}.{k}" if prefix else k
                self.log_metric(name, v)

    # ------------------------------------------------------------------
    def _log_event_rich(
        self,
        name: str,
        payload: dict | None = None,
        *,
        stage: str | None = None,
        actor: str | None = None,
        input_refs: list | None = None,
        output_refs: list | None = None,
    ) -> str:
        """Write enriched event to events.jsonl. Returns auto-assigned event_id."""
        import uuid
        self._seq += 1
        record: dict = {
            "ts": _now(),
            "run_id": self._config.run_id,
            "project": self._config.project,
            "suite": self._config.suite,
            "target": self._config.target,
            "method": self._config.method,
            "seed": self._config.seed,
            "event": name,
            "event_id": uuid.uuid4().hex,
            "sequence": self._seq,
            "payload": payload or {},
        }
        if stage is not None:
            record["stage"] = stage
        if actor is not None:
            record["actor"] = actor
        if input_refs:
            record["input_refs"] = input_refs
        if output_refs:
            record["output_refs"] = output_refs
        _append_jsonl(self._events_path, record)
        return record["event_id"]

    def log_event(self, name: str, payload: dict | None = None) -> None:
        self._log_event_rich(name, payload)

    # ------------------------------------------------------------------
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        p = Path(path)
        size = p.stat().st_size if p.exists() else None
        _append_jsonl(
            self._artifacts_path,
            {
                "ts": _now(),
                "path": str(p),
                "artifact_path": artifact_path,
                "exists": p.exists(),
                "size_bytes": size,
            },
        )

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        p = Path(path)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file():
                    self.log_artifact(child, artifact_path)
        elif p.is_file():
            self.log_artifact(p, artifact_path)
