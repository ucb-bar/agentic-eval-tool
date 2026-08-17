"""Live Codex trajectory recorder — fed timestamped JSONL lines one at a time.

The streaming counterpart to :func:`aet.trajectory.importers.codex.import_codex`. Chia (or any
harness) tees each ``codex exec --json`` stdout line to disk *and* into this recorder as it
arrives, with the wall offset at which the line was seen. The recorder wraps the same
:class:`~aet.trajectory.codex.CodexNormalizer` the batch importer uses — so a run killed mid-turn
and later re-imported from the raw file produces the same trajectory the live recorder had built,
minus the unfinished turn. One normalization code path, two entry points.

It is deliberately dependency-light and fail-open: feeding a malformed line never raises (it is
kept as an ``[UNPARSED]`` line), so a telemetry hiccup cannot crash the agent process.
"""
from __future__ import annotations

import json
from pathlib import Path

from aet.trajectory.classify import ActivityClassifier, ActivityConfig
from aet.trajectory.codex import CodexNormalizer, CodexRun
from aet.trajectory.importers.codex import build_trajectory_from_run
from aet.trajectory.model import RunTrajectory
from aet.trajectory.price_snapshot import PriceSnapshot


class CodexTrajectoryRecorder:
    """Accumulate a Codex stream live and materialize a :class:`RunTrajectory` on demand."""

    def __init__(self, *, run_id: str = "codex-live", model: str = "gpt-5-codex",
                 snapshot: PriceSnapshot | None = None,
                 classifier: ActivityClassifier | None = None,
                 billing_row: dict | None = None,
                 on_update=None) -> None:
        self.run_id = run_id
        self.model = model
        self.snapshot = snapshot or PriceSnapshot.default_openai()
        self.classifier = classifier or ActivityClassifier(ActivityConfig())
        self.classifier_cfg = self.classifier.config.to_dict()
        self.billing_row = billing_row or {"provider": "openai"}
        self._on_update = on_update
        self._norm = CodexNormalizer()

    # ------------------------------------------------------------------ feeding
    def feed_line(self, line: str, *, t_s: float | None = None) -> None:
        """Ingest one raw stdout line seen at wall offset ``t_s`` seconds. Never raises."""
        before = self._norm.run.normalized_event_count
        self._norm.feed_line(line, t_s=t_s)
        if self._on_update is not None and self._norm.run.normalized_event_count != before:
            try:
                self._on_update(self.trajectory())
            except Exception:
                pass  # fail-open: a plotting/callback error must not break the recorded run

    def feed_timestamped(self, record) -> None:
        """Ingest one *timestamped* record: either a ``(t_s, raw_line)`` pair or a mapping with a
        timestamp key (``t_s``/``ts``/``t``) plus a line key (``line``/``event``/``raw``).

        A mapping whose line value is itself a dict is re-serialized so the normalizer sees JSON.
        """
        t_s = None
        line = None
        if isinstance(record, (tuple, list)) and len(record) == 2:
            t_s, line = record
        elif isinstance(record, dict):
            for k in ("t_s", "ts", "t", "timestamp"):
                if k in record:
                    t_s = record[k]
                    break
            for k in ("line", "event", "raw"):
                if k in record:
                    line = record[k]
                    break
            if line is None:                       # the record *is* the event
                line = record
        else:
            line = record
        if not isinstance(line, str):
            line = json.dumps(line)
        try:
            t_s = None if t_s is None else float(t_s)
        except (TypeError, ValueError):
            t_s = None
        self.feed_line(line, t_s=t_s)

    def feed_file(self, path: str | Path) -> "CodexTrajectoryRecorder":
        """Replay a raw JSONL file line-by-line (equivalent to the batch importer, no timestamps)."""
        for ln in Path(path).read_text(errors="ignore").splitlines():
            self.feed_line(ln)
        return self

    # ------------------------------------------------------------------ readout
    @property
    def run(self) -> CodexRun:
        return self._norm.run

    def trajectory(self, *, calculated_at: str | None = None) -> RunTrajectory:
        """Build the trajectory from what has streamed so far.

        ``calculated_at`` defaults to empty (deterministic) so repeated calls are stable; pass a
        wall-clock string to stamp provenance for a finalized live run.
        """
        return build_trajectory_from_run(
            self._norm.run, run_id=self.run_id, classifier=self.classifier,
            classifier_cfg=self.classifier_cfg, model=self.model, snapshot=self.snapshot,
            billing_row=self.billing_row, calculated_at=(calculated_at or ""),
            source="stream:codex")
