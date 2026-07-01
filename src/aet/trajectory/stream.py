"""Live streaming — feed an in-flight ``stream-json`` transcript into a growing RunTrajectory.

The same data-model and builder used for batch imports drive the live view: each flush re-parses
the accumulated buffer (transcripts are small) and rebuilds a one-round trajectory, so there is
exactly one code path. Cost is authoritative only at the terminal ``result`` event — until then
the snapshot is ``provisional`` and cost is a list-price estimate.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from aet.tracking.claude_stream import parse_stream, parse_timestamped_stream
from aet.trajectory.build import append_round
from aet.trajectory.classify import ActivityClassifier
from aet.trajectory.model import RunTrajectory
from aet.trajectory.pricing import PriceTable


def _extract_ts(line: str) -> float | None:
    try:
        iso = json.loads(line).get("timestamp")
        if iso:
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    return None


def _saw_result(lines: list[str]) -> bool:
    for line in lines:
        try:
            if json.loads(line).get("type") == "result":
                return True
        except Exception:
            continue
    return False


def _clean_timeline(lines: list[str], fallback: list[float]) -> list[float]:
    """A monotone timeline for the buffered lines. Uses embedded ISO ``timestamp``s with
    carry-forward (lead-in lines borrow the first real one) so epoch seconds are never mixed with
    a synthetic clock; only when NO line carries a timestamp do we fall back to feed-time ts."""
    embedded = [_extract_ts(ln) for ln in lines]
    if not any(t is not None for t in embedded):
        return fallback
    first_real = next(t for t in embedded if t is not None)
    out: list[float] = []
    last = first_real
    for t in embedded:
        if t is not None:
            last = t
        out.append(last)
    return out


class TrajectoryStream:
    """Accumulate transcript lines; produce a fresh :class:`RunTrajectory` on demand."""

    def __init__(self, *, classifier: ActivityClassifier | None = None,
                 price_table: PriceTable | None = None,
                 on_update: Callable[[RunTrajectory], None] | None = None,
                 flush_every: int = 1, run_id: str = "stream") -> None:
        self.classifier = classifier or ActivityClassifier()
        self.price_table = price_table or PriceTable()
        self.on_update = on_update
        self.flush_every = max(1, flush_every)
        self.run_id = run_id
        self._lines: list[str] = []
        self._feed_ts: list[float] = []
        self._pending = 0

    def feed_line(self, line: str, ts: float | None = None) -> None:
        line = line.strip()
        if not line:
            return
        self._lines.append(line)
        self._feed_ts.append(ts if ts is not None else time.monotonic())
        self._pending += 1
        if self._pending >= self.flush_every and self.on_update is not None:
            self._pending = 0
            self.on_update(self.snapshot())

    def snapshot(self) -> RunTrajectory:
        traj = RunTrajectory(run_id=self.run_id, source="stream",
                             classifier_config=self.classifier.config.to_dict())
        if not self._lines:
            traj.provisional = True
            return traj
        timeline = _clean_timeline(self._lines, self._feed_ts)
        base = timeline[0]
        rebased = [(t - base, line) for t, line in zip(timeline, self._lines)]
        try:
            result = parse_timestamped_stream(rebased)
        except Exception:
            result = parse_stream("\n".join(self._lines))
        append_round(traj, result, classifier=self.classifier, price_table=self.price_table)
        traj.provisional = not _saw_result(self._lines)
        return traj

    def attach_file(self, path: str | Path, *, poll_s: float = 0.5, follow: bool = True,
                    max_seconds: float | None = None) -> RunTrajectory:
        """Tail ``path``: read existing lines, then (if ``follow``) poll for appended ones until
        the terminal result event lands (or ``max_seconds`` elapses)."""
        path = Path(path)
        start = time.monotonic()
        buf = ""
        with open(path) as f:
            while True:
                chunk = f.readline()
                if chunk:
                    buf += chunk
                    if buf.endswith("\n"):
                        self.feed_line(buf)
                        buf = ""
                    continue
                if not follow or _saw_result(self._lines):
                    break
                if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                    break
                time.sleep(poll_s)
        if buf.strip():
            self.feed_line(buf)
        return self.snapshot()
