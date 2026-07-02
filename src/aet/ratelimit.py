"""Rate-limit detection + wake-timing for unattended agent runs (project-agnostic).

The ``claude`` CLI, when a usage window is exhausted, emits a ``rate_limit_event`` whose
``rate_limit_info`` has ``status == "rejected"`` and a ``rateLimitType`` (``"five_hour"`` — the
short window that refreshes on a rolling clock — or a weekly/``"seven_day"`` window), plus a
``resetsAt`` epoch. The invocation does **zero tool work** when it is rejected at the door. It can
also surface as a terminal ``result`` with ``is_error`` and a "session limit"/"usage limit" message.

This module turns that raw signal into a :class:`RateLimitState` and computes *when to wake*:

  * the **five-hour** limit is recoverable soon → wait to ``resetsAt`` (or poll every ~20 min up to a
    5h20m cap when the epoch is missing/stale) and resume the same session;
  * the **weekly** limit (or exhausting the poll cap / a wait budget) is not → the caller should leave
    a resumable note and stop honestly.

Generalises the oscar-merlin capsule-bench ``_ratelimit.py`` (which only knew ``five_hour``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FIVE_HOUR_CAP_S = 5 * 3600 + 20 * 60   # poll at most 5h20m — one full five-hour window + a margin
DEFAULT_POLL_S = 20 * 60               # 20 minutes: resume within ≤20 min of an unknown reset
WAKE_JITTER_S = 30                     # wake a touch AFTER the reset so the window is definitely fresh


def _is_weekly(limit_type: str) -> bool:
    lt = (limit_type or "").lower()
    return "week" in lt or "seven" in lt or lt == "7d"


@dataclass
class RateLimitState:
    """The actionable rate-limit verdict for one invocation's transcript."""

    rejected: bool = False           # rejected at the door with no tool work → this attempt was burned
    saw_rejection: bool = False      # a rejection event appeared at all (even if some work happened)
    limit_type: str = ""             # "five_hour" | "weekly" | "seven_day" | ...
    resets_at: int | None = None     # epoch seconds from the rejected event's resetsAt

    @property
    def is_five_hour(self) -> bool:
        return self.limit_type.lower() == "five_hour"

    @property
    def is_weekly(self) -> bool:
        return _is_weekly(self.limit_type)


def _iter_events(lines: Iterable) -> "Iterable[dict]":
    for item in lines:
        if isinstance(item, dict):
            yield item
            continue
        s = str(item).strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except Exception:
            continue


def parse_rate_limit(events: Iterable) -> RateLimitState:
    """Derive a :class:`RateLimitState` from an iterable of stream-json events (dicts or json lines)."""
    st = RateLimitState()
    tool_uses = 0
    for e in _iter_events(events):
        t = e.get("type")
        if t == "rate_limit_event":
            ri = e.get("rate_limit_info", {}) or {}
            if ri.get("status") == "rejected":
                st.saw_rejection = True
                st.limit_type = str(ri.get("rateLimitType") or st.limit_type or "")
                ra = ri.get("resetsAt")
                if isinstance(ra, (int, float)):
                    st.resets_at = int(ra)
        elif t == "result":
            msg = str(e.get("result", "")).lower()
            if e.get("is_error") and ("session limit" in msg or "usage limit" in msg):
                st.saw_rejection = True
                if not st.limit_type:
                    st.limit_type = "five_hour"    # the common terminal-error form is the short window
        elif t == "assistant":
            for b in (e.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_uses += 1
    st.rejected = st.saw_rejection and tool_uses == 0
    return st


def rate_limit_from_transcript(path: str | Path) -> RateLimitState:
    """Parse a transcript file on disk into a :class:`RateLimitState`."""
    p = Path(path)
    if not p.exists():
        return RateLimitState()
    return parse_rate_limit(p.read_text(errors="ignore").splitlines())


def seconds_until_reset(state: RateLimitState, now: float, *, jitter: float = WAKE_JITTER_S) -> float | None:
    """Seconds to sleep to reach ``resets_at`` (+ jitter), or None when the epoch is missing/stale."""
    if state.resets_at is None:
        return None
    delta = (state.resets_at + jitter) - now
    if delta <= 0:
        return None                # the reset time is already in the past → poll instead
    return delta
