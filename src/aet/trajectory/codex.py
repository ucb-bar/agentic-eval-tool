"""Codex-CLI stdout JSONL — the lossless, structural normalizer shared by the batch importer
and the live recorder.

This is the Codex analogue of ``tracking/claude_stream.py``: it turns the ``codex exec --json``
event stream (verified against codex-cli 0.147.0) into typed records — per-turn token usage,
tool calls, file changes, agent/reasoning text — while preserving every byte losslessly.

**One event per line.** Each line is a JSON object whose ``type`` is a dotted name
(``thread.started`` / ``turn.completed`` / ``item.completed`` / ``turn.failed`` / ``error`` …).
Dispatch is **structural**: split ``type`` on ``"."`` into ``(domain, action)`` and route on the
tuple — never a regex, never a "line starts with" assumption (repo cardinal rule). An unknown
``type`` or ``item.type`` is kept verbatim in :attr:`CodexRun.unknown_events`, and a line that is
not JSON at all is kept verbatim in :attr:`CodexRun.unparsed_lines` (tagged ``[UNPARSED]``) — the
stream is never silently dropped, so a run killed mid-turn still reconciles.

**Token subset semantics (do NOT double count).** From ``turn.completed.usage``:
``cached_input_tokens`` ⊆ ``input_tokens``; ``cache_write_input_tokens`` (when present) is also an
input subset; ``reasoning_output_tokens`` ⊆ ``output_tokens``. An absent bucket stays ``None`` —
unknown is not zero. ``uncached_input = max(input - cache_read - cache_write, 0)``.

The normalizer is **streaming**: :meth:`CodexNormalizer.feed_line` takes one raw line (plus an
optional wall timestamp) and updates the in-progress :class:`CodexRun`. Feeding a whole file and
feeding it line-by-line yield byte-for-byte identical results — that single code path is what
makes the batch importer and the live recorder agree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Item types the 0.147.0 stream is known to emit. Unknown types are kept losslessly, never
# matched against this set to decide whether to drop them — it is documentation-of-intent only.
KNOWN_ITEM_TYPES = frozenset({
    "agent_message", "reasoning", "file_change", "command_execution",
    "mcp_tool_call", "web_search",
})

# Item types that are a tool span rather than model text (used for activity-band mapping).
_TOOL_ITEM_TYPES = frozenset({
    "command_execution", "file_change", "mcp_tool_call", "web_search",
})

UNPARSED_TAG = "[UNPARSED]"


def _as_int_or_none(v) -> int | None:
    """Coerce a usage bucket to int, preserving ``None`` (absent ≠ 0). Non-numeric → ``None``."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class CodexTurnUsage:
    """Token usage reported at one ``turn.completed`` event.

    Every bucket is nullable: a bucket the provider did not report stays ``None`` so a reader can
    tell "unknown" from "zero". ``cached_input_tokens`` / ``cache_write_input_tokens`` are subsets
    of ``input_tokens``; ``reasoning_output_tokens`` is a subset of ``output_tokens``.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    provider_reported: bool = True
    source_event: str = "turn.completed"
    event_index: int | None = None   # ordinal of the turn.completed line (for pseudo-time on import)
    t_s: float | None = None          # wall offset of the turn.completed line, when timestamped

    @property
    def uncached_input_tokens(self) -> int | None:
        """``max(input - cache_read - cache_write, 0)`` — the tokens billed at the full input rate.

        ``None`` when ``input_tokens`` itself is unknown (cannot be reconstructed). Absent cache
        buckets count as zero *here only* (they are genuine subsets, so an absent one removes
        nothing) — this is distinct from reporting them as zero elsewhere.
        """
        if self.input_tokens is None:
            return None
        cr = self.cached_input_tokens or 0
        cw = self.cache_write_input_tokens or 0
        return max(int(self.input_tokens) - int(cr) - int(cw), 0)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "provider_reported": self.provider_reported,
            "source_event": self.source_event,
            "event_index": self.event_index,
        }

    @classmethod
    def from_usage(cls, usage: dict, *, source_event: str = "turn.completed") -> "CodexTurnUsage":
        return cls(
            input_tokens=_as_int_or_none(usage.get("input_tokens")),
            cached_input_tokens=_as_int_or_none(usage.get("cached_input_tokens")),
            cache_write_input_tokens=_as_int_or_none(usage.get("cache_write_input_tokens")),
            output_tokens=_as_int_or_none(usage.get("output_tokens")),
            reasoning_output_tokens=_as_int_or_none(usage.get("reasoning_output_tokens")),
            provider_reported=True,
            source_event=source_event,
        )


@dataclass
class CodexToolCall:
    """One tool span (``command_execution`` / ``file_change`` / ``mcp_tool_call`` / ``web_search``).

    Built from structured item fields only — never by string-parsing the command. ``raw`` keeps the
    completed item verbatim so nothing is lost.
    """

    item_id: str
    kind: str
    status: str = ""
    command: str | None = None
    exit_code: int | None = None
    changes: list | None = None
    started_index: int | None = None   # event ordinal of item.started (for band timing)
    completed_index: int | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.exit_code is not None and self.exit_code != 0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id, "kind": self.kind, "status": self.status,
            "command": self.command, "exit_code": self.exit_code, "changes": self.changes,
            "t_start_s": self.t_start_s, "t_end_s": self.t_end_s,
        }


@dataclass
class CodexRun:
    """The normalized result of one Codex stream (one thread, possibly many turns)."""

    thread_id: str | None = None
    turns: list[CodexTurnUsage] = field(default_factory=list)
    tools: list[CodexToolCall] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    reasoning_texts: list[str] = field(default_factory=list)
    file_changes: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)          # turn.failed / error, lossless
    unknown_events: list[dict] = field(default_factory=list)   # unknown type / item.type, lossless
    unparsed_lines: list[str] = field(default_factory=list)    # non-JSON lines, verbatim
    raw_event_count: int = 0            # every non-blank input line
    normalized_event_count: int = 0     # every JSON line we dispatched (parsed OK)
    num_turns_started: int = 0

    # --------------------------------------------------------------- token roll-up
    def totals(self) -> dict:
        """Summed token buckets across turns. A bucket is ``None`` iff **no** turn reported it.

        Never fabricates zero for an all-absent bucket (unknown ≠ 0); a partially-absent bucket
        sums the turns that did report it (each absent turn contributes nothing, matching the
        subset semantics).
        """
        keys = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens")
        out: dict[str, int | None] = {}
        for k in keys:
            vals = [getattr(t, k) for t in self.turns]
            reported = [v for v in vals if v is not None]
            out[k] = sum(reported) if reported else None
        # uncached is derived, and only when input is known
        if out["input_tokens"] is None:
            out["uncached_input_tokens"] = None
        else:
            out["uncached_input_tokens"] = max(
                out["input_tokens"] - (out["cached_input_tokens"] or 0)
                - (out["cache_write_input_tokens"] or 0), 0)
        return out

    @property
    def final_output(self) -> str | None:
        """The last agent_message — the Codex analogue of ``--output-last-message``."""
        return self.agent_messages[-1] if self.agent_messages else None

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "turns": [t.to_dict() for t in self.turns],
            "tools": [t.to_dict() for t in self.tools],
            "agent_messages": list(self.agent_messages),
            "reasoning_texts": list(self.reasoning_texts),
            "file_changes": list(self.file_changes),
            "errors": list(self.errors),
            "unknown_events": list(self.unknown_events),
            "unparsed_lines": list(self.unparsed_lines),
            "totals": self.totals(),
            "raw_event_count": self.raw_event_count,
            "normalized_event_count": self.normalized_event_count,
            "num_turns_started": self.num_turns_started,
        }


class CodexNormalizer:
    """Streaming, structural, lossless normalizer for a single Codex JSONL stream.

    Feed it one raw line at a time (:meth:`feed_line`), optionally with a wall timestamp for band
    timing; read the accumulated :class:`CodexRun` at any point. There is exactly one code path, so
    a completed-file import and a live per-line recorder produce identical results.
    """

    def __init__(self) -> None:
        self.run = CodexRun()
        self._open_tools: dict[str, CodexToolCall] = {}   # item_id -> in-progress tool
        self._line_ordinal = 0

    # ------------------------------------------------------------------ public API
    def feed_line(self, line: str, *, t_s: float | None = None) -> None:
        """Ingest one raw stdout line. Never raises on malformed input."""
        raw = line.rstrip("\n")
        if not raw.strip():
            return
        self.run.raw_event_count += 1
        self._line_ordinal += 1
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # A non-JSON line (a torn write, a stray log line) is kept verbatim, tagged, not dropped.
            self.run.unparsed_lines.append(f"{UNPARSED_TAG} {raw}")
            return
        if not isinstance(event, dict):
            self.run.unparsed_lines.append(f"{UNPARSED_TAG} {raw}")
            return
        self.run.normalized_event_count += 1
        self._dispatch(event, t_s=t_s)

    def feed_text(self, text: str) -> "CodexNormalizer":
        for ln in text.splitlines():
            self.feed_line(ln)
        return self

    def result(self) -> CodexRun:
        return self.run

    # ------------------------------------------------------------------ dispatch (structural)
    def _dispatch(self, event: dict, *, t_s: float | None) -> None:
        etype = str(event.get("type", ""))
        domain, _, action = etype.partition(".")   # structural split, no regex
        if domain == "thread":
            if action == "started":
                self.run.thread_id = event.get("thread_id") or self.run.thread_id
            else:
                self._keep_unknown(event)
        elif domain == "turn":
            self._on_turn(action, event, t_s=t_s)
        elif domain == "item":
            self._on_item(action, event, t_s=t_s)
        elif domain == "error" or etype == "error":
            self.run.errors.append(dict(event))
        else:
            self._keep_unknown(event)

    def _on_turn(self, action: str, event: dict, *, t_s: float | None = None) -> None:
        if action == "started":
            self.run.num_turns_started += 1
        elif action == "completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                tu = CodexTurnUsage.from_usage(usage)
            else:
                # a turn.completed with no usage block is still a real turn — record an all-null one
                tu = CodexTurnUsage(provider_reported=False)
            tu.event_index = self._line_ordinal
            tu.t_s = t_s
            self.run.turns.append(tu)
        elif action == "failed":
            self.run.errors.append(dict(event))
        else:
            self._keep_unknown(event)

    def _on_item(self, action: str, event: dict, *, t_s: float | None) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            self._keep_unknown(event)
            return
        itype = str(item.get("type", ""))
        item_id = str(item.get("id", "")) or f"_anon_{self._line_ordinal}"

        if itype not in KNOWN_ITEM_TYPES:
            # keep the whole event losslessly, but do not crash / drop
            self._keep_unknown(event)
            return

        if itype == "agent_message":
            if action == "completed":
                self.run.agent_messages.append(item.get("text", "") or "")
            return
        if itype == "reasoning":
            if action == "completed":
                self.run.reasoning_texts.append(item.get("text", "") or "")
            return

        # tool-like items (command_execution / file_change / mcp_tool_call / web_search)
        if action == "started":
            tc = CodexToolCall(
                item_id=item_id, kind=itype, status=item.get("status", "in_progress"),
                command=item.get("command"), exit_code=item.get("exit_code"),
                changes=item.get("changes"), started_index=self._line_ordinal,
                t_start_s=t_s, raw=dict(item))
            self._open_tools[item_id] = tc
        elif action == "completed":
            tc = self._open_tools.pop(item_id, None)
            if tc is None:
                # completed with no observed start (mid-stream attach / one-shot) — synthesize
                tc = CodexToolCall(item_id=item_id, kind=itype, started_index=self._line_ordinal,
                                   t_start_s=t_s)
            tc.status = item.get("status", "completed")
            tc.command = item.get("command", tc.command)
            tc.exit_code = item.get("exit_code", tc.exit_code)
            tc.changes = item.get("changes", tc.changes)
            tc.completed_index = self._line_ordinal
            tc.t_end_s = t_s
            tc.raw = dict(item)
            self.run.tools.append(tc)
            if itype == "file_change":
                for ch in (item.get("changes") or []):
                    self.run.file_changes.append(dict(ch))
        else:
            self._keep_unknown(event)

    def _keep_unknown(self, event: dict) -> None:
        self.run.unknown_events.append(dict(event))


def normalize_text(text: str) -> CodexRun:
    """Convenience: normalize a whole JSONL blob in one call (batch path)."""
    return CodexNormalizer().feed_text(text).result()
