"""Parser for `claude --print --output-format stream-json` JSONL output.

Each line emitted by Claude Code is a JSON event.  We parse the full stream
into a structured ClaudeStreamResult that carries everything needed for
comprehensive OTel instrumentation:

  - per-turn token usage (input, output, cache_creation, cache_read)
  - per-tool-call name + input summary + result summary
  - total cost_usd, num_turns, duration_ms
  - session_id for replay, model name, finish reasons
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace


def _extract_file_paths(tool_name: str, input_dict: dict) -> list[str]:
    """Best-effort extraction of file paths touched by a tool call."""
    import re
    paths: list[str] = []
    if tool_name in ("Read", "Write", "Edit", "NotebookEdit", "MultiEdit"):
        if p := input_dict.get("file_path"):
            paths.append(str(p))
    elif tool_name == "Bash":
        cmd = input_dict.get("command", "")
        # explicit file args to common commands
        for m in re.findall(
            r'(?:^|\s)(?:cat|head|tail|less|wc|grep|sed|awk|cp|mv|rm|chmod|diff|sort|python3?)\s+((?:/[\w./\-]+))',
            cmd,
        ):
            paths.append(m)
        # redirect targets  > /path or >> /path
        for m in re.findall(r'>+\s*(/[\w./\-]+)', cmd):
            paths.append(m)
        # find . -name / -path style
        for m in re.findall(r'(?:find|ls)\s+(\/[\w./\-]+)', cmd):
            paths.append(m)
    return [p for p in dict.fromkeys(paths) if p]  # deduplicate, preserve order


@dataclass
class ToolCall:
    tool_use_id: str
    name: str
    input: dict
    result: str | None = None
    is_error: bool = False
    turn_index: int = 0
    duration_s: float = 0.0
    start_offset_s: float = 0.0   # seconds from stream start; used for OTel child span timestamps
    file_paths: list[str] = field(default_factory=list)
    is_mcp: bool = False
    output_size: int = 0
    reasoning_before: str = ""   # assistant text that immediately preceded this tool call

    def __post_init__(self) -> None:
        if not self.file_paths:
            self.file_paths = _extract_file_paths(self.name, self.input)
        if not self.is_mcp:
            self.is_mcp = self.name.startswith("mcp__")

    def input_summary(self, max_chars: int = 300) -> str:
        """Human-readable one-line summary of the tool input."""
        try:
            text = json.dumps(self.input, ensure_ascii=False)
        except Exception:
            text = str(self.input)
        return text[:max_chars] if len(text) > max_chars else text

    def result_summary(self, max_chars: int = 300) -> str:
        r = self.result or ""
        return r[:max_chars] if len(r) > max_chars else r


@dataclass
class TurnUsage:
    turn: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = ""
    message_id: str = ""
    finish_reasons: list[str] = field(default_factory=list)
    reasoning_text: str = ""   # assistant text blocks from this turn (no tool calls = planning turn)
    start_offset_s: float = 0.0  # seconds from stream start; used for OTel inference span timestamps
    has_thinking: bool = False   # this turn contained an extended-thinking content block

    @property
    def total_input_tokens(self) -> int:
        """input_tokens per Anthropic semconv: raw + cache_read + cache_creation."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class ModelUsage:
    """Per-model cost and token breakdown.

    Sourced either from the result event's ``modelUsage`` field (authoritative) or, for a
    truncated transcript, aggregated from per-turn :class:`TurnUsage` grouped by model
    (see :meth:`ClaudeStreamResult.per_model_usage`). ``activity_share`` is this model's fraction
    of the run's total billed tokens (0.0 until :meth:`~ClaudeStreamResult.per_model_usage` fills
    it in).
    """
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
    web_search_requests: int = 0
    activity_share: float = 0.0

    @property
    def total_billed_tokens(self) -> int:
        """Billed tokens for this model: raw input + output + cache read + cache creation."""
        return (self.input_tokens + self.output_tokens
                + self.cache_read_input_tokens + self.cache_creation_input_tokens)


@dataclass
class ClaudeStreamResult:
    success: bool
    result_text: str
    cost_usd: float
    num_turns: int
    duration_ms: int
    duration_api_ms: int
    session_id: str
    model: str
    tool_calls: list[ToolCall]
    turn_usage: list[TurnUsage]
    model_usage: list[ModelUsage] = field(default_factory=list)
    has_result_event: bool = False   # a terminal `result` event was seen (complete transcript)
    #: `per_token` or `subscription`. `cost_usd` is MONEY only in the first case; on a subscription
    #: seat the CLI still reports a dollar figure and it is what the tokens would have cost on the
    #: API. Defaults to `subscription`, the conservative direction: counting an unclassified run as
    #: metered would inflate reported spend and consume a budget cap no card is backing.
    billing_mode: str = "subscription"

    @property
    def total_input_tokens(self) -> int:
        return sum(t.total_input_tokens for t in self.turn_usage)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turn_usage)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(t.cache_creation_input_tokens for t in self.turn_usage)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(t.cache_read_input_tokens for t in self.turn_usage)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_error_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_error)

    @property
    def unique_tools_used(self) -> list[str]:
        seen: list[str] = []
        for tc in self.tool_calls:
            if tc.name not in seen:
                seen.append(tc.name)
        return seen

    def estimated_cost_usd(self, price_table=None) -> float | None:
        """List-price cost estimate summed over per-turn ``TurnUsage`` × ``PriceTable``, per model.

        This is the fallback for a **truncated/killed** transcript that never emitted a terminal
        ``result`` event, so the authoritative :attr:`cost_usd` (the CLI's ``total_cost_usd``) is
        absent and reads as ``0.0``. For a complete transcript prefer :attr:`cost_usd`; this is an
        ESTIMATE from list prices, not the billed number.

        Each turn is priced by its own model (falling back to the run's :attr:`model`). Returns
        ``None`` when *no* turn can be priced (the model is cost-unavailable) — never a fabricated
        ``$0``. When only some turns price, the unpriced ones contribute nothing (best-effort
        partial estimate). ``price_table`` defaults to :meth:`PriceTable.from_env`.
        """
        if price_table is None:
            from aet.trajectory.pricing import PriceTable
            price_table = PriceTable.from_env()
        total = 0.0
        any_priced = False
        for t in self.turn_usage:
            c = price_table.estimate_usd(
                t.input_tokens, t.output_tokens,
                t.cache_read_input_tokens, t.cache_creation_input_tokens,
                model=t.model or self.model,
            )
            if c is not None:
                total += c
                any_priced = True
        return total if any_priced else None

    def per_model_usage(self, price_table=None) -> list[ModelUsage]:
        """Per-model token/cost breakdown for this run, each carrying its ``activity_share``.

        The whole point is that a run mixing an orchestrator model with delegated sub-agent models
        (e.g. Opus orchestrator + Sonnet subagents + Haiku background) reports **all** of them
        distinctly, not just the run's primary model.

        Resolution:
          * when the authoritative result-event ``modelUsage`` is present (:attr:`model_usage`),
            prefer it verbatim — it is the billed split;
          * otherwise (a truncated/killed transcript that never emitted a ``result`` event, so
            :attr:`model_usage` is empty) fall back to aggregating the per-turn :class:`TurnUsage`
            grouped by :attr:`TurnUsage.model` (dedup is already handled by the parser) and price
            each model from its summed tokens via ``PriceTable`` — a list-price ESTIMATE. A
            cost-unavailable model keeps ``cost_usd = 0.0`` (never a fabricated price); its tokens
            and share are still reported.

        Every returned :class:`ModelUsage` gets ``activity_share`` = its fraction of the run's
        total billed tokens (0.0 when the run recorded no billed tokens). ``price_table`` defaults
        to :meth:`PriceTable.from_env`. Models appear in first-seen turn order (or ``modelUsage``
        iteration order).
        """
        if self.model_usage:
            items = [replace(mu) for mu in self.model_usage]
        else:
            if price_table is None:
                from aet.trajectory.pricing import PriceTable
                price_table = PriceTable.from_env()
            by_model: dict[str, ModelUsage] = {}
            order: list[str] = []
            for t in self.turn_usage:
                key = t.model or self.model
                mu = by_model.get(key)
                if mu is None:
                    mu = ModelUsage(
                        model=key, input_tokens=0, output_tokens=0,
                        cache_read_input_tokens=0, cache_creation_input_tokens=0, cost_usd=0.0,
                    )
                    by_model[key] = mu
                    order.append(key)
                mu.input_tokens += t.input_tokens
                mu.output_tokens += t.output_tokens
                mu.cache_read_input_tokens += t.cache_read_input_tokens
                mu.cache_creation_input_tokens += t.cache_creation_input_tokens
            for mu in by_model.values():
                c = price_table.estimate_usd(
                    mu.input_tokens, mu.output_tokens,
                    mu.cache_read_input_tokens, mu.cache_creation_input_tokens, model=mu.model,
                )
                if c is not None:
                    mu.cost_usd = c
            items = [by_model[k] for k in order]

        total_billed = sum(mu.total_billed_tokens for mu in items)
        for mu in items:
            mu.activity_share = (mu.total_billed_tokens / total_billed) if total_billed else 0.0
        return items


def _content_blocks(msg: dict) -> list[dict]:
    """The dict content-blocks of a message. Desktop/app session logs sometimes carry ``content``
    as a plain string (or a list mixing strings and dicts); this keeps only the dict blocks so
    callers can ``block.get(...)`` without a type check at every use site."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return []
    return [b for b in content if isinstance(b, dict)]


def parse_timestamped_stream(
    events: "list[tuple[float, str]]",
) -> "ClaudeStreamResult":
    """Parse a list of (wall_clock_s, json_line) tuples.

    Identical to parse_stream but computes tool-call duration_s from
    the wall-clock gap between each tool_use and its tool_result.
    Collect timestamped lines while reading the subprocess stdout:

        events = []
        for raw in proc.stdout:
            events.append((time.monotonic(), raw))
        result = parse_timestamped_stream(events)
    """
    # Build text + timestamp index: map line-index → timestamp
    lines_with_ts = [(ts, line.strip()) for ts, line in events if line.strip()]
    # We need timestamps per tool_use_id.  First pass: find which line index
    # each tool_use_id appears on so we can look up the timestamp.
    tool_use_ts: dict[str, float] = {}
    tool_result_ts: dict[str, float] = {}
    for ts, line in lines_with_ts:
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant":
            for block in _content_blocks(ev.get("message", {})):
                if block.get("type") == "tool_use":
                    tid = block.get("id", "")
                    if tid and tid not in tool_use_ts:   # first (of any duplicate) emission wins
                        tool_use_ts[tid] = ts
        elif ev.get("type") == "user":
            for block in _content_blocks(ev.get("message", {})):
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid and tid not in tool_result_ts:
                        tool_result_ts[tid] = ts

    text = "\n".join(line for _, line in lines_with_ts)
    result = parse_stream(text)

    # Patch in timing, turn_index, and start_offset_s
    base_ts = lines_with_ts[0][0] if lines_with_ts else 0.0
    stream_end_ts = lines_with_ts[-1][0] if lines_with_ts else 0.0
    # turn boundaries dedup by message id (first emission), to stay aligned with the deduped
    # turn_usage that parse_stream produces for re-emitting session-log transcripts.
    turn_boundaries: list[float] = []
    _seen_turn_ids: set[str] = set()
    for ts, line in lines_with_ts:
        if not _is_turn_boundary(line):
            continue
        try:
            mid = json.loads(line).get("message", {}).get("id", "")
        except Exception:
            mid = ""
        if mid and mid in _seen_turn_ids:
            continue
        if mid:
            _seen_turn_ids.add(mid)
        turn_boundaries.append(ts)

    # Patch start_offset_s onto each TurnUsage (assistant message start time)
    for i, tu in enumerate(result.turn_usage):
        if i < len(turn_boundaries):
            tu.start_offset_s = round(turn_boundaries[i] - base_ts, 3)

    for tc in result.tool_calls:
        t_use = tool_use_ts.get(tc.tool_use_id, 0.0)
        t_res = tool_result_ts.get(tc.tool_use_id, 0.0)
        if t_use and t_res and t_res > t_use:
            tc.duration_s = round(t_res - t_use, 3)
        tc.start_offset_s = round(t_use - base_ts, 3) if t_use else 0.0
        tc.turn_index = sum(1 for tb in turn_boundaries if tb <= t_use)

    # Store stream_duration_s on the result so callers can use it for span anchoring
    result._stream_duration_s = round(stream_end_ts - base_ts, 3)  # type: ignore[attr-defined]
    return result


def _is_turn_boundary(line: str) -> bool:
    """Return True if line is an assistant-message event (marks a new LLM turn)."""
    try:
        ev = json.loads(line)
        return ev.get("type") == "assistant"
    except Exception:
        return False


def parse_stream(stream_text: str) -> ClaudeStreamResult:
    """Parse a complete --output-format stream-json capture into ClaudeStreamResult."""
    tool_calls_by_id: dict[str, ToolCall] = {}
    turn_usage: list[TurnUsage] = []
    model_usage_list: list[ModelUsage] = []
    result_text = ""
    cost_usd = 0.0
    # Initialised here, not only inside the `result` branch: a transcript that ends without a
    # terminal result event (a killed run, a rate-limit cut) still has to produce a result object,
    # and a name bound only on the happy path would raise NameError on exactly those runs.
    billing_mode = "subscription"
    num_turns = 0
    duration_ms = 0
    duration_api_ms = 0
    session_id = ""
    model = ""
    success = False
    has_result_event = False
    turn_num = 0
    seen_msg_ids: set[str] = set()   # session-log transcripts re-emit a message id; count once

    for raw_line in stream_text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "system":
            session_id = session_id or str(event.get("session_id", ""))

        elif etype == "assistant":
            msg = event.get("message", {})
            if not model:
                model = msg.get("model", "")
            mid = msg.get("id", "")
            # A session-log transcript re-emits the same assistant message (same id, identical
            # usage, different event uuid). Count each turn — and its token usage — exactly once.
            is_dup = bool(mid) and mid in seen_msg_ids
            if mid:
                seen_msg_ids.add(mid)
            usage = msg.get("usage", {})
            blocks = _content_blocks(msg)
            # Collect text blocks first — used for both TurnUsage and ToolCall
            turn_text_blocks: list[str] = []
            for block in blocks:
                if block.get("type") == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        turn_text_blocks.append(t)
            reasoning = " | ".join(turn_text_blocks)[:600]
            has_thinking = any(b.get("type") == "thinking" for b in blocks)
            if not is_dup:
                turn_num += 1
                turn_usage.append(TurnUsage(
                    turn=turn_num,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    model=msg.get("model", model),
                    message_id=mid,
                    finish_reasons=[msg["stop_reason"]] if msg.get("stop_reason") else [],
                    reasoning_text=reasoning,
                    has_thinking=has_thinking,
                ))
            # tool_use blocks dedup by tool_use_id, so duplicate emissions collapse harmlessly
            for block in blocks:
                if block.get("type") == "tool_use":
                    tc = ToolCall(
                        tool_use_id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                        reasoning_before=reasoning,
                    )
                    if tc.tool_use_id and tc.tool_use_id not in tool_calls_by_id:
                        tool_calls_by_id[tc.tool_use_id] = tc

        elif etype == "user":
            msg = event.get("message", {})
            for block in _content_blocks(msg):
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid in tool_calls_by_id:
                        content = block.get("content", [])
                        if isinstance(content, list):
                            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                            tool_calls_by_id[tid].result = "\n".join(parts)[:500]
                        elif isinstance(content, str):
                            tool_calls_by_id[tid].result = content[:500]
                        tool_calls_by_id[tid].is_error = bool(block.get("is_error", False))

        elif etype == "result":
            has_result_event = True
            success = (event.get("subtype") == "success") and not event.get("is_error", False)
            result_text = event.get("result", "")
            # real CLI emits total_cost_usd; test fixtures use cost_usd
            #
            # NOT NECESSARILY MONEY. The CLI emits this field on a subscription seat too, where it
            # is what the tokens WOULD have cost on the API and no card is charged. `billing_mode`
            # records which kind this is so a downstream ledger can keep them apart; summing them
            # presents quota consumption as money, and a budget cap enforced against the sum is
            # enforced against a number partly nobody is paying. See aet.trajectory.billing.
            cost_usd = float(event.get("total_cost_usd") or event.get("cost_usd") or 0.0)
            from aet.trajectory.billing import billing_mode_of
            billing_mode = billing_mode_of(event)
            num_turns = int(event.get("num_turns") or 0)
            duration_ms = int(event.get("duration_ms") or 0)
            duration_api_ms = int(event.get("duration_api_ms") or 0)
            session_id = session_id or str(event.get("session_id", ""))
            # per-model cost/token breakdown (real CLI only)
            result_stop_reason = event.get("stop_reason", "")
            for m, mu in (event.get("modelUsage") or {}).items():
                model_usage_list.append(ModelUsage(
                    model=m,
                    input_tokens=int(mu.get("inputTokens") or 0),
                    output_tokens=int(mu.get("outputTokens") or 0),
                    cache_read_input_tokens=int(mu.get("cacheReadInputTokens") or 0),
                    cache_creation_input_tokens=int(mu.get("cacheCreationInputTokens") or 0),
                    cost_usd=float(mu.get("costUSD") or 0.0),
                    web_search_requests=int(mu.get("webSearchRequests") or 0),
                ))
            # patch finish_reason on the last turn if streaming left it null
            if result_stop_reason and turn_usage:
                last = turn_usage[-1]
                if not last.finish_reasons:
                    last.finish_reasons = [result_stop_reason]

    # A truncated/killed transcript never emitted a `result` event, so the CLI's authoritative
    # num_turns is absent (0). Fall back to the assistant turns actually seen so the run does not
    # report zero activity. (For a complete transcript the CLI value is authoritative — keep it.)
    if not has_result_event:
        num_turns = len(turn_usage)

    return ClaudeStreamResult(
        success=success,
        result_text=result_text,
        cost_usd=cost_usd,
        num_turns=num_turns,
        duration_ms=duration_ms,
        duration_api_ms=duration_api_ms,
        session_id=session_id,
        model=model,
        tool_calls=list(tool_calls_by_id.values()),
        turn_usage=turn_usage,
        model_usage=model_usage_list,
        billing_mode=billing_mode,
        has_result_event=has_result_event,
    )
