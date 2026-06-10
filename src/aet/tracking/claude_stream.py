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
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    tool_use_id: str
    name: str
    input: dict
    result: str | None = None
    is_error: bool = False

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

    @property
    def total_input_tokens(self) -> int:
        """input_tokens per Anthropic semconv: raw + cache_read + cache_creation."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


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


def parse_stream(stream_text: str) -> ClaudeStreamResult:
    """Parse a complete --output-format stream-json capture into ClaudeStreamResult."""
    tool_calls_by_id: dict[str, ToolCall] = {}
    turn_usage: list[TurnUsage] = []
    result_text = ""
    cost_usd = 0.0
    num_turns = 0
    duration_ms = 0
    duration_api_ms = 0
    session_id = ""
    model = ""
    success = False
    turn_num = 0

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
            usage = msg.get("usage", {})
            turn_num += 1
            turn_usage.append(TurnUsage(
                turn=turn_num,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                model=msg.get("model", model),
                message_id=msg.get("id", ""),
                finish_reasons=[msg["stop_reason"]] if msg.get("stop_reason") else [],
            ))
            for block in msg.get("content", []):
                if block.get("type") == "tool_use":
                    tc = ToolCall(
                        tool_use_id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    )
                    if tc.tool_use_id:
                        tool_calls_by_id[tc.tool_use_id] = tc

        elif etype == "user":
            msg = event.get("message", {})
            for block in msg.get("content", []):
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
            success = event.get("subtype") == "success"
            result_text = event.get("result", "")
            cost_usd = float(event.get("cost_usd") or 0.0)
            num_turns = int(event.get("num_turns") or 0)
            duration_ms = int(event.get("duration_ms") or 0)
            duration_api_ms = int(event.get("duration_api_ms") or 0)
            session_id = session_id or str(event.get("session_id", ""))

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
    )
