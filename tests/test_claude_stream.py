"""Tests for the Claude Code stream-json parser."""
import json
from aet.tracking.claude_stream import parse_stream


def _make_stream(*events) -> str:
    return "\n".join(json.dumps(e) for e in events)


_FULL_STREAM = _make_stream(
    {"type": "system", "subtype": "init", "session_id": "sess-abc", "tools": ["Bash", "Write"]},
    {"type": "assistant", "message": {
        "id": "msg1", "type": "message", "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll write the file."},
            {"type": "tool_use", "id": "tc1", "name": "Write",
             "input": {"file_path": "out.md", "content": "# Hello"}},
        ],
        "model": "claude-opus-4-5", "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 30,
                  "cache_creation_input_tokens": 512, "cache_read_input_tokens": 256}}},
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tc1",
         "content": [{"type": "text", "text": "File written successfully."}]}]}},
    {"type": "assistant", "message": {
        "id": "msg2", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
        "model": "claude-opus-4-5", "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 5,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 512}}},
    {"type": "result", "subtype": "success", "cost_usd": 0.0123,
     "duration_ms": 5000, "duration_api_ms": 4200,
     "num_turns": 2, "result": "Done.", "session_id": "sess-abc"},
)


def test_parse_stream_success_flag():
    r = parse_stream(_FULL_STREAM)
    assert r.success is True


def test_parse_stream_result_text():
    r = parse_stream(_FULL_STREAM)
    assert r.result_text == "Done."


def test_parse_stream_cost():
    r = parse_stream(_FULL_STREAM)
    assert abs(r.cost_usd - 0.0123) < 1e-9


def test_parse_stream_num_turns():
    r = parse_stream(_FULL_STREAM)
    assert r.num_turns == 2


def test_parse_stream_duration():
    r = parse_stream(_FULL_STREAM)
    assert r.duration_ms == 5000
    assert r.duration_api_ms == 4200


def test_parse_stream_session_id():
    r = parse_stream(_FULL_STREAM)
    assert r.session_id == "sess-abc"


def test_parse_stream_model():
    r = parse_stream(_FULL_STREAM)
    assert r.model == "claude-opus-4-5"


def test_parse_stream_tool_calls():
    r = parse_stream(_FULL_STREAM)
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.name == "Write"
    assert tc.tool_use_id == "tc1"
    assert "out.md" in tc.input_summary()
    assert "File written" in (tc.result or "")
    assert tc.is_error is False


def test_parse_stream_turn_usage():
    r = parse_stream(_FULL_STREAM)
    assert len(r.turn_usage) == 2
    t1 = r.turn_usage[0]
    assert t1.input_tokens == 100
    assert t1.output_tokens == 30
    assert t1.cache_creation_input_tokens == 512
    assert t1.cache_read_input_tokens == 256
    assert t1.total_input_tokens == 100 + 512 + 256  # per Anthropic semconv


def test_parse_stream_total_input_tokens():
    r = parse_stream(_FULL_STREAM)
    # turn1: 100+512+256=868  turn2: 200+0+512=712  total=1580
    assert r.total_input_tokens == 868 + 712


def test_parse_stream_total_output_tokens():
    r = parse_stream(_FULL_STREAM)
    assert r.total_output_tokens == 30 + 5


def test_parse_stream_cache_tokens():
    r = parse_stream(_FULL_STREAM)
    assert r.total_cache_creation_tokens == 512
    assert r.total_cache_read_tokens == 256 + 512


def test_parse_stream_unique_tools():
    r = parse_stream(_FULL_STREAM)
    assert r.unique_tools_used == ["Write"]


def test_parse_stream_tool_call_count():
    r = parse_stream(_FULL_STREAM)
    assert r.tool_call_count == 1
    assert r.tool_error_count == 0


def test_parse_stream_tool_error():
    stream = _make_stream(
        {"type": "assistant", "message": {
            "id": "m1", "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "bad cmd"}}],
            "model": "claude-opus-4-5", "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
             "content": [{"type": "text", "text": "command not found"}]}]}},
        {"type": "result", "subtype": "error", "cost_usd": 0.0,
         "duration_ms": 100, "num_turns": 1, "result": ""},
    )
    r = parse_stream(stream)
    assert r.tool_error_count == 1
    assert r.tool_calls[0].is_error is True


def test_parse_stream_empty():
    r = parse_stream("")
    assert r.success is False
    assert r.result_text == ""
    assert r.num_turns == 0
    assert r.tool_calls == []


def test_parse_stream_ignores_garbage_lines():
    stream = "not json\n" + _FULL_STREAM + "\nalso not json"
    r = parse_stream(stream)
    assert r.success is True
    assert r.num_turns == 2


def test_parse_stream_finish_reasons():
    r = parse_stream(_FULL_STREAM)
    assert r.turn_usage[0].finish_reasons == ["tool_use"]
    assert r.turn_usage[1].finish_reasons == ["end_turn"]


def test_dry_run_stream_parses():
    """The synthetic stream in claude_code_eval.py must parse cleanly."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from claude_code_eval import _DRY_RUN_STREAM
    r = parse_stream(_DRY_RUN_STREAM)
    assert r.success is True
    assert r.num_turns == 2
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "Write"
    assert r.total_cache_creation_tokens == 512
