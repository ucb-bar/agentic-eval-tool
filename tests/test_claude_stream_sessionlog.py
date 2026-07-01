"""Session-log transcripts re-emit assistant messages; the parser must count each turn once."""
import json

from aet.tracking.claude_stream import parse_stream


def _asst(mid, cache_read):
    return {"type": "assistant", "message": {
        "id": mid, "role": "assistant", "model": "claude-opus-4-8", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 100, "output_tokens": 10,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": cache_read}}}


def test_duplicate_message_ids_counted_once():
    # same message id emitted 3× (session-log), identical usage; only the event uuid differs
    events = [{"type": "system", "session_id": "s"},
              _asst("msgA", 1000), _asst("msgA", 1000), _asst("msgA", 1000),
              _asst("msgB", 2000),
              {"type": "result", "subtype": "success", "total_cost_usd": 0.5,
               "duration_ms": 1000, "num_turns": 2, "result": "ok", "session_id": "s"}]
    r = parse_stream("\n".join(json.dumps(e) for e in events))
    assert len(r.turn_usage) == 2                 # not 4
    assert r.total_cache_read_tokens == 3000      # 1000 + 2000, each message once
    assert r.total_input_tokens == 200 + 3000     # raw input (2×100) + cache read


def test_unique_ids_unaffected():
    events = [_asst("m1", 100), _asst("m2", 200),
              {"type": "result", "subtype": "success", "cost_usd": 0.1, "num_turns": 2}]
    r = parse_stream("\n".join(json.dumps(e) for e in events))
    assert len(r.turn_usage) == 2
    assert r.total_cache_read_tokens == 300
